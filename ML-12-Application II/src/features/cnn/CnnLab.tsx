import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { useCameraState } from '../../engine3d/Stage';
import { LabShell, Metrics } from '../../components/LabShell';
import { VisionInput } from '../vision/VisionInput';
import { Segments } from '../../engine3d/plots';
import { BOX, commitInstances, Instances, Label, setInstance, SPHERE } from '../../engine3d/primitives';
import { CLASS_COLORS, NEG_W, POS_W } from '../../engine3d/palette';
import { CNN, CnnTrainer, DEFAULT_CNN, cnnAccuracy, cnnShapes, loadDigits, type CnnLayer, type CnnTrace } from '../../ml/cnn';
import { DIGIT_LABELS, DIGIT_PIXELS } from '../../data/digits';
import pretrained from '../../data/cnnPretrained.json';
import { trainTestSplit } from '../../data/datasets';
import { classificationReport } from '../../ml/metrics';
import { Rng } from '../../lib/random';
import { useT } from '../../i18n/i18n';
import { Button, FloatingPanel, Icon, Section, Segmented, Select, Slider, Toggle, fmt, pct } from '../../components/ui';
import { ConfusionMatrix, LinePlot, Quiz, Tex } from '../../components/common';

type V3 = [number, number, number];
const INPUT = { c: 1, h: 16, w: 16 };
let DIGITS: ReturnType<typeof prepDigits> | null = null;
function prepDigits() {
  const { X, y } = loadDigits(DIGIT_PIXELS, DIGIT_LABELS);
  const sp = trainTestSplit(y, 0.2, 42);
  return { Xtr: sp.train.map((i) => X[i]), ytr: sp.train.map((i) => y[i]), Xte: sp.test.map((i) => X[i]), yte: sp.test.map((i) => y[i]) };
}
const sameArch = (a: CnnLayer[], b: CnnLayer[]) => JSON.stringify(a) === JSON.stringify(b);
const CLASSIC: { name: string; k: number[][] }[] = [
  { name: 'Sobel x', k: [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]] },
  { name: 'Sobel y', k: [[-1, -2, -1], [0, 0, 0], [1, 2, 1]] },
  { name: 'Laplacian', k: [[0, 1, 0], [1, -4, 1], [0, 1, 0]] },
  { name: 'Blur', k: [[1, 2, 1], [2, 4, 2], [1, 2, 1]].map((r) => r.map((v) => v / 16)) },
];

function Heat({ data, h, w, size = 96, signed }: { data: ArrayLike<number>; h: number; w: number; size?: number; signed?: boolean }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current?.getContext('2d');
    if (!c) return;
    const img = c.createImageData(w, h);
    let mx = 1e-9;
    for (let i = 0; i < h * w; i++) mx = Math.max(mx, Math.abs(data[i]));
    for (let i = 0; i < h * w; i++) {
      const v = data[i] / mx;
      const [r, g, b] = signed ? (v >= 0 ? [59 + 196 * v, 130 + 125 * v, 246] : [239, 68 + 187 * (1 + v), 68 + 187 * (1 + v)]) : [v * 0 + 7 + 248 * Math.max(0, v), 11 + 206 * Math.max(0, v), 20 + 235 * Math.max(0, v)];
      img.data.set([r, g, b, 255], i * 4);
    }
    c.putImageData(img, 0, 0);
  }, [data, h, w, signed]);
  return <canvas ref={ref} width={w} height={h} style={{ width: size, height: (size * h) / w, imageRendering: 'pixelated' }} className="rounded border border-line" />;
}

export default function CnnLab() {
  const t = useT();
  if (!DIGITS) DIGITS = prepDigits();
  const D = DIGITS;
  const [layers, setLayers] = useState<CnnLayer[]>(DEFAULT_CNN);
  const [train, setTrain] = useState({ lr: 0.003, batch: 16, optimizer: 'adam' as 'adam' | 'sgd' | 'momentum', seed: 7, maxEpochs: 10 });
  const [source, setSource] = useState<'pretrained' | 'trained'>('pretrained');
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [tick, setTick] = useState(0);
  const [sampleIdx, setSampleIdx] = useState(3);
  const [custom, setCustom] = useState<Float32Array | null>(null);
  const [selLayer, setSelLayer] = useState(0);
  const [selCh, setSelCh] = useState(0);
  const [scan, setScan] = useState(0);
  const [scanOn, setScanOn] = useState(true);
  const [inCh, setInCh] = useState(0);
  const [showDense, setShowDense] = useState(true);
  const [cam, setCam] = useCameraState();
  const v = useMemo(() => cnnShapes(layers, INPUT), [layers]);
  const valid = v.errors.length === 0;

  // Model creation is pure: source/seed changes create exactly one new session.
  const [resetNonce, setResetNonce] = useState(0);
  const [pendingStep, setPendingStep] = useState(false);
  const [trainError, setTrainError] = useState('');
  const model = useMemo(() => {
    if (!valid) return null;
    const src = source === 'pretrained' && sameArch(layers, DEFAULT_CNN) ? 'pretrained' : 'trained';
    const net = new CNN(layers, INPUT, train.seed);
    if (src === 'pretrained') {
      (pretrained.params as ({ W: number[]; b: number[] } | null)[]).forEach((p, i) => { if (p && net.params[i]) { net.params[i]!.W.set(p.W); net.params[i]!.b.set(p.b); } });
    }
    const optCfg = { name: train.optimizer, lr: train.lr, momentum: 0.9, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 };
    return { net, trainer: src === 'trained' ? new CnnTrainer(net, D.Xtr, D.ytr, optCfg, train.batch, train.seed) : null, history: [] as { epoch: number; testAcc: number; loss: number }[] };
  }, [layers, valid, source, train.seed, resetNonce, D]); // eslint-disable-line react-hooks/exhaustive-deps
  const effSource = model?.trainer ? 'trained' : 'pretrained';
  const advance = () => {
    const tr = model?.trainer;
    if (!tr || tr.epoch >= train.maxEpochs) return;
    const epoch = tr.epoch; tr.trainBatch();
    if (tr.epoch !== epoch) model!.history.push({ ...tr.history[tr.history.length - 1], testAcc: cnnAccuracy(model!.net, D.Xte, D.yte).acc });
  };
  useEffect(() => { setTrainError(''); }, [model]);
  useEffect(() => {
    if (!pendingStep || !model?.trainer) return;
    try { advance(); } catch (error) { setTrainError((error as Error).message); }
    setPendingStep(false); setTick((x) => x + 1);
  }, [pendingStep, model]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => {
    if (!running || !model?.trainer) return;
    let raf = 0, last = 0;
    const loop = (now: number) => {
      try {
        const tr = model.trainer!;
        if (tr.epoch >= train.maxEpochs) { setRunning(false); return; }
        const t0 = performance.now(), ep = tr.epoch;
        do { advance(); } while (performance.now() - t0 < Math.min(30, 8 * speed) && tr.epoch === ep);
        if (tr.epoch >= train.maxEpochs) { setRunning(false); setTick((x) => x + 1); return; }
        if (now - last > 150) { last = now; setTick((x) => x + 1); }
        raf = requestAnimationFrame(loop);
      } catch (error) { setTrainError((error as Error).message); setRunning(false); setTick((x) => x + 1); }
    };
    raf = requestAnimationFrame(loop); return () => cancelAnimationFrame(raf);
  }, [running, model, speed, train.maxEpochs, D]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { if (model?.trainer) { model.trainer.opt.cfg = { ...model.trainer.opt.cfg, lr: train.lr, name: train.optimizer }; model.trainer.batch = train.batch; } }, [train, model]);

  const x = custom ?? D.Xte[Math.min(sampleIdx, D.Xte.length - 1)];
  const label = custom ? null : D.yte[Math.min(sampleIdx, D.yte.length - 1)];
  const trace: CnnTrace | null = useMemo(() => (model ? model.net.forward(x) : null), [model, x, tick]); // eslint-disable-line react-hooks/exhaustive-deps
  const probs = trace ? Array.from(trace.a[layers.length]) : [];
  const pred = probs.length ? probs.indexOf(Math.max(...probs)) : -1;
  const evalRep = useMemo(() => (model ? (() => { const r = cnnAccuracy(model.net, D.Xte, D.yte); return classificationReport(D.yte, r.pred, 10); })() : null), [model, tick, D]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---------- 3D layout: each tensor is a stack of sheets along x ----------
  const geo = useMemo(() => {
    const shapes = v.shapes;
    const xs: number[] = [];
    let cx = 0;
    const cell = shapes.map((s) => (s.h > 1 ? Math.min(0.36, 6 / Math.max(s.h, s.w)) : 0));
    shapes.forEach((s, i) => {
      const depth = s.h > 1 ? s.c * 0.26 : 0.6;
      xs.push(cx + depth / 2);
      cx += depth + (i === 0 ? 2.4 : 2.0);
    });
    const shift = cx / 2;
    const cellPos = (li: number, c: number, r: number, col: number): V3 => {
      const s = shapes[li];
      if (s.h > 1) return [xs[li] - shift + (c - (s.c - 1) / 2) * 0.26, ((s.h - 1) / 2 - r) * cell[li], (col - (s.w - 1) / 2) * cell[li]];
      // vectors: column (≤ 32) or grid
      const n = s.c, cols = n > 32 ? Math.ceil(Math.sqrt(n)) : 1, rows = Math.ceil(n / cols), g = Math.min(0.42, 8 / rows);
      const j = c, rr = Math.floor(j / cols), cc = j % cols;
      return [xs[li] - shift, ((rows - 1) / 2 - rr) * g, (cc - (cols - 1) / 2) * g];
    };
    return { cellPos, cell, xs: xs.map((v) => v - shift) };
  }, [v.shapes]);

  const tensorMesh = useRef<THREE.InstancedMesh>(null);
  const vecMesh = useRef<THREE.InstancedMesh>(null);
  const capT = useMemo(() => v.shapes.reduce((s, sh, i) => s + (sh.h > 1 && (i === 0 || layers[i - 1]?.kind !== 'flatten') ? sh.c * sh.h * sh.w : 0), 0), [v.shapes, layers]);
  const capV = useMemo(() => v.shapes.reduce((s, sh) => s + (sh.h === 1 ? sh.c : 0), 0), [v.shapes]);

  // conv/pool layer chosen for the kernel animation
  const opLayers = layers.map((_, i) => i).filter((i) => layers[i].kind === 'conv' || layers[i].kind === 'pool');
  const li = opLayers.includes(selLayer) ? selLayer : opLayers[0] ?? -1;
  const L = li >= 0 ? (layers[li] as Extract<CnnLayer, { kind: 'conv' | 'pool' }>) : null;
  const so = li >= 0 ? v.shapes[li + 1] : null, si = li >= 0 ? v.shapes[li] : null;
  const ch = Math.min(selCh, (so?.c ?? 1) - 1);
  const nPos = so ? so.h * so.w : 1;
  const sp = Math.min(scan, nPos - 1);
  const oy = so ? Math.floor(sp / so.w) : 0, ox = so ? sp % so.w : 0;
  useEffect(() => {
    if (!scanOn || globalThis.window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return;
    const id = setInterval(() => setScan((s) => (s + 1) % nPos), 140);
    return () => clearInterval(id);
  }, [scanOn, nPos]);

  // receptive field of the current output cell + the multiply–sum breakdown
  const window = useMemo(() => {
    if (!L || !si || !so || !trace || !model) return null;
    const k = L.kind === 'conv' ? L.k : L.size, stride = L.stride, pad = L.kind === 'conv' && L.pad === 'same' ? Math.floor(k / 2) : 0;
    const cells: { c: number; r: number; col: number; x: number; w: number }[] = [];
    const chans = L.kind === 'conv' ? [...Array(si.c).keys()] : [ch];
    chans.forEach((c) => {
      for (let ky = 0; ky < k; ky++) for (let kx = 0; kx < k; kx++) {
        const r = oy * stride + ky - pad, col = ox * stride + kx - pad;
        const inside = r >= 0 && r < si.h && col >= 0 && col < si.w;
        const xv = inside ? trace.a[li][(c * si.h + r) * si.w + col] : 0;
        const w = L.kind === 'conv' ? model.net.params[li]!.W[((ch * si.c + c) * k + ky) * k + kx] : 1;
        cells.push({ c, r, col, x: xv, w });
      }
    });
    const o = (ch * so.h + oy) * so.w + ox;
    return { k, cells, bias: L.kind === 'conv' ? model.net.params[li]!.b[ch] : 0, z: L.kind === 'conv' ? trace.z[li]![o] : trace.a[li + 1][o], a: trace.a[li + 1][o] };
  }, [L, si, so, trace, model, li, ch, oy, ox]);

  useLayoutEffect(() => {
    const m = tensorMesh.current, vm = vecMesh.current;
    if (!m || !vm || !trace) return;
    let k = 0, kv = 0;
    const dim = new THREE.Color('#0d1424'), c = new THREE.Color();
    v.shapes.forEach((s, l) => {
      const a = trace.a[l];
      let mx = 1e-9;
      for (let i = 0; i < a.length; i++) mx = Math.max(mx, Math.abs(a[i]));
      if (s.h > 1 && (l === 0 || layers[l - 1]?.kind !== 'flatten')) {
        const tint = l === 0 ? '#E5E7EB' : layers[l - 1].kind === 'pool' ? '#A855F7' : '#00D9FF';
        for (let ch2 = 0; ch2 < s.c; ch2++) for (let r = 0; r < s.h; r++) for (let col = 0; col < s.w; col++) {
          const val = a[(ch2 * s.h + r) * s.w + col] / mx;
          const p = geo.cellPos(l, ch2, r, col);
          const hl = (l === li + 1 && ch2 === ch && r === oy && col === ox) || (l === li && window?.cells.some((q) => q.c === ch2 && q.r === r && q.col === col));
          c.set(hl ? '#ffffff' : val >= 0 ? tint : NEG_W).lerp(dim, hl ? 0.1 : 1 - (0.08 + 0.92 * Math.abs(val)));
          setInstance(m, k++, p[0], p[1], p[2], [0.06 + 0.12 * Math.abs(val), geo.cell[l] * 0.9, geo.cell[l] * 0.9], c);
        }
      } else if (s.h === 1) {
        for (let j = 0; j < s.c; j++) {
          const p = geo.cellPos(l, j, 0, 0);
          const val = a[j] / mx;
          if (l === v.shapes.length - 1) c.set(CLASS_COLORS[j % 6]).lerp(dim, 1 - (0.12 + 0.88 * a[j]));
          else c.set(val >= 0 ? '#22C55E' : NEG_W).lerp(dim, 1 - (0.1 + 0.9 * Math.abs(val)));
          setInstance(vm, kv++, p[0], p[1], p[2], l === v.shapes.length - 1 ? 0.2 + 0.25 * a[j] : 0.11, c);
        }
      }
    });
    commitInstances(m, k); commitInstances(vm, kv);
  });

  const lines = useMemo(() => {
    const segs: { a: V3; b: V3; c: string; c2?: string; i?: number }[] = [];
    if (!trace || !model) return segs;
    if (window && L) {
      const target = geo.cellPos(li + 1, ch, oy, ox);
      window.cells.forEach((q) => { if (q.r >= 0 && q.r < si!.h && q.col >= 0 && q.col < si!.w) segs.push({ a: geo.cellPos(li, q.c, q.r, q.col), b: target, c: '#ffffff', c2: '#00D9FF', i: 0.25 + 0.6 * Math.min(1, Math.abs(q.w * q.x)) }); });
    }
    if (showDense) {
      const rng = new Rng(5);
      layers.forEach((l, i) => {
        if (l.kind !== 'dense' && l.kind !== 'output') return;
        const p = model.net.params[i]!, nin = v.shapes[i].c * v.shapes[i].h * v.shapes[i].w, nout = l.units;
        const keep = Math.min(1, 700 / (nin * nout));
        let mx = 1e-9; for (const w of p.W) mx = Math.max(mx, Math.abs(w));
        for (let j = 0; j < nout; j++) for (let k = 0; k < nin; k++) {
          if (keep < 1 && rng.next() > keep) continue;
          const w = p.W[j * nin + k];
          const src = v.shapes[i].h > 1 ? geo.cellPos(i, Math.floor(k / (v.shapes[i].h * v.shapes[i].w)), Math.floor((k % (v.shapes[i].h * v.shapes[i].w)) / v.shapes[i].w), k % v.shapes[i].w) : geo.cellPos(i, k, 0, 0);
          segs.push({ a: src, b: geo.cellPos(i + 1, j, 0, 0), c: w >= 0 ? POS_W : NEG_W, i: 0.04 + 0.5 * Math.abs(w * trace.a[i][k]) / mx });
        }
      });
    }
    return segs;
  }, [trace, model, window, L, li, ch, oy, ox, geo, si, showDense, layers, v.shapes]);
  const denseShownPct = useMemo(() => {
    let tot = 0;
    layers.forEach((l, i) => { if (l.kind === 'dense' || l.kind === 'output') tot += l.units * v.shapes[i].c * v.shapes[i].h * v.shapes[i].w; });
    return tot ? Math.min(1, lines.length / tot) : 0;
  }, [lines, layers, v.shapes]);

  const scene = valid && trace ? (
    <group>
      <Instances ref={tensorMesh} capacity={Math.max(1, capT)} geometry={BOX} glow={0.45} />
      <Instances ref={vecMesh} capacity={Math.max(1, capV)} geometry={SPHERE} glow={0.6} />
      <Segments segs={lines} />
      {v.shapes.map((s, l) => {
        const top = s.h > 1 ? ((s.h - 1) / 2) * geo.cell[l] + 0.9 : Math.min(8, s.c * 0.42) / 2 + 0.9;
        const name = l === 0 ? 'Input' : layers[l - 1].kind === 'output' ? 'Output' : layers[l - 1].kind;
        return <Label key={l} position={[geo.xs[l], Math.min(top, 5.5), 0]} className={l === li + 1 ? 'text-cyan' : ''}>{name} <span className="text-muted">{s.h > 1 ? `${s.c}×${s.h}×${s.w}` : s.c}</span></Label>;
      })}
      {probs.map((p, j) => { const q = geo.cellPos(v.shapes.length - 1, j, 0, 0); return <Label key={j} position={[q[0] + 0.9, q[1], q[2]]}><span style={{ color: j === pred ? '#F59E0B' : undefined }}>{j}</span> {(p * 100).toFixed(0)}%</Label>; })}
    </group>
  ) : null;

  const setL = (i: number, patch: Partial<CnnLayer>) => { setRunning(false); setLayers(layers.map((l, j) => (j === i ? ({ ...l, ...patch } as CnnLayer) : l))); };
  const insert = (l: CnnLayer) => { setRunning(false); const fi = layers.findIndex((q) => q.kind === 'flatten'); const at = l.kind === 'conv' || l.kind === 'pool' ? (fi < 0 ? layers.length - 1 : fi) : layers.length - 1; setLayers([...layers.slice(0, at), l, ...layers.slice(at)]); };
  const tr = model?.trainer;
  const hist = model?.history ?? [];

  return (
    <LabShell crumb="CNN" title={t('CNN Lab — โครงข่ายคอนโวลูชัน', 'CNN Lab — Convolutional Networks')} chips={['Deep learning', t('ขั้นสูง', 'Advanced')]}
      experiment={{ model: 'CNN', source: custom ? t('ภาพที่วาดหรืออัปโหลด', 'Drawn or uploaded image') : `UCI handwritten digits · train ${D.Xtr.length} / test ${D.Xte.length}`,
        stages: [{ name: 'Input image', kind: 'tensor', shape: [16, 16, 1], values: x }, ...(valid && trace ? layers.map((layer, i) => ({ name: layer.kind, kind: v.shapes[i + 1].h > 1 ? 'tensor' as const : 'vector' as const, values: trace.a[i + 1], weights: layer.kind === 'dense' || layer.kind === 'output' ? model?.net.params[i]?.W : undefined, shape: [v.shapes[i + 1].h, v.shapes[i + 1].w, v.shapes[i + 1].c] as [number, number, number], labels: i === layers.length - 1 ? probs.map((_, j) => `P(${j})`) : undefined, highlight: i === layers.length - 1 ? [pred] : undefined })) : [])],
        metrics: [{ label: 'Prediction', value: pred >= 0 ? String(pred) : '—' }, { label: 'Target', value: label == null ? 'custom' : String(label) }, { label: 'Test accuracy', value: evalRep ? pct(evalRep.accuracy) : '—' }, { label: 'Epoch', value: String(tr?.epoch ?? (effSource === 'pretrained' ? pretrained.epochs : 0)) }],
        history: effSource === 'trained' ? [{ label: 'Training loss', values: model?.history.map((row) => row.loss) ?? [] }, { label: 'Test accuracy', values: model?.history.map((row) => row.testAcc) ?? [], color: '#F59E0B' }] : undefined,
        confusion: evalRep?.confusion,
        sample: { index: sampleIdx, count: D.Xte.length, onChange: (index) => { setCustom(null); setSampleIdx(index); } },
        output: pred >= 0 ? `${t('ตัวเลขที่ทำนาย', 'Predicted digit')}: ${pred} · P=${pct(probs[pred])}${label == null ? '' : ` · ${t('คำตอบจริง', 'target')}=${label}`}` : t('ยังไม่มีผลลัพธ์ ตรวจสถาปัตยกรรมก่อน', 'No output yet; check the architecture'),
        note: effSource === 'pretrained' ? t('ใช้โมเดลฝึกล่วงหน้า', 'Using pretrained weights') : t('ใช้โมเดลที่ฝึกในแล็บนี้', 'Using the model trained in this lab'),
      }}
      intro={t('ภาพตัวเลขเขียนมือจริง 16×16 ไหลผ่านชั้น convolution และ pooling กรอบสีขาวคือ receptive field ของช่องผลลัพธ์ที่กำลังคำนวณ วาดตัวเลขเองหรืออัปโหลดภาพได้', 'Real handwritten 16×16 digits flow through convolution and pooling layers. The white cells are the receptive field of the output cell being computed. Draw your own digit or upload an image.')}
      cam={cam} setCam={setCam} distance={22} scene={scene}
      blocked={valid ? undefined : <div className="grid h-full place-items-center px-6 text-center"><div><p className="font-display text-xl text-red">{t('สถาปัตยกรรมไม่ถูกต้อง', 'Invalid architecture')}</p><ul className="mt-2 text-muted">{v.errors.map((e, i) => <li key={i}>{t(e)}</li>)}</ul></div></div>}
      readout={{ title: effSource === 'pretrained' ? t('โมเดลฝึกล่วงหน้า', 'Pretrained model') : t('โมเดลที่ฝึกเอง', 'Your trained model'), rows: [
        [t('ภาพ', 'image'), custom ? t('ภาพของผู้ใช้', 'your image') : `test #${sampleIdx} (${t('จริง', 'true')} = ${label})`],
        [t('ทำนาย', 'prediction'), pred >= 0 ? `${pred} (${pct(probs[pred])})` : '—', pred >= 0 && label != null ? (pred === label ? '#22C55E' : '#EF4444') : undefined],
        [t('พารามิเตอร์', 'params'), model ? model.net.paramCount().toLocaleString() : '—'],
        ...(effSource === 'pretrained' ? [[t('ฝึกมาแล้ว', 'trained'), `${pretrained.epochs} epochs · test ${pct(pretrained.testAcc)}`] as [string, string]] : [['epoch', `${tr?.epoch ?? 0} · step ${tr?.steps ?? 0}`] as [string, string]]),
        ['test accuracy', evalRep ? pct(evalRep.accuracy) : '—'],
        ...(window && L ? [[`${L.kind} · ${t('ช่อง', 'ch')} ${ch} @ (${oy},${ox})`, L.kind === 'conv' ? `z=${fmt(window.z, 3)} → ${fmt(window.a, 3)}` : `${L.mode} = ${fmt(window.a, 3)}`] as [string, string]] : []),
        [t('เส้น dense ที่แสดง', 'dense links drawn'), showDense ? pct(denseShownPct) : '0%'],
      ] }}
      aside={window && L && (
        <FloatingPanel title={L.kind === 'conv' ? t('คูณทีละช่องแล้วรวม', 'Multiply element-wise, then sum') : t('หน้าต่าง pooling', 'Pooling window')}
          className="hidden max-w-[360px] lg:flex" actions={
          <div className="flex items-center gap-1 text-[0.9375rem]">
            <button type="button" onClick={() => setScanOn(!scanOn)} className="rounded border border-line px-2 py-0.5">{scanOn ? '❚❚' : '▶'}</button>
            <button type="button" onClick={() => { setScanOn(false); setScan((s) => (s + 1) % nPos); }} className="rounded border border-line px-2 py-0.5">→</button>
          </div>
        }>
          <div className="flex flex-col gap-2 pb-1">
            {L.kind === 'conv' && si!.c > 1 && <Slider label={t('ช่องอินพุตที่แสดง', 'Input channel shown')} value={Math.min(inCh, si!.c - 1)} min={0} max={si!.c - 1} onChange={setInCh} />}
            <div className="grid gap-0.5 font-mono text-[0.84375rem]" style={{ gridTemplateColumns: `repeat(${window.k}, minmax(0,1fr))` }}>
              {window.cells.filter((q) => q.c === (L.kind === 'conv' ? Math.min(inCh, si!.c - 1) : ch)).map((q, i) => (
                <span key={i} className="rounded px-1 py-0.5 text-center" style={{ background: `rgba(0,217,255,${Math.min(0.6, Math.abs(q.x * q.w))})` }}>
                  {L.kind === 'conv' ? <>{fmt(q.w, 2)}<br />×{fmt(q.x, 2)}</> : fmt(q.x, 2)}
                </span>
              ))}
            </div>
            <div className="font-mono text-[0.90625rem] text-muted">
              {L.kind === 'conv'
                ? <>Σ(w·x) {t('ทุกช่องอินพุต', 'over all channels')} = {fmt(window.z - window.bias, 3)} + b({fmt(window.bias, 3)}) = {fmt(window.z, 3)} → {L.act}: <b className="text-ink">{fmt(window.a, 3)}</b></>
                : <>{L.mode}({window.cells.filter((q) => q.c === ch).map((q) => fmt(q.x, 2)).join(', ')}) = <b className="text-ink">{fmt(window.a, 3)}</b></>}
            </div>
          </div>
        </FloatingPanel>
      )}
      playback={valid ? { playing: running,
        onToggle: () => { if (tr && tr.epoch >= train.maxEpochs) return; if (!tr) setSource('trained'); setRunning(!running); },
        onNext: () => { setRunning(false); if (!tr) { setSource('trained'); setPendingStep(true); } else { try { advance(); } catch (error) { setTrainError((error as Error).message); } setTick((x) => x + 1); } },
        onReset: () => { setRunning(false); setPendingStep(false); setSource('trained'); setResetNonce((n) => n + 1); },
        canNext: !tr || tr.epoch < train.maxEpochs, speed, onSpeed: setSpeed,
        status: trainError || `epoch ${tr?.epoch ?? 0} / ${train.maxEpochs}` } : undefined}
      controls={
        <>
          <Section title={t('ภาพอินพุต', 'Input image')}>
            <Slider label={`test #`} value={sampleIdx} min={0} max={D.Xte.length - 1} onChange={(v2) => { setSampleIdx(v2); setCustom(null); }} />
            <VisionInput image={x} onChange={setCustom} />
          </Section>
          <Section title={t('สถาปัตยกรรม', 'Architecture')}>
            <ol className="flex flex-col gap-1.5">
              {layers.map((l, i) => (
                <li key={i} className="rounded-lg border border-line p-2 text-[0.96875rem]">
                  <div className="flex items-center justify-between gap-2">
                    <b className="font-mono uppercase text-cyan">{l.kind}</b>
                    <span className="font-mono text-muted">→ {v.shapes[i + 1] ? (v.shapes[i + 1].h > 1 ? `${v.shapes[i + 1].c}×${v.shapes[i + 1].h}×${v.shapes[i + 1].w}` : v.shapes[i + 1].c) : '?'} · {v.params[i] ?? 0}p</span>
                    {l.kind !== 'output' && l.kind !== 'flatten' && <button type="button" onClick={() => { setRunning(false); setLayers(layers.filter((_, j) => j !== i)); }} title={t('ลบ', 'Remove')} aria-label={t('ลบ', 'Remove')} className="hover:text-red"><Icon name="trash" className="h-3.5 w-3.5" /></button>}
                  </div>
                  {l.kind === 'conv' && <div className="mt-1.5 grid grid-cols-2 gap-1.5">
                    <Select<number> label="filters" value={l.filters} onChange={(filters) => setL(i, { filters })} options={[1, 2, 4, 6, 8, 12, 16, 24, 32].map((n) => ({ value: n, label: String(n) }))} />
                    <Select<number> label="kernel" value={l.k} onChange={(k) => setL(i, { k })} options={[1, 3, 5].map((n) => ({ value: n, label: `${n}×${n}` }))} />
                    <Select<number> label="stride" value={l.stride} onChange={(stride) => setL(i, { stride })} options={[1, 2].map((n) => ({ value: n, label: String(n) }))} />
                    <Select<'same' | 'valid'> label="padding" value={l.pad} onChange={(pad) => setL(i, { pad })} options={[{ value: 'same', label: 'same' }, { value: 'valid', label: 'valid' }]} />
                    <Select<'relu' | 'tanh' | 'linear'> label="activation" value={l.act} onChange={(act) => setL(i, { act })} options={['relu', 'tanh', 'linear'].map((a) => ({ value: a as 'relu', label: a }))} />
                  </div>}
                  {l.kind === 'pool' && <div className="mt-1.5 grid grid-cols-3 gap-1.5">
                    <Select<'max' | 'avg'> label="mode" value={l.mode} onChange={(mode) => setL(i, { mode })} options={[{ value: 'max', label: 'max' }, { value: 'avg', label: 'avg' }]} />
                    <Select<number> label="size" value={l.size} onChange={(size) => setL(i, { size })} options={[2, 3].map((n) => ({ value: n, label: String(n) }))} />
                    <Select<number> label="stride" value={l.stride} onChange={(stride) => setL(i, { stride })} options={[1, 2, 3].map((n) => ({ value: n, label: String(n) }))} />
                  </div>}
                  {l.kind === 'dense' && <div className="mt-1.5 grid grid-cols-2 gap-1.5">
                    <Select<number> label="units" value={l.units} onChange={(units) => setL(i, { units })} options={[4, 8, 16, 24, 32, 64, 128].map((n) => ({ value: n, label: String(n) }))} />
                    <Select<'relu' | 'tanh' | 'linear'> label="activation" value={l.act} onChange={(act) => setL(i, { act })} options={['relu', 'tanh', 'linear'].map((a) => ({ value: a as 'relu', label: a }))} />
                  </div>}
                  {l.kind === 'dropout' && <Slider label="rate" value={l.rate} min={0} max={0.8} step={0.05} format={(r) => r.toFixed(2)} onChange={(rate) => setL(i, { rate })} />}
                </li>
              ))}
            </ol>
            <div className="flex flex-wrap gap-1.5">
              <Button size="sm" onClick={() => insert({ kind: 'conv', filters: 4, k: 3, stride: 1, pad: 'same', act: 'relu' })}><Icon name="plus" />Conv2D</Button>
              <Button size="sm" onClick={() => insert({ kind: 'pool', mode: 'max', size: 2, stride: 2 })}><Icon name="plus" />Pool</Button>
              <Button size="sm" onClick={() => insert({ kind: 'dense', units: 16, act: 'relu' })}><Icon name="plus" />Dense</Button>
              <Button size="sm" onClick={() => insert({ kind: 'dropout', rate: 0.2 })}><Icon name="plus" />Dropout</Button>
              <Button size="sm" onClick={() => { setRunning(false); setLayers(DEFAULT_CNN); setSource('pretrained'); }}>{t('ค่าเริ่มต้น', 'Default')}</Button>
            </div>
            {v.errors.map((e, i) => <p key={i} className="text-[0.9375rem] text-red">{t(e)}</p>)}
            <p className="text-[0.90625rem] text-muted">{t('padding "same" ใช้ pad = ⌊k/2⌋ · DepthwiseConv / Separable / BatchNorm ยังไม่รองรับในห้องนี้', 'padding “same” uses pad = ⌊k/2⌋ · Depthwise/Separable conv and BatchNorm are not supported in this lab yet')}</p>
          </Section>
          <Section title={t('การฝึก', 'Training')}>
            <Segmented label={t('น้ำหนักที่ใช้', 'Weights')} value={effSource} onChange={(s) => { setRunning(false); if (s === 'pretrained' && !sameArch(layers, DEFAULT_CNN)) setLayers(DEFAULT_CNN); setSource(s); setResetNonce((n) => n + 1); setTick((q) => q + 1); }}
              options={[{ value: 'pretrained', label: t('Pretrained', 'Pretrained') }, { value: 'trained', label: t('ฝึกเอง', 'Train here') }]} />
            <div className="rounded-lg border border-green/30 bg-green/5 p-2.5 text-[0.9375rem]">
              <div className="flex items-center justify-between gap-2">
                <div><b className="text-green">{t('โมเดลฝึกล่วงหน้า', 'Pretrained model')}</b><div className="mt-0.5 text-muted">{pretrained.epochs} epochs · test accuracy {pct(pretrained.testAcc)}</div></div>
                <Button size="sm" onClick={() => { setRunning(false); setLayers(DEFAULT_CNN); setSource('pretrained'); setResetNonce((n) => n + 1); setTick((q) => q + 1); }}>{t('โหลด', 'Load')}</Button>
              </div>
            </div>
            <Select<'adam' | 'sgd' | 'momentum'> label="Optimizer" value={train.optimizer} onChange={(optimizer) => setTrain({ ...train, optimizer })} options={[{ value: 'adam', label: 'Adam' }, { value: 'momentum', label: 'Momentum' }, { value: 'sgd', label: 'SGD' }]} />
            <Slider label="η" value={Math.log10(train.lr)} min={-4} max={-0.5} step={0.05} format={(q) => (10 ** q).toPrecision(2)} onChange={(q) => setTrain({ ...train, lr: Number((10 ** q).toPrecision(3)) })} />
            <Select<number> label="Batch" value={train.batch} onChange={(batch) => setTrain({ ...train, batch })} options={[8, 16, 32, 64].map((n) => ({ value: n, label: String(n) }))} />
            <Slider label={t('epoch สูงสุด', 'Max epochs')} value={train.maxEpochs} min={1} max={30} onChange={(maxEpochs) => setTrain({ ...train, maxEpochs })} />
            <Slider label="Init seed" value={train.seed} min={1} max={99} onChange={(seed) => { setRunning(false); setTrain({ ...train, seed }); }} />
            <p className="text-[0.90625rem] text-muted">{t(`ข้อมูล: ${D.Xtr.length} train / ${D.Xte.length} test จากชุดตัวเลขเขียนมือ UCI (ภาพจริง 8×8 ขยายเป็น 16×16)`, `Data: ${D.Xtr.length} train / ${D.Xte.length} test from the UCI handwritten digits (real 8×8 images upsampled to 16×16)`)}</p>
          </Section>
          <Section title={t('การแสดงผล', 'View')}>
            <Select<number> label={t('ชั้นที่แสดงการเลื่อน kernel', 'Layer for the kernel animation')} value={li} onChange={(q) => { setSelLayer(q); setScan(0); }} options={opLayers.map((q) => ({ value: q, label: `${q + 1}. ${layers[q].kind}` }))} />
            {so && <Slider label={t('ช่องผลลัพธ์ (filter)', 'Output channel (filter)')} value={ch} min={0} max={so.c - 1} onChange={(q) => { setSelCh(q); }} />}
            <Toggle label={t('เส้นเชื่อม dense (สุ่มตัวอย่าง)', 'Dense connections (sampled)')} checked={showDense} onChange={setShowDense} />
          </Section>
        </>
      }
      tabs={[
        { id: 'maps', label: t('ตรวจ feature map', 'Feature-map inspector'), content: trace && model ? (
          <div className="flex flex-col gap-8">
            {layers.map((l, i) => {
              if (l.kind !== 'conv' && l.kind !== 'pool') return null;
              const s = v.shapes[i + 1];
              return (
                <div key={i}>
                  <h3 className="label-xs mb-2">{i + 1}. {l.kind} → {s.c}×{s.h}×{s.w}</h3>
                  <div className="flex flex-wrap gap-3">
                    {Array.from({ length: s.c }, (_, c) => (
                      <button key={c} type="button" onClick={() => { setSelLayer(i); setSelCh(c); }} className={`flex flex-col items-center gap-1 rounded p-1 ${i === li && c === ch ? 'ring-1 ring-cyan' : ''}`}>
                        <Heat data={trace.a[i + 1].subarray(c * s.h * s.w, (c + 1) * s.h * s.w)} h={s.h} w={s.w} size={72} />
                        {l.kind === 'conv' && <Heat data={model.net.params[i]!.W.subarray(c * v.shapes[i].c * l.k * l.k, c * v.shapes[i].c * l.k * l.k + l.k * l.k)} h={l.k} w={l.k} size={30} signed />}
                        <span className="font-mono text-[0.875rem] text-muted">#{c}</span>
                      </button>
                    ))}
                  </div>
                </div>
              );
            })}
            <div>
              <h3 className="label-xs mb-2">{t('เทียบกับตัวกรองขอบแบบคลาสสิกบนภาพเดียวกัน', 'Compare with classic edge filters on the same image')}</h3>
              <div className="flex flex-wrap gap-4">
                {CLASSIC.map((f) => {
                  const out = new Float32Array(196);
                  for (let r = 0; r < 14; r++) for (let c = 0; c < 14; c++) { let s = 0; for (let a = 0; a < 3; a++) for (let b = 0; b < 3; b++) s += f.k[a][b] * x[(r + a) * 16 + c + b]; out[r * 14 + c] = s; }
                  return <div key={f.name} className="flex flex-col items-center gap-1"><Heat data={out} h={14} w={14} size={84} signed /><span className="font-mono text-[0.875rem] text-muted">{f.name}</span></div>;
                })}
              </div>
            </div>
          </div>
        ) : null },
        { id: 'curves', label: t('การฝึก', 'Training'), content: (
          <div className="grid gap-8 lg:grid-cols-2">
            <div>{hist.length ? <LinePlot xTitle="epoch" yTitle="test accuracy" yRange={[0, 1.02]} series={[{ name: 'test acc', x: hist.map((h) => h.epoch), y: hist.map((h) => h.testAcc), color: '#22C55E' }]} />
              : <p className="text-[1.0625rem] text-muted">{t('กด ▶ เพื่อฝึกจากน้ำหนักสุ่มในเบราว์เซอร์ ราว 1–3 วินาทีต่อ epoch สำหรับโมเดลเริ่มต้น', 'Press ▶ to train from random weights in the browser; about 1–3 s per epoch for the default model.')}</p>}</div>
            <div>{hist.length > 0 && <LinePlot xTitle="epoch" yTitle="train loss" series={[{ name: 'loss', x: hist.map((h) => h.epoch), y: hist.map((h) => h.loss), color: '#00D9FF' }]} />}</div>
          </div>
        ) },
        { id: 'eval', label: t('การประเมินผล', 'Evaluation'), content: evalRep ? (
          <div className="grid gap-8 lg:grid-cols-[1fr_1.3fr]">
            <Metrics items={[['Accuracy', pct(evalRep.accuracy)], ['Precision', pct(evalRep.precision)], ['Recall', pct(evalRep.recall)], ['F1', pct(evalRep.f1)]]} />
            <div><h3 className="label-xs mb-2">Confusion matrix (test, n = {evalRep.support})</h3><ConfusionMatrix m={evalRep.confusion} classNames={[...Array(10).keys()].map(String)} /></div>
          </div>
        ) : null },
        { id: 'math', label: t('คณิตศาสตร์', 'Math'), content: (
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="flex flex-col gap-3">
              <Tex block>{'z_{f,i,j} = b_f + \\sum_{c}\\sum_{u,v} W_{f,c,u,v}\\; x_{c,\\; i s + u - p,\\; j s + v - p}'}</Tex>
              <Tex block>{'H_{out} = \\Big\\lfloor \\frac{H + 2p - k}{s} \\Big\\rfloor + 1'}</Tex>
              <Tex block>{'\\text{maxpool: } y_{c,i,j} = \\max_{u,v<k} x_{c,\\, is+u,\\, js+v}'}</Tex>
            </div>
            <div className="flex flex-col gap-3">
              <Tex block>{'\\frac{\\partial L}{\\partial W_{f,c,u,v}} = \\sum_{i,j} \\delta_{f,i,j}\\; x_{c,\\, is+u-p,\\, js+v-p}'}</Tex>
              <p className="text-[1.0625rem] text-muted">{t('พารามิเตอร์ของ conv = filters × channels_in × k × k + filters ซึ่งน้อยกว่า dense มาก เพราะใช้น้ำหนักร่วมกันทั้งภาพ (weight sharing)', 'Conv parameters = filters × in_channels × k × k + filters, far fewer than dense because weights are shared across the image.')}</p>
            </div>
          </div>
        ) },
        { id: 'ex', label: t('แบบฝึกหัด', 'Exercises'), content: (
          <Quiz id="cnn" items={[
            { q: t('ภาพ 16×16, kernel 3×3, stride 1, padding valid ได้ผลลัพธ์ขนาดเท่าไร', '16×16 input, 3×3 kernel, stride 1, valid padding gives…'), options: ['16×16', '14×14', '8×8'], answer: 1, why: '(16 − 3)/1 + 1 = 14' },
            { q: t('max pooling ช่วยอะไร', 'What does max pooling help with?'), options: [t('เพิ่มจำนวนพารามิเตอร์', 'Adding parameters'), t('ลดขนาดและทนต่อการเลื่อนเล็กน้อย', 'Downsampling and tolerance to small shifts')], answer: 1, why: t('ไม่มีพารามิเตอร์ให้เรียนรู้ เก็บเฉพาะค่าสูงสุดในหน้าต่าง', 'It has no learnable parameters; it keeps the maximum in each window.') },
            { q: t('ทำไม CNN ใช้พารามิเตอร์น้อยกว่า MLP บนภาพ', 'Why does a CNN need fewer parameters than an MLP on images?'), options: [t('ใช้ kernel เดียวกันทุกตำแหน่ง', 'The same kernel is reused at every position'), t('ใช้ข้อมูลน้อยกว่า', 'It uses less data')], answer: 0, why: t('weight sharing และ local connectivity', 'Weight sharing and local connectivity.') },
          ]} />
        ) },
      ]}
    />
  );
}
