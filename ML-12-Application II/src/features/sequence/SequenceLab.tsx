import { useEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import { useCameraState } from '../../engine3d/Stage';
import { LabShell, Metrics } from '../../components/LabShell';
import { flowVector } from '../../engine3d/experiment';
import { PointCloud, Segments } from '../../engine3d/plots';
import { Label } from '../../engine3d/primitives';
import { GRAD, NEG_W, POS_W } from '../../engine3d/palette';
import { SeqModel, makeSeqBatch, type Cell, type SeqTask } from '../../ml/sequence';
import { Optimizer } from '../../ml/nn/optimizer';
import { Rng } from '../../lib/random';
import { useT } from '../../i18n/i18n';
import { Section, Segmented, Slider, fmt, pct } from '../../components/ui';
import { LinePlot, Quiz, Tex } from '../../components/common';

type V3 = [number, number, number];
const GATES = ['i', 'f', 'g', 'o'] as const;
const GATE_COL: Record<string, string> = { i: '#22C55E', f: '#F59E0B', g: '#00D9FF', o: '#A855F7' };

export default function SequenceLab({ initial = 'lstm' }: { initial?: Cell }) {
  const t = useT();
  const [cfg, setCfg] = useState({ cell: initial as Cell, hidden: 6, layers: 1, task: 'memory' as SeqTask, T: 12, lr: 0.01, clip: 5, batch: 8, seed: 3, maxSteps: 3000 });
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [selT, setSelT] = useState(0);
  const [selLayer, setSelLayer] = useState(0);
  const [exSeed, setExSeed] = useState(1);
  const [tick, setTick] = useState(0);
  const [nonce, setNonce] = useState(0);
  const [cam, setCam] = useCameraState();

  const st = useMemo(() => {
    const model = new SeqModel({ cell: cfg.cell, hidden: cfg.hidden, layers: cfg.layers, inDim: 1, outDim: 1, seed: cfg.seed });
    const evalSet = makeSeqBatch(cfg.task, 64, cfg.T, new Rng(999));
    return { model, opt: new Optimizer({ name: 'adam', lr: cfg.lr, momentum: 0.9, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }), rng: new Rng(cfg.seed + 7), steps: 0, history: [] as { step: number; loss: number; metric: number }[], evalSet };
  }, [cfg.cell, cfg.hidden, cfg.layers, cfg.task, cfg.T, cfg.seed, nonce]); // eslint-disable-line react-hooks/exhaustive-deps
  st.opt.cfg.lr = cfg.lr;

  const evaluate = () => {
    let loss = 0, ok = 0;
    st.evalSet.X.forEach((xs, i) => {
      const tr = st.model.forward(xs, cfg.task);
      loss += st.model.loss(tr, st.evalSet.Y[i], cfg.task);
      if (cfg.task === 'memory' && (tr.y[cfg.T - 1][0] > 0.5 ? 1 : 0) === st.evalSet.Y[i][cfg.T - 1][0]) ok++;
    });
    return { loss: loss / st.evalSet.X.length, metric: cfg.task === 'memory' ? ok / st.evalSet.X.length : loss / st.evalSet.X.length };
  };
  const trainStep = () => {
    const m = st.model;
    const { X, Y } = makeSeqBatch(cfg.task, cfg.batch, cfg.T, st.rng);
    const acc = m.P.map((p) => ({ Wx: new Float64Array(p.Wx.length), Wh: new Float64Array(p.Wh.length), b: new Float64Array(p.b.length) }));
    const gWy = new Float64Array(m.Wy.length), gby = new Float64Array(m.by.length);
    X.forEach((xs, s) => {
      const g = m.backward(m.forward(xs, cfg.task), Y[s], cfg.task);
      g.gP.forEach((gp, l) => (['Wx', 'Wh', 'b'] as const).forEach((k) => gp[k].forEach((v, i) => (acc[l][k][i] += v / X.length))));
      g.gWy.forEach((v, i) => (gWy[i] += v / X.length)); g.gby.forEach((v, i) => (gby[i] += v / X.length));
    });
    // global-norm gradient clipping
    let n2 = 0;
    acc.forEach((a) => (['Wx', 'Wh', 'b'] as const).forEach((k) => a[k].forEach((v) => (n2 += v * v))));
    gWy.forEach((v) => (n2 += v * v));
    gby.forEach((v) => (n2 += v * v));
    const sc = cfg.clip > 0 && Math.sqrt(n2) > cfg.clip ? cfg.clip / Math.sqrt(n2) : 1;
    st.opt.begin();
    acc.forEach((a, l) => (['Wx', 'Wh', 'b'] as const).forEach((k) => st.opt.step(`${k}${l}`, m.P[l][k], a[k].map((v) => v * sc))));
    st.opt.step('Wy', m.Wy, gWy.map((v) => v * sc)); st.opt.step('by', m.by, gby.map((v) => v * sc));
    st.steps++;
    if (st.steps % 25 === 0) { const e = evaluate(); st.history.push({ step: st.steps, ...e }); }
  };
  useEffect(() => {
    if (!running) return;
    let raf = 0, last = 0;
    const loop = (now: number) => {
      const t0 = performance.now();
      while (performance.now() - t0 < Math.min(25, 6 * speed) && st.steps < cfg.maxSteps) trainStep();
      if (st.steps >= cfg.maxSteps) { setRunning(false); setTick((x) => x + 1); return; }
      if (now - last > 120) { last = now; setTick((x) => x + 1); }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [running, speed, st, cfg]); // eslint-disable-line react-hooks/exhaustive-deps

  // the example sequence shown in 3D, traced with the current weights
  const ex = useMemo(() => makeSeqBatch(cfg.task, 1, cfg.T, new Rng(exSeed)), [cfg.task, cfg.T, exSeed]);
  const trace = useMemo(() => st.model.forward(ex.X[0], cfg.task), [st, ex, tick, cfg.task]); // eslint-disable-line react-hooks/exhaustive-deps
  const grads = useMemo(() => st.model.backward(trace, ex.Y[0], cfg.task), [st, trace, ex, cfg.task]);
  const exLoss = st.model.loss(trace, ex.Y[0], cfg.task);
  const last = st.history[st.history.length - 1];

  // ---------- 3D: unrolled through time along x ----------
  const H = cfg.hidden, T = cfg.T, lstm = cfg.cell === 'lstm';
  const dx = Math.min(1.6, 26 / T), gap = Math.min(0.75, 5.5 / H);
  const X0 = -((T - 1) * dx) / 2;
  const hPos = (l: number, ti: number, j: number): V3 => [X0 + ti * dx, (j - (H - 1) / 2) * gap + l * (H * gap + 1.2) - (cfg.layers - 1) * (H * gap + 1.2) / 2, 0];
  const topY = hPos(cfg.layers - 1, 0, H - 1)[1] + 1.4, botY = hPos(0, 0, 0)[1] - 1.4;
  const flow = useRef(0);
  const [fpos, setFpos] = useState(0);
  useEffect(() => { if (window.matchMedia?.('(prefers-reduced-motion: reduce)').matches) return; const id = setInterval(() => { flow.current = (flow.current + 1) % (T + 4); setFpos(flow.current); }, 220); return () => clearInterval(id); }, [T]);

  const nodes: { p: V3; c: THREE.Color; s: number }[] = [];
  const segs: { a: V3; b: V3; c: string; c2?: string; i?: number }[] = [];
  const val = (v: number) => new THREE.Color(v >= 0 ? '#00D9FF' : NEG_W).lerp(new THREE.Color('#0d1424'), 1 - (0.12 + 0.88 * Math.min(1, Math.abs(v))));
  const gmax = Math.max(1e-12, ...grads.gradNorms);
  for (let ti = 0; ti < T; ti++) {
    const lit = ti <= fpos;
    const xin: V3 = [X0 + ti * dx, botY, 0];
    nodes.push({ p: xin, c: val(ex.X[0][ti][0]), s: 0.22 });
    nodes.push({ p: [X0 + ti * dx, topY, 0], c: val(trace.y[ti][0] * (cfg.task === 'memory' ? 2 : 1) - (cfg.task === 'memory' ? 1 : 0)), s: cfg.task === 'memory' && ti < T - 1 ? 0.1 : 0.22 });
    nodes.push({ p: [X0 + ti * dx, botY - 0.75, 0], c: new THREE.Color(GRAD).multiplyScalar(0.15 + 0.85 * grads.gradNorms[ti] / gmax), s: 0.08 + 0.22 * grads.gradNorms[ti] / gmax });
    for (let l = 0; l < cfg.layers; l++) {
      const s = trace.steps[l][ti], p = st.model.P[l];
      for (let j = 0; j < H; j++) {
        const hp = hPos(l, ti, j);
        nodes.push({ p: hp, c: lit ? val(s.h[j]) : new THREE.Color('#111a2c'), s: ti === selT && l === selLayer ? 0.2 : 0.15 });
        if (lstm) {
          nodes.push({ p: [hp[0], hp[1], 0.9], c: val(Math.tanh(s.c[j])), s: 0.1 });
          GATES.forEach((gname, gi) => { const gv = s[gname]![j]; nodes.push({ p: [hp[0] + (gi - 1.5) * 0.12 * Math.min(1, dx), hp[1], -0.7], c: new THREE.Color(GATE_COL[gname]).multiplyScalar(gname === 'g' ? Math.abs(gv) : gv), s: 0.045 }); });
        }
        if (l === 0) segs.push({ a: xin, b: hp, c: '#64748B', i: lit ? 0.25 : 0.05 });
        else segs.push({ a: hPos(l - 1, ti, j), b: hp, c: '#64748B', i: lit ? 0.25 : 0.05 });
        if (l === cfg.layers - 1) segs.push({ a: hp, b: [X0 + ti * dx, topY, 0], c: '#64748B', i: lit ? 0.2 : 0.04 });
        if (ti > 0) for (let k = 0; k < H; k++) {
          const w = p.Wh[(lstm ? 2 * H + j : j) * H + k];
          segs.push({ a: hPos(l, ti - 1, k), b: hp, c: w >= 0 ? POS_W : NEG_W, i: (lit ? 0.5 : 0.08) * Math.min(1, Math.abs(w)) });
        }
        if (lstm && ti > 0) segs.push({ a: [hPos(l, ti - 1, j)[0], hp[1], 0.9], b: [hp[0], hp[1], 0.9], c: '#F59E0B', i: lit ? 0.9 * trace.steps[l][ti].f![j] : 0.05 });
      }
    }
  }

  const scene = (
    <group>
      <PointCloud points={nodes.map((n) => n.p)} colors={nodes.map((n) => n.c)} sizes={nodes.map((n) => n.s)} glow={0.6} />
      <Segments segs={segs} />
      <mesh position={[X0 + selT * dx, 0, 0]}><boxGeometry args={[dx * 0.8, topY - botY + 1.8, 2.4]} /><meshBasicMaterial color="#00D9FF" transparent opacity={0.05} depthWrite={false} /></mesh>
      {Array.from({ length: T }, (_, ti) => (
        <mesh key={ti} position={[X0 + ti * dx, 0, 0]} onClick={(e) => { e.stopPropagation(); setSelT(ti); }}>
          <boxGeometry args={[dx * 0.9, topY - botY + 1.8, 0.2]} /><meshBasicMaterial transparent opacity={0} depthWrite={false} />
        </mesh>
      ))}
      <Label position={[X0 - dx, botY, 0]} muted>x_t</Label>
      <Label position={[X0 - dx, topY, 0]} muted>{cfg.task === 'memory' ? 'ŷ (t = T)' : 'ŷ_t'}</Label>
      <Label position={[X0 - dx, botY - 0.75, 0]} muted>‖∂L/∂h_t‖</Label>
      {Array.from({ length: cfg.layers }, (_, l) => <Label key={l} position={[X0 - dx, hPos(l, 0, (H - 1) / 2)[1], 0]} muted>h{cfg.layers > 1 ? `⁽${l + 1}⁾` : ''}</Label>)}
      {lstm && <Label position={[X0 + (T - 1) * dx + dx, hPos(0, 0, H - 1)[1] + 0.5, 0.9]} muted>c_t</Label>}
      {[0, Math.floor(T / 2), T - 1].map((ti) => <Label key={ti} position={[X0 + ti * dx, botY - 1.4, 0]} muted>t={ti}</Label>)}
    </group>
  );

  const s = trace.steps[Math.min(selLayer, cfg.layers - 1)][selT];
  const reset = () => { setRunning(false); setNonce((n) => n + 1); };
  const set = (patch: Partial<typeof cfg>) => { setRunning(false); setCfg({ ...cfg, ...patch }); setSelT(0); };

  return (
    <LabShell crumb={cfg.cell === 'lstm' ? 'LSTM' : 'RNN'} title={cfg.cell === 'lstm' ? 'LSTM — Long Short-Term Memory' : 'Recurrent Neural Network (RNN)'}
      experiment={{ model: cfg.cell.toUpperCase(), source: cfg.task === 'memory' ? t('ลำดับ ±1 สำหรับงานจำค่าแรก', '±1 sequences for the first-value memory task') : t('ลำดับคลื่น sine สำหรับทำนายค่าถัดไป', 'Sine sequences for next-value prediction'),
        stages: [
          { name: 'Input sequence', values: ex.X[0].map((x) => x[0]), labels: ex.X[0].map((_, i) => `x[${i}]`) },
          { name: 'Hidden states hₜ', kind: 'tensor', shape: [H, T, cfg.layers], values: trace.steps.flatMap((layer) => Array.from({ length: H }, (_, h) => layer.map((step) => step.h[h])).flat()) },
          ...(lstm ? [
            { name: 'Cell memory cₜ', kind: 'tensor' as const, shape: [H, T, cfg.layers] as [number, number, number], values: trace.steps.flatMap((layer) => Array.from({ length: H }, (_, h) => layer.map((step) => step.c[h])).flat()) },
            { name: `i / f / g / o · t=${Math.min(selT, T - 1)}`, kind: 'tensor' as const, shape: [4, H, 1] as [number, number, number], values: [trace.steps[Math.min(selLayer, cfg.layers - 1)][Math.min(selT, T - 1)]].flatMap((step) => [...step.i!, ...step.f!, ...step.g!, ...step.o!]) },
          ] : []),
          { name: 'Readout ŷₜ', values: trace.y.map((y) => y[0]), labels: trace.y.map((_, i) => `ŷ[${i}]`) },
          { name: 'Prediction / target', values: [trace.y[T - 1][0], ex.Y[0][T - 1][0]], labels: ['prediction', 'target'] },
        ],
        metrics: [{ label: 'Training steps', value: String(st.steps) }, { label: 'Example loss', value: fmt(exLoss, 5) }, { label: cfg.task === 'memory' ? 'Eval accuracy' : 'Eval MSE', value: last ? cfg.task === 'memory' ? pct(last.metric) : fmt(last.metric, 5) : '—' }, { label: 'Sequence length', value: String(T) }],
        history: [{ label: 'Evaluation loss (every 25 steps)', values: st.history.map((row) => row.loss) }],
        sample: { index: exSeed - 1, count: 50, onChange: (index) => setExSeed(index + 1) },
        output: cfg.task === 'memory' ? `P(first=+1)=${fmt(trace.y[T - 1][0], 4)} · class=${trace.y[T - 1][0] > 0.5 ? 1 : 0} · target=${ex.Y[0][T - 1][0]}` : `ŷ=${flowVector(trace.y.map((v) => v[0]))} · loss=${fmt(exLoss, 4)}`,
      }}
      chips={[t('ข้อมูลลำดับ', 'Sequence model'), t('ขั้นสูง', 'Advanced')]}
      intro={t('เครือข่ายถูก "คลี่" ตามเวลาไปทางขวา แต่ละคอลัมน์คือ time step เดียวกันที่ใช้น้ำหนักชุดเดิม จุดสีม่วงด้านล่างคือขนาด gradient ที่ไหลย้อนกลับมาถึงแต่ละ step (BPTT) คลิกคอลัมน์เพื่อตรวจค่า', 'The network is unrolled through time to the right; every column is a time step sharing the same weights. Purple dots at the bottom are the gradient magnitude reaching each step (BPTT). Click a column to inspect it.')}
      cam={cam} setCam={setCam} distance={Math.max(16, T * dx * 0.9)} scene={scene}
      readout={{ title: `${cfg.cell.toUpperCase()} · ${cfg.task === 'memory' ? t('งานจำค่าแรก', 'memory task') : t('ทำนายค่าถัดไปของ sine', 'next-value sine')}`, rows: [
        [t('ขั้นการฝึก', 'training steps'), st.steps],
        [cfg.task === 'memory' ? t('ความแม่นยำ (eval 64 ลำดับ)', 'accuracy (64 eval seqs)') : 'MSE (eval)', last ? (cfg.task === 'memory' ? pct(last.metric) : fmt(last.metric, 4)) : '—', last && cfg.task === 'memory' && last.metric > 0.95 ? '#22C55E' : undefined],
        [t('loss ตัวอย่างนี้', 'example loss'), fmt(exLoss, 4)],
        [t('gradient ที่ t=0 / t=T−1', 'grad at t=0 / t=T−1'), `${fmt(grads.gradNorms[0], 4)} / ${fmt(grads.gradNorms[T - 1], 4)}`],
        [t('พารามิเตอร์', 'params'), (st.model.P.reduce((a, p) => a + p.Wx.length + p.Wh.length + p.b.length, 0) + st.model.Wy.length + 1).toLocaleString()],
      ] }}
      caption={cfg.task === 'memory' ? t(`โจทย์: ลำดับ ±1 ยาว ${T} ขั้น ต้องตอบที่ขั้นสุดท้ายว่าค่าแรกเป็น +1 หรือไม่ ข้อมูลต้องถูกจำผ่านทุก step`, `Task: a ±1 sequence of length ${T}; at the last step answer whether the first value was +1. The information must survive every step.`)
        : t('โจทย์: เห็นค่า sin ทีละขั้นแล้วทำนายค่าถัดไป ความถี่และเฟสสุ่มใหม่ทุกลำดับ', 'Task: read a sine wave step by step and predict the next value; frequency and phase are random per sequence.')}
      playback={{ playing: running, onToggle: () => setRunning(!running), onNext: () => { trainStep(); setTick((x) => x + 1); }, onReset: reset, speed, onSpeed: setSpeed, status: `step ${st.steps}` }}
      controls={
        <>
          <Section title={t('โมเดล', 'Model')}>
            <Segmented value={cfg.cell} onChange={(cell) => set({ cell })} options={[{ value: 'rnn', label: 'SimpleRNN' }, { value: 'lstm', label: 'LSTM' }]} />
            <Slider label={t('hidden units', 'Hidden units')} value={cfg.hidden} min={2} max={16} onChange={(hidden) => set({ hidden })} />
            <Slider label={t('จำนวนชั้น', 'Layers')} value={cfg.layers} min={1} max={2} onChange={(layers) => set({ layers })} />
            <p className="text-[0.90625rem] text-muted">{t('activation: tanh (RNN), σ/tanh (gate ของ LSTM) · bidirectional และ dropout ยังไม่รองรับในห้องนี้', 'Activation: tanh (RNN), σ/tanh (LSTM gates) · bidirectional and dropout are not supported here yet')}</p>
          </Section>
          <Section title={t('งาน', 'Task')}>
            <Segmented value={cfg.task} onChange={(task) => set({ task })} options={[{ value: 'memory', label: t('จำค่าแรก', 'Remember first') }, { value: 'sine', label: t('ทำนาย sine', 'Predict sine') }]} />
            <Slider label={t('ความยาวลำดับ T', 'Sequence length T')} value={cfg.T} min={3} max={30} onChange={(T2) => set({ T: T2 })} />
            <Slider label={t('ตัวอย่างที่แสดง', 'Example shown')} value={exSeed} min={1} max={50} onChange={setExSeed} />
          </Section>
          <Section title={t('การฝึก (Adam + BPTT)', 'Training (Adam + BPTT)')}>
            <Slider label="η" value={Math.log10(cfg.lr)} min={-4} max={-0.5} step={0.05} format={(v) => (10 ** v).toPrecision(2)} onChange={(v) => setCfg({ ...cfg, lr: Number((10 ** v).toPrecision(3)) })} />
            <Slider label="Batch" value={cfg.batch} min={1} max={32} onChange={(batch) => setCfg({ ...cfg, batch })} />
            <Slider label={t('Gradient clipping (0 = ปิด)', 'Gradient clipping (0 = off)')} value={cfg.clip} min={0} max={10} step={0.5} onChange={(clip) => setCfg({ ...cfg, clip })} />
            <Slider label={t('ขั้นสูงสุด', 'Max steps')} value={cfg.maxSteps} min={200} max={10000} step={200} onChange={(maxSteps) => setCfg({ ...cfg, maxSteps })} />
            <Slider label="Seed" value={cfg.seed} min={1} max={99} onChange={(seed) => set({ seed })} />
          </Section>
          <Section title={t('ตรวจ time step', 'Inspect a time step')}>
            <Slider label="t" value={selT} min={0} max={T - 1} onChange={setSelT} />
            {cfg.layers > 1 && <Slider label={t('ชั้น', 'Layer')} value={selLayer} min={0} max={cfg.layers - 1} onChange={setSelLayer} />}
          </Section>
        </>
      }
      tabs={[
        { id: 'inspect', label: t('ตรวจ gate และ state', 'Gates & state'), content: (
          <div className="grid gap-8 lg:grid-cols-[1.3fr_1fr]">
            <div className="scroll-thin overflow-x-auto">
              <p className="mb-2 font-mono text-[1rem] text-muted">t = {selT} · x_t = {fmt(s.x[0], 3)}</p>
              <table className="num w-full min-w-[460px] text-left font-mono text-[0.96875rem]">
                <thead className="text-muted"><tr className="border-b border-line"><th className="py-1">j</th><th>h_(t−1)</th>{lstm && <>{GATES.map((g) => <th key={g} style={{ color: GATE_COL[g] }}>{g}</th>)}<th>c_(t−1)</th><th>c_t</th></>}<th>h_t</th></tr></thead>
                <tbody>{s.h.map((h, j) => (
                  <tr key={j} className="border-b border-line/50"><td className="py-1">{j}</td><td>{fmt(s.hPrev[j], 3)}</td>
                    {lstm && <>{GATES.map((g) => <td key={g}>{fmt(s[g]![j], 3)}</td>)}<td>{fmt(s.cPrev[j], 3)}</td><td>{fmt(s.c[j], 3)}</td></>}
                    <td>{fmt(h, 3)}</td></tr>
                ))}</tbody>
              </table>
              {lstm && <Tex block>{`c_0 = f_0 c_{0}^{prev} + i_0 g_0 = (${fmt(s.f![0], 2)})(${fmt(s.cPrev[0], 2)}) + (${fmt(s.i![0], 2)})(${fmt(s.g![0], 2)}) = ${fmt(s.c[0], 3)}`}</Tex>}
            </div>
            <div className="flex flex-col gap-3">
              {lstm ? (
                <>
                  <Tex block>{'\\mathbf i = \\sigma(W_i[\\mathbf x_t, \\mathbf h_{t-1}] + \\mathbf b_i),\\ \\mathbf f = \\sigma(\\cdot),\\ \\mathbf o = \\sigma(\\cdot),\\ \\mathbf g = \\tanh(\\cdot)'}</Tex>
                  <Tex block>{'\\mathbf c_t = \\mathbf f\\odot\\mathbf c_{t-1} + \\mathbf i\\odot\\mathbf g,\\qquad \\mathbf h_t = \\mathbf o\\odot\\tanh(\\mathbf c_t)'}</Tex>
                  <p className="text-[1.0625rem] text-muted">{t('เส้นสีส้มตามแนวนอนคือ cell state ความสว่างคือค่า forget gate f ถ้า f ≈ 1 ข้อมูลไหลผ่านได้ไกลโดย gradient ไม่หาย', 'The horizontal orange line is the cell state; its brightness is the forget gate f. With f ≈ 1 information (and gradient) travels far without vanishing.')}</p>
                </>
              ) : (
                <>
                  <Tex block>{'\\mathbf h_t = \\tanh(W_x\\mathbf x_t + W_h\\mathbf h_{t-1} + \\mathbf b)'}</Tex>
                  <Tex block>{'\\frac{\\partial \\mathbf h_t}{\\partial \\mathbf h_{k}} = \\prod_{s=k+1}^{t} \\mathrm{diag}(1-\\mathbf h_s^2)\\,W_h'}</Tex>
                  <p className="text-[1.0625rem] text-muted">{t('ผลคูณของเมทริกซ์ซ้ำ ๆ นี้ทำให้ gradient ที่ย้อนไปไกลหดลงแบบเอกซ์โพเนนเชียล (vanishing gradient) ดูจุดสีม่วงที่ t ต้น ๆ', 'Repeated multiplication shrinks gradients exponentially as they go back in time (vanishing gradient): look at the purple dots at early t.')}</p>
                </>
              )}
            </div>
          </div>
        ) },
        { id: 'train', label: t('การฝึก', 'Training'), content: (
          <div className="grid gap-8 lg:grid-cols-2">
            <div><h3 className="label-xs mb-2">{cfg.task === 'memory' ? t('ความแม่นยำบนชุดประเมิน', 'Eval accuracy') : 'Eval MSE'}</h3>
              <LinePlot xTitle="step" yTitle={cfg.task === 'memory' ? 'accuracy' : 'MSE'} yRange={cfg.task === 'memory' ? [0, 1.02] : undefined} series={[{ name: cfg.cell, x: st.history.map((h) => h.step), y: st.history.map((h) => h.metric), color: '#22C55E' }]} /></div>
            <div><h3 className="label-xs mb-2">{cfg.task === 'sine' ? t('ตัวอย่าง: ค่าจริงเทียบกับที่ทำนาย', 'Example: target vs prediction') : t('ขนาด gradient ต่อ time step (ตัวอย่างนี้)', 'Gradient size per time step (this example)')}</h3>
              {cfg.task === 'sine'
                ? <LinePlot xTitle="t" yTitle="value" series={[{ name: 'target x_(t+1)', x: ex.Y[0].map((_, i) => i), y: ex.Y[0].map((v) => v[0]), color: '#94A3B8', dash: 'dot' }, { name: 'prediction', x: trace.y.map((_, i) => i), y: trace.y.map((v) => v[0]), color: '#00D9FF' }]} />
                : <LinePlot xTitle="t" yTitle="‖∂L/∂h_t‖" series={[{ name: cfg.cell, x: grads.gradNorms.map((_, i) => i), y: grads.gradNorms, color: GRAD }]} />}
            </div>
          </div>
        ) },
        { id: 'eval', label: t('สรุป', 'Summary'), content: (
          <Metrics items={[[t('ขั้น', 'Steps'), String(st.steps)], [cfg.task === 'memory' ? 'Accuracy' : 'MSE', last ? (cfg.task === 'memory' ? pct(last.metric) : fmt(last.metric, 4)) : '—'], ['grad t=0', fmt(grads.gradNorms[0], 4)], ['grad t=T−1', fmt(grads.gradNorms[T - 1], 4)]]} />
        ) },
        { id: 'ex', label: t('แบบฝึกหัด', 'Exercises'), content: (
          <div className="grid gap-8 lg:grid-cols-2">
            <ol className="list-decimal space-y-2 pl-5 text-[1.125rem]">
              <li>{t('งานจำค่าแรก T = 20: ฝึก SimpleRNN 3000 ขั้น แล้วเปลี่ยนเป็น LSTM ด้วยค่าเดิม เทียบความแม่นยำ', 'Memory task, T = 20: train SimpleRNN for 3000 steps, then LSTM with the same settings; compare accuracy.')}</li>
              <li>{t('ดูจุด gradient สีม่วง: ใน RNN ค่าที่ t = 0 เล็กกว่าที่ t = T−1 กี่เท่า', 'Look at the purple gradient dots: how much smaller is it at t = 0 than at t = T−1 for the RNN?')}</li>
              <li>{t('ปิด gradient clipping และเพิ่ม η ดูว่าการฝึกเสถียรไหม', 'Turn off gradient clipping and raise η; is training stable?')}</li>
            </ol>
            <Quiz id={`seq-${cfg.cell}`} items={[
              { q: t('ทำไม SimpleRNN จำข้อมูลระยะยาวได้ยาก', 'Why do SimpleRNNs struggle with long-range memory?'), options: [t('gradient หายเมื่อย้อนผ่านหลาย step', 'Gradients vanish through many steps'), t('มีพารามิเตอร์มากเกินไป', 'Too many parameters')], answer: 0, why: t('ผลคูณของ Jacobian ที่มีขนาดน้อยกว่า 1 ซ้ำ ๆ', 'Repeated products of Jacobians with norm below 1.') },
              { q: t('forget gate f = 1 และ input gate i = 0 หมายถึง', 'Forget gate f = 1 and input gate i = 0 means'), options: [t('cell state ถูกคงไว้เหมือนเดิม', 'the cell state is kept unchanged'), t('cell state ถูกล้าง', 'the cell state is erased')], answer: 0, why: 'c_t = 1·c_(t−1) + 0·g' },
              { q: t('weight sharing ใน RNN หมายถึง', 'Weight sharing in an RNN means'), options: [t('ทุก time step ใช้ W ชุดเดียวกัน', 'every time step uses the same W'), t('ทุกชั้นใช้ W เดียวกัน', 'every layer uses the same W')], answer: 0, why: t('จึงใช้กับลำดับความยาวเท่าใดก็ได้', 'So it works for any sequence length.') },
            ]} />
          </div>
        ) },
      ]}
    />
  );
}
