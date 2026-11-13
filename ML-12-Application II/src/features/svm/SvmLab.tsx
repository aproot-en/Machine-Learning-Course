import { useMemo, useState } from 'react';
import * as THREE from 'three';
import { useCameraState } from '../../engine3d/Stage';
import { LabShell, Metrics } from '../../components/LabShell';
import { kernel } from '../../ml/svm';
import { HeightSurface, OrientedPlane, PointCloud, Segments, boxClip, contour, planeMapper, rgb } from '../../engine3d/plots';
import { BOX, FrameBox, Label } from '../../engine3d/primitives';
import { CLASS_COLORS } from '../../engine3d/palette';
import { DATASETS, applyScaler, datasetInfo, fitScaler, makeDataset, trainTestSplit, type DatasetId } from '../../data/datasets';
import { SMO, type Kernel, type SvmParams } from '../../ml/svm';
import { classificationReport } from '../../ml/metrics';
import { useSettings, useT } from '../../i18n/i18n';
import { useInterval } from '../../lib/useLoop';
import { Button, Section, Segmented, Select, Slider, Toggle, fmt, pct } from '../../components/ui';
import { Callout, ConfusionMatrix, Quiz, Tex } from '../../components/common';

const R = 5;
const SVM_DATA: DatasetId[] = ['blobs2d', 'moons', 'circles', 'xor', 'blobs3d', 'iris'];

export default function SvmLab() {
  const t = useT();
  const level = useSettings((s) => s.level);
  const [ds, setDs] = useState({ id: 'moons' as DatasetId, n: 140, noise: 0.15, seed: 4, testRatio: 0.25 });
  const [dims, setDims] = useState<2 | 3>(2);
  const [p, setP] = useState<SvmParams>({ kernel: 'rbf', C: 2, gamma: 1, degree: 3, coef0: 1, tol: 1e-3, classWeight: [1, 1] });
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [view, setView] = useState<number | null>(null);
  const [surface, setSurface] = useState(true);
  const [query, setQuery] = useState<number[] | null>(null);
  const [nonce, setNonce] = useState(0);
  const [, setTick] = useState(0);
  const [cam, setCam] = useCameraState();
  const [experimentSample, setExperimentSample] = useState(0);
  const info = datasetInfo(ds.id);

  // ---- data: binary task; Iris uses versicolor (+1) vs virginica (−1) ----
  const data = useMemo(() => {
    const raw = makeDataset(ds.id, { n: ds.n, classes: 2, noise: ds.noise, seed: ds.seed });
    let idx = raw.X.map((_, i) => i);
    let pos = 0;
    if (ds.id === 'iris') { idx = idx.filter((i) => raw.y[i] > 0); pos = 1; }
    const fdim = dims === 3 && raw.X[0].length >= 3 ? 3 : 2;
    const feats = ds.id === 'iris' ? (fdim === 3 ? [1, 2, 3] : [2, 3]) : [0, 1, 2].slice(0, fdim);
    const X = idx.map((i) => feats.map((f) => raw.X[i][f]));
    const yy = idx.map((i) => (raw.y[i] === pos ? 1 : -1));
    const sp = trainTestSplit(yy.map((v) => (v > 0 ? 0 : 1)), ds.testRatio, ds.seed);
    const sc = fitScaler(sp.train.map((i) => X[i]));
    const Xs = X.map((r) => applyScaler(sc, r));
    const names = ds.id === 'iris' ? ['versicolor', 'virginica'] : ['A', 'B'];
    return { Xtr: sp.train.map((i) => Xs[i]), ytr: sp.train.map((i) => yy[i]), Xte: sp.test.map((i) => Xs[i]), yte: sp.test.map((i) => yy[i]), fdim, names, featNames: feats.map((f) => raw.featureNames[f]) };
  }, [ds, dims]);

  const model = useMemo(() => {
    const m = new SMO(data.Xtr, data.ytr, p, ds.seed);
    return { m, snaps: [{ alpha: m.alpha.slice(), b: m.b, iter: 0 }] };
  }, [data, p, ds.seed, nonce]); // eslint-disable-line react-hooks/exhaustive-deps
  const { m, snaps } = model;
  const live = view == null || view >= snaps.length - 1;
  const shown = live ? snaps[snaps.length - 1] : snaps[view!];
  // a lightweight view of the model at the displayed pass
  const f = useMemo(() => {
    const alpha = live ? m.alpha : shown.alpha, b = live ? m.b : shown.b;
    const sv = [...alpha.keys()].filter((i) => alpha[i] > 1e-6);
    const k = (a: number[], c: number[]) => {
      if (p.kernel === 'rbf') { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - c[i]) ** 2; return Math.exp(-p.gamma * s); }
      let d = 0; for (let i = 0; i < a.length; i++) d += a[i] * c[i];
      return p.kernel === 'linear' ? d : p.kernel === 'poly' ? (p.gamma * d + p.coef0) ** p.degree : Math.tanh(p.gamma * d + p.coef0);
    };
    const decision = (x: number[]) => sv.reduce((s, i) => s + alpha[i] * m.y[i] * k(m.X[i], x), b);
    const w = Array(data.fdim).fill(0);
    sv.forEach((i) => m.X[i].forEach((v, d) => (w[d] += alpha[i] * m.y[i] * v)));
    return { decision, sv, alpha, b, w };
  }, [m, shown, live, p, data.fdim, snaps.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const step = () => {
    if (m.done) return false;
    m.pass();
    snaps.push({ alpha: m.alpha.slice(), b: m.b, iter: m.iterations });
    setTick((x) => x + 1);
    return !m.done;
  };
  useInterval(running, 160, speed, step, () => setRunning(false));

  const map = useMemo(() => planeMapper([...data.Xtr, ...data.Xte], R, 0.4), [data]);
  const scale3 = useMemo(() => R / Math.max(1e-6, ...[...data.Xtr, ...data.Xte].flat().map(Math.abs)), [data]);
  const toScene = (x: number[]): [number, number, number] => {
    if (data.fdim === 3) return [x[0] * scale3, x[1] * scale3, x[2] * scale3];
    const [a, c] = map.toScene(x);
    return [a, 0.02, c];
  };

  // decision surface over the 2D plane
  const G = 56;
  const grid = useMemo(() => {
    if (data.fdim !== 2) return null;
    const vals = new Float32Array((G + 1) * (G + 1)), heights = new Float32Array(vals.length), cols = new Float32Array(vals.length * 3);
    const A = rgb(CLASS_COLORS[0]), B = rgb(CLASS_COLORS[1]);
    for (let r = 0; r <= G; r++) for (let c = 0; c <= G; c++) {
      const v = r * (G + 1) + c;
      const x = map.toData((c / G - 0.5) * 2 * R, (r / G - 0.5) * 2 * R);
      const fv = f.decision(x);
      vals[v] = fv;
      heights[v] = Math.max(-2.5, Math.min(2.5, fv)) * 0.9;
      const t = 1 / (1 + Math.exp(-2 * fv)), base = t > 0.5 ? A : B, s = 0.25 + 0.6 * Math.abs(t - 0.5) * 2;
      cols[v * 3] = base[0] * s; cols[v * 3 + 1] = base[1] * s; cols[v * 3 + 2] = base[2] * s;
    }
    const seg = (lvl: number) => contour(G, vals, lvl).map(([c0, r0, c1, r1]) => [[(c0 / G - 0.5) * 2 * R, 0.03, (r0 / G - 0.5) * 2 * R], [(c1 / G - 0.5) * 2 * R, 0.03, (r1 / G - 0.5) * 2 * R]] as [[number, number, number], [number, number, number]]);
    return { heights, cols, zero: seg(0), plus: seg(1), minus: seg(-1) };
  }, [f, map, data.fdim]);

  // boundary samples for non-linear 3D models
  const boundary3 = useMemo(() => {
    if (data.fdim !== 3 || p.kernel === 'linear') return [];
    const N = 22, pts: [number, number, number][] = [];
    for (let a = 0; a < N; a++) for (let b = 0; b < N; b++) for (let c = 0; c < N; c++) {
      const s = [((a + 0.5) / N - 0.5) * 2 * R, ((b + 0.5) / N - 0.5) * 2 * R, ((c + 0.5) / N - 0.5) * 2 * R];
      if (Math.abs(f.decision(s.map((v) => v / scale3))) < 0.12) pts.push(s as [number, number, number]);
    }
    return pts;
  }, [f, data.fdim, p.kernel, scale3]);

  const predTr = data.Xtr.map((x) => (f.decision(x) >= 0 ? 1 : -1));
  const predTe = data.Xte.map((x) => (f.decision(x) >= 0 ? 1 : -1));
  const accTr = predTr.filter((v, i) => v === data.ytr[i]).length / data.ytr.length;
  const report = classificationReport(data.yte.map((v) => (v > 0 ? 0 : 1)), predTe.map((v) => (v > 0 ? 0 : 1)), 2);
  const nw = Math.hypot(...f.w);
  const qf = query ? f.decision(query) : null;
  const clip = useMemo(() => boxClip(R), []);

  const scene = (
    <group>
      <FrameBox size={R} labels={data.fdim === 3 ? data.featNames : [data.featNames[0], 'f(x)', data.featNames[1]]} />
      <PointCloud points={data.Xtr.map(toScene)} colors={data.ytr.map((v) => CLASS_COLORS[v > 0 ? 0 : 1])} sizes={0.16} />
      <PointCloud points={data.Xte.map(toScene)} colors={data.yte.map((v) => new THREE.Color(CLASS_COLORS[v > 0 ? 0 : 1]).multiplyScalar(0.7))} sizes={0.2} geometry={BOX} glow={0.15} />
      <PointCloud points={f.sv.map((i) => toScene(m.X[i]))} colors={f.sv.map((i) => CLASS_COLORS[m.y[i] > 0 ? 0 : 1])} sizes={0.34} basic wire opacity={0.7} />
      <PointCloud points={data.Xtr.map(toScene).filter((_, i) => predTr[i] !== data.ytr[i])} colors={data.Xtr.map(() => '#EF4444')} sizes={0.3} geometry={BOX} basic wire opacity={0.9} />
      {grid && surface && <HeightSurface G={G} heights={grid.heights} colors={grid.cols} size={2 * R} opacity={0.55} />}
      {grid && <Segments segs={[...grid.zero.map(([a, b]) => ({ a, b, c: '#ffffff' })), ...grid.plus.map(([a, b]) => ({ a, b, c: CLASS_COLORS[0], i: 0.6 })), ...grid.minus.map(([a, b]) => ({ a, b, c: CLASS_COLORS[1], i: 0.6 }))]} />}
      {data.fdim === 3 && p.kernel === 'linear' && nw > 1e-6 && (
        <>
          <OrientedPlane normal={[f.w[0], f.w[1], f.w[2]]} d={-f.b * scale3} color="#E5E7EB" clip={clip} />
          <OrientedPlane normal={[f.w[0], f.w[1], f.w[2]]} d={(1 - f.b) * scale3} color={CLASS_COLORS[0]} opacity={0.05} wire={0.1} clip={clip} />
          <OrientedPlane normal={[f.w[0], f.w[1], f.w[2]]} d={(-1 - f.b) * scale3} color={CLASS_COLORS[1]} opacity={0.05} wire={0.1} clip={clip} />
        </>
      )}
      {boundary3.length > 0 && <PointCloud points={boundary3} colors={boundary3.map(() => '#E5E7EB')} sizes={0.06} basic opacity={0.6} />}
      {data.fdim === 2 && (
        <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, 0, 0]} onClick={(e) => { e.stopPropagation(); setQuery(map.toData(e.point.x, e.point.z)); }}>
          <planeGeometry args={[2 * R, 2 * R]} /><meshBasicMaterial transparent opacity={0} depthWrite={false} />
        </mesh>
      )}
      {query && qf != null && (() => { const s = toScene(query); return (
        <group>
          <mesh position={[s[0], 0.25, s[2]]}><octahedronGeometry args={[0.3]} /><meshStandardMaterial color={CLASS_COLORS[qf >= 0 ? 0 : 1]} emissive={CLASS_COLORS[qf >= 0 ? 0 : 1]} emissiveIntensity={0.8} /></mesh>
          <Label position={[s[0], 0.9, s[2]]}>f(x) = {qf.toFixed(3)} → {data.names[qf >= 0 ? 0 : 1]}</Label>
        </group>
      ); })()}
    </group>
  );

  const reset = () => { setRunning(false); setView(null); setNonce((n) => n + 1); };
  const set = (patch: Partial<SvmParams>) => { setP({ ...p, ...patch }); setView(null); setRunning(false); };
  const logSlider = (label: string, v: number, lo: number, hi: number, on: (v: number) => void) => (
    <Slider label={label} value={Math.log10(v)} min={lo} max={hi} step={0.05} format={(x) => (10 ** x).toPrecision(3)} onChange={(x) => on(Number((10 ** x).toPrecision(3)))} />
  );

  return (
    <LabShell crumb="SVM" title="Support Vector Machine" chips={[t('การเรียนรู้แบบมีผู้สอน', 'Supervised'), t('ปานกลาง', 'Intermediate')]}
      experiment={{ model: 'SVM', source: `${t(info.name)} · ${data.Xtr.length + data.Xte.length} ${t('ตัวอย่าง', 'samples')}`,
        stages: [
          { name: 'Input x', values: query ?? data.Xte[Math.min(experimentSample, data.Xte.length - 1)] },
          { name: 'Support vectors', kind: 'points', points: f.sv.map((i) => data.Xtr[i]), values: f.sv.map((i) => f.alpha[i]), classes: f.sv.map((i) => data.ytr[i] > 0 ? 0 : 1) },
          { name: `${p.kernel} K(xᵢ,x)`, values: f.sv.map((i) => kernel(data.Xtr[i], query ?? data.Xte[Math.min(experimentSample, data.Xte.length - 1)], p)) },
          { name: 'αᵢ yᵢ K(xᵢ,x)', values: f.sv.map((i) => f.alpha[i] * data.ytr[i] * kernel(data.Xtr[i], query ?? data.Xte[Math.min(experimentSample, data.Xte.length - 1)], p)) },
          { name: 'Σ + b → f(x)', values: [f.decision(query ?? data.Xte[Math.min(experimentSample, data.Xte.length - 1)])], labels: ['f(x)'] },
          { name: 'Class', values: [f.decision(query ?? data.Xte[Math.min(experimentSample, data.Xte.length - 1)]) >= 0 ? 1 : 0, f.decision(query ?? data.Xte[Math.min(experimentSample, data.Xte.length - 1)]) < 0 ? 1 : 0], labels: data.names },
        ],
        metrics: [{ label: 'Train accuracy', value: pct(accTr) }, { label: 'Test accuracy', value: pct(report.accuracy) }, { label: 'SMO passes', value: String(shown.iter) }, { label: 'Support vectors', value: String(f.sv.length) }],
        confusion: report.confusion,
        sample: { index: experimentSample, count: data.Xte.length, onChange: (index) => { setExperimentSample(index); setQuery(null); } },
        output: `${t('คลาส', 'Class')}: ${data.names[f.decision(query ?? data.Xte[Math.min(experimentSample, data.Xte.length - 1)]) >= 0 ? 0 : 1]} · test accuracy ${pct(report.accuracy)}`,
      }}
      intro={t('กด ▶ เพื่อฝึกด้วย SMO ทีละรอบ พื้นผิวคือค่า f(x) ที่คำนวณจาก support vector เส้นสีขาวคือ f(x) = 0 คลิกที่พื้นเพื่อจำแนกจุดใหม่', 'Press ▶ to train with SMO pass by pass. The surface is f(x) computed from the support vectors; the white line is f(x) = 0. Click the floor to classify a new point.')}
      cam={cam} setCam={setCam} distance={17} scene={scene}
      readout={{ title: `SVM · ${p.kernel}`, rows: [
        [t('ชุดข้อมูล', 'dataset'), `${t(info.name)}`],
        ['C · γ', `${p.C} · ${p.kernel === 'linear' ? '—' : p.gamma}`],
        [t('รอบ SMO', 'SMO passes'), `${shown.iter}${m.done && live ? ' ✓' : ''}`],
        ['support vectors', `${f.sv.length} / ${data.Xtr.length}`],
        ...(p.kernel === 'linear' ? [['margin 2/‖w‖', nw > 1e-9 ? fmt(2 / nw, 3) : '—'] as [string, string]] : []),
        ['b', fmt(f.b, 3)],
        ['acc train / test', `${pct(accTr)} / ${pct(report.accuracy)}`],
      ] }}
      playback={{ playing: running, onToggle: () => { if (!live) { setView(null); } setRunning(!running); }, onNext: () => { if (!live) setView(view! + 1); else step(); },
        onPrev: () => { setRunning(false); setView(Math.max(0, (view ?? snaps.length - 1) - 1)); }, canPrev: (view ?? snaps.length - 1) > 0, onReset: reset, speed, onSpeed: setSpeed,
        status: live ? (m.done ? t('ลู่เข้าแล้ว', 'converged') : `pass ${m.iterations}`) : t(`ดูรอบที่ ${shown.iter}`, `viewing pass ${shown.iter}`) }}
      caption={<>{t('รูปทรงลวด = support vector (α > 0) · กรอบแดง = จุด train ที่จำแนกผิด · ลูกบาศก์ = จุด test', 'Wire spheres = support vectors (α > 0) · red frames = misclassified training points · cubes = test points')}</>}
      controls={
        <>
          <Section title={t('ชุดข้อมูล', 'Dataset')}>
            <Select label={t('ชุดข้อมูล', 'Dataset')} value={ds.id} onChange={(id) => { setDs({ ...ds, id }); setQuery(null); }} options={DATASETS.filter((d) => SVM_DATA.includes(d.id)).map((d) => ({ value: d.id, label: t(d.name) + (d.id === 'iris' ? ' · versicolor vs virginica' : '') }))} />
            {info.synthetic && <Slider label={t('จำนวนตัวอย่าง', 'Samples')} value={ds.n} min={40} max={300} step={10} onChange={(n) => setDs({ ...ds, n })} />}
            {info.synthetic && <Slider label="Noise" value={ds.noise} min={0} max={0.6} step={0.05} format={(v) => v.toFixed(2)} onChange={(noise) => setDs({ ...ds, noise })} />}
            <Slider label="Seed" value={ds.seed} min={1} max={99} onChange={(seed) => setDs({ ...ds, seed })} />
            <Segmented label={t('จำนวนฟีเจอร์', 'Features')} value={String(dims) as '2' | '3'} onChange={(v) => setDims(Number(v) as 2 | 3)} options={[{ value: '2', label: t('2 ฟีเจอร์ + พื้นผิว f(x)', '2 + f(x) surface') }, { value: '3', label: t('3 ฟีเจอร์', '3 features') }]} />
            {dims === 3 && info.features < 3 && <p className="text-[0.9375rem] text-orange">{t('ชุดข้อมูลนี้มี 2 ฟีเจอร์ เลือก Blobs 3D หรือ Iris', 'This dataset has 2 features; choose Blobs 3D or Iris')}</p>}
          </Section>
          <Section title={t('โมเดล', 'Model')}>
            <Select<Kernel> label="Kernel" value={p.kernel} onChange={(kernel) => set({ kernel })} options={[{ value: 'linear', label: 'Linear' }, { value: 'rbf', label: 'RBF (Gaussian)' }, { value: 'poly', label: 'Polynomial' }, { value: 'sigmoid', label: 'Sigmoid' }]} />
            {logSlider('C', p.C, -2, 3, (C) => set({ C }))}
            {p.kernel !== 'linear' && logSlider('γ (gamma)', p.gamma, -2, 1.5, (gamma) => set({ gamma }))}
            {p.kernel === 'poly' && <Slider label="degree" value={p.degree} min={2} max={5} onChange={(degree) => set({ degree })} />}
            {(p.kernel === 'poly' || p.kernel === 'sigmoid') && <Slider label="coef0" value={p.coef0} min={-2} max={2} step={0.1} format={(v) => v.toFixed(1)} onChange={(coef0) => set({ coef0 })} />}
            {level >= 2 && <Slider label={t(`น้ำหนักคลาส ${data.names[1]}`, `Class weight ${data.names[1]}`)} value={p.classWeight[1]} min={0.2} max={5} step={0.1} format={(v) => v.toFixed(1)} onChange={(v) => set({ classWeight: [1, v] })} />}
            {level >= 3 && logSlider('tolerance', p.tol, -4, -1, (tol) => set({ tol }))}
            {p.kernel === 'sigmoid' && <p className="text-[0.9375rem] text-muted">{t('kernel sigmoid ไม่ได้เป็น positive semi-definite เสมอ SMO อาจไม่ลู่เข้าสู่คำตอบที่ดี', 'The sigmoid kernel is not always positive semi-definite; SMO may not reach a good solution.')}</p>}
          </Section>
          <Section title={t('การแสดงผล', 'View')}>
            <Toggle label={t('พื้นผิว f(x)', 'f(x) surface')} checked={surface} onChange={setSurface} />
            {query && <Button size="sm" onClick={() => setQuery(null)}>{t('ลบจุดใหม่', 'Clear new point')}</Button>}
            {data.fdim === 3 && p.kernel !== 'linear' && <p className="text-[0.9375rem] text-muted">{t('จุดขาวคือตำแหน่งบนกริด 22³ ที่ |f(x)| < 0.12 ซึ่งประมาณผิว f(x) = 0', 'White dots are 22³ grid points with |f(x)| < 0.12, approximating the surface f(x) = 0.')}</p>}
          </Section>
        </>
      }
      tabs={[
        { id: 'concept', label: t('แนวคิด', 'Concept'), content: (
          <div className="grid max-w-5xl gap-8 text-[1.125rem] leading-7 lg:grid-cols-2">
            <div className="space-y-3">
              <p>{t('SVM หาเส้นแบ่ง (หรือระนาบ) ที่อยู่ห่างจากจุดใกล้สุดของทั้งสองคลาสมากที่สุด ระยะนั้นเรียกว่า margin จุดที่อยู่บนขอบ margin หรือล้ำเข้ามาคือ support vector มีเพียงจุดเหล่านี้ที่มี α > 0 และกำหนดคำตอบ', 'SVM finds the boundary that stays as far as possible from the nearest points of both classes; that distance is the margin. Points on or inside the margin are support vectors, the only points with α > 0, and they alone define the solution.')}</p>
              <p>{t('Soft margin: ค่า C กำหนดว่ายอมให้จุดล้ำ margin ได้มากแค่ไหน C ใหญ่ = เข้มงวด margin แคบ C เล็ก = margin กว้างขึ้น ทนต่อ noise', 'Soft margin: C sets how much margin violation is tolerated. Large C = strict, narrow margin; small C = wider margin, more tolerant of noise.')}</p>
            </div>
            <div className="space-y-3">
              <p><b>Kernel trick:</b> {t('แทนที่จะแปลงข้อมูลไปมิติสูงจริง ๆ SVM ใช้เพียงผลคูณภายใน K(x, x′) ในมิตินั้น เส้นแบ่งที่เห็นโค้งในภาพนี้คือระนาบตรงในพื้นที่ฟีเจอร์ของ kernel (ซึ่งเป็นมิติสูงหรือไม่จำกัดสำหรับ RBF)', 'Instead of mapping data to a high-dimensional space, SVM only needs inner products K(x, x′) there. The curved boundary you see is a flat hyperplane in the kernel’s feature space (infinite-dimensional for RBF).')}</p>
              <Callout>{t('พื้นผิวที่ยกขึ้นคือค่าของฟังก์ชันตัดสินใจ f(x) จริง ไม่ใช่การฉายข้อมูลไปมิติสูง', 'The raised surface is the real decision function f(x), not a projection of the data into a higher dimension.')}</Callout>
            </div>
          </div>
        ) },
        { id: 'math', label: t('คณิตศาสตร์', 'Math'), content: (
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="flex flex-col gap-3">
              <h3 className="label-xs">Primal (soft margin)</h3>
              <Tex block>{'\\min_{\\mathbf w,b,\\xi}\\; \\tfrac12\\|\\mathbf w\\|^2 + C\\sum_i \\xi_i \\quad\\text{s.t. } y_i(\\mathbf w\\cdot\\mathbf x_i+b)\\ge 1-\\xi_i,\\ \\xi_i\\ge0'}</Tex>
              <h3 className="label-xs">Dual · {t('สิ่งที่ SMO แก้', 'what SMO solves')}</h3>
              <Tex block>{'\\max_{\\alpha}\\; \\sum_i\\alpha_i - \\tfrac12\\sum_{i,j}\\alpha_i\\alpha_j y_i y_j K(\\mathbf x_i,\\mathbf x_j),\\quad 0\\le\\alpha_i\\le C,\\ \\sum_i \\alpha_i y_i = 0'}</Tex>
              <h3 className="label-xs">{t('ฟังก์ชันตัดสินใจ', 'Decision function')}</h3>
              <Tex block>{'f(\\mathbf x) = \\sum_{i\\in SV}\\alpha_i y_i K(\\mathbf x_i, \\mathbf x) + b'}</Tex>
            </div>
            <div className="flex flex-col gap-3">
              <h3 className="label-xs">Kernels</h3>
              <Tex block>{'K_{lin} = \\mathbf x\\cdot\\mathbf x\',\\quad K_{rbf} = e^{-\\gamma\\|\\mathbf x-\\mathbf x\'\\|^2}'}</Tex>
              <Tex block>{'K_{poly} = (\\gamma\\,\\mathbf x\\cdot\\mathbf x\' + c_0)^d,\\quad K_{sig} = \\tanh(\\gamma\\,\\mathbf x\\cdot\\mathbf x\' + c_0)'}</Tex>
              <p className="font-mono text-[1rem] text-muted">{t('ค่า dual objective ตอนนี้', 'Current dual objective')}: {live ? fmt(m.dualObjective(), 4) : '—'}</p>
              {p.kernel === 'linear' && <Tex block>{`\\mathbf w = \\sum \\alpha_i y_i \\mathbf x_i = (${f.w.map((v) => fmt(v, 3)).join(', ')}),\\ \\text{margin} = 2/\\|\\mathbf w\\| = ${nw > 1e-9 ? fmt(2 / nw, 3) : '\\infty'}`}</Tex>}
            </div>
          </div>
        ) },
        { id: 'inspect', label: t('ตรวจการคำนวณ', 'Calculation inspector'), content: (
          <div className="grid gap-8 lg:grid-cols-2">
            <div className="scroll-thin max-h-[420px] overflow-auto">
              <table className="num w-full text-left font-mono text-[0.96875rem]">
                <thead className="sticky top-0 bg-bg text-muted"><tr><th className="py-1.5">#</th><th>y</th><th>α</th><th>y·f(x)</th><th>{t('สถานะ', 'status')}</th></tr></thead>
                <tbody>{f.sv.map((i) => { const yf = m.y[i] * f.decision(m.X[i]); return (
                  <tr key={i} className="border-b border-line/50"><td className="py-1">{i}</td><td style={{ color: CLASS_COLORS[m.y[i] > 0 ? 0 : 1] }}>{m.y[i] > 0 ? '+1' : '−1'}</td><td>{fmt(f.alpha[i], 4)}</td><td>{fmt(yf, 3)}</td>
                    <td className="text-muted">{f.alpha[i] >= m.Ci(i) - 1e-6 ? (yf < 0 ? t('ผิดฝั่ง (α = C)', 'wrong side (α = C)') : t('ในขอบ (α = C)', 'inside margin (α = C)')) : t('บนขอบ margin', 'on the margin')}</td></tr>
                ); })}</tbody>
              </table>
              <p className="mt-2 text-[0.96875rem] text-muted">{t(`จุดที่เหลือ ${data.Xtr.length - f.sv.length} จุดมี α = 0 ลบทิ้งได้โดยคำตอบไม่เปลี่ยน`, `The other ${data.Xtr.length - f.sv.length} points have α = 0 and could be removed without changing the answer.`)}</p>
            </div>
            <div className="flex flex-col gap-3">
              <h3 className="label-xs">{t('จุดใหม่ (คลิกที่พื้นในฉาก)', 'New point (click the floor)')}</h3>
              {query && qf != null ? (
                <>
                  <p className="font-mono text-[1rem]">x = ({query.map((v) => fmt(v, 3)).join(', ')})</p>
                  <Tex block>{`f(\\mathbf x) = ${f.sv.slice(0, 3).map((i) => `(${fmt(f.alpha[i], 2)})(${m.y[i] > 0 ? '+1' : '-1'})K_{${i}}`).join(' + ')}${f.sv.length > 3 ? ' + \\dots' : ''} + (${fmt(f.b, 2)}) = ${fmt(qf, 4)}`}</Tex>
                  <p>{t('ผลทำนาย', 'Prediction')}: <b style={{ color: CLASS_COLORS[qf >= 0 ? 0 : 1] }}>{data.names[qf >= 0 ? 0 : 1]}</b> ({t('ระยะจากเส้นแบ่งในหน่วยของ margin', 'distance in margin units')} |f| = {fmt(Math.abs(qf), 3)})</p>
                </>
              ) : <p className="text-[1.0625rem] text-muted">{t('ใช้ได้ในโหมด 2 ฟีเจอร์', 'Available in 2-feature mode')}</p>}
            </div>
          </div>
        ) },
        { id: 'eval', label: t('การประเมินผล', 'Evaluation'), content: (
          <div className="grid gap-8 lg:grid-cols-2">
            <Metrics items={[[t('ความแม่นยำ test', 'Test accuracy'), pct(report.accuracy)], ['Precision', pct(report.precision)], ['Recall', pct(report.recall)], ['F1', pct(report.f1)]]} />
            <div><h3 className="label-xs mb-2">Confusion matrix (test)</h3><ConfusionMatrix m={report.confusion} classNames={data.names} /></div>
          </div>
        ) },
        { id: 'ex', label: t('แบบฝึกหัด', 'Exercises'), content: (
          <div className="grid gap-8 lg:grid-cols-2">
            <ol className="list-decimal space-y-2 pl-5 text-[1.125rem]">
              <li>{t('ใช้ kernel Linear กับชุด Circles แล้วสังเกตความแม่นยำ จากนั้นเปลี่ยนเป็น RBF', 'Use a linear kernel on Circles and note the accuracy; then switch to RBF.')}</li>
              <li>{t('ปรับ C จาก 0.01 ถึง 100 นับจำนวน support vector และสังเกตความกว้างของ margin', 'Sweep C from 0.01 to 100; count support vectors and watch the margin width.')}</li>
              <li>{t('เพิ่ม γ จนเส้นแบ่งล้อมจุดแต่ละจุด ดูว่า test accuracy เป็นอย่างไร (overfitting)', 'Raise γ until the boundary wraps individual points; what happens to test accuracy? (overfitting)')}</li>
            </ol>
            <Quiz id="svm" items={[
              { q: t('จุดใดมี α > 0', 'Which points have α > 0?'), options: [t('ทุกจุด', 'All points'), t('เฉพาะ support vector', 'Only support vectors'), t('เฉพาะจุดที่จำแนกผิด', 'Only misclassified points')], answer: 1, why: t('จุดที่อยู่นอก margin อย่างชัดเจนมี α = 0 และไม่ส่งผลต่อ f(x)', 'Points safely outside the margin have α = 0 and do not affect f(x).') },
              { q: t('ลด C ลงจะทำให้เกิดอะไร', 'Decreasing C…'), options: [t('margin แคบลง', 'narrows the margin'), t('margin กว้างขึ้น ยอมให้ล้ำมากขึ้น', 'widens the margin, allowing more violations'), t('ไม่มีผล', 'has no effect')], answer: 1, why: t('C คือน้ำหนักของโทษ ξ ยิ่งน้อยยิ่งให้ความสำคัญกับ ‖w‖ เล็ก', 'C weights the ξ penalty; smaller C favours a small ‖w‖.') },
              { q: t('γ ของ RBF ใหญ่มากหมายถึง', 'A very large RBF γ means'), options: [t('อิทธิพลของแต่ละจุดแผ่ไปไกล', 'each point influences far away'), t('อิทธิพลแคบมาก เสี่ยง overfit', 'very local influence, risk of overfitting')], answer: 1, why: t('K ลดลงเร็วตามระยะ ฟังก์ชันจึงเป็นหย่อม ๆ รอบ support vector', 'K decays fast with distance, so f becomes islands around support vectors.') },
            ]} />
          </div>
        ) },
      ]}
    />
  );
}
