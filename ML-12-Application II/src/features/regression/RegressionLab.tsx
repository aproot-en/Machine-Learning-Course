import { useMemo, useState } from 'react';
import { useCameraState } from '../../engine3d/Stage';
import { LabShell, Metrics } from '../../components/LabShell';
import { HeightSurface, PointCloud, Polyline, Segments, contour, planeMapper, rgb } from '../../engine3d/plots';
import { FrameBox, Label } from '../../engine3d/primitives';
import { CLASS_COLORS, GRAD } from '../../engine3d/palette';
import { applyScaler, fitScaler, makeDataset, trainTestSplit, type DatasetId } from '../../data/datasets';
import { GDRegression, closedForm, makeRegressionData, r2, type RegKind } from '../../ml/regression';
import type { OptimizerName } from '../../ml/nn/types';
import { classificationReport } from '../../ml/metrics';
import { useSettings, useT } from '../../i18n/i18n';
import { useInterval } from '../../lib/useLoop';
import { NumberField, Section, Segmented, Select, Slider, Toggle, fmt, pct } from '../../components/ui';
import { ConfusionMatrix, LinePlot, Quiz, Tex } from '../../components/common';

const R = 5;

export default function RegressionLab({ initial = 'linear' }: { initial?: RegKind }) {
  const t = useT();
  const level = useSettings((s) => s.level);
  const [kind, setKind] = useState<RegKind>(initial);
  const [d, setD] = useState<1 | 2>(1);
  const [ds, setDs] = useState({ n: 60, noise: 0.5, seed: 3, nonlinear: false, cls: 'blobs2d' as DatasetId, testRatio: 0.25 });
  const [cfg, setCfg] = useState({ optimizer: 'sgd' as OptimizerName, lr: 0.05, batchSize: 0, l2: 0, l1: 0, init: 'random' as 'zeros' | 'random', maxIter: 500 });
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [view, setView] = useState<number | null>(null);
  const [showLoss, setShowLoss] = useState(true);
  const [nonce, setNonce] = useState(0);
  const [, setTick] = useState(0);
  const [cam, setCam] = useCameraState();
  const [experimentSample, setExperimentSample] = useState(0);
  const dims = kind === 'logistic' ? 2 : d;

  const data = useMemo(() => {
    if (kind === 'linear') {
      const r = makeRegressionData(ds.n, d, ds.noise, ds.seed, ds.nonlinear);
      const sp = trainTestSplit(r.X.map(() => 0), ds.testRatio, ds.seed);
      return { Xtr: sp.train.map((i) => r.X[i]), ytr: sp.train.map((i) => r.y[i]), Xte: sp.test.map((i) => r.X[i]), yte: sp.test.map((i) => r.y[i]) };
    }
    const raw = makeDataset(ds.cls, { n: ds.n * 2, classes: 2, noise: 0.2 + ds.noise * 0.3, seed: ds.seed });
    const sp = trainTestSplit(raw.y, ds.testRatio, ds.seed);
    const sc = fitScaler(sp.train.map((i) => raw.X[i]));
    const X = raw.X.map((r) => applyScaler(sc, r));
    return { Xtr: sp.train.map((i) => X[i]), ytr: sp.train.map((i) => raw.y[i]), Xte: sp.test.map((i) => X[i]), yte: sp.test.map((i) => raw.y[i]) };
  }, [kind, d, ds]);

  const optCfg = { name: cfg.optimizer, lr: cfg.lr, momentum: 0.9, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 };
  const model = useMemo(() => new GDRegression(data.Xtr, data.ytr, { kind, optimizer: optCfg, batchSize: cfg.batchSize, l2: cfg.l2, l1: cfg.l1, seed: ds.seed, init: cfg.init }),
    [data, kind, cfg.init, cfg.optimizer, ds.seed, nonce]); // eslint-disable-line react-hooks/exhaustive-deps
  model.cfg = { ...model.cfg, optimizer: optCfg, batchSize: cfg.batchSize, l2: cfg.l2, l1: cfg.l1 };
  const H = model.history;
  const cur = view == null ? H[H.length - 1] : H[Math.min(view, H.length - 1)];
  const exact = useMemo(() => (kind === 'linear' ? closedForm(data.Xtr, data.ytr) : null), [kind, data]);
  const diverged = !Number.isFinite(cur.loss) || cur.loss > 1e6;

  const step = () => {
    if (model.iter >= cfg.maxIter || diverged) return false;
    model.step(); setTick((x) => x + 1);
    return model.iter < cfg.maxIter;
  };
  useInterval(running, 60, speed, step, () => setRunning(false));

  const predict = (x: number[], w = cur.w, b = cur.b) => { const z = b + x.reduce((s, v, k) => s + v * w[k], 0); return kind === 'logistic' ? 1 / (1 + Math.exp(-z)) : z; };
  const pTr = data.Xtr.map((x) => predict(x)), pTe = data.Xte.map((x) => predict(x));
  const mse = (p: number[], y: number[]) => p.reduce((s, v, i) => s + (v - y[i]) ** 2, 0) / Math.max(1, p.length);
  const report = kind === 'logistic' ? classificationReport(data.yte, pTe.map((v) => (v >= 0.5 ? 1 : 0)), 2) : null;

  // ---------- scene mapping ----------
  const all = [...data.Xtr, ...data.Xte];
  const ys = [...data.ytr, ...data.yte];
  const yLo = Math.min(...ys), yHi = Math.max(...ys);
  const sy = kind === 'logistic' ? 3.5 : (2 * R * 0.8) / Math.max(1e-6, yHi - yLo);
  const yMid = kind === 'logistic' ? 0.5 : (yLo + yHi) / 2;
  const Y = (v: number) => (v - yMid) * sy;
  const map = useMemo(() => planeMapper(dims === 1 ? all.map((x) => [x[0], 0]) : all, R, 0.3), [data, dims]); // eslint-disable-line react-hooks/exhaustive-deps
  const pos = (x: number[], y: number): [number, number, number] => { const [a, c] = map.toScene(dims === 1 ? [x[0], 0] : x); return [a, Y(y), dims === 1 ? 0 : c]; };

  const G = 40;
  const surf = useMemo(() => {
    if (dims !== 2) return null;
    const h = new Float32Array((G + 1) ** 2), cols = new Float32Array(h.length * 3), p = new Float32Array(h.length);
    const A = rgb(CLASS_COLORS[0]), B = rgb(CLASS_COLORS[1]);
    for (let r = 0; r <= G; r++) for (let c = 0; c <= G; c++) {
      const v = r * (G + 1) + c, x = map.toData((c / G - 0.5) * 2 * R, (r / G - 0.5) * 2 * R), val = predict(x);
      p[v] = val; h[v] = Math.max(-R, Math.min(R, Y(val)));
      const tt = kind === 'logistic' ? val : 0.5;
      cols[v * 3] = (A[0] * (1 - tt) + B[0] * tt) * 0.7; cols[v * 3 + 1] = (A[1] * (1 - tt) + B[1] * tt) * 0.7; cols[v * 3 + 2] = (A[2] * (1 - tt) + B[2] * tt) * 0.7;
    }
    const boundary = kind === 'logistic' ? contour(G, p, 0.5).map(([c0, r0, c1, r1]) => ({ a: [(c0 / G - 0.5) * 2 * R, -R + 0.02, (r0 / G - 0.5) * 2 * R] as [number, number, number], b: [(c1 / G - 0.5) * 2 * R, -R + 0.02, (r1 / G - 0.5) * 2 * R] as [number, number, number], c: '#ffffff' })) : [];
    return { h, cols, boundary };
  }, [cur, map, dims, kind]); // eslint-disable-line react-hooks/exhaustive-deps

  // loss landscape J(w, b) for 1-feature linear regression, centred on the exact solution
  const LG = 36, LS = 8, LX = 2 * R + 4;
  const land = useMemo(() => {
    if (kind !== 'linear' || dims !== 1 || !exact) return null;
    const wr = 3, br = 3, h = new Float32Array((LG + 1) ** 2), cols = new Float32Array(h.length * 3);
    let mx = 0;
    const J = (w: number, b: number) => model.loss([w], b);
    for (let r = 0; r <= LG; r++) for (let c = 0; c <= LG; c++) { const v = r * (LG + 1) + c; h[v] = Math.log1p(J(exact.w[0] + (c / LG - 0.5) * 2 * wr, exact.b + (r / LG - 0.5) * 2 * br)); mx = Math.max(mx, h[v]); }
    const k = 5 / mx, lo = rgb('#3B82F6'), hi = rgb('#F59E0B');
    h.forEach((v, i) => { const tt = v / mx; h[i] = v * k; cols[i * 3] = lo[0] + (hi[0] - lo[0]) * tt; cols[i * 3 + 1] = lo[1] + (hi[1] - lo[1]) * tt; cols[i * 3 + 2] = lo[2] + (hi[2] - lo[2]) * tt; });
    const toL = (w: number, b: number): [number, number, number] => [LX + ((w - exact.w[0]) / (2 * wr)) * LS, Math.log1p(J(w, b)) * k + 0.08 - 2, ((b - exact.b) / (2 * br)) * LS];
    return { h, cols, toL };
  }, [kind, dims, exact, model]);

  const histShown = H.slice(0, (view ?? H.length - 1) + 1);
  const scene = (
    <group>
      <FrameBox size={R} labels={dims === 1 ? ['x', 'y', ''] : ['x₁', kind === 'logistic' ? 'p(y=1)' : 'y', 'x₂']} flat={false} />
      <PointCloud points={data.Xtr.map((x, i) => pos(x, data.ytr[i]))} colors={data.ytr.map((v) => (kind === 'logistic' ? CLASS_COLORS[v] : '#00D9FF'))} sizes={0.15} />
      <PointCloud points={data.Xte.map((x, i) => pos(x, data.yte[i]))} colors={data.yte.map((v) => (kind === 'logistic' ? CLASS_COLORS[v] : '#A855F7'))} sizes={0.15} glow={0.1} />
      {kind === 'linear' && <Segments segs={data.Xtr.map((x, i) => { const r = data.ytr[i] - pTr[i]; return { a: pos(x, data.ytr[i]), b: pos(x, pTr[i]), c: r >= 0 ? '#22C55E' : '#EF4444', i: 0.8 }; })} />}
      {dims === 1 && <Polyline points={Array.from({ length: 41 }, (_, k) => { const sx = -R + (k / 40) * 2 * R; const x = map.toData(sx, 0)[0]; return [sx, Math.max(-R, Math.min(R, Y(predict([x])))), 0] as [number, number, number]; })} color="#E5E7EB" />}
      {exact && dims === 1 && <Polyline points={[-R, R].map((sx) => { const x = map.toData(sx, 0)[0]; return [sx, Math.max(-R, Math.min(R, Y(exact.w[0] * x + exact.b))), 0] as [number, number, number]; })} color="#F59E0B" opacity={0.35} />}
      {surf && <HeightSurface G={G} heights={surf.h} colors={surf.cols} size={2 * R} opacity={0.5} />}
      {surf && surf.boundary.length > 0 && <Segments segs={surf.boundary} />}
      {land && showLoss && (
        <group>
          <HeightSurface G={LG} heights={land.h} colors={land.cols} size={LS} position={[LX, -2, 0]} opacity={0.75} />
          <Polyline points={histShown.map((s) => land.toL(s.w[0], s.b))} color="#ffffff" colorEnd="#00D9FF" />
          {(() => { const p = land.toL(cur.w[0], cur.b); return (
            <>
              <mesh position={p}><sphereGeometry args={[0.18, 20, 14]} /><meshStandardMaterial color="#00D9FF" emissive="#00D9FF" emissiveIntensity={0.9} /></mesh>
              {cur.iter > 0 && <Segments segs={[{ a: p, b: land.toL(cur.w[0] - 0.4 * cur.gw[0], cur.b - 0.4 * cur.gb), c: GRAD }]} />}
            </>
          ); })()}
          <Label position={[LX, 4.2, 0]} muted>{t('พื้นผิว loss J(w, b)', 'Loss surface J(w, b)')}</Label>
          <Label position={[LX + LS / 2 + 0.6, -2, 0]} muted>w</Label>
          <Label position={[LX, -2, LS / 2 + 0.6]} muted>b</Label>
        </group>
      )}
    </group>
  );

  const reset = () => { setRunning(false); setView(null); setNonce((n) => n + 1); };
  const fTex = kind === 'logistic'
    ? `p = \\sigma(${fmt(cur.w[0], 3)}x_1 ${cur.w[1] >= 0 ? '+' : ''}${fmt(cur.w[1] ?? 0, 3)}x_2 ${cur.b >= 0 ? '+' : ''}${fmt(cur.b, 3)})`
    : `\\hat y = ${cur.w.map((w, k) => `${fmt(w, 3)}x_${k + 1}`).join(' + ')} ${cur.b >= 0 ? '+' : ''}${fmt(cur.b, 3)}`;

  return (
    <LabShell crumb={kind === 'linear' ? 'Linear Regression' : 'Logistic Regression'} title={kind === 'linear' ? 'Linear Regression' : 'Logistic Regression'}
      experiment={{ model: kind === 'linear' ? 'Linear Regression' : 'Logistic Regression',
        stages: [
          { name: 'Test input x', values: data.Xte[Math.min(experimentSample, data.Xte.length - 1)] },
          { name: 'xᵢ × wᵢ', values: data.Xte[Math.min(experimentSample, data.Xte.length - 1)].map((v, i) => v * cur.w[i]) },
          { name: 'Σ + b', values: [cur.b + data.Xte[Math.min(experimentSample, data.Xte.length - 1)].reduce((sum, v, i) => sum + v * cur.w[i], 0)] },
          { name: kind === 'linear' ? 'ŷ = z' : 'Sigmoid → P(y)', values: kind === 'linear' ? [pTe[Math.min(experimentSample, pTe.length - 1)]] : [1 - pTe[Math.min(experimentSample, pTe.length - 1)], pTe[Math.min(experimentSample, pTe.length - 1)]], labels: kind === 'linear' ? ['ŷ'] : ['P(0)', 'P(1)'] },
          { name: 'Prediction / target', values: [kind === 'linear' ? pTe[Math.min(experimentSample, pTe.length - 1)] : pTe[Math.min(experimentSample, pTe.length - 1)] >= 0.5 ? 1 : 0, data.yte[Math.min(experimentSample, data.yte.length - 1)]], labels: ['prediction', 'target'] },
        ],
        metrics: [{ label: 'Iteration', value: String(cur.iter) }, { label: 'Train loss', value: fmt(cur.loss, 4) }, { label: kind === 'linear' ? 'Test MSE' : 'Test accuracy', value: kind === 'linear' ? fmt(mse(pTe, data.yte), 4) : pct(report!.accuracy) }, { label: 'Test samples', value: String(data.Xte.length) }],
        note: diverged ? t('การฝึกลู่ออก ผลทำนายใช้ไม่ได้ กรุณาลด learning rate แล้วเริ่มใหม่', 'Training diverged; predictions are invalid. Lower the learning rate and reset.') : undefined,
        history: [{ label: 'Training loss', values: H.slice(0, cur.iter + 1).map((h) => h.loss) }], confusion: report?.confusion,
        sample: { index: experimentSample, count: data.Xte.length, onChange: setExperimentSample },
        source: `${data.Xtr.length + data.Xte.length} ${t('ตัวอย่าง', 'samples')} · ${dims} features`,
        output: kind === 'linear' ? `ŷ=${fmt(pTe[Math.min(experimentSample, pTe.length - 1)], 4)} · y=${fmt(data.yte[Math.min(experimentSample, data.yte.length - 1)], 4)} · test MSE=${fmt(mse(pTe, data.yte), 4)}` : `p(y=1)=${fmt(pTe[Math.min(experimentSample, pTe.length - 1)], 4)} · class=${pTe[Math.min(experimentSample, pTe.length - 1)] >= 0.5 ? 1 : 0} · test accuracy=${pct(report!.accuracy)}`,
      }}
      chips={[t('การเรียนรู้แบบมีผู้สอน', 'Supervised'), t('พื้นฐาน', 'Beginner')]}
      intro={kind === 'linear'
        ? t('กด ▶ เพื่อปรับ w และ b ด้วย gradient descent เส้นสีเขียว/แดงคือ residual ทางขวาคือพื้นผิว loss และเส้นทางที่ optimizer เดิน', 'Press ▶ to fit w and b by gradient descent. Green/red lines are residuals; on the right is the loss surface and the optimizer’s path.')
        : t('กด ▶ เพื่อฝึก พื้นผิวคือ p(y=1) = σ(w·x + b) เส้นสีขาวที่พื้นคือเส้นแบ่ง p = 0.5', 'Press ▶ to train. The surface is p(y=1) = σ(w·x + b); the white line on the floor is the p = 0.5 boundary.')}
      cam={cam} setCam={setCam} distance={land && showLoss ? 22 : 16} target={land && showLoss ? [LX / 2, 0, 0] : [0, 0, 0]} scene={scene}
      readout={{ title: kind === 'linear' ? 'Linear regression · GD' : 'Logistic regression · GD', rows: [
        [t('รอบที่', 'iteration'), `${cur.iter}${view != null ? ` / ${H.length - 1}` : ''}`],
        ['w', `(${cur.w.map((v) => fmt(v, 3)).join(', ')})`], ['b', fmt(cur.b, 3)],
        [kind === 'linear' ? 'MSE (train)' : 'cross-entropy', diverged ? t('ลู่ออก!', 'diverged!') : fmt(cur.loss, 4), diverged ? '#EF4444' : undefined],
        ['‖∇‖', fmt(Math.hypot(...cur.gw, cur.gb), 4)],
        ...(kind === 'linear' ? [['R² (test)', fmt(r2(data.yte, pTe), 3)] as [string, string], [t('คำตอบแบบปิด', 'closed form'), `w=(${exact!.w.map((v) => fmt(v, 2)).join(', ')}) b=${fmt(exact!.b, 2)}`] as [string, string]]
          : [['acc (test)', pct(report!.accuracy)] as [string, string]]),
      ] }}
      playback={{ playing: running, onToggle: () => { setView(null); setRunning(!running); }, onNext: () => { if (view != null) setView(view + 1 >= H.length - 1 ? null : view + 1); else step(); },
        onPrev: () => { setRunning(false); setView(Math.max(0, (view ?? H.length - 1) - 1)); }, canPrev: (view ?? H.length - 1) > 0, onReset: reset, speed, onSpeed: setSpeed,
        status: `iter ${cur.iter} · ${cfg.optimizer}` }}
      controls={
        <>
          <Section title={t('โมเดล', 'Model')}>
            <Segmented value={kind} onChange={(k) => { setKind(k); reset(); }} options={[{ value: 'linear', label: 'Linear' }, { value: 'logistic', label: 'Logistic' }]} />
            {kind === 'linear' && <Segmented label={t('จำนวนฟีเจอร์', 'Features')} value={String(d) as '1' | '2'} onChange={(v) => { setD(Number(v) as 1 | 2); reset(); }} options={[{ value: '1', label: t('1 → เส้นตรง', '1 → line') }, { value: '2', label: t('2 → ระนาบ', '2 → plane') }]} />}
          </Section>
          <Section title={t('ชุดข้อมูล', 'Dataset')}>
            {kind === 'logistic' && <Select<DatasetId> label={t('ชุดข้อมูล', 'Dataset')} value={ds.cls} onChange={(cls) => setDs({ ...ds, cls })} options={[{ value: 'blobs2d', label: 'Blobs' }, { value: 'moons', label: 'Two Moons' }, { value: 'xor', label: 'XOR' }]} />}
            <Slider label={t('จำนวนตัวอย่าง', 'Samples')} value={ds.n} min={20} max={200} step={10} onChange={(n) => setDs({ ...ds, n })} />
            <Slider label="Noise" value={ds.noise} min={0} max={2} step={0.05} format={(v) => v.toFixed(2)} onChange={(noise) => setDs({ ...ds, noise })} />
            {kind === 'linear' && <Toggle label={t('ความสัมพันธ์ไม่เป็นเส้นตรง (มีพจน์ x²)', 'Non-linear truth (x² term)')} checked={ds.nonlinear} onChange={(nonlinear) => setDs({ ...ds, nonlinear })} />}
            <Slider label="Seed" value={ds.seed} min={1} max={99} onChange={(seed) => setDs({ ...ds, seed })} />
          </Section>
          <Section title={t('การฝึก', 'Training')}>
            <Select<OptimizerName> label="Optimizer" value={cfg.optimizer} onChange={(optimizer) => setCfg({ ...cfg, optimizer })} options={[{ value: 'sgd', label: 'Gradient descent' }, { value: 'momentum', label: 'Momentum' }, { value: 'rmsprop', label: 'RMSprop' }, { value: 'adam', label: 'Adam' }]} />
            <Slider label="Learning rate η" value={Math.log10(cfg.lr)} min={-3} max={0.3} step={0.05} format={(v) => (10 ** v).toPrecision(2)} onChange={(v) => setCfg({ ...cfg, lr: Number((10 ** v).toPrecision(3)) })} />
            <Select<number> label="Batch" value={cfg.batchSize} onChange={(batchSize) => setCfg({ ...cfg, batchSize })} options={[{ value: 0, label: t('ทั้งชุด (batch GD)', 'Full batch') }, { value: 1, label: '1 (SGD)' }, { value: 8, label: '8' }, { value: 16, label: '16' }]} />
            <Select<'zeros' | 'random'> label={t('ค่าเริ่มต้น', 'Initialisation')} value={cfg.init} onChange={(init) => setCfg({ ...cfg, init })} options={[{ value: 'random', label: 'random' }, { value: 'zeros', label: 'zeros' }]} />
            {level >= 2 && <NumberField label="L2 (ridge)" value={cfg.l2} min={0} max={5} step={0.01} onChange={(l2) => setCfg({ ...cfg, l2: Math.max(0, l2) })} />}
            {level >= 2 && <NumberField label="L1 (lasso)" value={cfg.l1} min={0} max={5} step={0.01} onChange={(l1) => setCfg({ ...cfg, l1: Math.max(0, l1) })} />}
            <NumberField label={t('รอบสูงสุด', 'Max iterations')} value={cfg.maxIter} min={1} max={5000} onChange={(maxIter) => setCfg({ ...cfg, maxIter: Math.max(1, Math.round(maxIter)) })} />
            {land && <Toggle label={t('แสดงพื้นผิว loss', 'Show loss surface')} checked={showLoss} onChange={setShowLoss} />}
          </Section>
        </>
      }
      tabs={[
        { id: 'concept', label: t('แนวคิด', 'Concept'), content: (
          <div className="grid max-w-5xl gap-8 text-[1.125rem] leading-7 lg:grid-cols-2">
            <p>{kind === 'linear'
              ? t('Linear regression หาเส้นตรง (หรือระนาบ) ที่ทำให้ผลรวมกำลังสองของ residual (ระยะแนวตั้งจากจุดถึงเส้น) น้อยที่สุด ปัญหานี้มีคำตอบแบบปิด (normal equation) แต่ที่นี่ใช้ gradient descent เพื่อให้เห็นการเรียนรู้ทีละก้าว เส้นสีส้มจาง ๆ คือคำตอบแบบปิดสำหรับเทียบ', 'Linear regression finds the line (or plane) that minimises the sum of squared residuals (vertical distances to the line). It has a closed-form answer (the normal equation), but here gradient descent shows learning step by step. The faint orange line is the closed-form answer for comparison.')
              : t('Logistic regression ใช้ sigmoid แปลงผลรวมเชิงเส้นเป็นความน่าจะเป็น แล้วเรียนรู้ด้วย cross-entropy เส้นแบ่งการตัดสินใจ p = 0.5 จึงเป็นเส้นตรงเสมอ ลองใช้ชุด XOR แล้วดูว่าทำไมแยกไม่ได้', 'Logistic regression passes a linear score through a sigmoid to get a probability, and learns with cross-entropy. Its p = 0.5 boundary is therefore always a straight line; try XOR to see why it cannot separate it.')}</p>
            <ul className="list-disc space-y-1.5 pl-5">
              <li>{t('η ใหญ่เกินไป: loss แกว่งหรือลู่ออก (ลอง η ≈ 1)', 'η too large: the loss oscillates or diverges (try η ≈ 1).')}</li>
              <li>{t('SGD (batch = 1) เดินซิกแซก แต่แต่ละก้าวถูกกว่า', 'SGD (batch = 1) zig-zags, but each step is cheaper.')}</li>
              <li>{t('L2 ดึงน้ำหนักเข้าหา 0 ทำให้เส้นราบลง L1 ทำให้บางน้ำหนักเป็น 0 พอดี', 'L2 pulls weights toward 0 (flatter line); L1 can set some weights exactly to 0.')}</li>
              <li>{t('เปิด "ความสัมพันธ์ไม่เป็นเส้นตรง" แล้วดูรูปแบบของ residual ที่ไม่สุ่ม (underfitting)', 'Turn on the non-linear truth and look for patterned, non-random residuals (underfitting).')}</li>
            </ul>
          </div>
        ) },
        { id: 'math', label: t('คณิตศาสตร์', 'Math'), content: (
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="flex flex-col gap-3">
              <h3 className="label-xs">{t('โมเดลตอนนี้', 'Current model')}</h3>
              <Tex block>{fTex}</Tex>
              <h3 className="label-xs">Loss</h3>
              <Tex block>{kind === 'linear' ? 'J = \\frac1n\\sum_i (\\hat y_i - y_i)^2 + \\tfrac{\\lambda_2}{2}\\|\\mathbf w\\|^2 + \\lambda_1\\|\\mathbf w\\|_1' : 'J = -\\frac1n\\sum_i \\big[y_i\\log p_i + (1-y_i)\\log(1-p_i)\\big] + \\tfrac{\\lambda_2}{2}\\|\\mathbf w\\|^2'}</Tex>
            </div>
            <div className="flex flex-col gap-3">
              <h3 className="label-xs">{t('Gradient และการอัปเดต', 'Gradient and update')}</h3>
              <Tex block>{kind === 'linear' ? '\\frac{\\partial J}{\\partial w_k} = \\frac2n\\sum_i(\\hat y_i - y_i)x_{ik} + \\lambda_2 w_k,\\quad \\frac{\\partial J}{\\partial b} = \\frac2n\\sum_i(\\hat y_i - y_i)' : '\\frac{\\partial J}{\\partial w_k} = \\frac1n\\sum_i(p_i - y_i)x_{ik} + \\lambda_2 w_k'}</Tex>
              <Tex block>{`\\mathbf w \\leftarrow \\mathbf w - \\eta\\nabla J,\\quad \\nabla J = (${cur.gw.map((v) => fmt(v, 3)).join(', ')};\\ ${fmt(cur.gb, 3)}),\\ \\eta = ${cfg.lr}`}</Tex>
              {exact && <Tex block>{`\\text{Normal equation: } (X^\\top X)^{-1}X^\\top y \\Rightarrow \\mathbf w = (${exact.w.map((v) => fmt(v, 3)).join(', ')}),\\ b = ${fmt(exact.b, 3)}`}</Tex>}
            </div>
          </div>
        ) },
        { id: 'eval', label: t('การประเมินผล', 'Evaluation'), content: (
          <div className="grid gap-8 lg:grid-cols-2">
            <div className="flex flex-col gap-5">
              {kind === 'linear'
                ? <Metrics items={[['MSE train', fmt(mse(pTr, data.ytr), 4)], ['MSE test', fmt(mse(pTe, data.yte), 4)], ['R² train', fmt(r2(data.ytr, pTr), 3)], ['R² test', fmt(r2(data.yte, pTe), 3)]]} />
                : <><Metrics items={[['Accuracy', pct(report!.accuracy)], ['Precision', pct(report!.precision)], ['Recall', pct(report!.recall)], ['F1', pct(report!.f1)]]} /><ConfusionMatrix m={report!.confusion} classNames={['A', 'B']} /></>}
            </div>
            <div><h3 className="label-xs mb-2">{t('loss ต่อรอบ', 'Loss per iteration')}</h3>
              <LinePlot xTitle="iteration" yTitle="loss" markerX={view ?? undefined} series={[{ name: 'train loss', x: H.map((h) => h.iter), y: H.map((h) => h.loss), color: '#00D9FF' }]} /></div>
          </div>
        ) },
        { id: 'ex', label: t('แบบฝึกหัด', 'Exercises'), content: (
          <Quiz id={`reg-${kind}`} items={kind === 'linear' ? [
            { q: t('ถ้า loss เพิ่มขึ้นทุกรอบ ควรทำอะไร', 'If the loss rises every iteration, you should…'), options: [t('เพิ่ม η', 'increase η'), t('ลด η', 'decrease η'), t('เพิ่มจำนวนรอบ', 'add iterations')], answer: 1, why: t('ก้าวใหญ่เกินไปทำให้ข้ามจุดต่ำสุดไปเรื่อย ๆ ลองดูเส้นทางบนพื้นผิว loss', 'Steps that are too big overshoot the minimum repeatedly; watch the path on the loss surface.') },
            { q: t('residual คืออะไร', 'What is a residual?'), options: ['y − ŷ', 'w · x', '∂J/∂w'], answer: 0, why: t('ความต่างระหว่างค่าจริงกับค่าที่ทำนาย เส้นสีเขียว/แดงในฉาก', 'The difference between the true and predicted values: the green/red lines in the scene.') },
            { q: t('GD แบบ full batch บน MSE ของ linear regression จะเจอ local minimum ไหม', 'Can full-batch GD on linear-regression MSE get stuck in a local minimum?'), options: [t('ได้', 'Yes'), t('ไม่ได้ เพราะ loss เป็น convex', 'No, the loss is convex')], answer: 1, why: t('พื้นผิว J(w,b) เป็นชามโค้งรูปเดียว ดูได้จากฉากทางขวา', 'J(w,b) is a single bowl, as the surface on the right shows.') },
          ] : [
            { q: t('เส้นแบ่งของ logistic regression เป็นรูปร่างแบบใด', 'What shape is a logistic-regression boundary?'), options: [t('เส้นตรง/ระนาบ', 'A line/plane'), t('วงกลม', 'A circle'), t('อะไรก็ได้', 'Anything')], answer: 0, why: t('p = 0.5 เมื่อ w·x + b = 0 ซึ่งเป็นสมการเชิงเส้น', 'p = 0.5 exactly when w·x + b = 0, a linear equation.') },
            { q: t('ทำไมไม่ใช้ MSE กับ logistic regression', 'Why not use MSE for logistic regression?'), options: [t('ทำให้ loss ไม่ convex และ gradient เล็กเมื่อทายผิดมาก', 'It makes the loss non-convex with tiny gradients when very wrong'), t('คำนวณไม่ได้', 'It cannot be computed')], answer: 0, why: t('cross-entropy กับ sigmoid ให้ gradient = p − y ที่เรียบง่ายและไม่หายไป', 'Cross-entropy with sigmoid gives the simple, non-vanishing gradient p − y.') },
          ]} />
        ) },
      ]}
    />
  );
}
