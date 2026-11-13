import type React from 'react';
import { useMemo, useState } from 'react';
import * as THREE from 'three';
import { useCameraState } from '../../engine3d/Stage';
import { LabShell, Metrics } from '../../components/LabShell';
import { HeightSurface, PointCloud, planeMapper, rgb } from '../../engine3d/plots';
import { BOX, Label } from '../../engine3d/primitives';
import { CLASS_COLORS } from '../../engine3d/palette';
import { applyScaler, datasetInfo, fitScaler, makeDataset, trainTestSplit, type DatasetId } from '../../data/datasets';
import { argmax, fitModel, type ModelKind, type ModelParams } from '../../ml/classifiers';
import { classificationReport } from '../../ml/metrics';
import { useT } from '../../i18n/i18n';
import { Button, Section, Select, Slider, Toggle, fmt, pct } from '../../components/ui';
import { Callout, ConfusionMatrix, Quiz, download } from '../../components/common';

const R = 4, GAP = 1.2, G = 40;
const SIDE_X = [-(R + GAP / 2), R + GAP / 2];
const DATA: DatasetId[] = ['moons', 'circles', 'xor', 'spiral', 'blobs2d', 'iris'];
const NAMES: Record<ModelKind, string> = { knn: 'KNN', svm: 'SVM', logistic: 'Logistic Regression', tree: 'Decision Tree', forest: 'Random Forest', mlp: 'Neural Network (MLP)' };
const SIDE_COL = ['#00D9FF', '#F59E0B'];
const base = (kind: ModelKind): ModelParams => ({ kind, k: 7, kernel: 'rbf', C: 2, gamma: 1, maxDepth: 6, nTrees: 15, hidden: 10, epochs: 120, seed: 1 });

export default function ComparePage() {
  const t = useT();
  const [ds, setDs] = useState({ id: 'moons' as DatasetId, n: 240, classes: 3, noise: 0.25, seed: 4, testRatio: 0.3 });
  const [models, setModels] = useState<ModelParams[]>([base('knn'), base('svm')]);
  const [view, setView] = useState({ surface: true, test: true, disagree: true });
  const [cam, setCam] = useCameraState();
  const info = datasetInfo(ds.id);

  // one shared, standardised split: both models see exactly the same data
  const data = useMemo(() => {
    const raw = makeDataset(ds.id, { n: ds.n, classes: ds.classes, noise: ds.noise, seed: ds.seed });
    const feats = ds.id === 'iris' ? [2, 3] : [0, 1];
    const X0 = raw.X.map((r) => feats.map((f) => r[f]));
    const sp = trainTestSplit(raw.y, ds.testRatio, ds.seed);
    const sc = fitScaler(sp.train.map((i) => X0[i]));
    const X = X0.map((r) => applyScaler(sc, r));
    return { Xtr: sp.train.map((i) => X[i]), ytr: sp.train.map((i) => raw.y[i]), Xte: sp.test.map((i) => X[i]), yte: sp.test.map((i) => raw.y[i]), K: raw.nClasses, names: raw.classNames, feat: feats.map((f) => raw.featureNames[f]) };
  }, [ds]);
  const map = useMemo(() => planeMapper([...data.Xtr, ...data.Xte], R, 0.3), [data]);

  const fitted = useMemo(() => models.map((p) => {
    const t0 = performance.now();
    const m = fitModel(p, data.Xtr, data.ytr, data.K);
    const fitMs = performance.now() - t0;
    const t1 = performance.now();
    const predTe = data.Xte.map((x) => argmax(m.proba(x)));
    const inferMs = (performance.now() - t1) / Math.max(1, data.Xte.length);
    const predTr = data.Xtr.map((x) => argmax(m.proba(x)));
    const report = classificationReport(data.yte, predTe, data.K);
    const trainAcc = predTr.filter((c, i) => c === data.ytr[i]).length / data.ytr.length;
    // decision surface: height = confidence of the predicted class, colour = probability-weighted class colour
    const heights = new Float32Array((G + 1) ** 2), cols = new Float32Array(heights.length * 3);
    const C = CLASS_COLORS.map(rgb);
    for (let r = 0; r <= G; r++) for (let c = 0; c <= G; c++) {
      const v = r * (G + 1) + c, pr = m.proba(map.toData((c / G - 0.5) * 2 * R, (r / G - 0.5) * 2 * R));
      const conf = Math.max(...pr);
      heights[v] = conf * 2.2;
      let rr = 0, gg = 0, bb = 0;
      pr.forEach((q, k) => { rr += q * C[k][0]; gg += q * C[k][1]; bb += q * C[k][2]; });
      const s = 0.25 + 0.6 * conf;
      cols[v * 3] = rr * s; cols[v * 3 + 1] = gg * s; cols[v * 3 + 2] = bb * s;
    }
    return { m, fitMs, inferMs, predTe, report, trainAcc, heights, cols };
  }), [models, data, map]);

  const disagree = data.Xte.map((_, i) => fitted[0].predTe[i] !== fitted[1].predTe[i]);
  const agreement = 1 - disagree.filter(Boolean).length / Math.max(1, disagree.length);
  const P = (x: number[], side: number, y = 2.45): [number, number, number] => { const [a, c] = map.toScene(x); return [a + SIDE_X[side], y, c]; };

  const scene = (
    <group>
      {fitted.map((f, s) => (
        <group key={s}>
          <gridHelper args={[2 * R, 8, '#1d2a4a', '#131c33']} position={[SIDE_X[s], 0, 0]} />
          <lineSegments position={[SIDE_X[s], 1.2, 0]}><edgesGeometry args={[new THREE.BoxGeometry(2 * R, 2.4, 2 * R)]} /><lineBasicMaterial color={SIDE_COL[s]} transparent opacity={0.35} /></lineSegments>
          {view.surface && <HeightSurface G={G} heights={f.heights} colors={f.cols} size={2 * R} position={[SIDE_X[s], 0, 0]} opacity={0.6} wire={0.05} />}
          <PointCloud points={data.Xtr.map((x) => P(x, s))} colors={data.ytr.map((c) => CLASS_COLORS[c])} sizes={0.1} />
          {view.test && <PointCloud points={data.Xte.map((x) => P(x, s))} colors={data.Xte.map((_, i) => (f.predTe[i] === data.yte[i] ? new THREE.Color(CLASS_COLORS[data.yte[i]]).multiplyScalar(0.8) : new THREE.Color('#EF4444')))} sizes={0.15} geometry={BOX} glow={0.3} />}
          {view.disagree && <PointCloud points={data.Xte.filter((_, i) => disagree[i]).map((x) => P(x, s, 2.6))} colors={data.Xte.filter((_, i) => disagree[i]).map(() => '#ffffff')} sizes={0.2} basic wire opacity={0.45} />}
          <Label position={[SIDE_X[s], 3.4, -R]}><span style={{ color: SIDE_COL[s] }}>{s === 0 ? 'A' : 'B'} · {NAMES[models[s].kind]}</span> <span className="text-muted">test {pct(f.report.accuracy)}</span></Label>
        </group>
      ))}
    </group>
  );

  const setM = (s: number, patch: Partial<ModelParams>) => setModels(models.map((m, i) => (i === s ? { ...m, ...patch } : m)));
  const modelControls = (s: number) => {
    const p = models[s];
    return (
      <Section key={s} title={`${s === 0 ? 'A' : 'B'} · ${s === 0 ? t('ฟ้า', 'cyan') : t('ส้ม', 'orange')}`}>
        <Select<ModelKind> label={t('อัลกอริทึม', 'Algorithm')} value={p.kind} onChange={(kind) => setM(s, { kind })} options={(Object.keys(NAMES) as ModelKind[]).map((k) => ({ value: k, label: NAMES[k] }))} />
        {p.kind === 'knn' && <Slider label="K" value={p.k} min={1} max={25} onChange={(k) => setM(s, { k })} />}
        {p.kind === 'svm' && <>
          <Select label="Kernel" value={p.kernel} onChange={(kernel) => setM(s, { kernel })} options={[{ value: 'rbf', label: 'RBF' }, { value: 'linear', label: 'Linear' }, { value: 'poly', label: 'Polynomial' }]} />
          <Slider label="C" value={Math.log10(p.C)} min={-2} max={2} step={0.1} format={(v) => (10 ** v).toPrecision(2)} onChange={(v) => setM(s, { C: Number((10 ** v).toPrecision(3)) })} />
          {p.kernel !== 'linear' && <Slider label="γ" value={Math.log10(p.gamma)} min={-2} max={1.5} step={0.1} format={(v) => (10 ** v).toPrecision(2)} onChange={(v) => setM(s, { gamma: Number((10 ** v).toPrecision(3)) })} />}
        </>}
        {(p.kind === 'tree' || p.kind === 'forest') && <Slider label="Max depth" value={p.maxDepth} min={1} max={15} onChange={(maxDepth) => setM(s, { maxDepth })} />}
        {p.kind === 'forest' && <Slider label={t('จำนวนต้นไม้', 'Trees')} value={p.nTrees} min={1} max={40} onChange={(nTrees) => setM(s, { nTrees })} />}
        {p.kind === 'mlp' && <>
          <Slider label={t('นิวรอนต่อชั้น (2 ชั้นซ่อน)', 'Units per layer (2 hidden)')} value={p.hidden} min={2} max={32} onChange={(hidden) => setM(s, { hidden })} />
          <Slider label="Epochs" value={p.epochs} min={10} max={400} step={10} onChange={(epochs) => setM(s, { epochs })} />
        </>}
        {(p.kind === 'forest' || p.kind === 'mlp' || p.kind === 'tree') && <Slider label="Seed" value={p.seed} min={1} max={99} onChange={(seed) => setM(s, { seed })} />}
      </Section>
    );
  };

  const PRESETS: [ModelKind, ModelKind, Partial<ModelParams>?, Partial<ModelParams>?][] = [
    ['knn', 'svm'], ['logistic', 'mlp'], ['tree', 'forest'], ['svm', 'svm', { kernel: 'linear' }, { kernel: 'rbf' }], ['knn', 'knn', { k: 1 }, { k: 25 }], ['logistic', 'svm', undefined, { kernel: 'rbf' }],
  ];
  const presetLabel = (a: ModelKind, b: ModelKind, pa?: Partial<ModelParams>, pb?: Partial<ModelParams>) => {
    const n = (k: ModelKind, p?: Partial<ModelParams>) => (k === 'svm' && p?.kernel ? `SVM ${p.kernel}` : k === 'knn' && p?.k ? `KNN K=${p.k}` : NAMES[k].replace(' (MLP)', '').replace(' Regression', ''));
    return `${n(a, pa)} vs ${n(b, pb)}`;
  };
  const picker = (
    <div className="flex flex-col gap-3 rounded-xl border border-line bg-surface/60 p-3 md:flex-row md:items-end md:gap-4">
      <div className="flex flex-1 flex-wrap items-end gap-3">
        {[0, 1].map((s) => (
          <label key={s} className="flex min-w-[200px] flex-1 flex-col gap-1 text-[1rem]">
            <span style={{ color: SIDE_COL[s] }}>{s === 0 ? t('โมเดล A (ซ้าย)', 'Model A (left)') : t('โมเดล B (ขวา)', 'Model B (right)')}</span>
            <select value={models[s].kind} onChange={(e) => setM(s, { kind: e.target.value as ModelKind })}
              className="rounded-lg border bg-bg px-3 py-2 text-[1.125rem] font-medium text-ink" style={{ borderColor: SIDE_COL[s] }}>
              {(Object.keys(NAMES) as ModelKind[]).map((k) => <option key={k} value={k}>{NAMES[k]}</option>)}
            </select>
          </label>
        )).reduce<React.ReactNode[]>((acc, el, i) => (i ? [...acc, <span key="vs" className="pb-2 font-display text-lg text-muted">vs</span>, el] : [el]), [])}
        <button type="button" onClick={() => setModels([models[1], models[0]])} className="rounded-lg border border-line px-3 py-2 text-[1rem] text-muted hover:text-ink" title={t('สลับฝั่ง', 'Swap sides')}>⇄ {t('สลับ', 'Swap')}</button>
        <Button onClick={() => download('model-comparison.json', JSON.stringify({ schema: 1, dataset: ds, featureNames: data.feat, classNames: data.names, preprocessing: 'z-score fitted on train only; exported X values are standardized', train: { X: data.Xtr, y: data.ytr }, test: { X: data.Xte, y: data.yte }, agreement, models: models.map((parameters, i) => ({ parameters, trainAccuracy: fitted[i].trainAcc, testReport: fitted[i].report, testPredictions: fitted[i].predTe, fitMs: fitted[i].fitMs, inferenceMsPerSample: fitted[i].inferMs, scoreType: parameters.kind === 'svm' ? 'uncalibrated normalized decision scores' : 'class distribution' })) }, null, 2))}>{t('ส่งออกผล JSON', 'Export results JSON')}</Button>
      </div>
      <div className="flex flex-col gap-1">
        <span className="label-xs">{t('คู่ที่น่าสนใจ', 'Suggested pairs')}</span>
        <div className="flex flex-wrap gap-1.5">
          {PRESETS.map(([a, b, pa, pb]) => (
            <button key={presetLabel(a, b, pa, pb)} type="button" onClick={() => setModels([{ ...base(a), ...pa }, { ...base(b), ...pb }])} className="chip hover:border-cyan hover:text-ink">{presetLabel(a, b, pa, pb)}</button>
          ))}
        </div>
      </div>
    </div>
  );

  const row = (label: string, f: (i: number) => string, better?: (a: number, b: number) => number, val?: (i: number) => number): [string, string] => {
    const w = better && val ? better(val(0), val(1)) : 0;
    return [label, `${f(0)}${w < 0 ? ' ◀' : ''}  |  ${f(1)}${w > 0 ? ' ◀' : ''}`];
  };
  const hi = (a: number, b: number) => (a > b ? -1 : b > a ? 1 : 0);

  return (
    <LabShell crumb={t('เปรียบเทียบ', 'Compare')} title={t('เปรียบเทียบโมเดล', 'Model Comparison')} chips={[t('ชุดข้อมูลเดียวกัน', 'Shared dataset'), t('ซิงก์มุมกล้อง', 'Synchronised view')]} toolbar={picker}
      intro={t('สองโมเดลฝึกบนชุดข้อมูลและการแบ่ง train/test เดียวกัน แสดงคู่กันในฉาก 3 มิติเดียว หมุนกล้องครั้งเดียวเห็นทั้งสองฝั่ง ความสูงของพื้นผิวคือความมั่นใจของโมเดล วงสีขาวคือจุด test ที่สองโมเดลทายไม่ตรงกัน', 'Two models are trained on the same dataset and the same train/test split, shown side by side in one 3D scene so one camera moves both. Surface height is model confidence; white rings mark test points where the two models disagree.')}
      cam={cam} setCam={setCam} distance={19} target={[0, 1, 0]} scene={scene}
      readout={{ title: `A: ${NAMES[models[0].kind]}  |  B: ${NAMES[models[1].kind]}`, rows: [
        [t('ชุดข้อมูล', 'dataset'), `${t(info.name)} · ${data.Xtr.length}/${data.Xte.length}`],
        row('test accuracy', (i) => pct(fitted[i].report.accuracy), hi, (i) => fitted[i].report.accuracy),
        row('F1 (macro)', (i) => pct(fitted[i].report.f1), hi, (i) => fitted[i].report.f1),
        row('train accuracy', (i) => pct(fitted[i].trainAcc)),
        row(t('เวลาฝึก', 'fit time'), (i) => `${fmt(fitted[i].fitMs, 1)} ms`, (a, b) => hi(b, a), (i) => fitted[i].fitMs),
        row(t('เวลาทำนาย/จุด', 'predict / point'), (i) => `${fmt(fitted[i].inferMs * 1000, 0)} µs`, (a, b) => hi(b, a), (i) => fitted[i].inferMs),
        [t('ทายตรงกัน', 'agreement'), pct(agreement)],
      ] }}
      caption={t('◀ = ดีกว่าในตัวชี้วัดนั้น · ลูกบาศก์แดง = จุด test ที่โมเดลฝั่งนั้นทายผิด · เวลาวัดจริงในเบราว์เซอร์นี้', '◀ = better on that metric · red cubes = test points that side got wrong · times are measured in this browser')}
      controls={
        <>
          <Section title={t('ชุดข้อมูล (ใช้ร่วมกัน)', 'Shared dataset')}>
            <Select<DatasetId> label={t('ชุดข้อมูล', 'Dataset')} value={ds.id} onChange={(id) => setDs({ ...ds, id })} options={DATA.map((id) => ({ value: id, label: t(datasetInfo(id).name) + (id === 'iris' ? ' · petal L/W' : '') }))} />
            {info.synthetic && <Slider label={t('จำนวนตัวอย่าง', 'Samples')} value={ds.n} min={60} max={500} step={20} onChange={(n) => setDs({ ...ds, n })} />}
            {info.fixedClasses == null && <Slider label={t('จำนวนคลาส', 'Classes')} value={ds.classes} min={2} max={5} onChange={(classes) => setDs({ ...ds, classes })} />}
            {info.synthetic && <Slider label="Noise" value={ds.noise} min={0} max={0.8} step={0.05} format={(v) => v.toFixed(2)} onChange={(noise) => setDs({ ...ds, noise })} />}
            <Slider label={t('สัดส่วน test', 'Test ratio')} value={ds.testRatio} min={0.1} max={0.5} step={0.05} format={pct} onChange={(testRatio) => setDs({ ...ds, testRatio })} />
            <Slider label="Seed" value={ds.seed} min={1} max={99} onChange={(seed) => setDs({ ...ds, seed })} />
          </Section>
          {modelControls(0)}
          {modelControls(1)}
          <Section title={t('การแสดงผล', 'View')}>
            <Toggle label={t('พื้นผิวความมั่นใจ', 'Confidence surface')} checked={view.surface} onChange={(surface) => setView({ ...view, surface })} />
            <Toggle label={t('จุด test (ลูกบาศก์)', 'Test points (cubes)')} checked={view.test} onChange={(test) => setView({ ...view, test })} />
            <Toggle label={t('ไฮไลต์จุดที่ทายไม่ตรงกัน', 'Highlight disagreements')} checked={view.disagree} onChange={(disagree) => setView({ ...view, disagree })} />
          </Section>
        </>
      }
      tabs={[
        { id: 'eval', label: t('ผลการประเมิน', 'Evaluation'), content: (
          <div className="grid gap-10 lg:grid-cols-2">
            {fitted.map((f, s) => (
              <div key={s} className="flex flex-col gap-4">
                <h3 className="font-display text-lg font-semibold" style={{ color: SIDE_COL[s] }}>{s === 0 ? 'A' : 'B'} · {NAMES[models[s].kind]}</h3>
                <Metrics items={[['Accuracy', pct(f.report.accuracy)], ['Precision', pct(f.report.precision)], ['Recall', pct(f.report.recall)], ['F1', pct(f.report.f1)]]} />
                <p className="font-mono text-[0.96875rem] text-muted">train {pct(f.trainAcc)} · fit {fmt(f.fitMs, 1)} ms · {f.m.complexity}</p>
                <ConfusionMatrix m={f.report.confusion} classNames={data.names} />
              </div>
            ))}
          </div>
        ) },
        { id: 'params', label: t('พารามิเตอร์', 'Parameters'), content: (
          <div className="scroll-thin overflow-x-auto">
            <table className="num w-full min-w-[520px] text-left font-mono text-[1rem]">
              <thead className="text-muted"><tr className="border-b border-line"><th className="py-2" /><th style={{ color: SIDE_COL[0] }}>A</th><th style={{ color: SIDE_COL[1] }}>B</th></tr></thead>
              <tbody>
                {([[t('อัลกอริทึม', 'algorithm'), (p: ModelParams) => NAMES[p.kind]], [t('ค่าหลัก', 'key settings'), (p: ModelParams) => (p.kind === 'knn' ? `K=${p.k}` : p.kind === 'svm' ? `${p.kernel}, C=${p.C}${p.kernel !== 'linear' ? `, γ=${p.gamma}` : ''}` : p.kind === 'tree' ? `depth ≤ ${p.maxDepth}` : p.kind === 'forest' ? `${p.nTrees} trees, depth ≤ ${p.maxDepth}` : p.kind === 'mlp' ? `2×${p.hidden} tanh, ${p.epochs} epochs` : 'Adam, 400 steps, L2=0.001')]] as [string, (p: ModelParams) => string][]).map(([k, f]) => (
                  <tr key={k} className="border-b border-line/50"><td className="py-1.5 text-muted">{k}</td><td>{f(models[0])}</td><td>{f(models[1])}</td></tr>
                ))}
                <tr className="border-b border-line/50"><td className="py-1.5 text-muted">{t('ขนาดโมเดล', 'model size')}</td><td>{fitted[0].m.complexity}</td><td>{fitted[1].m.complexity}</td></tr>
                <tr className="border-b border-line/50"><td className="py-1.5 text-muted">{t('ช่องว่าง train − test', 'train − test gap')}</td>{fitted.map((f, s) => <td key={s}>{pct(f.trainAcc - f.report.accuracy)}</td>)}</tr>
                <tr><td className="py-1.5 text-muted">{t('ทายตรงกันบน test', 'test agreement')}</td><td colSpan={2}>{pct(agreement)} ({disagree.filter(Boolean).length} {t('จุดไม่ตรงกัน', 'disagreements')})</td></tr>
              </tbody>
            </table>
          </div>
        ) },
        { id: 'concept', label: t('เปรียบเทียบอย่างยุติธรรม', 'Comparing fairly'), content: (
          <div className="grid max-w-5xl gap-8 text-[1.125rem] leading-7 lg:grid-cols-2">
            <ul className="list-disc space-y-1.5 pl-5">
              <li>{t('ใช้ชุดข้อมูล การแบ่ง และ seed เดียวกัน หน้านี้บังคับให้เป็นอย่างนั้นเสมอ', 'Use the same data, split and seed; this page always does.')}</li>
              <li>{t('ปรับมาตรฐานด้วยค่าจาก train เท่านั้น แล้วใช้กับ test', 'Standardise using train statistics only, then apply them to test.')}</li>
              <li>{t('ดู "ช่องว่าง train − test" ถ้ามากแปลว่า overfit', 'Watch the train − test gap; a large gap means overfitting.')}</li>
              <li>{t('ชุด test เล็กทำให้ความแม่นยำต่างกันได้เพราะความบังเอิญ ลองเปลี่ยน seed หลายค่าก่อนสรุป', 'A small test set can differ by chance; try several seeds before concluding.')}</li>
            </ul>
            <Callout>{t('รูปร่างพื้นผิวบอกลักษณะของแต่ละอัลกอริทึม: KNN เป็นหย่อม ๆ, ต้นไม้เป็นขั้นบันไดตั้งฉากกับแกน, Logistic มีรูป sigmoid, SVM RBF และ MLP โค้งเรียบ คะแนน SVM เป็นคะแนนตัดสินใจที่ปรับสเกล ไม่ใช่ความน่าจะเป็นที่ผ่านการ calibrate', 'Surface shapes reveal each algorithm: KNN is patchy, trees are axis-aligned steps, logistic has a sigmoid shape, RBF SVM and MLP are smooth curves. SVM confidence uses rescaled decision scores, not calibrated probabilities.')}</Callout>
          </div>
        ) },
        { id: 'ex', label: t('แบบฝึกหัด', 'Exercises'), content: (
          <Quiz id="compare" items={[
            { q: t('บน XOR ทำไม Logistic Regression แพ้ Decision Tree', 'On XOR, why does logistic regression lose to a decision tree?'), options: [t('เส้นแบ่งของ logistic เป็นเส้นตรง', 'Its boundary is a straight line'), t('ข้อมูลน้อยเกินไป', 'Too little data')], answer: 0, why: t('ต้นไม้แบ่งพื้นที่ได้หลายส่วนตั้งฉากกับแกน ซึ่งตรงกับรูปแบบ XOR', 'A tree carves several axis-aligned regions, which matches XOR.') },
            { q: t('Decision tree depth 15 ได้ train 100% แต่ test 85% หมายถึง', 'A depth-15 tree scores 100% train but 85% test. That indicates'), options: ['underfitting', 'overfitting'], answer: 1, why: t('เทียบกับ Random Forest ที่ depth เดียวกันบนหน้านี้', 'Compare with a random forest of the same depth on this page.') },
            { q: t('ถ้าสองโมเดลมี test accuracy เท่ากัน แปลว่าทายเหมือนกันทุกจุดหรือไม่', 'If two models have equal test accuracy, do they predict identically?'), options: [t('ใช่', 'Yes'), t('ไม่จำเป็น ดูค่า agreement', 'Not necessarily; check agreement')], answer: 1, why: t('อาจผิดคนละจุด วงสีขาวในฉากแสดงจุดเหล่านั้น', 'They may err on different points; the white rings show them.') },
          ]} />
        ) },
      ]}
    />
  );
}
