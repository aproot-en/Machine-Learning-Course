import { useMemo, useState } from 'react';
import * as THREE from 'three';
import { useCameraState } from '../../engine3d/Stage';
import { LabShell, Metrics } from '../../components/LabShell';
import { PointCloud, Segments, planeMapper } from '../../engine3d/plots';
import { BOX, Label } from '../../engine3d/primitives';
import { CLASS_COLORS } from '../../engine3d/palette';
import { datasetInfo, makeDataset, trainTestSplit, type DatasetId } from '../../data/datasets';
import { buildForest, buildTree, featureImportance, predictForest, predictTree, treeNodes, type TreeNode, type TreeParams } from '../../ml/tree';
import { classificationReport } from '../../ml/metrics';
import { useSettings, useT } from '../../i18n/i18n';
import { useInterval } from '../../lib/useLoop';
import { Section, Segmented, Select, Slider, Toggle, fmt, pct } from '../../components/ui';
import { ConfusionMatrix, Quiz, Tex } from '../../components/common';

const R = 5, FLOOR = -3.2;
const TILE = new THREE.BoxGeometry(2, 0.02, 2);
const DATA: DatasetId[] = ['blobs2d', 'moons', 'circles', 'xor', 'spiral', 'iris'];
type Mode = 'tree' | 'forest';
type V3 = [number, number, number];

/** In-order leaf positions; parents centred over children. */
function layoutTree(root: TreeNode, width: number, top: number, cx: number, z: number, maxOrder: number) {
  const pos = new Map<number, V3>();
  const visible = (n: TreeNode) => n.order < maxOrder;
  const leaves: TreeNode[] = [];
  const isLeaf = (n: TreeNode) => !n.right || !visible(n.right);
  const collect = (n: TreeNode) => { if (isLeaf(n)) leaves.push(n); else { collect(n.left!); collect(n.right!); } };
  collect(root);
  let depth = 0;
  treeNodes(root).forEach((n) => { if (visible(n)) depth = Math.max(depth, n.depth); });
  const dy = Math.min(1.5, 7 / Math.max(1, depth));
  const place = (n: TreeNode): number => {
    let x: number;
    if (isLeaf(n)) x = cx + (leaves.length === 1 ? 0 : (leaves.indexOf(n) / (leaves.length - 1) - 0.5) * width);
    else x = (place(n.left!) + place(n.right!)) / 2;
    pos.set(n.id, [x, top - n.depth * dy, z]);
    return x;
  };
  place(root);
  return pos;
}

export default function TreeLab({ initial = 'tree' }: { initial?: Mode }) {
  const t = useT();
  const level = useSettings((s) => s.level);
  const [mode, setMode] = useState<Mode>(initial);
  const [ds, setDs] = useState({ id: 'moons' as DatasetId, n: 200, classes: 3, noise: 0.25, seed: 4, testRatio: 0.25, fx: 2, fy: 3 });
  const [p, setP] = useState<TreeParams>({ maxDepth: 5, minSamplesSplit: 2, minSamplesLeaf: 1, criterion: 'gini', maxFeatures: 'all', ccpAlpha: 0 });
  const [rf, setRf] = useState({ nTrees: 12, bootstrap: true, seed: 2, featureMode: 'sqrt' as 'sqrt' | 'all' });
  const [grow, setGrow] = useState(Infinity);
  const [sel, setSel] = useState(0);
  const [query, setQuery] = useState<number[] | null>(null);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [showRegions, setShowRegions] = useState(true);
  const [cam, setCam] = useCameraState();
  const [experimentSample, setExperimentSample] = useState(0);
  const info = datasetInfo(ds.id);

  const data = useMemo(() => {
    const raw = makeDataset(ds.id, { n: ds.n, classes: ds.classes, noise: ds.noise, seed: ds.seed });
    const feats = ds.id === 'iris' ? [ds.fx, ds.fy] : [0, 1];
    const X = raw.X.map((r) => feats.map((f) => r[f]));
    const sp = trainTestSplit(raw.y, ds.testRatio, ds.seed);
    return { Xtr: sp.train.map((i) => X[i]), ytr: sp.train.map((i) => raw.y[i]), Xte: sp.test.map((i) => X[i]), yte: sp.test.map((i) => raw.y[i]), K: raw.nClasses, names: raw.classNames, feat: feats.map((f) => raw.featureNames[f]) };
  }, [ds]);

  const tree = useMemo(() => (mode === 'tree' ? buildTree(data.Xtr, data.ytr, data.K, p) : null), [mode, data, p]);
  const forest = useMemo(() => (mode === 'forest' ? buildForest(data.Xtr, data.ytr, data.K, { ...p, maxFeatures: rf.featureMode === 'sqrt' ? 'sqrt' : 'all' }, rf.nTrees, rf.bootstrap, rf.seed) : null), [mode, data, p, rf]);
  const nodes = useMemo(() => (tree ? treeNodes(tree) : []), [tree]);
  const total = mode === 'tree' ? nodes.length : rf.nTrees;
  const g = Math.min(grow, total);
  const done = g >= total;
  useInterval(running, mode === 'tree' ? 450 : 600, speed, () => { if (g >= total) return false; setGrow((q) => Math.min(total, Math.min(q, total) + (mode === 'tree' ? 2 : 1))); return true; }, () => setRunning(false));

  const predict = (x: number[]) => (mode === 'tree' ? predictTree(tree!, x, g).cls : predictForest(forest!, x, data.K, Math.max(1, g)).cls);
  const predTe = data.Xte.map(predict), predTr = data.Xtr.map(predict);
  const report = classificationReport(data.yte, predTe, data.K);
  const trainAcc = predTr.filter((v, i) => v === data.ytr[i]).length / data.ytr.length;

  const map = useMemo(() => planeMapper([...data.Xtr, ...data.Xte], R, 0.4), [data]);
  const P = (x: number[]): V3 => { const [a, c] = map.toScene(x); return [a, FLOOR + 0.1, c]; };
  const regions = useMemo(() => {
    if (!showRegions) return [];
    const N = 50, out: { p: V3; c: string }[] = [];
    for (let a = 0; a < N; a++) for (let b = 0; b < N; b++) {
      const sx = ((a + 0.5) / N - 0.5) * 2 * R, sz = ((b + 0.5) / N - 0.5) * 2 * R;
      out.push({ p: [sx, FLOOR, sz], c: CLASS_COLORS[predict(map.toData(sx, sz))] });
    }
    return out;
  }, [showRegions, map, tree, forest, g]); // eslint-disable-line react-hooks/exhaustive-deps

  const shownTree = mode === 'tree' ? tree! : forest!.trees[Math.min(sel, forest!.trees.length - 1)];
  const order = mode === 'tree' ? g : Infinity;
  const layout = useMemo(() => layoutTree(shownTree, 2 * R, 7, 0, -R - 1.5, order), [shownTree, order]);
  const path = query ? predictTree(shownTree, query, order).path : [];
  const visibleNodes = treeNodes(shownTree).filter((n) => n.order < order);
  const bootSet = mode === 'forest' ? new Set(forest!.bootstraps[Math.min(sel, rf.nTrees - 1)]) : null;
  const importance = useMemo(() => {
    if (mode === 'tree') return featureImportance(tree!, 2);
    const acc = [0, 0];
    forest!.trees.forEach((tr) => featureImportance(tr, 2, tr.samples.length).forEach((v, k) => (acc[k] += v / forest!.trees.length)));
    return acc;
  }, [mode, tree, forest]);
  const fq = query && forest ? predictForest(forest, query, data.K, Math.max(1, g)) : null;

  const nodeColor = (n: TreeNode) => { const tot = n.counts.reduce((a, b) => a + b, 0) || 1; return new THREE.Color(CLASS_COLORS[n.prediction]).multiplyScalar(0.35 + 0.65 * (n.counts[n.prediction] / tot)); };
  const scene = (
    <group>
      <gridHelper args={[2 * R, 10, '#1d2a4a', '#131c33']} position={[0, FLOOR - 0.01, 0]} />
      {regions.length > 0 && <PointCloud points={regions.map((r) => r.p)} colors={regions.map((r) => r.c)} sizes={(2 * R) / 50 / 2} geometry={TILE} basic opacity={0.22} />}
      <PointCloud points={data.Xtr.map(P)} colors={data.ytr.map((c, i) => (bootSet && !bootSet.has(i) ? new THREE.Color(CLASS_COLORS[c]).multiplyScalar(0.25) : CLASS_COLORS[c]))} sizes={data.Xtr.map((_, i) => (bootSet && !bootSet.has(i) ? 0.09 : 0.13))} />
      <PointCloud points={data.Xte.map(P)} colors={data.yte.map((c) => new THREE.Color(CLASS_COLORS[c]).multiplyScalar(0.7))} sizes={0.17} geometry={BOX} glow={0.1} />
      {/* the tree itself, standing behind the data floor */}
      <Segments segs={visibleNodes.filter((n) => n.right && n.right.order < order).flatMap((n) => [n.left!, n.right!].map((c) => ({ a: layout.get(n.id)!, b: layout.get(c.id)!, c: path.includes(c.id) ? '#ffffff' : '#4b5d85', i: path.includes(c.id) ? 1 : 0.6 })))} />
      <PointCloud points={visibleNodes.map((n) => layout.get(n.id)!)} colors={visibleNodes.map(nodeColor)} sizes={visibleNodes.map((n) => 0.12 + 0.3 * Math.sqrt(n.samples.length / data.Xtr.length) + (path.includes(n.id) ? 0.06 : 0))} glow={0.6} />
      {visibleNodes.length <= 31 && visibleNodes.map((n) => {
        const q = layout.get(n.id)!;
        const split = n.feature != null && n.right && n.right.order < order;
        return <Label key={n.id} position={[q[0], q[1] + 0.45, q[2]]} className={path.includes(n.id) ? 'text-cyan' : ''}>{split ? `${data.feat[n.feature!].replace(' (cm)', '')} ≤ ${fmt(n.threshold!, 2)}` : `→ ${data.names[n.prediction]}`}</Label>;
      })}
      {/* split lines of the shown tree projected on the floor */}
      {mode === 'tree' && <Segments segs={visibleNodes.filter((n) => n.feature != null && n.right && n.right.order < order).map((n) => {
        const xs = n.samples.map((i) => data.Xtr[i]);
        const lo = [Math.min(...xs.map((v) => v[0])), Math.min(...xs.map((v) => v[1]))], hi = [Math.max(...xs.map((v) => v[0])), Math.max(...xs.map((v) => v[1]))];
        const a = n.feature === 0 ? [n.threshold!, lo[1]] : [lo[0], n.threshold!], b = n.feature === 0 ? [n.threshold!, hi[1]] : [hi[0], n.threshold!];
        return { a: P(a), b: P(b), c: '#ffffff', i: path.includes(n.id) ? 1 : 0.35 };
      })} />}
      {query && <><mesh position={P(query)}><octahedronGeometry args={[0.3]} /><meshStandardMaterial color="#ffffff" emissive="#ffffff" emissiveIntensity={0.6} /></mesh>
        <Label position={[P(query)[0], FLOOR + 0.9, P(query)[2]]}>→ {data.names[predict(query)]}</Label></>}
      <mesh position={[0, FLOOR, 0]} rotation={[-Math.PI / 2, 0, 0]} onClick={(e) => { e.stopPropagation(); setQuery(map.toData(e.point.x, e.point.z)); }}>
        <planeGeometry args={[2 * R, 2 * R]} /><meshBasicMaterial transparent opacity={0} depthWrite={false} />
      </mesh>
      {mode === 'forest' && forest!.trees.map((tr, k) => {
        if (k === sel || k >= Math.max(1, g)) return null;
        const col = k % 6, row = Math.floor(k / 6);
        const lp = layoutTree(tr, 1.7, 3.4 - row * 2.6, (col - 2.5) * 2.2, -R - 5, Infinity);
        const ns = treeNodes(tr);
        return (
          <group key={k} scale={1}>
            <Segments segs={ns.filter((n) => n.left).flatMap((n) => [n.left!, n.right!].map((c) => ({ a: lp.get(n.id)!, b: lp.get(c.id)!, c: '#334466', i: 0.5 })))} />
            <PointCloud points={ns.map((n) => lp.get(n.id)!)} colors={ns.map(nodeColor)} sizes={0.06} glow={0.4} />
          </group>
        );
      })}
    </group>
  );

  const reset = () => { setRunning(false); setGrow(mode === 'tree' ? 1 : 1); };
  const leaves = visibleNodes.filter((n) => !n.right || n.right.order >= order).length;

  return (
    <LabShell crumb={mode === 'tree' ? 'Decision Tree' : 'Random Forest'} title={mode === 'tree' ? 'Decision Tree' : 'Random Forest'}
      experiment={{ model: mode === 'tree' ? 'Decision Tree' : 'Random Forest', source: `${t(info.name)} · ${data.Xtr.length + data.Xte.length} ${t('ตัวอย่าง', 'samples')}`,
        stages: [
          { name: 'Test input x', values: query ?? data.Xte[Math.min(experimentSample, data.Xte.length - 1)], labels: data.feat },
          { name: mode === 'tree' ? 'Decision path' : 'Selected tree path', kind: 'tree', points: visibleNodes.map((n) => { const point = layout.get(n.id)!; return [point[0], point[1] - 3.5, 0]; }), values: visibleNodes.map((n) => n.impurity),
            highlight: visibleNodes.flatMap((n, i) => predictTree(shownTree, query ?? data.Xte[Math.min(experimentSample, data.Xte.length - 1)], order).path.includes(n.id) ? [i] : []),
            labels: visibleNodes.map((n) => n.feature == null ? `class ${data.names[n.prediction]}` : `${data.feat[n.feature]} ≤ ${fmt(n.threshold!, 2)}`),
            edges: visibleNodes.flatMap((n, i) => [n.left, n.right].flatMap((child) => { const target = visibleNodes.findIndex((v) => v.id === child?.id); return target >= 0 ? [[i, target] as [number, number]] : []; })),
          },
          ...(mode === 'forest' ? [{ name: 'Each tree prediction', values: predictForest(forest!, query ?? data.Xte[Math.min(experimentSample, data.Xte.length - 1)], data.K, Math.max(1, g)).each }] : []),
          { name: mode === 'tree' ? 'Leaf distribution' : 'Forest votes', values: mode === 'tree' ? predictTree(tree!, query ?? data.Xte[Math.min(experimentSample, data.Xte.length - 1)], g).probs : predictForest(forest!, query ?? data.Xte[Math.min(experimentSample, data.Xte.length - 1)], data.K, Math.max(1, g)).votes, labels: data.names },
          { name: 'Output class', values: [predict(query ?? data.Xte[Math.min(experimentSample, data.Xte.length - 1)])] },
        ],
        metrics: [{ label: 'Train accuracy', value: pct(trainAcc) }, { label: 'Test accuracy', value: pct(report.accuracy) }, { label: mode === 'tree' ? 'Visible nodes' : 'Trees used', value: String(mode === 'tree' ? visibleNodes.length : Math.max(1, g)) }, { label: 'Test samples', value: String(data.Xte.length) }],
        confusion: report.confusion,
        sample: { index: experimentSample, count: data.Xte.length, onChange: (index) => { setExperimentSample(index); setQuery(null); } },
        output: `${t('คลาส', 'Class')}: ${data.names[predict(query ?? data.Xte[Math.min(experimentSample, data.Xte.length - 1)])]} · test accuracy=${pct(report.accuracy)}`,
      }}
      chips={[t('การเรียนรู้แบบมีผู้สอน', 'Supervised'), t('ปานกลาง', 'Intermediate')]}
      intro={mode === 'tree' ? t('กด ↺ แล้ว ▶ เพื่อดูต้นไม้โตทีละโหนด สีของพื้นคือพื้นที่ตัดสินใจ คลิกที่พื้นเพื่อดูเส้นทางการตัดสินใจในต้นไม้', 'Press ↺ then ▶ to grow the tree node by node. The floor colours are decision regions; click the floor to trace a decision path.')
        : t('แต่ละต้นฝึกบน bootstrap sample ของตัวเอง (จุดจาง = ไม่ถูกสุ่มเลือกให้ต้นที่เลือก) ผลทำนายคือการโหวตของทุกต้น', 'Each tree is trained on its own bootstrap sample (faded points = not drawn for the selected tree). The prediction is the vote of all trees.')}
      cam={cam} setCam={setCam} distance={21} target={[0, 0.5, -2]} scene={scene}
      readout={{ title: mode === 'tree' ? 'CART' : 'Random Forest', rows: [
        [t('ชุดข้อมูล', 'dataset'), t(info.name)],
        ...(mode === 'tree' ? [[t('โหนด', 'nodes'), `${visibleNodes.length} / ${nodes.length}`] as [string, string], [t('ใบ', 'leaves'), String(leaves)] as [string, string]]
          : [[t('ต้นไม้ที่ใช้', 'trees used'), `${Math.max(1, g)} / ${rf.nTrees}`] as [string, string], ['OOB accuracy', forest!.oob == null ? '—' : pct(forest!.oob)] as [string, string]]),
        [`${p.criterion} (root)`, fmt(shownTree.impurity, 3)],
        ['acc train / test', `${pct(trainAcc)} / ${pct(report.accuracy)}`],
        [t('ความสำคัญฟีเจอร์', 'importance'), importance.map((v, k) => `${data.feat[k].replace(' (cm)', '')} ${pct(v)}`).join(' · ')],
        ...(fq ? [[t('โหวต', 'votes'), fq.votes.map((v, c) => `${data.names[c]}:${v}`).join(' ')] as [string, string]] : []),
      ] }}
      playback={{ playing: running, onToggle: () => { if (done) setGrow(1); setRunning(!running); }, onNext: () => setGrow(Math.min(total, g + (mode === 'tree' ? 2 : 1))), onPrev: () => { setRunning(false); setGrow(Math.max(1, g - (mode === 'tree' ? 2 : 1))); },
        canPrev: g > 1, canNext: !done, onReset: reset, speed, onSpeed: setSpeed, status: mode === 'tree' ? `${t('โหนด', 'node')} ${g}/${total}` : `${t('ต้น', 'tree')} ${Math.max(1, g)}/${total}` }}
      controls={
        <>
          <Section title={t('โมเดล', 'Model')}>
            <Segmented value={mode} onChange={(m) => { setMode(m); setGrow(Infinity); setSel(0); }} options={[{ value: 'tree', label: 'Decision Tree' }, { value: 'forest', label: 'Random Forest' }]} />
          </Section>
          <Section title={t('ชุดข้อมูล', 'Dataset')}>
            <Select<DatasetId> label={t('ชุดข้อมูล', 'Dataset')} value={ds.id} onChange={(id) => { setDs({ ...ds, id }); setQuery(null); }} options={DATA.map((id) => ({ value: id, label: t(datasetInfo(id).name) }))} />
            {ds.id === 'iris' && (
              <div className="grid grid-cols-2 gap-2">
                {(['fx', 'fy'] as const).map((k) => <Select<number> key={k} label={k === 'fx' ? 'x' : 'y'} value={ds[k]} onChange={(v) => setDs({ ...ds, [k]: v })} options={['sepal L', 'sepal W', 'petal L', 'petal W'].map((n, i) => ({ value: i, label: n, disabled: i === ds[k === 'fx' ? 'fy' : 'fx'] }))} />)}
              </div>
            )}
            {info.synthetic && <Slider label={t('จำนวนตัวอย่าง', 'Samples')} value={ds.n} min={60} max={500} step={20} onChange={(n) => setDs({ ...ds, n })} />}
            {info.fixedClasses == null && <Slider label={t('จำนวนคลาส', 'Classes')} value={ds.classes} min={2} max={5} onChange={(classes) => setDs({ ...ds, classes })} />}
            {info.synthetic && <Slider label="Noise" value={ds.noise} min={0} max={0.8} step={0.05} format={(v) => v.toFixed(2)} onChange={(noise) => setDs({ ...ds, noise })} />}
            <Slider label="Seed" value={ds.seed} min={1} max={99} onChange={(seed) => setDs({ ...ds, seed })} />
          </Section>
          <Section title={t('การแบ่ง', 'Splitting')}>
            <Slider label="Max depth" value={p.maxDepth} min={1} max={12} onChange={(maxDepth) => setP({ ...p, maxDepth })} />
            <Slider label="Min samples split" value={p.minSamplesSplit} min={2} max={40} onChange={(minSamplesSplit) => setP({ ...p, minSamplesSplit })} />
            <Slider label="Min samples leaf" value={p.minSamplesLeaf} min={1} max={30} onChange={(minSamplesLeaf) => setP({ ...p, minSamplesLeaf })} />
            <Segmented label="Criterion" value={p.criterion} onChange={(criterion) => setP({ ...p, criterion })} options={[{ value: 'gini', label: 'Gini' }, { value: 'entropy', label: 'Entropy' }]} />
            {level >= 2 && <Slider label={t('Pruning α (ccp)', 'Pruning α (ccp)')} value={p.ccpAlpha} min={0} max={0.05} step={0.001} format={(v) => v.toFixed(3)} onChange={(ccpAlpha) => setP({ ...p, ccpAlpha })} />}
          </Section>
          {mode === 'forest' && (
            <Section title="Forest">
              <Slider label={t('จำนวนต้นไม้', 'Number of trees')} value={rf.nTrees} min={1} max={30} onChange={(nTrees) => { setRf({ ...rf, nTrees }); setSel(0); }} />
              <Segmented label="Max features" value={rf.featureMode} onChange={(featureMode) => setRf({ ...rf, featureMode })} options={[{ value: 'sqrt', label: '√d' }, { value: 'all', label: t('ทั้งหมด', 'all') }]} />
              <Toggle label="Bootstrap" checked={rf.bootstrap} onChange={(bootstrap) => setRf({ ...rf, bootstrap })} />
              <Slider label="Forest seed" value={rf.seed} min={1} max={99} onChange={(seed) => setRf({ ...rf, seed })} />
              <Slider label={t('ต้นที่แสดงขนาดใหญ่', 'Tree shown large')} value={sel} min={0} max={rf.nTrees - 1} onChange={setSel} />
            </Section>
          )}
          <Section title={t('การแสดงผล', 'View')}>
            <Toggle label={t('พื้นที่ตัดสินใจ', 'Decision regions')} checked={showRegions} onChange={setShowRegions} />
          </Section>
        </>
      }
      tabs={[
        { id: 'concept', label: t('แนวคิด', 'Concept'), content: (
          <div className="grid max-w-5xl gap-8 text-[1.125rem] leading-7 lg:grid-cols-2">
            <p>{t('Decision tree (CART) แบ่งข้อมูลซ้ำ ๆ ด้วยคำถามแบบ "ฟีเจอร์ ≤ ค่า" โดยทุกโหนดลองทุกฟีเจอร์และทุกค่าแบ่ง แล้วเลือกตัวที่ลด impurity ได้มากที่สุด เส้นแบ่งจึงตั้งฉากกับแกนเสมอ ต้นที่ลึกเกินไปจำ noise ได้ (overfitting)', 'A decision tree (CART) repeatedly splits data with “feature ≤ value” questions. Each node tries every feature and threshold and keeps the one that reduces impurity most, so boundaries are always axis-aligned. Very deep trees memorise noise (overfitting).')}</p>
            <p>{t('Random forest ฝึกต้นไม้หลายต้น แต่ละต้นเห็น bootstrap sample ต่างกันและสุ่มฟีเจอร์ที่ใช้แบ่ง ต้นไม้จึงผิดพลาดต่างกัน การโหวตรวมช่วยลด variance ความแม่นยำ OOB วัดจากจุดที่แต่ละต้นไม่ได้เห็น', 'A random forest trains many trees, each on a different bootstrap sample with random feature subsets, so their errors differ. Majority voting lowers variance. OOB accuracy is measured on points each tree never saw.')}</p>
          </div>
        ) },
        { id: 'math', label: t('คณิตศาสตร์', 'Math'), content: (
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="flex flex-col gap-3">
              <Tex block>{'\\text{Gini}(S) = 1 - \\sum_k p_k^2,\\qquad H(S) = -\\sum_k p_k\\log_2 p_k'}</Tex>
              <Tex block>{'\\Delta = I(S) - \\frac{|S_L|}{|S|}I(S_L) - \\frac{|S_R|}{|S|}I(S_R)'}</Tex>
              <Tex block>{'\\hat y_{forest}(\\mathbf x) = \\text{mode}\\{\\hat y_1(\\mathbf x), \\dots, \\hat y_T(\\mathbf x)\\}'}</Tex>
            </div>
            <div className="scroll-thin max-h-[360px] overflow-auto">
              <table className="num w-full text-left font-mono text-[0.96875rem]">
                <thead className="sticky top-0 bg-bg text-muted"><tr><th className="py-1">#</th><th>depth</th><th>n</th><th>{t('จำนวนต่อคลาส', 'class counts')}</th><th>{p.criterion}</th><th>{t('การแบ่ง', 'split')}</th><th>Δ</th></tr></thead>
                <tbody>{visibleNodes.map((n) => (
                  <tr key={n.id} className={`border-b border-line/50 ${path.includes(n.id) ? 'bg-cyan/10' : ''}`}>
                    <td className="py-1">{n.id}</td><td>{n.depth}</td><td>{n.samples.length}</td><td>[{n.counts.join(', ')}]</td><td>{fmt(n.impurity, 3)}</td>
                    <td>{n.feature != null && n.right && n.right.order < order ? `${data.feat[n.feature].replace(' (cm)', '')} ≤ ${fmt(n.threshold!, 3)}` : t('ใบ', 'leaf')}</td><td>{n.gain != null && n.right && n.right.order < order ? fmt(n.gain, 3) : ''}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          </div>
        ) },
        { id: 'eval', label: t('การประเมินผล', 'Evaluation'), content: (
          <div className="grid gap-8 lg:grid-cols-2">
            <div className="flex flex-col gap-4">
              <Metrics items={[['Accuracy', pct(report.accuracy)], ['Precision', pct(report.precision)], ['Recall', pct(report.recall)], ['F1', pct(report.f1)]]} />
              <div>
                <h3 className="label-xs mb-2">{t('ความสำคัญของฟีเจอร์ (ลด impurity เฉลี่ย)', 'Feature importance (mean impurity decrease)')}</h3>
                {importance.map((v, k) => (
                  <div key={k} className="flex items-center gap-3 font-mono text-[1rem]"><span className="w-36 truncate">{data.feat[k]}</span><span className="h-2 flex-1 rounded bg-line"><span className="block h-full rounded bg-cyan" style={{ width: pct(v) }} /></span><span className="w-14 text-right">{pct(v)}</span></div>
                ))}
              </div>
              {fq && <div className="font-mono text-[1rem]">{t('การโหวตของแต่ละต้นสำหรับจุดที่คลิก', 'Per-tree votes for the clicked point')}: {fq.each.map((c, i) => <span key={i} style={{ color: CLASS_COLORS[c] }}>{data.names[c]} </span>)}</div>}
            </div>
            <div><h3 className="label-xs mb-2">Confusion matrix (test)</h3><ConfusionMatrix m={report.confusion} classNames={data.names} /></div>
          </div>
        ) },
        { id: 'ex', label: t('แบบฝึกหัด', 'Exercises'), content: (
          <Quiz id={`tree-${mode}`} items={[
            { q: t('ทำไมเส้นแบ่งของ decision tree ตั้งฉากกับแกนเสมอ', 'Why are tree boundaries always axis-aligned?'), options: [t('เพราะแต่ละโหนดถามเพียงฟีเจอร์เดียว ≤ ค่าหนึ่ง', 'Each node tests a single feature against a threshold'), t('เพราะข้อมูลถูกปรับมาตรฐาน', 'Because data is standardised')], answer: 0, why: t('ลองชุด Circles แล้วดูขั้นบันไดของพื้นที่ตัดสินใจ', 'Try Circles and look at the staircase regions.') },
            { q: t('max depth สูงมากกับข้อมูลมี noise จะเกิดอะไร', 'Very large max depth on noisy data causes…'), options: ['underfitting', 'overfitting'], answer: 1, why: t('train accuracy ใกล้ 100% แต่ test ลดลง', 'Train accuracy nears 100% while test drops.') },
            { q: t('ทำไม random forest มักดีกว่าต้นเดียว', 'Why does a random forest usually beat a single tree?'), options: [t('แต่ละต้นลึกกว่า', 'Each tree is deeper'), t('ต้นที่ผิดพลาดต่างกันโหวตรวมกันแล้ว variance ลดลง', 'Trees with different errors vote, reducing variance')], answer: 1, why: t('bootstrap และการสุ่มฟีเจอร์ทำให้ต้นไม้ไม่เหมือนกัน', 'Bootstrapping and random features decorrelate the trees.') },
          ]} />
        ) },
      ]}
    />
  );
}
