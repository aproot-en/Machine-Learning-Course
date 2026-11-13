import { useMemo, useState } from 'react';
import * as THREE from 'three';
import { useCameraState } from '../../engine3d/Stage';
import { LabShell, Metrics } from '../../components/LabShell';
import { distance } from '../../ml/knn';
import { PointCloud, Polyline, Segments } from '../../engine3d/plots';
import { FrameBox, Label } from '../../engine3d/primitives';
import { CLASS_COLORS } from '../../engine3d/palette';
import { applyScaler, datasetInfo, fitScaler, makeDataset, type DatasetId } from '../../data/datasets';
import { dbscanEvents, dbscanReplay, initCentroids, kmeansNext, silhouette, type KMeansState } from '../../ml/clustering';
import { useT } from '../../i18n/i18n';
import { useInterval } from '../../lib/useLoop';
import { Section, Segmented, Select, Slider, Toggle, fmt } from '../../components/ui';
import { LinePlot, Quiz, Tex } from '../../components/common';

const R = 5;
const OCTA = new THREE.OctahedronGeometry(1, 0);
const TILE = new THREE.BoxGeometry(2, 2, 0.02);
type Mode = 'kmeans' | 'dbscan';
const DATA: DatasetId[] = ['blobs3d', 'blobs2d', 'moons', 'circles', 'iris'];

export default function ClusterLab({ initial = 'kmeans' }: { initial?: Mode }) {
  const t = useT();
  const [mode, setMode] = useState<Mode>(initial);
  const [ds, setDs] = useState({ id: (initial === 'dbscan' ? 'moons' : 'blobs3d') as DatasetId, n: 240, classes: 4, noise: 0.2, seed: 5 });
  const [km, setKm] = useState({ k: 4, init: 'kmeans++' as 'kmeans++' | 'random', maxIter: 50, tol: 1e-4, seed: 1 });
  const [db, setDb] = useState({ eps: 0.3, minPts: 5, metric: 'euclidean' as 'euclidean' | 'manhattan' });
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [pos, setPos] = useState(0);
  const [regions, setRegions] = useState(true);
  const [cam, setCam] = useCameraState();
  const [experimentSample, setExperimentSample] = useState(0);
  const info = datasetInfo(ds.id);

  const X = useMemo(() => {
    const raw = makeDataset(ds.id, { n: ds.n, classes: ds.classes, noise: ds.noise, seed: ds.seed });
    const sel = raw.X.map((r) => r.slice(0, 3));
    const sc = fitScaler(sel);
    return sel.map((r) => applyScaler(sc, r));
  }, [ds]);
  const dims = X[0].length;
  const scale = useMemo(() => (R * 0.95) / Math.max(1e-6, ...X.flat().map(Math.abs)), [X]);
  const S = (v: number[]): [number, number, number] => [v[0] * scale, v[1] * scale, (v[2] ?? 0) * scale];

  // ---------- K-Means: every half-step recorded ----------
  const kmStates = useMemo(() => {
    const s0: KMeansState = { iter: 0, phase: 'assign', centroids: initCentroids(X, km.k, km.init, km.seed), assign: X.map(() => -1), inertia: null, changed: 0, displacement: 0, converged: false };
    return [s0];
  }, [X, km.k, km.init, km.seed]);
  // ---------- DBSCAN: event log ----------
  const events = useMemo(() => dbscanEvents(X, db.eps, db.minPts, db.metric), [X, db]);
  const [, setTick] = useState(0);
  const total = mode === 'kmeans' ? kmStates.length - 1 : events.length;
  const p = Math.min(pos, total);

  const step = () => {
    if (mode === 'kmeans') {
      const last = kmStates[kmStates.length - 1];
      if (p < kmStates.length - 1) { setPos(p + 1); return true; }
      if (last.converged) return false;
      kmStates.push(kmeansNext(X, last, km.tol, km.maxIter));
      setTick((x) => x + 1);
      setPos(kmStates.length - 1);
      return true;
    }
    if (p >= events.length) return false;
    // skip quickly through the bookkeeping events so each tick shows one point decision
    let q = p + 1;
    while (q < events.length && events[q - 1].type === 'cluster') q++;
    setPos(q);
    return q < events.length;
  };
  useInterval(running, mode === 'kmeans' ? 700 : 90, speed, step, () => setRunning(false));

  const kmS = kmStates[Math.min(p, kmStates.length - 1)];
  const dbS = useMemo(() => dbscanReplay(X.length, events, p), [X, events, p]);
  const labels = mode === 'kmeans' ? kmS.assign : dbS.labels;
  const sil = useMemo(() => (labels.every((l) => l >= -1) ? silhouette(X, labels) : null), [X, labels]);

  // elbow curve (final inertia for K = 1…8) and k-distance curve for choosing ε
  const elbow = useMemo(() => {
    if (mode !== 'kmeans') return [];
    return Array.from({ length: 8 }, (_, i) => {
      let s: KMeansState = { iter: 0, phase: 'assign', centroids: initCentroids(X, i + 1, 'kmeans++', km.seed), assign: X.map(() => -1), inertia: null, changed: 0, displacement: 0, converged: false };
      for (let g = 0; g < 200 && !s.converged; g++) s = kmeansNext(X, s, 1e-6, 100);
      return { k: i + 1, inertia: s.inertia ?? 0 };
    });
  }, [mode, X, km.seed]);
  const kdist = useMemo(() => {
    if (mode !== 'dbscan') return [];
    return X.map((x) => X.map((y) => Math.hypot(...x.map((v, k) => v - y[k]))).sort((a, b) => a - b)[Math.min(db.minPts - 1, X.length - 1)]).sort((a, b) => b - a);
  }, [mode, X, db.minPts]);

  // Voronoi-style regions on the floor (2D data only)
  const voronoi = useMemo(() => {
    if (mode !== 'kmeans' || dims !== 2 || !regions) return [];
    const N = 48, out: { p: [number, number, number]; c: string }[] = [];
    for (let a = 0; a < N; a++) for (let b = 0; b < N; b++) {
      const sx = ((a + 0.5) / N - 0.5) * 2 * R, sy = ((b + 0.5) / N - 0.5) * 2 * R;
      let best = 0, bd = Infinity;
      kmS.centroids.forEach((c, j) => { const d = (c[0] * scale - sx) ** 2 + (c[1] * scale - sy) ** 2; if (d < bd) { bd = d; best = j; } });
      out.push({ p: [sx, sy, -0.05], c: CLASS_COLORS[best % CLASS_COLORS.length] });
    }
    return out;
  }, [mode, dims, regions, kmS, scale]);

  const colorOf = (l: number) => (l >= 0 ? CLASS_COLORS[l % CLASS_COLORS.length] : l === -1 ? '#7f1d1d' : '#475569');
  const segs = mode === 'kmeans' && kmS.assign[0] >= 0
    ? X.map((x, i) => ({ a: S(x), b: S(kmS.centroids[kmS.assign[i]]), c: colorOf(kmS.assign[i]), i: 0.18 }))
    : mode === 'dbscan' && dbS.current != null ? dbS.neighbors.map((j) => ({ a: S(X[dbS.current!]), b: S(X[j]), c: '#ffffff', i: 0.6 })) : [];
  const trails = mode === 'kmeans' ? kmS.centroids.map((_, j) => kmStates.slice(0, p + 1).filter((s) => s.phase === 'assign').map((s) => S(s.centroids[j]))) : [];
  const sizes = mode === 'dbscan' ? dbS.kind.map((k) => (k === 'core' ? 0.17 : k === 'border' ? 0.12 : 0.1)) : 0.14;

  const scene = (
    <group>
      <FrameBox size={R} labels={['z₁', 'z₂', 'z₃']} flat={dims === 2} />
      {voronoi.length > 0 && <PointCloud points={voronoi.map((v) => v.p)} colors={voronoi.map((v) => v.c)} sizes={(2 * R) / 48 / 2} geometry={TILE} basic opacity={0.14} />}
      <PointCloud points={X.map(S)} colors={labels.map(colorOf)} sizes={sizes} />
      <Segments segs={segs} />
      {mode === 'kmeans' && <PointCloud points={kmS.centroids.map(S)} colors={kmS.centroids.map((_, j) => colorOf(j))} sizes={0.42} geometry={OCTA} glow={0.8} />}
      {trails.map((tr, j) => tr.length > 1 && <Polyline key={j} points={tr} color="#ffffff" colorEnd={colorOf(j)} />)}
      {mode === 'dbscan' && dbS.current != null && (dims === 3
        ? <mesh position={S(X[dbS.current])} scale={db.eps * scale}><sphereGeometry args={[1, 24, 16]} /><meshBasicMaterial color="#ffffff" wireframe transparent opacity={0.18} /></mesh>
        : <mesh position={S(X[dbS.current])} scale={db.eps * scale}><ringGeometry args={[0.97, 1, 64]} /><meshBasicMaterial color="#ffffff" transparent opacity={0.7} /></mesh>)}
      {mode === 'dbscan' && dbS.current != null && <Label position={(() => { const s = S(X[dbS.current!]); return [s[0], s[1] + 0.6, s[2]] as [number, number, number]; })()}>{dbS.neighbors.length} {t('เพื่อนบ้าน', 'neighbours')}</Label>}
    </group>
  );

  const reset = () => { setRunning(false); setPos(0); if (mode === 'kmeans') kmStates.splice(1); };
  const kmPhaseText = kmS.converged ? t('ลู่เข้าแล้ว', 'converged') : kmS.phase === 'assign' ? t('ถัดไป: assign จุดเข้ากลุ่มที่ใกล้ที่สุด', 'next: assign points to the nearest centroid') : t('ถัดไป: ย้าย centroid ไปที่ค่าเฉลี่ย', 'next: move centroids to the mean');
  const dbEvent = p > 0 ? events[p - 1] : null;
  const dbText = !dbEvent ? t('เริ่ม: ทุกจุดยังไม่ถูกเยี่ยม (สีเทา)', 'Start: every point is unvisited (grey)')
    : dbEvent.type === 'visit' ? t(`เยี่ยมจุด ${dbEvent.p}: พบ ${dbEvent.neighbors.length} จุดในรัศมี ε (ต้องการ ${db.minPts})`, `Visit point ${dbEvent.p}: ${dbEvent.neighbors.length} points within ε (need ${db.minPts})`)
      : dbEvent.type === 'core' ? t(`จุด ${dbEvent.p} เป็น core point ขยายคลัสเตอร์ ${dbEvent.c}`, `Point ${dbEvent.p} is a core point; expand cluster ${dbEvent.c}`)
        : dbEvent.type === 'border' ? t(`จุด ${dbEvent.p} เป็น border point ของคลัสเตอร์ ${dbEvent.c}`, `Point ${dbEvent.p} is a border point of cluster ${dbEvent.c}`)
          : dbEvent.type === 'noise' ? t(`จุด ${dbEvent.p} มีเพื่อนบ้านไม่พอ → noise (ชั่วคราว)`, `Point ${dbEvent.p} has too few neighbours → noise (for now)`)
            : t(`เริ่มคลัสเตอร์ใหม่ ${dbEvent.c}`, `Start new cluster ${dbEvent.c}`);
  const nNoise = dbS.labels.filter((l) => l === -1).length;

  return (
    <LabShell crumb={mode === 'kmeans' ? 'K-Means' : 'DBSCAN'} title={mode === 'kmeans' ? 'K-Means Clustering' : 'DBSCAN'}
      experiment={{ model: mode === 'kmeans' ? 'K-Means' : 'DBSCAN', source: `${t(info.name)} · ${X.length} ${t('จุด ไม่มี label สำหรับฝึก', 'points, no training labels')}`,
        stages: [
          { name: 'Input data X', kind: 'points', points: X, values: X.map(() => 1), highlight: [Math.min(experimentSample, X.length - 1)] },
          { name: `x[${Math.min(experimentSample, X.length - 1)}]`, values: X[Math.min(experimentSample, X.length - 1)] },
          mode === 'kmeans' ? { name: 'Distance to centroids', values: kmS.centroids.map((c) => distance(X[Math.min(experimentSample, X.length - 1)], c, 'euclidean')) }
            : { name: `Distance / ε=${db.eps}`, values: X.map((x) => distance(x, X[Math.min(experimentSample, X.length - 1)], db.metric)), highlight: X.flatMap((x, i) => distance(x, X[Math.min(experimentSample, X.length - 1)], db.metric) <= db.eps ? [i] : []) },
          mode === 'kmeans' ? { name: 'Current centroids', kind: 'points', points: kmS.centroids, values: kmS.centroids.map(() => 1), classes: kmS.centroids.map((_, i) => i) }
            : { name: 'ε neighborhood', kind: 'points', points: X.filter((x) => distance(x, X[Math.min(experimentSample, X.length - 1)], db.metric) <= db.eps), values: X.filter((x) => distance(x, X[Math.min(experimentSample, X.length - 1)], db.metric) <= db.eps).map(() => 1) },
          { name: 'Cluster assignments', kind: 'points', points: X, values: labels, classes: labels, highlight: [Math.min(experimentSample, X.length - 1)] },
        ],
        metrics: [{ label: 'Samples', value: String(X.length) }, { label: mode === 'kmeans' ? 'Iteration' : 'Replay events', value: String(mode === 'kmeans' ? kmS.iter : p) }, { label: mode === 'kmeans' ? 'Inertia' : 'Noise points', value: mode === 'kmeans' ? kmS.inertia == null ? '—' : fmt(kmS.inertia, 3) : String(nNoise) }, { label: 'Silhouette', value: sil == null ? '—' : fmt(sil, 3) }],
        history: mode === 'kmeans' ? [{ label: 'Inertia', values: kmStates.slice(0, p + 1).filter((state) => state.inertia != null).map((state) => state.inertia!) }] : undefined,
        sample: { index: experimentSample, count: X.length, onChange: setExperimentSample },
        output: `${t('กลุ่มของจุดที่เลือก', 'Selected point cluster')}: ${labels[Math.min(experimentSample, labels.length - 1)] >= 0 ? labels[Math.min(experimentSample, labels.length - 1)] : labels[Math.min(experimentSample, labels.length - 1)] === -1 && mode === 'dbscan' ? 'noise' : t('ยังไม่จัดกลุ่ม', 'not assigned')} · ${t('จัดกลุ่มแล้ว', 'assigned')} ${labels.filter((v) => v >= 0).length}/${X.length}${sil == null ? '' : ` · silhouette=${fmt(sil)}`}`,
        note: t('ผลลัพธ์คือกลุ่มที่ค้นพบ ไม่ใช่คลาสคำตอบเดิม', 'Outputs are discovered clusters, not ground-truth classes'),
      }}
      chips={[t('การเรียนรู้แบบไม่มีผู้สอน', 'Unsupervised'), t('พื้นฐาน', 'Beginner')]}
      intro={mode === 'kmeans' ? t('กด → เพื่อเดินทีละครึ่งขั้น (assign / update) หรือ ▶ เพื่อเล่นต่อเนื่อง ทรงแปดหน้าคือ centroid เส้นขาวคือเส้นทางที่ centroid เคลื่อนที่', 'Press → for one half-step (assign / update) or ▶ to play. Octahedra are centroids; white trails are their paths.')
        : t('DBSCAN หาคลัสเตอร์จากความหนาแน่น กด ▶ เพื่อดูการเยี่ยมแต่ละจุด วงกลมคือรัศมี ε จุดใหญ่ = core จุดเล็ก = border แดงเข้ม = noise', 'DBSCAN finds clusters by density. Press ▶ to watch each point being visited. The circle is the ε radius; large = core, small = border, dark red = noise.')}
      cam={cam} setCam={setCam} distance={dims === 2 ? 15 : 18} lockRotation={false} scene={scene}
      readout={{ title: mode === 'kmeans' ? 'K-Means' : 'DBSCAN', rows: mode === 'kmeans' ? [
        [t('ชุดข้อมูล', 'dataset'), t(info.name)], ['K', km.k], [t('รอบที่', 'iteration'), kmS.iter],
        [t('จุดที่ย้ายกลุ่ม', 'points reassigned'), kmS.assign[0] >= 0 ? kmS.changed : '—'],
        [t('centroid ขยับสูงสุด', 'max centroid shift'), fmt(kmS.displacement, 4)],
        ['inertia (WCSS)', kmS.inertia == null ? '—' : fmt(kmS.inertia, 2)],
        ['silhouette', sil == null ? '—' : fmt(sil, 3)],
      ] : [
        [t('ชุดข้อมูล', 'dataset'), t(info.name)], ['ε · MinPts', `${db.eps} · ${db.minPts}`], [t('เหตุการณ์', 'event'), `${p} / ${events.length}`],
        [t('คลัสเตอร์', 'clusters'), dbS.clusters], ['core / border', `${dbS.kind.filter((k) => k === 'core').length} / ${dbS.kind.filter((k) => k === 'border').length}`],
        ['noise', nNoise], ['silhouette', sil == null ? '—' : fmt(sil, 3)],
      ] }}
      caption={mode === 'kmeans' ? kmPhaseText : dbText}
      playback={{ playing: running, onToggle: () => setRunning(!running), onNext: () => step(), onPrev: () => { setRunning(false); setPos(Math.max(0, p - 1)); }, canPrev: p > 0,
        canNext: mode === 'kmeans' ? !(kmS.converged && p >= kmStates.length - 1) : p < events.length, onReset: reset, speed, onSpeed: setSpeed, status: `${p}/${mode === 'kmeans' ? kmStates.length - 1 : events.length}` }}
      controls={
        <>
          <Section title={t('อัลกอริทึม', 'Algorithm')}>
            <Segmented value={mode} onChange={(m) => { setMode(m); setPos(0); setRunning(false); }} options={[{ value: 'kmeans', label: 'K-Means' }, { value: 'dbscan', label: 'DBSCAN' }]} />
          </Section>
          <Section title={t('ชุดข้อมูล', 'Dataset')}>
            <Select<DatasetId> label={t('ชุดข้อมูล', 'Dataset')} value={ds.id} onChange={(id) => { setDs({ ...ds, id }); setPos(0); }} options={DATA.map((id) => ({ value: id, label: t(datasetInfo(id).name) }))} />
            {info.synthetic && <Slider label={t('จำนวนตัวอย่าง', 'Samples')} value={ds.n} min={60} max={500} step={20} onChange={(n) => { setDs({ ...ds, n }); setPos(0); }} />}
            {info.fixedClasses == null && <Slider label={t('จำนวนกลุ่มจริง', 'True groups')} value={ds.classes} min={2} max={6} onChange={(classes) => { setDs({ ...ds, classes }); setPos(0); }} />}
            {info.synthetic && <Slider label="Noise" value={ds.noise} min={0} max={0.8} step={0.05} format={(v) => v.toFixed(2)} onChange={(noise) => { setDs({ ...ds, noise }); setPos(0); }} />}
            <Slider label="Data seed" value={ds.seed} min={1} max={99} onChange={(seed) => { setDs({ ...ds, seed }); setPos(0); }} />
          </Section>
          {mode === 'kmeans' ? (
            <Section title="K-Means">
              <Slider label="K" value={km.k} min={1} max={8} onChange={(k) => { setKm({ ...km, k }); setPos(0); }} />
              <Select<'kmeans++' | 'random'> label={t('การวาง centroid เริ่มต้น', 'Initialisation')} value={km.init} onChange={(init) => { setKm({ ...km, init }); setPos(0); }} options={[{ value: 'kmeans++', label: 'k-means++' }, { value: 'random', label: t('สุ่มจากจุดข้อมูล', 'Random points') }]} />
              <Slider label={t('รอบสูงสุด', 'Max iterations')} value={km.maxIter} min={1} max={100} onChange={(maxIter) => { setKm({ ...km, maxIter }); setPos(0); }} />
              <Slider label="Tolerance (log₁₀)" value={Math.log10(km.tol)} min={-6} max={-1} step={1} format={(v) => `1e${v}`} onChange={(v) => { setKm({ ...km, tol: 10 ** v }); setPos(0); }} />
              <Slider label="Init seed" value={km.seed} min={1} max={99} onChange={(seed) => { setKm({ ...km, seed }); setPos(0); }} />
              {dims === 2 && <Toggle label={t('พื้นที่แบบ Voronoi', 'Voronoi regions')} checked={regions} onChange={setRegions} />}
            </Section>
          ) : (
            <Section title="DBSCAN">
              <Slider label="ε (epsilon)" value={db.eps} min={0.05} max={1.5} step={0.01} format={(v) => v.toFixed(2)} onChange={(eps) => { setDb({ ...db, eps }); setPos(0); }} />
              <Slider label="MinPts" value={db.minPts} min={2} max={20} onChange={(minPts) => { setDb({ ...db, minPts }); setPos(0); }} />
              <Select<'euclidean' | 'manhattan'> label={t('วิธีวัดระยะ', 'Metric')} value={db.metric} onChange={(metric) => { setDb({ ...db, metric }); setPos(0); }} options={[{ value: 'euclidean', label: 'Euclidean' }, { value: 'manhattan', label: 'Manhattan' }]} />
              <p className="text-[0.9375rem] text-muted">{t('ข้อมูลถูกปรับมาตรฐานก่อน ε จึงอยู่ในหน่วย z-score', 'Data is standardised, so ε is in z-score units.')}</p>
            </Section>
          )}
        </>
      }
      tabs={[
        { id: 'concept', label: t('แนวคิด', 'Concept'), content: (
          <div className="grid max-w-5xl gap-8 text-[1.125rem] leading-7 lg:grid-cols-2">
            <div className="space-y-3">
              <h3 className="font-display text-lg font-semibold">K-Means</h3>
              <p>{t('สลับสองขั้นซ้ำ: (1) Assign จุดเข้ากลุ่มของ centroid ที่ใกล้สุด (2) Update ย้าย centroid ไปที่ค่าเฉลี่ยของกลุ่ม ทุกขั้นทำให้ WCSS ลดลงหรือคงที่ จึงลู่เข้าเสมอ แต่อาจเป็น local minimum ขึ้นกับจุดเริ่ม', 'Alternate (1) assign each point to its nearest centroid and (2) move each centroid to its cluster mean. Each step lowers or keeps WCSS, so it always converges, possibly to a local minimum depending on initialisation.')}</p>
              <p>{t('K-Means สมมติว่ากลุ่มเป็นก้อนกลม ขนาดใกล้กัน ลองชุด Moons แล้วดูว่าตัดผิดอย่างไร', 'K-Means assumes round, similar-sized clusters; try Moons to see how it cuts wrongly.')}</p>
            </div>
            <div className="space-y-3">
              <h3 className="font-display text-lg font-semibold">DBSCAN</h3>
              <p>{t('จุดที่มีเพื่อนบ้านในรัศมี ε อย่างน้อย MinPts จุดคือ core point คลัสเตอร์ขยายจาก core ไปยังจุดที่เชื่อมถึงกันได้ จุดที่อยู่ติด core แต่ตัวเองไม่ใช่ core คือ border ที่เหลือคือ noise', 'A point with at least MinPts neighbours within ε is a core point. Clusters grow from core points through reachable points. Points next to a core but not core themselves are border points; the rest are noise.')}</p>
              <p>{t('ไม่ต้องกำหนดจำนวนคลัสเตอร์ล่วงหน้า และหารูปร่างแปลก ๆ ได้ แต่ไวต่อค่า ε', 'No cluster count is needed and odd shapes are found, but it is sensitive to ε.')}</p>
            </div>
          </div>
        ) },
        { id: 'math', label: t('คณิตศาสตร์', 'Math'), content: (
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="flex flex-col gap-3">
              <Tex block>{'\\text{WCSS} = \\sum_{k=1}^{K}\\sum_{\\mathbf x\\in C_k}\\|\\mathbf x-\\boldsymbol\\mu_k\\|^2'}</Tex>
              <Tex block>{'c_i \\leftarrow \\arg\\min_k \\|\\mathbf x_i - \\boldsymbol\\mu_k\\|^2,\\qquad \\boldsymbol\\mu_k \\leftarrow \\frac{1}{|C_k|}\\sum_{i\\in C_k}\\mathbf x_i'}</Tex>
              {mode === 'kmeans' && <div className="font-mono text-[0.96875rem]">{kmS.centroids.map((c, j) => <div key={j} style={{ color: colorOf(j) }}>μ{j} = ({c.map((v) => fmt(v, 3)).join(', ')}) · n = {kmS.assign.filter((a) => a === j).length}</div>)}</div>}
            </div>
            <div className="flex flex-col gap-3">
              <Tex block>{'N_\\varepsilon(\\mathbf p) = \\{\\mathbf q : d(\\mathbf p,\\mathbf q)\\le\\varepsilon\\},\\quad \\mathbf p\\text{ is core} \\iff |N_\\varepsilon(\\mathbf p)| \\ge \\text{MinPts}'}</Tex>
              <Tex block>{'s(i) = \\frac{b(i)-a(i)}{\\max(a(i), b(i))}\\quad\\text{(silhouette)}'}</Tex>
            </div>
          </div>
        ) },
        { id: 'eval', label: t('การประเมินผล', 'Evaluation'), content: mode === 'kmeans' ? (
          <div className="grid gap-8 lg:grid-cols-2">
            <div><h3 className="label-xs mb-2">{t('Elbow: WCSS หลังลู่เข้า เทียบกับ K', 'Elbow: converged WCSS vs K')}</h3>
              <LinePlot xTitle="K" yTitle="WCSS" markerX={km.k} series={[{ name: 'WCSS', x: elbow.map((e) => e.k), y: elbow.map((e) => e.inertia), color: '#00D9FF' }]} /></div>
            <div><h3 className="label-xs mb-2">{t('WCSS ในแต่ละรอบ', 'WCSS by iteration')}</h3>
              <LinePlot xTitle="iteration" yTitle="WCSS" series={[{ name: 'WCSS', x: kmStates.filter((s) => s.inertia != null).map((_, i) => i), y: kmStates.filter((s) => s.inertia != null).map((s) => s.inertia!), color: '#F59E0B' }]} /></div>
          </div>
        ) : (
          <div className="grid gap-8 lg:grid-cols-2">
            <div><h3 className="label-xs mb-2">{t(`ระยะถึงเพื่อนบ้านลำดับที่ ${db.minPts} (เรียงมากไปน้อย) — "ข้อศอก" ของกราฟคือ ε ที่เหมาะ`, `Distance to the ${db.minPts}-th neighbour (sorted) — the knee suggests a good ε`)}</h3>
              <LinePlot xTitle={t('จุด', 'point')} yTitle="k-distance" series={[{ name: 'k-distance', x: kdist.map((_, i) => i), y: kdist, color: '#00D9FF' }, { name: 'ε', x: [0, kdist.length], y: [db.eps, db.eps], color: '#F59E0B', dash: 'dash' }]} /></div>
            <Metrics items={[[t('คลัสเตอร์', 'Clusters'), String(dbS.clusters)], ['Noise', String(nNoise)], ['Silhouette', sil == null ? '—' : fmt(sil, 3)], [t('เหตุการณ์', 'Events'), String(events.length)]]} />
          </div>
        ) },
        { id: 'ex', label: t('แบบฝึกหัด', 'Exercises'), content: (
          <Quiz id={`cluster-${mode}`} items={mode === 'kmeans' ? [
            { q: t('K-Means รับประกันคำตอบที่ดีที่สุดหรือไม่', 'Does K-Means guarantee the global optimum?'), options: [t('ใช่', 'Yes'), t('ไม่ ขึ้นกับจุดเริ่ม', 'No, it depends on initialisation')], answer: 1, why: t('ลองใช้การสุ่มธรรมดาและเปลี่ยน Init seed หลายค่า', 'Try random init with several seeds.') },
            { q: t('กราฟ elbow ใช้ทำอะไร', 'What is the elbow plot for?'), options: [t('เลือก K', 'Choosing K'), t('เลือก learning rate', 'Choosing the learning rate')], answer: 0, why: t('WCSS ลดลงเสมอเมื่อ K เพิ่ม จุดที่ลดลงช้าลงชัดเจนคือ K ที่สมเหตุสมผล', 'WCSS always falls as K grows; where the drop flattens is a sensible K.') },
          ] : [
            { q: t('ถ้าเพิ่ม ε มากเกินไป จะเกิดอะไร', 'If ε is too large…'), options: [t('ทุกจุดกลายเป็น noise', 'everything becomes noise'), t('คลัสเตอร์รวมกันเป็นก้อนเดียว', 'clusters merge into one')], answer: 1, why: t('ทุกจุดมีเพื่อนบ้านมากพอ ความหนาแน่นจึงเชื่อมถึงกันหมด', 'Every point has enough neighbours, so density connects everything.') },
            { q: t('border point คือ', 'A border point is…'), options: [t('จุดที่อยู่ในรัศมีของ core แต่ตัวเองมีเพื่อนบ้านไม่ถึง MinPts', 'within ε of a core point but with fewer than MinPts neighbours itself'), t('จุดที่ไกลจากทุกคลัสเตอร์', 'far from every cluster')], answer: 0, why: t('มันถูกนับเข้าคลัสเตอร์ แต่ไม่ขยายคลัสเตอร์ต่อ', 'It joins a cluster but does not expand it.') },
          ]} />
        ) },
      ]}
    />
  );
}
