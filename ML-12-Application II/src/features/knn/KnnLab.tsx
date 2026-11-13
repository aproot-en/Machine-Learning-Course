import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import { Stage, StageToolbar, useCameraState } from '../../engine3d/Stage';
import { KnnScene, R } from './KnnScene';
import { DATASETS, datasetInfo, fitScaler, applyScaler, makeDataset, trainTestSplit, type DatasetId } from '../../data/datasets';
import { accuracyByK, knnPredict, type KnnOptions, type Metric } from '../../ml/knn';
import { classificationReport } from '../../ml/metrics';
import { useSettings, useT } from '../../i18n/i18n';
import { FloatingPanel, Readout, Section, Segmented, Select, Slider, Toggle, pct, Button, Icon } from '../../components/ui';
import { PlaybackBar } from '../../components/PlaybackBar';
import { type ModelExperiment } from '../../engine3d/experiment';
import { ExperimentLab } from '../../components/ExperimentLab';
import { CLASS_COLORS } from '../../engine3d/palette';
import { KnnLessons, StageCaption, STAGES } from './knnContent';

export default function KnnLab() {
  const t = useT();
  const level = useSettings((s) => s.level);
  const [ds, setDs] = useState({ id: 'blobs3d' as DatasetId, n: 180, classes: 3, noise: 0.25, seed: 7, testRatio: 0.25, normalize: true });
  const info = datasetInfo(ds.id);
  const [axes, setAxes] = useState<number[]>([0, 1, 2]);
  const [view, setView] = useState<'3d' | '2d'>('3d');
  const [model, setModel] = useState<KnnOptions>({ k: 5, metric: 'euclidean', p: 3, weighting: 'uniform' });
  const [query, setQuery] = useState<number[]>([0.4, 0.3, -0.5]);
  const [mode, setMode] = useState<'continuous' | 'step'>('step');
  const [stage, setStage] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [showBoundary, setShowBoundary] = useState(false);
  const [showTest, setShowTest] = useState(false);
  const [showLabels, setShowLabels] = useState(true);
  const [edits, setEdits] = useState<Record<number, number[]>>({});
  const [selected, setSelected] = useState<number | null>(null);
  const [experimentSample, setExperimentSample] = useState(0);
  const [dragging, setDragging] = useState(false);
  const [cam, setCam] = useCameraState();
  const box = useRef<HTMLDivElement>(null);

  // ---------- data pipeline (all derived from the same arrays the model uses) ----------
  const raw = useMemo(() => makeDataset(ds.id, ds), [ds]);
  const nF = raw.X[0].length;
  const dims: 2 | 3 = view === '2d' || nF < 3 ? 2 : 3;
  const feats = useMemo(() => {
    const a = axes.filter((f) => f < nF);
    const base = a.length >= dims ? a : [0, 1, 2].slice(0, nF);
    return base.slice(0, dims);
  }, [axes, nF, dims]);
  const split = useMemo(() => trainTestSplit(raw.y, ds.testRatio, ds.seed), [raw, ds.testRatio, ds.seed]);
  const prepared = useMemo(() => {
    const sel = raw.X.map((r) => feats.map((f) => r[f]));
    const scaler = ds.normalize ? fitScaler(split.train.map((i) => sel[i])) : null;
    const X = sel.map((r) => (scaler ? applyScaler(scaler, r) : r));
    const maxAbs = Math.max(1e-6, ...X.flat().map(Math.abs));
    return { X, scale: (R * 0.92) / maxAbs, scaler };
  }, [raw, feats, split, ds.normalize]);
  useEffect(() => { setEdits({}); setSelected(null); }, [prepared]);
  const Xtr = useMemo(() => split.train.map((i, k) => edits[k] ?? prepared.X[i]), [split, prepared, edits]);
  const ytr = useMemo(() => split.train.map((i) => raw.y[i]), [split, raw]);
  const Xte = useMemo(() => split.test.map((i) => prepared.X[i]), [split, prepared]);
  const yte = useMemo(() => split.test.map((i) => raw.y[i]), [split, raw]);
  const q = useMemo(() => query.slice(0, dims).concat(Array(Math.max(0, dims - query.length)).fill(0)), [query, dims]);
  const lim = R / prepared.scale;

  const result = useMemo(() => knnPredict(Xtr, ytr, raw.nClasses, q, model), [Xtr, ytr, raw.nClasses, q, model]);
  const testPred = useMemo(() => Xte.map((x) => knnPredict(Xtr, ytr, raw.nClasses, x, model).predicted), [Xte, Xtr, ytr, raw.nClasses, model]);
  const report = useMemo(() => classificationReport(yte, testPred, raw.nClasses), [yte, testPred, raw.nClasses]);
  const curve = useMemo(() => accuracyByK(Xtr, ytr, Xte, yte, raw.nClasses, model, 25), [Xtr, ytr, Xte, yte, raw.nClasses, model.metric, model.p, model.weighting]); // eslint-disable-line react-hooks/exhaustive-deps
  const boundary = useMemo(() => {
    if (!showBoundary) return null;
    const G = dims === 2 ? 64 : 14;
    const n = dims === 2 ? G * G : G * G * G;
    const classes = new Uint8Array(n), conf = new Float32Array(n);
    const step = (2 * R) / G / prepared.scale, lo = -R / prepared.scale;
    for (let k = 0; k < n; k++) {
      const pt = dims === 2
        ? [lo + step * ((k % G) + 0.5), lo + step * (Math.floor(k / G) + 0.5)]
        : [lo + step * (Math.floor(k / (G * G)) + 0.5), lo + step * (Math.floor(k / G) % G + 0.5), lo + step * ((k % G) + 0.5)];
      const r = knnPredict(Xtr, ytr, raw.nClasses, pt, model);
      classes[k] = r.predicted;
      conf[k] = r.votes[r.predicted] / r.votes.reduce((a, b) => a + b, 0);
    }
    return { grid: G, classes, conf };
  }, [showBoundary, dims, prepared.scale, Xtr, ytr, raw.nClasses, model]);

  // ---------- playback ----------
  const effStage = mode === 'continuous' ? 5 : stage;
  useEffect(() => {
    if (!playing) return;
    if (mode === 'step') {
      if (stage >= 5) { setPlaying(false); return; }
      const id = setTimeout(() => setStage((s) => Math.min(5, s + 1)), 1400 / speed);
      return () => clearTimeout(id);
    }
    let raf = 0, t0 = performance.now(), last = t0;
    const tick = (now: number) => {
      const dt = (now - last) / 1000; last = now;
      const a = ((now - t0) / 1000) * 0.25 * speed;
      const r = lim * 0.75;
      setQuery([r * Math.sin(a * 1.3), r * 0.7 * Math.sin(a * 0.9 + 1), r * Math.cos(a)]);
      if (dt < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, mode, stage, speed, lim]);
  const playback = {
    playing,
    onToggle: () => { if (mode === 'step' && stage >= 5 && !playing) setStage(0); setPlaying((p) => !p); },
    onNext: () => { setMode('step'); setStage((s) => Math.min(5, s + (mode === 'continuous' ? 0 : 1))); },
    onPrev: () => setStage((s) => Math.max(0, s - 1)),
    canPrev: mode === 'step' && stage > 0,
    canNext: mode === 'continuous' || stage < 5,
    onReset: () => { setPlaying(false); setStage(0); },
    speed, onSpeed: setSpeed,
  };

  const setQ = useCallback((v: number[]) => setQuery((old) => (v.length === 2 ? [v[0], v[1], old[2] ?? 0] : v)), []);
  const cls = raw.classNames;
  const axisLabels = feats.map((f) => raw.featureNames[f] + (ds.normalize ? ' (z)' : ''));
  const votesTotal = result.votes.reduce((a, b) => a + b, 0);

  const experiment: ModelExperiment = {
    model: 'KNN', source: `${t(info.name)} · train ${Xtr.length} / test ${Xte.length}`,
    output: `${t('ผลทำนาย', 'Prediction')}: ${cls[result.predicted]} · ${t('คะแนนโหวต', 'Vote share')} ${pct(result.votes[result.predicted] / votesTotal)}`,
    stages: [
      { name: 'Query x', values: q, labels: axisLabels },
      { name: 'Training memory', kind: 'points', points: Xtr, values: Xtr.map(() => 1), classes: ytr, highlight: result.neighbors.map((n) => n.index) },
      { name: `${model.metric} distances`, values: result.ranked.map((n) => n.distance), highlight: result.neighbors.map((_, i) => i), labels: result.ranked.map((n) => `train[${n.index}]`) },
      { name: `K=${result.neighbors.length} neighbors`, kind: 'points', points: result.neighbors.map((n) => Xtr[n.index]), values: result.neighbors.map((n) => n.weight), classes: result.neighbors.map((n) => n.label) },
      { name: 'Class votes', values: result.votes, labels: cls, highlight: [result.predicted] },
      { name: 'Output distribution', values: result.votes.map((v) => v / votesTotal), labels: cls, highlight: [result.predicted] },
    ],
    metrics: [{ label: 'Test accuracy', value: pct(report.accuracy) }, { label: 'Precision', value: pct(report.precision) }, { label: 'Recall', value: pct(report.recall) }, { label: 'F1', value: pct(report.f1) }],
    history: [{ label: 'Train accuracy vs K (index 0 = K1)', values: curve.map((point) => point.train) }, { label: 'Test accuracy vs K (index 0 = K1)', values: curve.map((point) => point.test), color: '#F59E0B' }],
    confusion: report.confusion,
    sample: { index: experimentSample, count: Xte.length, onChange: (index) => { setExperimentSample(index); setQuery([...Xte[index], ...query.slice(dims)]); setSelected(null); } },
    note: t('เปลี่ยนจุดคำถามหรือ K เพื่อทดลองใหม่ ผลโหวตและชุดทดสอบคำนวณใหม่ทันที', 'Change the query or K to run another experiment; votes and test results update immediately'),
  };
  const nativeStage = (<>
      <div ref={box} className="relative h-[calc(100svh-190px)] min-h-[560px] overflow-hidden border-y border-line bg-bg">
        <div className="absolute inset-0">
          <Stage cam={cam} distance={dims === 2 ? 15 : 19} lockRotation={view === '2d'} controlsEnabled={!dragging} onPointerMissed={() => setSelected(null)}>
            <KnnScene dims={dims} scale={prepared.scale} train={Xtr} ytr={ytr} test={Xte} yte={yte} testWrong={testPred.map((p, i) => p !== yte[i])}
              showTest={showTest} query={q} result={result} stage={effStage} metric={model.metric} p={model.p} boundary={boundary}
              showLabels={showLabels} axisLabels={axisLabels} classNames={cls} selected={selected} onSelect={setSelected}
              onDragQuery={setQ} onDragPoint={(i, v) => setEdits((e) => ({ ...e, [i]: v }))} setDragging={setDragging} allowPointDrag />
          </Stage>
        </div>

        <div className="pointer-events-none absolute inset-0 z-30 flex flex-col p-3 sm:p-4">
          <div className="flex items-start justify-between gap-3">
            <div className="flex flex-col gap-2">
              <Readout title={`KNN · ${t(STAGES[effStage].name)}`} rows={[
                [t('ชุดข้อมูล', 'Dataset'), `${t(info.name)}`],
                ['train / test', `${Xtr.length} / ${Xte.length}`],
                ['K · ' + t('ระยะ', 'metric'), `${model.k} · ${model.metric}${model.metric === 'minkowski' ? ` p=${model.p}` : ''}`],
                [t('จุดคำถาม', 'Query'), `(${q.map((v) => v.toFixed(2)).join(', ')})`],
                ...(effStage >= 4 ? result.votes.map((v, c) => [`${t('โหวต', 'votes')} ${cls[c]}`, `${model.weighting === 'distance' ? v.toFixed(2) : v}  ${'▮'.repeat(Math.round((v / Math.max(votesTotal, 1e-9)) * 10))}`, CLASS_COLORS[c]] as [string, string, string]) : []),
                ...(effStage >= 5 ? [[t('ผลทำนาย', 'Prediction'), cls[result.predicted] + (result.tie ? t(' (เสมอ)', ' (tie)') : ''), CLASS_COLORS[result.predicted]] as [string, string, string]] : []),
                [t('ความแม่นยำ test', 'Test accuracy'), pct(report.accuracy)],
              ]} />
              <StageToolbar cam={cam} setCam={setCam} container={box} allowPresets={view === '3d'} />
            </div>
            <FloatingPanel title={t('ตัวควบคุม', 'Controls')} className="hidden w-[300px] md:flex" >
              {renderControls()}
            </FloatingPanel>
          </div>
          <div className="mt-auto flex flex-col items-center gap-2">
            <StageCaption stage={effStage} k={model.k} />
            <PlaybackBar {...playback} status={mode === 'step' ? `${t('ขั้น', 'step')} ${stage}/5` : t('ต่อเนื่อง', 'continuous')}>
              <div className="mx-1"><Segmented value={mode} onChange={(m) => { setMode(m); setPlaying(false); if (m === 'step') setStage(0); }}
                options={[{ value: 'step', label: t('ทีละขั้น', 'Step') }, { value: 'continuous', label: t('ต่อเนื่อง', 'Continuous') }]} /></div>
            </PlaybackBar>
          </div>
        </div>
      </div>

      <div className="border-b border-line bg-surface/40 p-4 md:hidden">
        {renderControls()}
      </div>

  </>);

  return (
    <div>
      <header className="mx-auto flex max-w-[1600px] flex-col gap-2 px-4 pb-4 pt-6 sm:px-6 md:flex-row md:items-end md:justify-between">
        <div>
          <nav className="text-[1rem] text-muted"><Link to="/algorithms" className="hover:text-ink">{t('อัลกอริทึม', 'Algorithms')}</Link> / KNN</nav>
          <h1 className="mt-1 font-display text-3xl font-semibold">K-Nearest Neighbors</h1>
          <p className="mt-1 max-w-[70ch] text-[1.125rem] text-muted">{t('ส่งจุดคำถามผ่านข้อมูล train ระยะทาง เพื่อนบ้าน และคะแนนโหวตในฉาก 3D เปลี่ยนตัวอย่างหรือ K เพื่อดูผลทำนายและความแม่นยำ หรือเปิด “สำรวจอัลกอริทึม” เพื่อลากจุดโดยตรง', 'Follow a query through training memory, distances, neighbors and votes in 3D. Change the sample or K to inspect predictions and accuracy, or open Algorithm detail to drag points directly.')}</p>
        </div>
        <div className="flex gap-2"><span className="chip">{t('การเรียนรู้แบบมีผู้สอน', 'Supervised')}</span><span className="chip">{t('พื้นฐาน', 'Beginner')}</span></div>
      </header>


      <ExperimentLab data={experiment} controls={renderControls()} playback={playback} detail={nativeStage} />

      <section className="mx-auto max-w-[1600px] px-4 py-10 sm:px-6">
        <KnnLessons model={model} result={result} Xtr={Xtr} ytr={ytr} q={q} classNames={cls} report={report} curve={curve}
          featureNames={axisLabels} onPlaceQuery={(v) => { setQuery(v); setMode('step'); setStage(3); setPlaying(false); }} onReveal={() => setStage(5)} lim={lim} dims={dims} nClasses={raw.nClasses} level={level}
          selected={selected} />
      </section>
    </div>
  );

  function renderControls() {
    return (
      <>
        <Section title={t('ชุดข้อมูล', 'Dataset')}>
          <Select label={t('เลือกชุดข้อมูล', 'Dataset')} value={ds.id} onChange={(id) => { setDs({ ...ds, id }); setAxes([0, 1, 2]); }}
            options={DATASETS.map((d) => ({ value: d.id, label: t(d.name) }))} />
          {info.synthetic && <Slider label={t('จำนวนตัวอย่าง', 'Samples')} value={ds.n} min={30} max={400} step={10} onChange={(n) => setDs({ ...ds, n })} />}
          {info.fixedClasses == null && <Slider label={t('จำนวนคลาส', 'Classes')} value={ds.classes} min={2} max={6} onChange={(classes) => setDs({ ...ds, classes })} />}
          {info.synthetic && <Slider label={t('สัญญาณรบกวน (noise)', 'Noise')} value={ds.noise} min={0} max={1} step={0.05} format={(v) => v.toFixed(2)} onChange={(noise) => setDs({ ...ds, noise })} />}
          <Slider label={t('สัดส่วน test', 'Test ratio')} value={ds.testRatio} min={0.1} max={0.5} step={0.05} format={pct} onChange={(testRatio) => setDs({ ...ds, testRatio })} />
          <div className="flex items-end gap-2">
            <div className="flex-1"><Slider label="Random seed" value={ds.seed} min={1} max={99} onChange={(seed) => setDs({ ...ds, seed })} /></div>
            <Button size="sm" onClick={() => setDs({ ...ds, seed: (ds.seed % 99) + 1 })}>{t('สุ่มใหม่', 'Reseed')}</Button>
          </div>
          <Toggle label={t('ปรับมาตรฐาน (z-score จาก train)', 'Standardise (z-score fit on train)')} checked={ds.normalize} onChange={(normalize) => setDs({ ...ds, normalize })} />
          {nF > dims && (
            <div className="grid grid-cols-3 gap-2">
              {Array.from({ length: dims }, (_, a) => (
                <Select key={a} label={['x', 'y', 'z'][a]} value={feats[a]} onChange={(f) => { const n = [...feats]; n[a] = f; setAxes(n); }}
                  options={raw.featureNames.map((f, i) => ({ value: i, label: f.replace(' (cm)', ''), disabled: feats.includes(i) && feats[a] !== i }))} />
              ))}
            </div>
          )}
          {nF > dims && <p className="text-[0.9375rem] text-muted">{t('KNN ใช้เฉพาะฟีเจอร์ที่แสดงบนแกน', 'KNN uses exactly the features shown on the axes')}</p>}
          {Object.keys(edits).length > 0 && <Button size="sm" onClick={() => setEdits({})}><Icon name="reset" />{t(`คืนค่าจุดที่ลาก (${Object.keys(edits).length})`, `Undo dragged points (${Object.keys(edits).length})`)}</Button>}
        </Section>
        <Section title={t('โมเดล', 'Model')}>
          <Slider label="K" value={model.k} min={1} max={Math.min(25, Xtr.length)} onChange={(k) => setModel({ ...model, k })} />
          <Select<Metric> label={t('วิธีวัดระยะ', 'Distance metric')} value={model.metric} onChange={(metric) => setModel({ ...model, metric })}
            options={[{ value: 'euclidean', label: 'Euclidean (L2)' }, { value: 'manhattan', label: 'Manhattan (L1)' }, { value: 'minkowski', label: 'Minkowski (Lp)' }]} />
          {model.metric === 'minkowski' && <Slider label="Minkowski p" value={model.p} min={1} max={8} step={0.5} onChange={(p) => setModel({ ...model, p })} />}
          {level >= 2 && <Segmented label={t('น้ำหนักโหวต', 'Vote weighting')} value={model.weighting} onChange={(weighting) => setModel({ ...model, weighting })}
            options={[{ value: 'uniform', label: t('เท่ากัน', 'Uniform') }, { value: 'distance', label: '1 / d' }]} />}
        </Section>
        <Section title={t('จุดคำถาม', 'Query point')}>
          {q.map((v, a) => (
            <Slider key={a} label={axisLabels[a]} value={Number(v.toFixed(2))} min={-Number(lim.toFixed(2))} max={Number(lim.toFixed(2))} step={0.01} format={(x) => x.toFixed(2)}
              onChange={(x) => { const n = [...q]; n[a] = x; setQ(n); }} />
          ))}
        </Section>
        <Section title={t('การแสดงผล', 'View')}>
          <Segmented value={view} onChange={(v) => { setView(v); setCam({ ...cam, preset: v === '2d' ? 'front' : 'iso', ortho: v === '2d', nonce: cam.nonce + 1 }); }}
            options={[{ value: '3d', label: '3D' }, { value: '2d', label: '2D' }]} />
          <Toggle label={t('พื้นที่ตัดสินใจ (decision region)', 'Decision region')} checked={showBoundary} onChange={setShowBoundary} />
          <Toggle label={t('แสดงจุด test (ลูกบาศก์)', 'Show test points (cubes)')} checked={showTest} onChange={setShowTest} />
          <Toggle label={t('ป้ายระยะทาง', 'Distance labels')} checked={showLabels} onChange={setShowLabels} />
          {showBoundary && <p className="text-[0.9375rem] text-muted">{dims === 2 ? t('คำนวณ KNN จริงที่ 64×64 จุด', 'Real KNN predictions on a 64×64 grid') : t('คำนวณ KNN จริงที่ 14×14×14 จุด ขนาดกล่อง = สัดส่วนคะแนนโหวต', 'Real KNN predictions at 14×14×14 points; box size = vote share')}</p>}
          {showTest && <p className="text-[0.9375rem] text-muted">{t('กรอบสีแดง = จุด test ที่ทำนายผิด', 'Red frame = misclassified test point')}</p>}
        </Section>
      </>
    );
  }
}

