import { useCallback, useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { Stage, StageToolbar, useCameraState } from '../../engine3d/Stage';
import { NetworkScene, type FlowState, type SampleTrace, type Selection, type ViewSettings } from './NetworkScene';
import { BlueprintScene } from './BlueprintScene';
import { BuilderPanel } from './BuilderPanel';
import { Inspector } from './Inspector';
import { DecisionMap, MlpExercises, MlpMath, ModelSummary, architectureSvg } from './mlpParts';
import { outputFor, presetLayers, useArch, useModelArch, type PresetName } from './archStore';
import { applyScaler, DATASETS, datasetInfo, fitScaler, makeDataset, trainTestSplit, type DatasetId } from '../../data/datasets';
import { Network } from '../../ml/nn/network';
import { Trainer, type PreparedData } from '../../ml/nn/trainer';
import { validateArchitecture } from '../../ml/nn/validate';
import { parseBuilderConfig } from '../../ml/nn/config';
import { type Loss, type OptimizerName } from '../../ml/nn/types';
import { classificationReport } from '../../ml/metrics';
import { argClass } from '../../ml/nn/network';
import { Rng } from '../../lib/random';
import { useSettings, useT } from '../../i18n/i18n';
import { PlaybackBar } from '../../components/PlaybackBar';
import { type ModelExperiment } from '../../engine3d/experiment';
import { ExperimentLab } from '../../components/ExperimentLab';
import { Button, FloatingPanel, NumberField, Readout, Segmented, Select, Slider, Tabs, Toggle, fmt, pct } from '../../components/ui';
import { ConfusionMatrix, LinePlot, download } from '../../components/common';
import { CLASS_COLORS } from '../../engine3d/palette';

const reduceMotion = typeof window !== 'undefined' && !!window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
const SAVE_KEY = 'mlvl-mlp-saves';
const readSaves = (): Record<string, unknown> => { try { const stored = JSON.parse(localStorage.getItem(SAVE_KEY) ?? '{}'); return stored && typeof stored === 'object' && !Array.isArray(stored) ? stored : {}; } catch { return {}; } };

export default function NetworkBuilder({ mode = 'neural' }: { mode?: 'neural' | 'model' }) {
  const t = useT();
  const level = useSettings((s) => s.level);
  const archStore = mode === 'model' ? useModelArch : useArch;
  const { layers, selectedId, select: selectLayer, set: setLayers } = archStore();
  const [data, setData] = useState({ id: 'circles' as DatasetId, n: 300, classes: 3, noise: 0.15, seed: 3, testRatio: 0.2, valRatio: 0.15, standardize: true });
  const [tc, setTc] = useState({ optimizer: 'adam' as OptimizerName, lr: 0.02, batchSize: 16, loss: 'bce' as Loss, momentum: 0.9, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, shuffle: true, seed: 1, earlyStopping: false, patience: 20, maxEpochs: 500 });
  const [view, setView] = useState<ViewSettings>({ spacing: 4.2, neuronSize: 0.26, showConnections: true, maxEdges: 4000, opacity: 1, weightColors: true, activationColors: true, labels: true, isolate: false });
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [tick, setTick] = useState(0);
  const [resetNonce, setResetNonce] = useState(0);
  const [viewEpoch, setViewEpoch] = useState<number | null>(null);
  const [sel, setSel] = useState<Selection | null>(null);
  const [sampleIdx, setSampleIdx] = useState(0);
  const [custom, setCustom] = useState<number[] | null>(null);
  const [customTarget, setCustomTarget] = useState(0);
  const [trainMode, setTrainMode] = useState(false);
  const [edgeStats, setEdgeStats] = useState<[number, number]>([0, 0]);
  const [rightTab, setRightTab] = useState<'data' | 'train' | 'view' | 'inspect'>('train');
  const [notice, setNotice] = useState<string | null>(null);
  const [saves, setSaves] = useState<string[]>(() => Object.keys(readSaves()));
  const flow = useRef<FlowState>({ mode: reduceMotion ? 'static' : 'forward', pos: 0, loop: !reduceMotion, speed: 1 });
  const [, forceFlow] = useState(0);
  const [cam, setCam] = useCameraState({ preset: 'iso' });
  const box = useRef<HTMLDivElement>(null);
  const trainerRef = useRef<Trainer | null>(null);

  // ---------- data ----------
  const info = datasetInfo(data.id);
  const raw = useMemo(() => makeDataset(data.id, data), [data]);
  const nF = raw.X[0].length;
  const prepared = useMemo(() => {
    const sp = trainTestSplit(raw.y, data.testRatio, data.seed);
    const inner = trainTestSplit(sp.train.map((i) => raw.y[i]), data.valRatio, data.seed + 1);
    const trIdx = inner.train.map((k) => sp.train[k]);
    const vaIdx = inner.test.map((k) => sp.train[k]);
    const scaler = data.standardize ? fitScaler(trIdx.map((i) => raw.X[i])) : null;
    const tf = (r: number[]) => (scaler ? applyScaler(scaler, r) : r.slice());
    const pd: PreparedData = {
      Xtr: trIdx.map((i) => tf(raw.X[i])), ytr: trIdx.map((i) => raw.y[i]),
      Xval: vaIdx.map((i) => tf(raw.X[i])), yval: vaIdx.map((i) => raw.y[i]),
      Xte: sp.test.map((i) => tf(raw.X[i])), yte: sp.test.map((i) => raw.y[i]), nClasses: raw.nClasses,
    };
    return { pd, tf, trIdx, vaIdx, teIdx: sp.test };
  }, [raw, data.testRatio, data.valRatio, data.seed, data.standardize]);

  // keep Input/Output (and the loss) consistent with the dataset when it changes
  useEffect(() => {
    setCustom(null); setCustomTarget(0); setSampleIdx(0);
    let loss = tc.loss;
    if (raw.nClasses > 2 && loss === 'bce') loss = 'cce';
    if (loss !== tc.loss) setTc((c) => ({ ...c, loss }));
    const cur = archStore.getState().layers;
    const out = outputFor(raw.nClasses, loss);
    const inL = cur[0], outL = cur[cur.length - 1];
    if (inL?.units !== nF || outL?.units !== out.units || outL?.activation !== out.activation) {
      setLayers(cur.map((l, i) => (i === 0 ? { ...l, units: nF } : i === cur.length - 1 && l.kind === 'output' ? { ...l, ...out } : l)), false);
      setNotice(t(`ปรับ Input = ${nF} ฟีเจอร์ และ Output = ${out.units} (${out.activation}) ให้ตรงกับชุดข้อมูล`, `Set Input = ${nF} features and Output = ${out.units} (${out.activation}) to match the dataset`));
    }
  }, [raw.nClasses, nF, mode]); // eslint-disable-line react-hooks/exhaustive-deps

  const validation = useMemo(() => validateArchitecture(layers, { nFeatures: nF, nClasses: raw.nClasses }, tc.loss), [layers, nF, raw.nClasses, tc.loss]);
  const canTrain = mode === 'neural' && validation.runnable;

  // ---------- trainer lifecycle ----------
  const optCfg = useMemo(() => ({ name: tc.optimizer, lr: tc.lr, momentum: tc.momentum, beta1: tc.beta1, beta2: tc.beta2, epsilon: tc.epsilon }), [tc.optimizer, tc.lr, tc.momentum, tc.beta1, tc.beta2, tc.epsilon]);
  const trainer = useMemo(() => canTrain
      ? new Trainer(layers, prepared.pd, { optimizer: { ...optCfg }, batchSize: tc.batchSize, loss: tc.loss, shuffle: tc.shuffle, seed: tc.seed, earlyStopping: tc.earlyStopping, patience: tc.patience })
      : null, [layers, prepared, tc.loss, tc.seed, tc.shuffle, canTrain, resetNonce]); // eslint-disable-line react-hooks/exhaustive-deps
  trainerRef.current = trainer;
  useEffect(() => {
    setRunning(false); setViewEpoch(null); setSel(null); setTick((x) => x + 1);
  }, [trainer]);
  // hyper-parameters that can change mid-training without re-initialising
  useEffect(() => {
    const tr = trainerRef.current;
    if (!tr) return;
    tr.opt.cfg = { ...optCfg };
    Object.assign(tr.cfg, { optimizer: tr.opt.cfg, batchSize: tc.batchSize, earlyStopping: tc.earlyStopping, patience: tc.patience });
  }, [optCfg, tc.batchSize, tc.earlyStopping, tc.patience, tick]);

  useEffect(() => {
    if (!running) return;
    let raf = 0, lastUi = 0;
    const loop = (now: number) => {
      const tr = trainerRef.current;
      if (!tr) { setRunning(false); return; }
      if (tr.stopped || tr.epoch >= tc.maxEpochs) { setRunning(false); return; }
      const budget = Math.min(14, 2.5 * speed);
      const t0 = performance.now();
      do { tr.trainBatch(); } while (performance.now() - t0 < budget && !tr.stopped && tr.epoch < tc.maxEpochs);
      if (tr.stopped || tr.epoch >= tc.maxEpochs) { setRunning(false); setTick((x) => x + 1); return; }
      if (now - lastUi > 120) { lastUi = now; setTick((x) => x + 1); }
      raf = requestAnimationFrame(loop);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [running, speed, tc.maxEpochs]);

  const liveEpoch = trainer?.epoch ?? 0;
  const shownEpoch = viewEpoch == null ? liveEpoch : trainer?.snapshotAt(viewEpoch)?.epoch ?? liveEpoch;
  const viewingPast = viewEpoch != null && shownEpoch !== liveEpoch;
  const displayNet = useMemo(() => {
    if (!trainer) return null;
    if (!viewingPast) return trainer.net;
    const snap = trainer.snapshotAt(shownEpoch);
    if (!snap) return trainer.net;
    const n = new Network(layers, tc.seed);
    n.params = snap.params.map((p) => (p ? { ...p, W: p.W.slice(), b: p.b.slice() } : null));
    return n;
  }, [trainer, viewingPast, shownEpoch, layers, tc.seed]);


  // ---------- traced sample ----------
  const pd = prepared.pd;
  const sample: SampleTrace | null = useMemo(() => {
    if (!displayNet) return null;
    const pool = pd.Xte.length ? pd.Xte : pd.Xtr;
    const labels = pd.Xte.length ? pd.yte : pd.ytr;
    const x = custom ? prepared.tf(Array.from({ length: nF }, (_, i) => custom[i] ?? 0)) : pool[Math.min(sampleIdx, pool.length - 1)];
    const label = custom ? Math.max(0, Math.min(raw.nClasses - 1, customTarget)) : labels[Math.min(sampleIdx, labels.length - 1)];
    const units = displayNet.widths[displayNet.widths.length - 1];
    const target = units === 1 ? [label] : Array.from({ length: units }, (_, k) => (k === label ? 1 : 0));
    const trace = displayNet.forward(x, trainMode, new Rng(sampleIdx * 17 + tick));
    return { x, label, target, trace, grads: displayNet.backward(trace, target, tc.loss), custom: !!custom };
  }, [displayNet, pd, custom, customTarget, sampleIdx, trainMode, tc.loss, tick, prepared]);

  const reports = useMemo(() => {
    if (!displayNet) return null;
    const pred = (X: number[][]) => X.map((x) => argClass(displayNet.output(x)));
    return {
      test: classificationReport(pd.yte, pred(pd.Xte), pd.nClasses),
      val: classificationReport(pd.yval, pred(pd.Xval), pd.nClasses),
      train: classificationReport(pd.ytr, pred(pd.Xtr), pd.nClasses),
    };
  }, [displayNet, pd, tick]); // eslint-disable-line react-hooks/exhaustive-deps

  // ---------- actions ----------
  const resumeFromView = () => { if (trainer && viewingPast) trainer.restore(shownEpoch); setViewEpoch(null); };
  const playback = {
    playing: running,
    onToggle: () => { if (!trainer || (!viewingPast && (trainer.stopped || trainer.epoch >= tc.maxEpochs))) return; resumeFromView(); setRunning((r) => !r); },
    onNext: () => {
      if (!trainer) return;
      if (viewingPast) { const n = trainer.snapshots.find((snapshot) => snapshot.epoch > shownEpoch)?.epoch ?? liveEpoch; setViewEpoch(n >= liveEpoch ? null : n); return; }
      if (trainer.stopped || trainer.epoch >= tc.maxEpochs) return;
      setRunning(false); trainer.trainBatch(); setTick((x) => x + 1);
    },
    onPrev: () => { if (!trainer) return; setRunning(false); setViewEpoch([...trainer.snapshots].reverse().find((snapshot) => snapshot.epoch < shownEpoch)?.epoch ?? 0); },
    canPrev: !!trainer && (shownEpoch) > 0,
    canNext: !!trainer && (viewingPast || (!trainer.stopped && trainer.epoch < tc.maxEpochs)),
    onReset: () => { setResetNonce((n) => n + 1); },
    speed, onSpeed: setSpeed,
  };
  const applyPreset = (p: PresetName) => { setLayers(presetLayers(p, nF, raw.nClasses, tc.loss)); setSel(null); };
  const setFlow = (mode: FlowState['mode']) => { flow.current.mode = mode; flow.current.pos = mode === 'backward' ? (displayNet?.widths.length ?? 1) - 1 : 0; forceFlow((x) => x + 1); };
  const onSelect = useCallback((s: Selection | null) => { setSel(s); if (s) setRightTab('inspect'); }, []);

  const config = () => ({ version: 1, app: 'ML Visual Lab', architecture: layers, dataset: data, training: tc });
  const importConfig = (text: string): string | null => {
    try {
      const c = parseBuilderConfig(text, mode);
      if (c.dataset) setData((d) => ({ ...d, ...c.dataset }));
      if (c.training) setTc((x) => ({ ...x, ...c.training }));
      setLayers(c.architecture);
      return null;
    } catch (error) { return `${(error as Error).message}. ` + t('อ่านไฟล์ JSON ไม่ได้', 'Could not parse the JSON file'); }
  };

  const hist = trainer?.history ?? [];
  const last = hist[hist.length - 1];
  const viewRec = viewingPast ? hist.find((h) => h.epoch === shownEpoch) : last;
  const nBatches = Math.ceil(pd.Xtr.length / Math.max(1, tc.batchSize));
  const status = validation.ok && !canTrain ? t('Blueprint · พร้อมส่งออก', 'Blueprint · ready to export')
    : !trainer ? t('สถาปัตยกรรมไม่ถูกต้อง', 'invalid architecture')
    : viewingPast ? t(`ดู epoch ${shownEpoch} จาก ${liveEpoch}`, `viewing epoch ${shownEpoch} of ${liveEpoch}`)
      : trainer.stopped === 'non-finite-gradient' ? t('หยุดฝึก: loss หรือ gradient ไม่เป็นจำนวนจำกัด กรุณาปรับอัตราการเรียนรู้หรือข้อมูล', 'Training stopped: non-finite loss or gradient. Adjust the learning rate or data.')
      : trainer.stopped ? t(`หยุดก่อนกำหนดที่ epoch ${liveEpoch}`, `early-stopped at epoch ${liveEpoch}`)
        : `epoch ${liveEpoch} · batch ${trainer.steps % nBatches}/${nBatches}`;
  const featureNames = raw.featureNames;

  const experiment: ModelExperiment = {
    model: 'MLP', source: `${t(info.name)} · train ${pd.Xtr.length} / val ${pd.Xval.length} / test ${pd.Xte.length}`,
    stages: sample ? layers.map((layer, i) => ({ name: `${layer.name}${layer.activation ? ` · ${layer.activation}` : ''}`, values: sample.trace.a[i], weights: displayNet?.params[i]?.W, highlight: i === layers.length - 1 ? [argClass(sample.trace.a[i])] : undefined })) : [{ name: 'Input', values: pd.Xte[0] ?? [] }, { name: 'Invalid architecture', values: [] }],
    output: sample ? `${t('ผลทำนาย', 'Prediction')}: ${raw.classNames[argClass(sample.trace.a[sample.trace.a.length - 1])]} · ${t('คำตอบจริง', 'Target')}: ${raw.classNames[sample.label]} · loss=${fmt(sample.grads.loss, 4)}` : t('ตรวจสถาปัตยกรรมก่อนเริ่มทดลอง', 'Check the architecture before running'),
    metrics: [{ label: 'Epoch', value: String(shownEpoch) }, { label: 'Train accuracy', value: reports ? pct(reports.train.accuracy) : '—' }, { label: 'Validation accuracy', value: reports && pd.Xval.length ? pct(reports.val.accuracy) : '—' }, { label: 'Test accuracy', value: reports ? pct(reports.test.accuracy) : '—' }],
    history: [{ label: 'Training loss', values: hist.filter((row) => row.epoch <= (shownEpoch)).map((row) => row.loss) }, { label: 'Validation loss', values: hist.filter((row) => row.epoch <= (shownEpoch)).map((row) => row.valLoss), color: '#F59E0B' }],
    confusion: reports?.test.confusion,
    sample: { index: sampleIdx, count: pd.Xte.length, onChange: (index) => { setCustom(null); setSampleIdx(index); } },
    note: `${tc.optimizer} · ${status}`,
  };
  const nativeStage = (<>
      <div ref={box} className="relative h-[calc(100svh-190px)] min-h-[600px] overflow-hidden border-y border-line bg-bg">
        <div className="absolute inset-0">
          {validation.ok && !canTrain ? (
            <Stage cam={cam} distance={Math.max(16, layers.length * 2.15)} preserveDrawingBuffer onPointerMissed={() => selectLayer(null)}>
              <BlueprintScene layers={layers} selectedId={selectedId} onSelect={selectLayer} />
            </Stage>
          ) : displayNet && canTrain ? (
            <Stage cam={cam} distance={Math.max(16, displayNet.widths.length * view.spacing * 1.15)} preserveDrawingBuffer onPointerMissed={() => setSel(null)}>
              <NetworkScene net={displayNet} layers={layers} sample={sample} flow={flow} view={view} selected={sel} onSelect={onSelect}
                classNames={raw.classNames} onEdgeStats={(d, tot) => setEdgeStats([d, tot])} />
            </Stage>
          ) : (
            <div className="grid h-full place-items-center px-6 text-center">
              <div className="max-w-md">
                <p className="font-display text-xl text-red">{t('ยังฝึกโมเดลนี้ไม่ได้', 'This model cannot run yet')}</p>
                <ul className="mt-3 flex flex-col gap-1 text-[1.0625rem] text-muted">{validation.errors.map((e, i) => <li key={i}>{t(e.msg)}</li>)}</ul>
              </div>
            </div>
          )}
        </div>

        <div className="pointer-events-none absolute inset-0 z-30 flex gap-3 p-3 sm:p-4">
          <div className="hidden w-[350px] shrink-0 flex-col gap-2 xl:flex">
            <FloatingPanel title={mode === 'model' ? t('สถาปัตยกรรมโมเดล', 'Model architecture') : t('สถาปัตยกรรม Neural Network', 'Neural Network architecture')} className="min-h-0">
              <BuilderPanel validation={validation} onPreset={applyPreset} store={archStore} universal={mode === 'model'} />
            </FloatingPanel>
          </div>
          <div className="flex min-w-0 flex-1 flex-col">
            <div className="flex flex-wrap items-start justify-between gap-2">
              <Readout {...(mode === 'model' ? {
                title: t('Blueprint โมเดล', 'Model blueprint'), rows: [
                  [t('จำนวน Layer', 'layers'), layers.length],
                  [t('องค์ประกอบ', 'components'), [...new Set(layers.map((l) => l.kind))].join(' · ')],
                  [t('พารามิเตอร์โดยประมาณ', 'estimated params'), validation.totalParams.toLocaleString()],
                  [t('สถานะ', 'status'), validation.ok ? t('พร้อมบันทึกและส่งออก', 'ready to save and export') : t('ต้องแก้โครงสร้าง', 'needs fixes')],
                ] as [string, ReactNode, string?][],
              } : {
                title: t('การฝึก', 'Training'), rows: [
                  [t('ชุดข้อมูล', 'dataset'), `${t(info.name)}`],
                  ['train / val / test', `${pd.Xtr.length} / ${pd.Xval.length} / ${pd.Xte.length}`],
                  ['epoch', viewingPast ? `${shownEpoch} / ${liveEpoch}` : liveEpoch],
                  ['loss (train / val)', viewRec ? `${fmt(viewRec.loss, 4)} / ${fmt(viewRec.valLoss, 4)}` : '—'],
                  ['acc (train / val)', viewRec ? `${pct(viewRec.acc)} / ${pct(viewRec.valAcc)}` : '—'],
                  ['acc (test)', reports ? pct(reports.test.accuracy) : '—', reports && reports.test.accuracy > 0.9 ? '#22C55E' : undefined],
                  [t('พารามิเตอร์', 'params'), validation.totalParams.toLocaleString()],
                  [t('เส้นที่แสดง', 'connections drawn'), edgeStats[1] ? `${edgeStats[0].toLocaleString()} (${pct(edgeStats[0] / edgeStats[1])})` : '—'],
                ] as [string, ReactNode, string?][],
              })} />
              <StageToolbar cam={cam} setCam={setCam} container={box} />
            </div>
            {canTrain && <div className="mt-2 flex flex-wrap items-center gap-1.5 self-start">
              <div className="glass pointer-events-auto flex items-center gap-1 p-1 text-[0.9375rem]">
                <span className="px-1.5 text-muted">{t('การไหลของสัญญาณ', 'Signal flow')}</span>
                <button type="button" onClick={() => setFlow('forward')} className={`rounded px-2 py-1 ${flow.current.mode === 'forward' ? 'bg-cyan/15 text-cyan' : 'text-muted hover:text-ink'}`}>Forward ▸</button>
                <button type="button" onClick={() => setFlow('backward')} className={`rounded px-2 py-1 ${flow.current.mode === 'backward' ? 'bg-purple/20 text-purple' : 'text-muted hover:text-ink'}`}>◂ Backward</button>
                <button type="button" onClick={() => setFlow('static')} className={`rounded px-2 py-1 ${flow.current.mode === 'static' ? 'bg-white/10 text-ink' : 'text-muted hover:text-ink'}`}>{t('หยุดภาพ', 'Freeze')}</button>
                <label className="flex items-center gap-1 px-1.5 text-muted"><input type="checkbox" checked={flow.current.loop} onChange={(e) => { flow.current.loop = e.target.checked; forceFlow((x) => x + 1); }} className="accent-cyan" />{t('วน', 'loop')}</label>
              </div>
            </div>}
            <div className="mt-auto flex flex-col items-center gap-2">
              {viewingPast && <div className="glass pointer-events-auto px-3 py-1.5 text-[0.96875rem] text-orange">{t('กำลังดูสถานะที่บันทึกไว้ กด ▶ เพื่อฝึกต่อจากจุดนี้ (epoch หลังจากนี้จะถูกแทนที่)', 'Viewing a recorded state. Press ▶ to continue training from here (later epochs will be replaced).')}</div>}
              {canTrain ? <PlaybackBar {...playback} status={status}>
                <button type="button" onClick={() => { if (!trainer) return; resumeFromView(); setRunning(false); trainer.trainEpoch(); setTick((x) => x + 1); }} disabled={!trainer}
                  className="rounded-lg px-2 py-1.5 text-[0.9375rem] text-muted hover:bg-white/5 hover:text-ink disabled:opacity-40">+1 epoch</button>
              </PlaybackBar> : validation.ok && <div className="glass pointer-events-auto px-4 py-2 text-[0.96875rem] text-cyan">{status} · {t('บันทึกหรือส่งออก JSON ได้จากแท็บสรุปโมเดล', 'Save or export JSON from Model summary')}</div>}
            </div>
          </div>
          {mode === 'neural' && <div className="hidden w-[340px] shrink-0 flex-col xl:flex">
            <FloatingPanel title={t('ตั้งค่า', 'Settings')} className="min-h-0">
              <div className="pb-2 pt-1"><Segmented value={rightTab} onChange={setRightTab} options={[
                { value: 'data', label: t('ข้อมูล', 'Data') }, { value: 'train', label: t('ฝึก', 'Train') }, { value: 'view', label: t('มุมมอง', 'View') }, { value: 'inspect', label: t('ตรวจ', 'Inspect') },
              ]} /></div>
              {renderRight()}
            </FloatingPanel>
          </div>}
        </div>
      </div>

      <div className="grid gap-4 border-b border-line bg-surface/40 p-4 xl:hidden">
        <BuilderPanel validation={validation} onPreset={applyPreset} store={archStore} universal={mode === 'model'} />
        {mode === 'neural' && <><Segmented value={rightTab} onChange={setRightTab} options={[{ value: 'data', label: t('ข้อมูล', 'Data') }, { value: 'train', label: t('ฝึก', 'Train') }, { value: 'view', label: t('มุมมอง', 'View') }, { value: 'inspect', label: t('ตรวจ', 'Inspect') }]} />{renderRight()}</>}
      </div>

  </>);

  return (
    <div>
      <header className="mx-auto flex max-w-[1600px] flex-col gap-2 px-4 pb-4 pt-6 sm:px-6 md:flex-row md:items-end md:justify-between">
        <div>
          <nav className="text-[1rem] text-muted"><Link to="/algorithms" className="hover:text-ink">{t('อัลกอริทึม', 'Algorithms')}</Link> / {mode === 'model' ? 'Model Builder' : 'Neural Network Builder'}</nav>
          <h1 className="mt-1 font-display text-3xl font-semibold">{mode === 'model' ? t('ตัวสร้างสถาปัตยกรรมโมเดล', 'Model Architecture Builder') : t('ตัวสร้างโครงข่ายประสาทเทียม', 'Neural Network Builder')}</h1>
          <p className="mt-1 max-w-[76ch] text-[1.125rem] text-muted">{mode === 'model'
            ? t('ลาก Conv, Recurrent, Attention, Normalization และ Merge มาต่อเป็น Blueprint ของโมเดลใหม่ โดยแยกจาก Neural Network Builder', 'Drag Conv, recurrent, attention, normalization and merge components into a new model blueprint, separate from the Neural Network Builder.')
            : t('สร้างและฝึกโครงข่าย Dense/MLP ดู forward/backward propagation และตรวจค่าน้ำหนักจริงทุกชั้น', 'Build and train Dense/MLP networks, watch forward/backward propagation, and inspect real weights in every layer.')}</p>
        </div>
        {notice && <button type="button" onClick={() => setNotice(null)} className="chip max-w-md border-cyan/50 text-left text-cyan">{notice} ✕</button>}
      </header>


      {mode === 'neural' ? <ExperimentLab data={experiment} controls={<><BuilderPanel validation={validation} onPreset={applyPreset} store={archStore} /><Segmented value={rightTab} onChange={setRightTab} options={[{ value: 'data', label: t('ข้อมูล', 'Data') }, { value: 'train', label: t('ฝึก', 'Train') }, { value: 'view', label: t('มุมมอง', 'View') }, { value: 'inspect', label: t('ตรวจ', 'Inspect') }]} />{renderRight()}</>} playback={{ ...playback, status }} detail={nativeStage} /> : nativeStage}

      {mode === 'model' && (
        <section className="mx-auto max-w-[1600px] px-4 py-10 sm:px-6">
          <h2 className="mb-5 font-display text-xl font-semibold">{t('บันทึกและส่งออก Blueprint', 'Save and export blueprint')}</h2>
          <ModelSummary layers={layers} validation={validation}
            onExportJson={() => download('model-blueprint.json', JSON.stringify(config(), null, 2))}
            onImport={importConfig}
            onSvg={() => download('architecture.svg', architectureSvg(layers, layers.map((l) => Math.max(1, validation.info[l.id]?.outShape ?? l.units ?? 1))), 'image/svg+xml')}
            onPng={() => { const c = box.current?.querySelector('canvas'); if (c) download('model-blueprint.png', c.toDataURL('image/png')); }}
            saves={saves}
            onSave={(name) => { const all = readSaves(); all[name] = config(); try { localStorage.setItem(SAVE_KEY, JSON.stringify(all)); } catch { /* storage unavailable */ } setSaves(Object.keys(all)); }}
            onLoad={(name) => { const c = readSaves()[name]; if (c) importConfig(JSON.stringify(c)); }}
            onDelete={(name) => { const all = readSaves(); delete all[name]; try { localStorage.setItem(SAVE_KEY, JSON.stringify(all)); } catch { /* storage unavailable */ } setSaves(Object.keys(all)); }} />
        </section>
      )}

      {mode === 'neural' && <section className="mx-auto max-w-[1600px] px-4 py-10 sm:px-6">
        <Tabs tabs={[
          {
            id: 'curves', label: t('กราฟการเรียนรู้', 'Learning curves'), content: (
              <div className="grid gap-8 lg:grid-cols-2">
                <div><h3 className="label-xs mb-2">Loss</h3>
                  <LinePlot xTitle="epoch" yTitle="loss" markerX={viewingPast ? shownEpoch : undefined} series={[
                    { name: 'train', x: hist.map((h) => h.epoch), y: hist.map((h) => h.loss), color: '#00D9FF' },
                    { name: 'validation', x: hist.map((h) => h.epoch), y: hist.map((h) => h.valLoss), color: '#F59E0B' },
                  ]} /></div>
                <div><h3 className="label-xs mb-2">{t('ความแม่นยำ', 'Accuracy')}</h3>
                  <LinePlot xTitle="epoch" yTitle="accuracy" yRange={[0, 1.02]} markerX={viewingPast ? shownEpoch : undefined} series={[
                    { name: 'train', x: hist.map((h) => h.epoch), y: hist.map((h) => h.acc), color: '#00D9FF' },
                    { name: 'validation', x: hist.map((h) => h.epoch), y: hist.map((h) => h.valAcc), color: '#F59E0B' },
                    { name: 'test', x: hist.map((h) => h.epoch), y: hist.map((h) => h.testAcc), color: '#22C55E', dash: 'dot' },
                  ]} /></div>
                {!hist.length && <p className="text-[1.0625rem] text-muted">{t('กด ▶ เพื่อเริ่มฝึก กราฟจะอัปเดตทุก epoch', 'Press ▶ to start training; the charts update every epoch.')}</p>}
              </div>
            ),
          },
          {
            id: 'map', label: t('เส้นแบ่งการตัดสินใจ', 'Decision boundary'), content: nF === 2 && displayNet ? (
              <div className="grid gap-8 lg:grid-cols-[380px_1fr]">
                <DecisionMap net={displayNet} tick={Math.floor(tick / 3)} X={raw.X} y={raw.y} train={prepared.trIdx} test={prepared.teIdx}
                  toModel={prepared.tf} classNames={raw.classNames} picked={custom} onPick={(p) => { setCustom(p); setCustomTarget(argClass(displayNet.output(prepared.tf(p)))); }} />
                <div className="max-w-[60ch] space-y-3 text-[1.125rem] leading-7 text-ink/90">
                  <p>{t('สีของแต่ละพิกเซลคือการผสมสีของคลาสตามความน่าจะเป็นที่โมเดลทำนาย สียิ่งเข้มยิ่งมั่นใจ ภาพนี้คำนวณจาก network ตัวเดียวกับที่แสดงในฉาก 3D', 'Each pixel blends class colours by the predicted probabilities; stronger colour means more confidence. It is computed by the same network shown in 3D.')}</p>
                  <p>{t('ช่วงแรกเส้นแบ่งจะเกือบเป็นเส้นตรง ระหว่างฝึก ชั้นซ่อนจะค่อย ๆ ดัดเส้นแบ่งให้โค้งตามข้อมูล', 'Early on the boundary is almost straight; during training the hidden layers bend it to follow the data.')}</p>
                </div>
              </div>
            ) : <p className="text-[1.0625rem] text-muted">{t('แผนที่นี้ต้องใช้ชุดข้อมูล 2 ฟีเจอร์ ชุดข้อมูลปัจจุบันมี ' + nF + ' ฟีเจอร์ ใช้ฉาก 3D และแท็บตรวจแทน', `This map needs a 2-feature dataset; the current one has ${nF}. Use the 3D scene and Inspect panel instead.`)}</p>,
          },
          {
            id: 'eval', label: t('การประเมินผล', 'Evaluation'), content: reports ? (
              <div className="grid gap-10 lg:grid-cols-[1fr_1fr]">
                <div className="scroll-thin overflow-x-auto">
                  <table className="num w-full min-w-[420px] font-mono text-[1rem]">
                    <thead className="text-muted"><tr className="border-b border-line text-left"><th className="py-2">split</th><th>n</th><th>accuracy</th><th>precision</th><th>recall</th><th>F1</th></tr></thead>
                    <tbody>{(['train', 'val', 'test'] as const).map((s) => (
                      <tr key={s} className="border-b border-line/50"><td className="py-1.5">{s}</td><td>{reports[s].support}</td><td>{pct(reports[s].accuracy)}</td><td>{pct(reports[s].precision)}</td><td>{pct(reports[s].recall)}</td><td>{pct(reports[s].f1)}</td></tr>
                    ))}</tbody>
                  </table>
                  <p className="mt-3 text-[1rem] text-muted">{t('precision / recall / F1 เป็นค่าเฉลี่ยแบบ macro ข้ามคลาส', 'Precision / recall / F1 are macro-averaged over classes.')}</p>
                </div>
                <div><h3 className="label-xs mb-2">Confusion matrix (test)</h3><ConfusionMatrix m={reports.test.confusion} classNames={raw.classNames} /></div>
              </div>
            ) : null,
          },
          {
            id: 'summary', label: t('สรุปโมเดล · ส่งออก', 'Model summary · Export'), content: (
              <ModelSummary layers={layers} validation={validation}
                onExportJson={() => download('ml-visual-lab-model.json', JSON.stringify(config(), null, 2))}
                onImport={importConfig}
                onSvg={() => download('architecture.svg', architectureSvg(layers, displayNet?.widths ?? layers.map((l) => Math.max(1, validation.info[l.id]?.outShape ?? l.units ?? 1))), 'image/svg+xml')}
                onPng={() => { const c = box.current?.querySelector('canvas'); if (c) download('network.png', c.toDataURL('image/png')); }}
                saves={saves}
                onSave={(name) => { const all = readSaves(); all[name] = config(); try { localStorage.setItem(SAVE_KEY, JSON.stringify(all)); } catch { /* storage unavailable */ } setSaves(Object.keys(all)); }}
                onLoad={(name) => { const c = readSaves()[name]; if (c) importConfig(JSON.stringify(c)); }}
                onDelete={(name) => { const all = readSaves(); delete all[name]; try { localStorage.setItem(SAVE_KEY, JSON.stringify(all)); } catch { /* storage unavailable */ } setSaves(Object.keys(all)); }} />
            ),
          },
          { id: 'math', label: t('คณิตศาสตร์', 'Math'), content: <MlpMath level={level} /> },
          { id: 'ex', label: t('แบบฝึกหัด', 'Exercises'), content: <MlpExercises live={{ dataset: data.id, nClasses: raw.nClasses, params: validation.totalParams, testAcc: reports?.test.accuracy ?? 0, epoch: liveEpoch }} /> },
        ]} />
      </section>}
    </div>
  );

  function renderRight() {
    if (rightTab === 'data') return (
      <div className="flex flex-col gap-3 pb-3">
        <Select label={t('ชุดข้อมูล', 'Dataset')} value={data.id} onChange={(id) => { setData({ ...data, id }); setCustom(null); }} options={DATASETS.map((d) => ({ value: d.id, label: t(d.name) }))} />
        <p className="text-[0.9375rem] text-muted">{t(info.note)}</p>
        {info.synthetic && <Slider label={t('จำนวนตัวอย่าง', 'Samples')} value={data.n} min={40} max={800} step={20} onChange={(n) => setData({ ...data, n })} />}
        {info.fixedClasses == null && <Slider label={t('จำนวนคลาส', 'Classes')} value={data.classes} min={2} max={6} onChange={(classes) => setData({ ...data, classes })} />}
        {info.synthetic && <Slider label="Noise" value={data.noise} min={0} max={0.8} step={0.05} format={(v) => v.toFixed(2)} onChange={(noise) => setData({ ...data, noise })} />}
        <Slider label={t('สัดส่วน test', 'Test ratio')} value={data.testRatio} min={0.1} max={0.5} step={0.05} format={pct} onChange={(testRatio) => setData({ ...data, testRatio })} />
        <Slider label={t('สัดส่วน validation (จาก train)', 'Validation split (of train)')} value={data.valRatio} min={0} max={0.4} step={0.05} format={pct} onChange={(valRatio) => setData({ ...data, valRatio })} />
        <Slider label="Data seed" value={data.seed} min={1} max={99} onChange={(seed) => setData({ ...data, seed })} />
        <Toggle label={t('ปรับมาตรฐาน (fit บน train)', 'Standardise (fit on train)')} checked={data.standardize} onChange={(standardize) => setData({ ...data, standardize })} />
      </div>
    );
    if (rightTab === 'train') return (
      <div className="flex flex-col gap-3 pb-3">
        <Select<OptimizerName> label="Optimizer" value={tc.optimizer} onChange={(optimizer) => setTc({ ...tc, optimizer })} options={[{ value: 'sgd', label: 'SGD' }, { value: 'momentum', label: 'SGD + Momentum' }, { value: 'rmsprop', label: 'RMSprop' }, { value: 'adam', label: 'Adam' }]} />
        <Slider label="Learning rate η" value={Math.log10(tc.lr)} min={-4} max={0} step={0.05} format={(v) => (10 ** v).toPrecision(2)} onChange={(v) => setTc({ ...tc, lr: Number((10 ** v).toPrecision(3)) })} />
        <Select<number> label="Batch size" value={tc.batchSize} onChange={(batchSize) => setTc({ ...tc, batchSize })} options={[1, 4, 8, 16, 32, 64, 128, 100000].map((v) => ({ value: v, label: v === 100000 ? t('ทั้งชุด (full batch)', 'Full batch') : String(v) }))} />
        <Select<Loss> label="Loss" value={tc.loss} onChange={(loss) => { setTc({ ...tc, loss }); const cur = archStore.getState().layers; setLayers(cur.map((l, i) => (i === cur.length - 1 ? { ...l, ...outputFor(raw.nClasses, loss) } : l))); }}
          options={[{ value: 'bce', label: 'Binary cross-entropy', disabled: raw.nClasses !== 2 }, { value: 'cce', label: 'Categorical cross-entropy' }, { value: 'mse', label: 'Mean squared error' }]} />
        {level >= 2 && tc.optimizer === 'momentum' && <Slider label="Momentum μ" value={tc.momentum} min={0} max={0.99} step={0.01} format={(v) => v.toFixed(2)} onChange={(momentum) => setTc({ ...tc, momentum })} />}
        {level >= 2 && (tc.optimizer === 'adam' || tc.optimizer === 'rmsprop') && (
          <>
            {tc.optimizer === 'adam' && <Slider label="β₁" value={tc.beta1} min={0} max={0.999} step={0.001} format={(v) => v.toFixed(3)} onChange={(beta1) => setTc({ ...tc, beta1 })} />}
            <Slider label="β₂" value={tc.beta2} min={0.9} max={0.9999} step={0.0001} format={(v) => v.toFixed(4)} onChange={(beta2) => setTc({ ...tc, beta2 })} />
            {level >= 3 && <Slider label="ε (log₁₀)" value={Math.log10(tc.epsilon)} min={-10} max={-4} step={1} format={(v) => `1e${v}`} onChange={(v) => setTc({ ...tc, epsilon: 10 ** v })} />}
          </>
        )}
        <NumberField label={t('จำนวน epoch สูงสุด', 'Max epochs')} value={tc.maxEpochs} min={1} max={5000} onChange={(maxEpochs) => setTc({ ...tc, maxEpochs: Math.max(1, Math.round(maxEpochs)) })} />
        <Toggle label="Shuffle" checked={tc.shuffle} onChange={(shuffle) => setTc({ ...tc, shuffle })} />
        <Toggle label="Early stopping (val loss)" checked={tc.earlyStopping} onChange={(earlyStopping) => setTc({ ...tc, earlyStopping })} />
        {tc.earlyStopping && <Slider label="Patience" value={tc.patience} min={3} max={100} onChange={(patience) => setTc({ ...tc, patience })} />}
        <div className="flex items-end gap-2">
          <div className="flex-1"><Slider label={t('seed การสุ่มน้ำหนัก', 'Weight-init seed')} value={tc.seed} min={1} max={99} onChange={(seed) => setTc({ ...tc, seed })} /></div>
        </div>
        <p className="text-[0.9375rem] text-muted">{t('η, batch, optimizer และ early stopping เปลี่ยนได้ระหว่างฝึก ส่วน loss, seed และสถาปัตยกรรมจะเริ่มโมเดลใหม่', 'η, batch size, optimizer and early stopping apply mid-training; loss, seed and architecture changes re-initialise the model.')}</p>
      </div>
    );
    if (rightTab === 'view') return (
      <div className="flex flex-col gap-3 pb-3">
        <Slider label={t('ระยะห่างระหว่างชั้น', 'Layer spacing')} value={view.spacing} min={2} max={8} step={0.2} format={(v) => v.toFixed(1)} onChange={(spacing) => setView({ ...view, spacing })} />
        <Slider label={t('ขนาดนิวรอน', 'Neuron size')} value={view.neuronSize} min={0.08} max={0.5} step={0.01} format={(v) => v.toFixed(2)} onChange={(neuronSize) => setView({ ...view, neuronSize })} />
        <Slider label={t('ความทึบของเส้น', 'Connection opacity')} value={view.opacity} min={0.1} max={1.5} step={0.05} format={(v) => v.toFixed(2)} onChange={(opacity) => setView({ ...view, opacity })} />
        <Select<number> label={t('จำนวนเส้นที่วาดสูงสุด', 'Max connections drawn')} value={view.maxEdges} onChange={(maxEdges) => setView({ ...view, maxEdges })} options={[500, 1000, 2000, 4000, 8000].map((v) => ({ value: v, label: v.toLocaleString() }))} />
        <Toggle label={t('แสดงเส้นเชื่อม', 'Show connections')} checked={view.showConnections} onChange={(showConnections) => setView({ ...view, showConnections })} />
        <Toggle label={t('สีตามน้ำหนัก (ฟ้า + / แดง −)', 'Weight colours (blue + / red −)')} checked={view.weightColors} onChange={(weightColors) => setView({ ...view, weightColors })} />
        <Toggle label={t('สีตาม activation', 'Activation colours')} checked={view.activationColors} onChange={(activationColors) => setView({ ...view, activationColors })} />
        <Toggle label={t('ป้ายชื่อชั้น', 'Layer labels')} checked={view.labels} onChange={(labels) => setView({ ...view, labels })} />
        <Toggle label={t('แยกเฉพาะชั้นที่เลือก', 'Isolate selected layer')} checked={view.isolate} onChange={(isolate) => setView({ ...view, isolate })} />
        <Slider label={t('ความเร็วการไหลของสัญญาณ', 'Signal flow speed')} value={flow.current.speed} min={0.25} max={4} step={0.25} format={(v) => `${v}×`} onChange={(v) => { flow.current.speed = v; forceFlow((x) => x + 1); }} />
        {edgeStats[1] > edgeStats[0] && <p className="text-[0.9375rem] text-orange">{t(`วาดเส้นแบบสุ่มตัวอย่าง ${pct(edgeStats[0] / edgeStats[1])} ของทั้งหมด ${edgeStats[1].toLocaleString()} เส้น (การคำนวณยังใช้ทุกเส้น)`, `Drawing a ${pct(edgeStats[0] / edgeStats[1])} sample of ${edgeStats[1].toLocaleString()} connections (computation still uses all of them)`)}</p>}
      </div>
    );
    const poolN = pd.Xte.length || pd.Xtr.length;
    return (
      <div className="flex flex-col gap-3 pb-3">
        <div className="label-xs">{t('ตัวอย่างที่ส่งผ่านโครงข่าย', 'Sample fed through the network')}</div>
        {custom ? (
          <div className="flex flex-col gap-2 text-[1rem]">
            <span className="font-mono">{t('จุดที่เลือกเอง', 'custom point')} ({custom.map((v) => v.toFixed(2)).join(', ')})</span>
            <Select<number> label={t('คลาสเป้าหมายสำหรับคำนวณ gradient', 'Target class for the gradient')} value={customTarget} onChange={setCustomTarget} options={raw.classNames.map((c, i) => ({ value: i, label: c }))} />
            <Button size="sm" onClick={() => setCustom(null)}>{t('กลับไปใช้ข้อมูล test', 'Back to test samples')}</Button>
          </div>
        ) : (
          <div className="flex items-end gap-2">
            <div className="flex-1"><Slider label={`${pd.Xte.length ? 'test' : 'train'} #`} value={Math.min(sampleIdx, poolN - 1)} min={0} max={Math.max(0, poolN - 1)} onChange={setSampleIdx} /></div>
            <Button size="sm" onClick={() => setSampleIdx(Math.floor(Math.random() * poolN))}>{t('สุ่ม', 'Random')}</Button>
          </div>
        )}
        {sample && (
          <div className="font-mono text-[0.9375rem]">
            <div className="flex justify-between"><span className="text-muted">x</span><span>({sample.x.map((v) => v.toFixed(2)).join(', ')})</span></div>
            <div className="flex justify-between"><span className="text-muted">{t('ค่าจริง / ทำนาย', 'true / predicted')}</span>
              <span><span style={{ color: CLASS_COLORS[sample.label] }}>{sample.custom ? '?' : raw.classNames[sample.label]}</span> / <span style={{ color: CLASS_COLORS[argClass(sample.trace.a[sample.trace.a.length - 1])] }}>{raw.classNames[argClass(sample.trace.a[sample.trace.a.length - 1])]}</span></span></div>
            <div className="flex justify-between"><span className="text-muted">{t('loss ของตัวอย่าง', 'sample loss')}</span><span>{fmt(sample.grads.loss, 4)}</span></div>
          </div>
        )}
        {layers.some((l) => l.kind === 'dropout') && <Toggle label={t('จำลอง forward แบบตอนฝึก (dropout ทำงาน)', 'Training-mode forward (dropout active)')} checked={trainMode} onChange={setTrainMode} />}
        <div className="border-t border-line pt-3">
          {sel && displayNet ? (
            <Inspector sel={sel} net={displayNet} layers={layers} sample={sample} trainer={trainer} viewingPast={viewingPast} onSelect={onSelect} featureNames={featureNames} />
          ) : <p className="text-[0.96875rem] text-muted">{t('คลิกนิวรอนหรือเส้นเชื่อมในฉาก 3D เพื่อดูอินพุต น้ำหนัก ผลรวม activation และ gradient', 'Click a neuron or connection in the 3D scene to see its inputs, weights, weighted sum, activation and gradient.')}</p>}
        </div>
      </div>
    );
  }
}
