import { useEffect, useMemo, useState } from 'react';
import { Link } from 'react-router-dom';
import { ExperimentLab } from '../../components/ExperimentLab';
import { Button, Section, Select, Slider, Tabs, fmt, pct } from '../../components/ui';
import { Callout, download } from '../../components/common';
import { VISION_MODELS, type VisionId } from '../../content/visionModels';
import { DIGIT_LABELS, DIGIT_PIXELS } from '../../data/digits';
import { trainTestSplit } from '../../data/datasets';
import { loadDigits } from '../../ml/cnn';
import { VisionBackbone, VisionTrainer, type VisionTrace, type VTensor } from '../../ml/vision';
import { Rng } from '../../lib/random';
import { useT } from '../../i18n/i18n';
import { flowVector, type ExperimentStage, type ModelExperiment } from '../../engine3d/experiment';
import { VisionAlgorithmDetail } from './VisionAlgorithmDetail';
import { visionGroupIndices } from '../../ml/visionDetail';
import type { PlaybackProps } from '../../components/PlaybackBar';
import { VisionTheory } from './VisionTheory';
import { TensorImage, VisionInput } from './VisionInput';

let digits: ReturnType<typeof loadDigits> | undefined;
function prepareData(n: number, seed: number) {
  digits ??= loadDigits(DIGIT_PIXELS, DIGIT_LABELS);
  const { X, y } = digits, split = trainTestSplit(y, 0.2, seed), rng = new Rng(seed + 2);
  const choose = (pool: number[], count: number) => rng.shuffle(Array.from({ length: 10 }, (_, label) => rng.shuffle(pool.filter((i) => y[i] === label)).slice(0, count / 10)).flat());
  const train = choose(split.train, n), test = choose(split.test, 100);
  return { Xtr: train.map((i) => X[i]), ytr: train.map((i) => y[i]), Xte: test.map((i) => X[i]), yte: test.map((i) => y[i]), trainIndices: train, testIndices: test };
}

function stageFor(name: string, tensor: VTensor, vit: boolean): ExperimentStage {
  if (vit && tensor.h > 1 && tensor.w === 1) {
    const values = Float64Array.from({ length: tensor.a.length }, (_, i) => tensor.a[(i % tensor.c) * tensor.h + Math.floor(i / tensor.c)]);
    return { name, kind: 'tensor', shape: [tensor.h, tensor.c, 1], values };
  }
  return { name, kind: tensor.h > 1 || tensor.w > 1 ? 'tensor' : 'vector', shape: [tensor.h, tensor.w, tensor.c], values: tensor.a };
}

export function visionStages(backbone: VisionBackbone, trace: VisionTrace, trainer: VisionTrainer, group: string): ExperimentStage[] {
  const indices = visionGroupIndices(backbone, group);
  const indexMap = new Map(indices.map((index, i) => [index, i]));
  const stages = indices.map((index, i): ExperimentStage => {
    const node = backbone.nodes[index], stage = stageFor(node.name, trace.tensors[index], backbone.id === 'vit');
    if (group !== 'overview') stage.inputs = node.inputs.flatMap((parent) => indexMap.has(parent) ? [indexMap.get(parent)!] : []);
    if (node.op === 'dense' && node.shape.h === 1 && i > 0 && indices[i - 1] === node.inputs[0]) stage.weights = node.W;
    return stage;
  });
  if (group === 'overview') {
    stages.push({ name: 'Train-only feature scaling', kind: 'vector', values: trainer.normalize(trace.features) });
    stages.push({ name: 'Classifier logits', values: trainer.logits(trace.features), weights: trainer.W });
    const probs = trainer.predict(trace.features), predicted = probs.indexOf(Math.max(...probs));
    stages.push({ name: 'Softmax · digits 0–9', values: probs, labels: Array.from({ length: 10 }, (_, i) => `P(${i})`), highlight: [predicted] });
  }
  return stages;
}

export default function VisionLab({ modelId }: { modelId: VisionId }) {
  const t = useT(), definition = VISION_MODELS[modelId];
  const [cfg, setCfg] = useState({ seed: 7, n: 300, width: 1, patch: 4, lr: 0.01, batch: 32, l2: 0.001, maxEpochs: 40, optimizer: 'adam' as 'adam' | 'sgd' });
  const [sample, setSample] = useState(0), [custom, setCustom] = useState<Float32Array | null>(null);
  const [running, setRunning] = useState(false), [speed, setSpeed] = useState(1), [tick, setTick] = useState(0), [nonce, setNonce] = useState(0);
  const [focus, setFocus] = useState('overview'), [inspect, setInspect] = useState(1), [channel, setChannel] = useState(0), [attentionHead, setAttentionHead] = useState(0);
  const [error, setError] = useState('');
  const data = useMemo(() => prepareData(cfg.n, cfg.seed), [cfg.n, cfg.seed]);
  const backbone = useMemo(() => new VisionBackbone(modelId, cfg.seed, cfg.width, cfg.patch), [modelId, cfg.seed, cfg.width, cfg.patch, nonce]);
  const trainer = useMemo(() => new VisionTrainer(backbone, data, cfg.lr, cfg.seed), [backbone, data]); // eslint-disable-line react-hooks/exhaustive-deps
  useEffect(() => { setRunning(false); setFocus('overview'); setInspect(1); setError(''); setTick((v) => v + 1); }, [trainer]);
  useEffect(() => { trainer.opt.cfg.lr = cfg.lr; trainer.opt.cfg.name = cfg.optimizer; }, [trainer, cfg.lr, cfg.optimizer]);
  const runStep = () => {
    if (!trainer.ready) trainer.prepareOne();
    else if (trainer.epoch < cfg.maxEpochs) trainer.trainBatch(cfg.batch, cfg.l2);
  };
  useEffect(() => {
    if (!running) return;
    let raf = 0, last = 0;
    const frame = (now: number) => {
      try {
        const start = performance.now();
        do { runStep(); } while (performance.now() - start < Math.min(16, 5 * speed) && trainer.epoch < cfg.maxEpochs);
        if (trainer.epoch >= cfg.maxEpochs) { setRunning(false); setTick((v) => v + 1); return; }
        if (now - last > 120) { last = now; setTick((v) => v + 1); }
        raf = requestAnimationFrame(frame);
      } catch (cause) { setRunning(false); setError((cause as Error).message); }
    };
    raf = requestAnimationFrame(frame); return () => cancelAnimationFrame(raf);
  }, [running, trainer, speed, cfg.batch, cfg.l2, cfg.maxEpochs]); // eslint-disable-line react-hooks/exhaustive-deps
  const index = Math.min(sample, data.Xte.length - 1), image = custom ?? data.Xte[index], target = custom ? null : data.yte[index];
  const trace = useMemo(() => backbone.forward(image), [backbone, image]);
  const probs = trainer.predict(trace.features), prediction = probs.indexOf(Math.max(...probs));
  const evaluation = useMemo(() => trainer.ready ? trainer.evaluate(true) : null, [trainer, tick]);
  const last = trainer.history[trainer.history.length - 1];
  const inspected = Math.min(inspect, backbone.nodes.length - 1), tensor = trace.tensors[inspected], node = backbone.nodes[inspected];
  const groups = [...new Set(backbone.nodes.map((n) => n.group))].filter((g) => g !== 'input');
  const maps = trace.attention.get(inspected);
  const displayChannel = Math.min(channel, tensor.c - 1), pixels = tensor.a.subarray(displayChannel * tensor.h * tensor.w, (displayChannel + 1) * tensor.h * tensor.w);
  const preparedStatus = trainer.ready ? `${t('ฝึกหัวจำแนก', 'Train classifier head')} · epoch ${trainer.epoch}/${cfg.maxEpochs}` : `${t('เตรียม features', 'Prepare features')} ${trainer.prepared}/${trainer.totalSamples}`;
  const experiment: ModelExperiment = {
    model: `${definition.name} · ${t('โมเดลสาธิตขนาดย่อ', 'compact teaching model')}`,
    source: `UCI digits · train ${data.Xtr.length} / test ${data.Xte.length} · grayscale 16×16`,
    stages: visionStages(backbone, trace, trainer, focus),
    output: `${t('ทำนาย', 'Prediction')}: ${prediction} · P=${pct(probs[prediction])}${target == null ? ` · ${t('ภาพของผู้ใช้ ไม่ทราบคำตอบจริง', 'custom image; target unknown')}` : ` · ${t('คำตอบจริง', 'Target')}: ${target}`}`,
    note: `${t('Backbone สุ่มและตรึงไว้ ฝึกเฉพาะหัวจำแนก ไม่มีน้ำหนัก ImageNet', 'Frozen random backbone; only the classifier head trains; no ImageNet weights')} · ${preparedStatus}`,
    metrics: [{ label: 'Epoch', value: String(trainer.epoch) }, { label: 'Test accuracy', value: evaluation ? pct(evaluation.report.accuracy) : '—' }, { label: 'Train loss', value: last ? fmt(last.loss, 5) : '—' }, { label: 'Test loss', value: evaluation ? fmt(evaluation.loss, 5) : '—' }, { label: 'Precision / Recall', value: evaluation ? `${pct(evaluation.report.precision)} / ${pct(evaluation.report.recall)}` : '—' }, { label: 'F1', value: evaluation ? pct(evaluation.report.f1) : '—' }, { label: 'Frozen weights stored', value: backbone.parameterCount.toLocaleString() }, { label: 'Trainable head params', value: (trainer.W.length + trainer.b.length).toLocaleString() }],
    history: [{ label: 'Train cross-entropy', values: trainer.history.map((row) => row.loss) }, { label: 'Test accuracy', values: trainer.history.map((row) => row.testAcc), color: '#F59E0B' }],
    confusion: evaluation?.report.confusion,
    sample: { index, count: data.Xte.length, onChange: (next) => { setCustom(null); setSample(next); } },
  };
  const exportExperiment = () => download(`${modelId}-experiment.json`, JSON.stringify({ schema: 1, modelId, mode: 'compact-frozen-random-backbone-trained-head', config: cfg, preprocessing: '16x16 grayscale 0..1; train-only feature standardization', trainIndices: data.trainIndices, testIndices: data.testIndices, history: trainer.history, evaluation, head: { W: Array.from(trainer.W), b: Array.from(trainer.b), mean: Array.from(trainer.mean), std: Array.from(trainer.std) } }, null, 2));
  const controls = <>
    <Section title={t('การทดลองจริง', 'Live experiment')}>
      <Select label={t('การแสดงผล', 'Computation view')} value={focus} onChange={(group) => { setFocus(group); const attention = backbone.nodes.findIndex((n) => n.group === group && n.op === 'attention'); setInspect(attention >= 0 ? attention : Math.max(0, backbone.nodes.findIndex((n) => n.group === group))); setChannel(0); }} options={[{ value: 'overview', label: t('ภาพรวมการไหล', 'Execution overview') }, ...groups.map((g) => ({ value: g, label: g }))]} />
      <Select label={t('จำนวนตัวอย่าง train', 'Training samples')} value={cfg.n} onChange={(n) => setCfg({ ...cfg, n })} options={[100, 300, 600].map((n) => ({ value: n, label: String(n) }))} />
      <Select label={t('ความกว้างสาธิต', 'Teaching channel width')} value={cfg.width} onChange={(width) => setCfg({ ...cfg, width })} options={[{ value: 1, label: '1× compact' }, { value: 2, label: '2× compact' }]} />
      <Slider label="Seed" value={cfg.seed} min={1} max={99} onChange={(seed) => setCfg({ ...cfg, seed })} />
      {modelId === 'vit' && <Select label={t('ขนาด patch', 'Patch size')} value={cfg.patch} onChange={(patch) => setCfg({ ...cfg, patch })} options={[{ value: 4, label: '4×4 → 17 tokens' }, { value: 8, label: '8×8 → 5 tokens' }]} />}
    </Section>
    <Section title={t('ฝึกเฉพาะหัวจำแนก', 'Train classifier head only')}>
      <Select label="Optimizer" value={cfg.optimizer} onChange={(optimizer) => setCfg({ ...cfg, optimizer })} options={[{ value: 'adam', label: 'Adam' }, { value: 'sgd', label: 'SGD' }]} />
      <Slider label="Learning rate" value={cfg.lr} min={0.001} max={0.1} step={0.001} format={(v) => v.toFixed(3)} onChange={(lr) => setCfg({ ...cfg, lr })} />
      <Select label="Batch size" value={cfg.batch} onChange={(batch) => setCfg({ ...cfg, batch })} options={[8, 16, 32, 64].map((n) => ({ value: n, label: String(n) }))} />
      <Slider label="L2" value={cfg.l2} min={0} max={0.02} step={0.001} onChange={(l2) => setCfg({ ...cfg, l2 })} />
      <Slider label={t('จำนวน epoch สูงสุด', 'Maximum epochs')} value={cfg.maxEpochs} min={5} max={100} step={5} onChange={(maxEpochs) => setCfg({ ...cfg, maxEpochs })} />
      <p className="text-sm text-cyan" role="status">{preparedStatus}</p>
      <Button size="sm" onClick={exportExperiment}>{t('ส่งออกผลและน้ำหนักหัว JSON', 'Export results and head JSON')}</Button>
    </Section>
    <Section title={t('ตรวจ tensor และสาขา', 'Inspect tensors and branches')}>
      <Select label={t('ชั้นที่คำนวณจริง', 'Computed node')} value={inspected} onChange={(value) => { setInspect(value); setChannel(0); }} options={backbone.nodes.map((n, i) => ({ value: i, label: `${i}. ${n.group} / ${n.name}` }))} />
      <p className="break-words font-mono text-sm">{node.op} · {tensor.h}×{tensor.w}×{tensor.c}<br />parents: {node.inputs.join(', ') || 'input'}</p>
      <Slider label={t('ช่องที่แสดง', 'Displayed channel')} value={displayChannel} min={0} max={Math.max(0, tensor.c - 1)} onChange={setChannel} />
      <TensorImage values={pixels} h={tensor.h} w={tensor.w} />
      <p className="break-words font-mono text-xs">{flowVector(pixels, 8)}</p>
      {maps && <><Select label="Attention head" value={Math.min(attentionHead, maps.length - 1)} onChange={setAttentionHead} options={maps.map((_, i) => ({ value: i, label: `Head ${i + 1}` }))} /><TensorImage values={maps[Math.min(attentionHead, maps.length - 1)]} h={tensor.h} w={tensor.h} /><p className="text-sm text-muted">{t('แถวคือ query คอลัมน์คือ key แถวละรวมเป็น 1', 'Rows are queries, columns are keys; each row sums to 1.')}</p></>}
    </Section>
  </>;
  const playback: PlaybackProps = { playing: running, onToggle: () => { if (trainer.epoch >= cfg.maxEpochs) return; setRunning((v) => !v); }, onNext: () => { setRunning(false); try { runStep(); setTick((v) => v + 1); } catch (cause) { setError((cause as Error).message); } }, onReset: () => { setRunning(false); setNonce((v) => v + 1); }, canNext: trainer.epoch < cfg.maxEpochs, speed, onSpeed: setSpeed, status: preparedStatus };
  return <div>
    <header className="mx-auto max-w-[1600px] px-4 pb-5 pt-6 sm:px-6"><Link to="/algorithms" className="text-sm text-muted hover:text-cyan">Algorithms / {definition.name}</Link><h1 className="mt-2 font-display text-3xl font-semibold">{definition.name} · {t('ห้องทดลองภาพ 3D', '3D vision lab')}</h1><p className="mt-2 max-w-[90ch] text-base text-muted">{t(definition.summary)}</p>
      <div className="mt-3 flex flex-wrap gap-2"><span className="chip">{t('รายละเอียดอัลกอริทึมจากการคำนวณจริง', 'Live Algorithm detail')}</span><span className="chip">{t('ฝึกเฉพาะหัว ไม่ใช่ ImageNet pretrained', 'Head training; no ImageNet pretrained weights')}</span></div>
    </header>
    {error && <div className="mx-auto max-w-[1600px] px-4 pb-4"><p role="alert" className="text-red">{error}</p></div>}
    <ExperimentLab data={experiment} controls={controls} detail={<VisionAlgorithmDetail key={modelId} backbone={backbone} trace={trace} trainer={trainer} nodeIndex={inspected} onNodeChange={(next) => { setInspect(next); setChannel(0); }} experiment={experiment} controls={controls} playback={playback} image={image} onImage={setCustom} customImage={custom !== null} />} detailLabel={t(`รายละเอียดอัลกอริทึม ${definition.name}`, `${definition.name} Algorithm detail`)} playback={playback} />
    <section className="mx-auto max-w-[1600px] px-4 py-8 sm:px-6"><Tabs tabs={[
      { id: 'theory', label: t('หลักการ สมการ และแบบทดสอบ', 'Theory, math and quiz'), content: <VisionTheory modelId={modelId} /> },
      { id: 'input', label: t('วาดหรืออัปโหลดภาพ', 'Draw or upload'), content: <div className="grid gap-6 md:grid-cols-2"><VisionInput image={image} onChange={setCustom} /><div><h3 className="mb-3 font-display text-xl">{t('ผลจากภาพปัจจุบัน', 'Current image prediction')}</h3><TensorImage values={image} h={16} w={16} /><p className="mt-3 text-base">{experiment.output}</p><Button className="mt-3" onClick={() => setCustom(null)}>{t('กลับไปใช้ภาพทดสอบ', 'Use test image')}</Button></div></div> },
      { id: 'architecture', label: t('รายการคำนวณและน้ำหนัก', 'Operations and weights'), content: <div className="space-y-4"><Callout>{t('รายการนี้เป็นโครงข่ายสาธิตที่รันจริง ใช้ CHW ภายในเอนจิน รูปร่างแสดง H×W×C เส้นทาง parent ใช้แสดง shortcut และสาขาคู่ขนาน', 'This is the executed teaching graph. Internally tensors use CHW; shapes below use H×W×C. Parent indices define real shortcuts and parallel branches.')}</Callout><div className="scroll-thin max-h-[640px] overflow-auto"><table className="w-full text-left text-sm"><thead><tr><th className="p-2">#</th><th className="p-2">Block / operation</th><th className="p-2">H×W×C</th><th className="p-2">Parents</th><th className="p-2">Frozen params</th></tr></thead><tbody>{backbone.nodes.map((n, i) => <tr key={i} className="border-t border-line"><td className="p-2">{i}</td><td className="p-2">{n.group} / {n.name}</td><td className="p-2 font-mono">{n.shape.h}×{n.shape.w}×{n.shape.c}</td><td className="p-2">{n.inputs.join(', ')}</td><td className="p-2">{(n.W?.length ?? 0) + (n.b?.length ?? 0)}</td></tr>)}</tbody></table></div></div> },
    ]} /></section>
  </div>;
}
