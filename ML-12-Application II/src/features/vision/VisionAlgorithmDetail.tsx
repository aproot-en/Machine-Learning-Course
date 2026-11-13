import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';
import { Stage, StageToolbar, useCameraState } from '../../engine3d/Stage';
import type { ModelExperiment } from '../../engine3d/experiment';
import { Button, FloatingPanel, NumberField, Readout, Section, Select, Slider, fmt, pct } from '../../components/ui';
import { PlaybackBar, type PlaybackProps } from '../../components/PlaybackBar';
import { Tex } from '../../components/common';
import { VISION_MODELS } from '../../content/visionModels';
import type { VisionBackbone, VisionTrace, VisionTrainer } from '../../ml/vision';
import { visionCalculation, visionNetwork } from '../../ml/visionDetail';
import { useT } from '../../i18n/i18n';
import { TensorImage, VisionInput } from './VisionInput';
import { VisionNetworkScene } from './VisionNetworkScene';

interface Props {
  backbone: VisionBackbone; trace: VisionTrace; trainer: VisionTrainer;
  nodeIndex: number; onNodeChange: (index: number) => void;
  experiment: ModelExperiment; controls: ReactNode; playback: PlaybackProps;
  image: Float32Array; onImage: (image: Float32Array) => void; customImage: boolean;
}

/** Live, model-specific operation view. All values come from the active inference trace. */
export function VisionAlgorithmDetail(p: Props) {
  const t = useT(), { backbone, trace, trainer, nodeIndex } = p, model = VISION_MODELS[backbone.id];
  const [cell, setCell] = useState(0), [inputChannel, setInputChannel] = useState(0), [page, setPage] = useState(0), [head, setHead] = useState(0);
  const [scanning, setScanning] = useState(() => !(typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches));
  const [scanSpeed, setScanSpeed] = useState(1);
  const [cam, setCam] = useCameraState({ preset: 'iso' }), box = useRef<HTMLDivElement>(null);
  const node = backbone.nodes[nodeIndex], output = trace.tensors[nodeIndex], spatial = output.h * output.w;
  const index = Math.min(cell, output.a.length - 1), channel = Math.floor(index / spatial), position = index % spatial;
  const calculation = useMemo(() => visionCalculation(backbone, trace, nodeIndex, index), [backbone, trace, nodeIndex, index]);
  useEffect(() => { setCell(0); setInputChannel(0); setPage(0); setHead(0); }, [backbone, nodeIndex]);
  useEffect(() => {
    if (!scanning) return;
    const timer = setInterval(() => setCell((current) => (current + 1) % output.a.length), 650 / scanSpeed);
    return () => clearInterval(timer);
  }, [scanning, scanSpeed, output.a.length, nodeIndex, backbone]);
  const network = useMemo(() => visionNetwork(backbone, trace, trainer, nodeIndex), [backbone, trace, trainer, nodeIndex, p.experiment]);
  let footprint = calculation.terms.filter((term) => term.index >= 0).map(({ parent, index }) => ({ parent, index }));
  if (node.op === 'attention') footprint = Array.from({ length: trace.tensors[node.inputs[0]].a.length }, (_, index) => ({ parent: node.inputs[0], index }));
  if (node.op === 'ln') footprint = Array.from({ length: output.c }, (_, c) => ({ parent: node.inputs[0], index: c * spatial + position }));
  const groups = [...new Set(backbone.nodes.map((n) => n.group))];
  const source = trace.tensors[node.inputs[0]], convolution = node.op === 'conv' || node.op === 'depthwise';
  const shownInput = node.op === 'depthwise' ? channel : Math.min(inputChannel, (source?.c ?? 1) - 1);
  const kernelTerms = convolution ? calculation.terms.filter((term) => term.label.startsWith(`c${shownInput} `)) : [];
  const maps = trace.attention.get(nodeIndex), attentionHead = Math.min(head, (maps?.length ?? 1) - 1);
  const termPages = Math.max(1, Math.ceil(calculation.terms.length / 48)), shownPage = Math.min(page, termPages - 1);
  const probs = trainer.predict(trace.features), logits = trainer.logits(trace.features);
  const go = (next: number) => { setCell(0); setPage(0); p.onNodeChange(next); };
  return <section aria-label={`${model.name} Algorithm detail`}>
    <header className="mx-auto max-w-[1600px] space-y-3 px-4 pb-5 sm:px-6">
      <h2 className="font-display text-2xl font-semibold">{model.name} · {t('รายละเอียดอัลกอริทึม', 'Algorithm detail')}</h2>
      <p className="max-w-[95ch] text-base leading-relaxed">{t(model.idea)}</p>
      <p className="text-sm text-muted">{t('ภาพอินพุต → feature maps และชั้นของโมเดล → หัวจำแนก → ผลทำนาย 0–9 แสดงพร้อมกันในฉากเดียวแบบ CNN เลือกชั้นเพื่อดู receptive field โดยยังเห็นเส้นทางตั้งแต่ภาพจนถึงผลลัพธ์', 'Input image → model feature maps and layers → classifier → digit predictions, together in one CNN-style scene. Select an operation to inspect its receptive field while keeping the complete input-to-output path visible.')}</p>
      <p className="text-sm text-orange">{t('คำนวณบนโมเดลสาธิตขนาดย่อ Backbone สุ่มและตรึงไว้ ฝึกเฉพาะหัวจำแนก ไม่มีน้ำหนัก ImageNet', 'Executed compact teaching model: frozen random backbone, trainable classifier head; no ImageNet weights.')}</p>
      <div className="grid gap-3 md:grid-cols-2">
        <Select label={t('เลือก block', 'Detail block')} value={node.group} onChange={(group) => go(backbone.nodes.findIndex((n) => n.group === group))} options={groups.map((group) => ({ value: group, label: group }))} />
        <Select label={t('ขั้นตอนอัลกอริทึม', 'Detail operation')} value={nodeIndex} onChange={go} options={backbone.nodes.map((n, i) => ({ value: i, label: `${i}. ${n.group} / ${n.name}` }))} />
      </div>
    </header>
    <div ref={box} className="relative h-[calc(100svh-190px)] min-h-[680px] overflow-hidden border-y border-line bg-bg">
      <Stage cam={cam} distance={Math.max(28, network.length * 6)} fogEnabled={false}>
        <VisionNetworkScene layers={network} modelName={model.name} vit={backbone.id === 'vit'} selected={nodeIndex} cell={index} footprint={footprint} onSelect={go} playing={scanning} speed={scanSpeed} cam={cam} />
      </Stage>
      <div className="pointer-events-none absolute inset-0 flex flex-col justify-between gap-4 p-3 sm:p-4">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div className="flex max-w-[280px] flex-col gap-2">
            <Readout title={`${model.name} · ${t('อินพุตและผลลัพธ์', 'Input and output')}`} rows={[
              [t('ภาพ', 'Image'), p.customImage ? t('ภาพของผู้ใช้', 'custom image') : `test #${p.experiment.sample?.index ?? 0}`],
              [t('ทำนาย', 'Prediction'), `${probs.indexOf(Math.max(...probs))} · ${pct(Math.max(...probs))}`, '#F59E0B'],
              ['Epoch', String(trainer.epoch)],
            ]} />
            <div className="glass flex items-center gap-3 p-2"><svg viewBox="0 0 16 16" width="88" height="88" role="img" aria-label={`${model.name} input image`} className="shrink-0 rounded border border-line" style={{ imageRendering: 'pixelated' }}>{Array.from(p.image, (value, pixel) => <rect key={pixel} x={pixel % 16} y={Math.floor(pixel / 16)} width="1" height="1" fill={`rgb(${Math.round(value * 255)},${Math.round(value * 255)},${Math.round(value * 255)})`} />)}</svg><p className="text-sm text-muted">{t('ภาพอินพุตจริง', 'Actual input image')}<br />16×16 grayscale</p></div>
            <div className="pointer-events-auto"><StageToolbar cam={cam} setCam={setCam} container={box} /></div>
          </div>
          <FloatingPanel title={t('ภาพและโครงข่าย', 'Image and network')} className="hidden max-h-[350px] w-[300px] lg:flex">
            <NumberField label={t('ตัวอย่างข้อมูล', 'Scene data sample')} value={p.experiment.sample?.index ?? 0} min={0} max={(p.experiment.sample?.count ?? 1) - 1} onChange={(value) => p.experiment.sample?.onChange(Math.trunc(value))} />
            <VisionInput image={p.image} onChange={p.onImage} />
          </FloatingPanel>
        </div>
        <div className="flex flex-col items-center gap-2">
          <p className="max-w-full break-words rounded bg-bg/95 px-3 py-2 text-center text-sm text-cyan">{p.experiment.output}</p>
          <div className="pointer-events-auto flex flex-wrap justify-center gap-2 rounded-lg border border-line bg-bg/95 p-2">
            <Button size="sm" onClick={() => setScanning((v) => !v)}>{scanning ? t('หยุดการสแกน', 'Pause scan') : t('เล่นการสแกน', 'Play scan')}</Button>
            <Button size="sm" onClick={() => { setScanning(false); setCell((index + 1) % output.a.length); }}>{t('ตำแหน่งถัดไป', 'Next position')}</Button>
            <Button size="sm" disabled={nodeIndex === 0} onClick={() => go(nodeIndex - 1)}>{t('ชั้นก่อนหน้า', 'Previous operation')}</Button>
            <Button size="sm" disabled={nodeIndex === backbone.nodes.length - 1} onClick={() => go(nodeIndex + 1)}>{t('ชั้นถัดไป', 'Next operation')}</Button>
          </div>
          <PlaybackBar {...p.playback} />
        </div>
      </div>
    </div>
    <div className="mx-auto grid max-w-[1600px] gap-6 px-4 py-6 sm:px-6 xl:grid-cols-[340px_minmax(0,1fr)]">
      <aside className="min-w-0">
        <Section title={t('ภาพอินพุต', 'Detail input image')}>
          {p.experiment.sample && <NumberField label={t('ตัวอย่างข้อมูล', 'Detail data sample')} value={p.experiment.sample.index} min={0} max={p.experiment.sample.count - 1} onChange={(value) => p.experiment.sample?.onChange(Math.trunc(value))} />}
          <VisionInput image={p.image} onChange={p.onImage} />
        </Section>
        <Section title={t('ตรวจตำแหน่งคำนวณ', 'Inspect computation position')}>
          <Slider label={t('ช่องผลลัพธ์', 'Detail output channel')} value={channel} min={0} max={output.c - 1} onChange={(c) => { setScanning(false); setCell(c * spatial + position); }} />
          <Slider label={t('ตำแหน่งผลลัพธ์', 'Detail output position')} value={position} min={0} max={spatial - 1} onChange={(pos) => { setScanning(false); setCell(channel * spatial + pos); }} />
          <Select label={t('ความเร็วการสแกน', 'Scan speed')} value={scanSpeed} onChange={setScanSpeed} options={[0.5, 1, 2, 4].map((speed) => ({ value: speed, label: `${speed}×` }))} />
          {convolution && node.op !== 'depthwise' && source.c > 1 && <Slider label={t('ช่องอินพุตของ kernel', 'Kernel input channel')} value={shownInput} min={0} max={source.c - 1} onChange={setInputChannel} />}
          <p className="text-sm text-muted">{t('สีขาวแสดงช่องรับข้อมูลและค่าที่กำลังคำนวณ ภาพต้นทางแสดงเป็น grayscale แผ่นสีฟ้าคือ feature maps สีม่วงคือค่าลบ คอลัมน์ด้านขวาคือผลลัพธ์ทุกคลาส ตัวเลขด้านขวารวมทุกค่าที่เกี่ยวข้อง', 'White marks the input footprint and current output cell. The source image is grayscale, cyan sheets show feature maps, purple shows negative values, and the final column shows every output class. Numeric calculations include all contributing values.')}</p>
        </Section>
        <details className="rounded-lg border border-line p-3"><summary className="cursor-pointer text-cyan">{t('ตั้งค่าข้อมูลและการฝึก', 'Data and training settings')}</summary>{p.controls}</details>
      </aside>
      <div className="min-w-0 space-y-6">
        <div className="rounded-xl border border-line bg-surface p-4 sm:p-5">
          <h3 className="font-display text-xl">{node.name} · {t('การคำนวณจริง', 'Actual calculation')}</h3>
          <Tex block>{model.formula}</Tex>
          <p className="mt-2 break-words font-mono text-base text-cyan">{calculation.formula}</p>
          <p className="mt-2 text-sm text-muted">{t('รับข้อมูลจาก', 'Inputs from')}: {node.inputs.map((i) => `#${i} ${backbone.nodes[i].name}`).join(' + ') || t('ภาพอินพุต', 'input image')}</p>
          {calculation.statistics && <p className="mt-2 break-words font-mono text-sm">{calculation.statistics}</p>}
          <div className="mt-4 flex flex-wrap items-start gap-5">
            <div><p className="mb-2 text-sm">{t('แผนที่ผลลัพธ์ของช่องที่เลือก', 'Selected output channel')}</p><TensorImage values={output.a.subarray(channel * spatial, (channel + 1) * spatial)} h={output.h} w={output.w} /></div>
            {kernelTerms.length > 0 && <div className="min-w-0 max-w-full"><p className="mb-2 text-sm">{t('อินพุต × kernel', 'Input × kernel')} · c{shownInput}</p><div className="grid max-w-full gap-1 overflow-x-auto font-mono text-sm" style={{ gridTemplateColumns: `repeat(${node.kw}, minmax(48px, 1fr))` }}>{kernelTerms.map((term, i) => <div key={i} className="rounded border border-line bg-cyan/5 p-1.5 text-center"><span>{fmt(term.value, 3)}</span><br /><span className="text-muted">× {fmt(term.weight, 3)}</span><br /><b className="text-cyan">{fmt(term.value * term.weight, 3)}</b></div>)}</div></div>}
          </div>
          {maps && <div className="mt-5 space-y-2"><Select label={t('หัว attention', 'Detail attention head')} value={attentionHead} onChange={setHead} options={maps.map((_, i) => ({ value: i, label: `Head ${i + 1}` }))} /><TensorImage values={maps[attentionHead]} h={output.h} w={output.h} /><p className="text-sm text-muted">{t('แถวคือ query คอลัมน์คือ key แต่ละแถวรวมเป็น 1 ตารางด้านล่างรวม A × V × Wₒ ของทุกหัวเพื่อคำนวณช่องผลลัพธ์ที่เลือก', 'Rows are queries, columns are keys; each row sums to 1. The table sums A × V × Wₒ across all heads for the selected output channel.')}</p></div>}
          <div className="mt-5 rounded-lg border border-cyan/30 bg-cyan/5 p-3 font-mono text-base" role="status">
            <p>{t('จำนวนค่าที่รวม', 'Terms')}: {calculation.terms.length} · bias = {fmt(calculation.bias, 6)}</p>
            <p>{t('คำนวณได้', 'Calculated')} = {fmt(calculation.result, 6)} · {t('ค่าจาก forward จริง', 'Actual forward value')} = {fmt(calculation.actual, 6)}</p>
          </div>
          <div className="mt-4 max-w-xs"><NumberField label={t('หน้าตารางค่าคำนวณ', 'Calculation table page')} value={shownPage + 1} min={1} max={termPages} onChange={(value) => setPage(Math.trunc(value) - 1)} /></div>
          <div className="scroll-thin mt-3 max-h-[360px] overflow-auto"><table className="w-full text-left font-mono text-sm"><thead><tr><th className="p-2">{t('แหล่งข้อมูล', 'Source')}</th><th className="p-2">x</th><th className="p-2">{t('สัมประสิทธิ์', 'Coefficient')}</th><th className="p-2">x × w</th></tr></thead><tbody>{calculation.terms.slice(shownPage * 48, (shownPage + 1) * 48).map((term, i) => <tr key={i} className="border-t border-line"><td className="p-2">{term.label}</td><td className="p-2">{fmt(term.value, 6)}</td><td className="p-2">{fmt(term.weight, 6)}</td><td className="p-2 text-cyan">{fmt(term.value * term.weight, 6)}</td></tr>)}</tbody></table></div>
        </div>
        <div className="rounded-xl border border-line bg-surface p-4"><h3 className="font-display text-xl">{t('ผลหัวจำแนกจากภาพปัจจุบัน', 'Classifier output for the current image')}</h3><p className="mt-2 text-sm text-muted">{t('feature → ปรับสเกลจาก train → linear → softmax ตารางอัปเดตเมื่อเปลี่ยนภาพหรือฝึกหัวจำแนก', 'Features → train-fitted scaling → linear → softmax. Values update when the image or classifier weights change.')}</p><div className="mt-3 grid grid-cols-2 gap-2 sm:grid-cols-5">{Array.from(probs, (probability, digit) => <div key={digit} className="rounded border border-line p-2 font-mono text-sm"><b>{digit}: {pct(probability)}</b><p className="text-muted">z={fmt(logits[digit], 4)}</p></div>)}</div></div>
      </div>
    </div>
  </section>;
}
