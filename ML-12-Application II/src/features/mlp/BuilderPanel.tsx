import { Fragment, useState } from 'react';
import { useArch, type ArchStore, type PresetName } from './archStore';
import { useSettings, useT } from '../../i18n/i18n';
import { ACTIVATIONS, BUILDABLE_LAYER_KINDS, INITS, type LayerKind, type LayerSpec } from '../../ml/nn/types';
import type { Validation } from '../../ml/nn/validate';
import { MAX_UNITS } from '../../ml/nn/validate';
import { Button, Icon, NumberField, Select, Slider, Toggle } from '../../components/ui';

const KIND_TONE: Record<string, string> = {
  input: 'text-ink', dense: 'text-cyan', output: 'text-green', dropout: 'text-muted', activation: 'text-purple',
  conv1d: 'text-orange', conv2d: 'text-orange', maxpool2d: 'text-orange', avgpool2d: 'text-orange', globalavgpool2d: 'text-orange', flatten: 'text-orange',
  batchnorm: 'text-blue-300', layernorm: 'text-blue-300', embedding: 'text-green', rnn: 'text-purple', lstm: 'text-purple', gru: 'text-purple',
  attention: 'text-pink-400', add: 'text-yellow-300', concatenate: 'text-yellow-300',
};

const LAYER_GROUPS: { label: string; items: [LayerKind, string, string][] }[] = [
  { label: 'Core', items: [['dense', 'Dense', 'W·a+b'], ['dropout', 'Dropout', 'regularize'], ['activation', 'Activation', 'f(x)']] },
  { label: 'Convolution', items: [['conv1d', 'Conv1D', 'sequence'], ['conv2d', 'Conv2D', 'image'], ['maxpool2d', 'MaxPool', 'downsample'], ['avgpool2d', 'AvgPool', 'downsample'], ['globalavgpool2d', 'GlobalAvg', 'reduce'], ['flatten', 'Flatten', 'reshape']] },
  { label: 'Sequence', items: [['embedding', 'Embedding', 'tokens'], ['rnn', 'RNN', 'recurrent'], ['lstm', 'LSTM', 'memory'], ['gru', 'GRU', 'gated'], ['attention', 'Attention', 'multi-head']] },
  { label: 'Utility & Merge', items: [['batchnorm', 'BatchNorm', 'normalize'], ['layernorm', 'LayerNorm', 'normalize'], ['add', 'Add', 'residual'], ['concatenate', 'Concat', 'merge']] },
];

export function BuilderPanel({ validation, onPreset, store = useArch, universal = false }: { validation: Validation; onPreset: (p: PresetName) => void; store?: ArchStore; universal?: boolean }) {
  const t = useT();
  const level = useSettings((s) => s.level);
  const { layers, selectedId, select, update, add, insert, remove, duplicate, move, undo, redo, past, future } = store();
  const [dragFrom, setDragFrom] = useState<number | null>(null);
  const [dragKind, setDragKind] = useState<LayerSpec['kind'] | null>(null);
  const sel = layers.find((l) => l.id === selectedId) ?? null;
  const issuesFor = (id: string) => [...validation.errors, ...validation.warnings].filter((i) => i.layerId === id);

  return (
    <div className="flex flex-col gap-3 pb-2">
      <div className="flex flex-wrap items-end gap-2">
        <div className="min-w-[150px] flex-1">
          <Select<string> label={t('สถาปัตยกรรมสำเร็จรูป', 'Preset architecture')} value="" onChange={(v) => v && onPreset(v as PresetName)} options={universal ? [
            { value: '', label: t('— เลือก —', '— choose —') },
              { value: 'cnn', label: 'CNN image classifier' },
              { value: 'sequence', label: 'Embedding + LSTM' },
              { value: 'transformer', label: 'Transformer encoder' },
            ] : [
              { value: '', label: t('— เลือก —', '— choose —') },
              { value: 'perceptron', label: 'Single Perceptron' },
              { value: 'mlp2', label: t('MLP 2 ชั้นซ่อน', 'MLP, 2 hidden layers') },
              { value: 'mlp5', label: t('MLP 5 ชั้นซ่อน', 'MLP, 5 hidden layers') },
              { value: 'dropout', label: t('MLP + Dropout', 'MLP + Dropout') },
            ]} />
        </div>
        <Button size="sm" onClick={undo} disabled={!past.length} title={t('ย้อนกลับ', 'Undo')}><Icon name="undo" /></Button>
        <Button size="sm" onClick={redo} disabled={!future.length} title={t('ทำซ้ำ', 'Redo')}><Icon name="redo" /></Button>
      </div>

      <div className="rounded-xl border border-cyan/25 bg-cyan/[0.035] p-2.5">
        <div className="mb-2 flex items-baseline justify-between gap-2">
          <span className="label-xs text-cyan">{t('กล่องเครื่องมือ Layer', 'Layer toolbox')}</span>
          <span className="text-[0.84375rem] text-muted">{t('ลากไปวางที่เส้นเชื่อม', 'Drag onto a connector')}</span>
        </div>
        <div className="scroll-thin max-h-[310px] space-y-2.5 overflow-y-auto pr-1">
          {(universal ? LAYER_GROUPS : LAYER_GROUPS.slice(0, 1)).map((group) => (
            <div key={group.label}>
              <div className="mb-1 font-mono text-[0.8125rem] uppercase tracking-wider text-muted">{group.label}</div>
              <div className="grid grid-cols-3 gap-1.5">
                {group.items.map(([kind, name, note]) => (
                  <button key={kind} type="button" draggable
                    onDragStart={(e) => { e.dataTransfer.setData('application/x-ml-layer', kind); e.dataTransfer.effectAllowed = 'copy'; setDragKind(kind); }}
                    onDragEnd={() => setDragKind(null)} onClick={() => add(kind)}
                    className="cursor-grab rounded-lg border border-line bg-bg px-2 py-2 text-left transition-colors hover:border-cyan/60 hover:bg-cyan/5 active:cursor-grabbing"
                    title={t(`ลาก ${name} ไปวางระหว่าง Layer หรือคลิกเพื่อเพิ่ม`, `Drag ${name} between layers, or click to add`)}>
                    <span className={`block truncate font-mono text-[0.84375rem] ${KIND_TONE[kind]}`}>{name}</span>
                    <span className="mt-0.5 block truncate font-mono text-[0.8125rem] text-muted">{note}</span>
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>

      <ol className="flex flex-col gap-1" aria-label={t('ชั้นของโครงข่าย', 'Network layers')}>
        {layers.map((l, i) => {
          const movable = l.kind !== 'input' && l.kind !== 'output';
          const info = validation.info[l.id];
          const iss = issuesFor(l.id);
          const hasErr = validation.errors.some((e) => e.layerId === l.id);
          return (
            <Fragment key={l.id}>
            <li draggable={movable}
              onDragStart={(e) => { e.dataTransfer.effectAllowed = 'move'; setDragFrom(i); }} onDragEnd={() => setDragFrom(null)}
              className={`group rounded-lg border transition-colors ${l.id === selectedId ? 'border-cyan/70 bg-cyan/5' : hasErr ? 'border-red/50' : 'border-line hover:border-line/80'} `}>
              <div className="flex items-center gap-1.5 px-2 py-1.5">
                <span className={`cursor-grab text-muted ${movable ? '' : 'invisible'}`} aria-hidden="true"><Icon name="grip" className="h-3.5 w-3.5" /></span>
                <button type="button" onClick={() => select(l.id === selectedId ? null : l.id)} className="flex min-w-0 flex-1 items-baseline gap-2 text-left">
                  <span className={`font-mono text-[0.875rem] uppercase ${KIND_TONE[l.kind]}`}>{l.kind}</span>
                  <span className="truncate text-[1.03125rem]">{l.name}</span>
                  <span className="ml-auto whitespace-nowrap font-mono text-[0.90625rem] text-muted">→{info?.outShape ?? '?'} · {info?.params.toLocaleString() ?? 0}p</span>
                </button>
                {movable && (
                  <span className="flex opacity-70 group-hover:opacity-100">
                    <button type="button" className="p-1 hover:text-cyan disabled:opacity-30" disabled={i <= 1} onClick={() => move(i, i - 1)} title={t('เลื่อนขึ้น', 'Move up')} aria-label={t('เลื่อนขึ้น', 'Move up')}><Icon name="up" className="h-3.5 w-3.5" /></button>
                    <button type="button" className="p-1 hover:text-cyan disabled:opacity-30" disabled={i >= layers.length - 2} onClick={() => move(i, i + 1)} title={t('เลื่อนลง', 'Move down')} aria-label={t('เลื่อนลง', 'Move down')}><Icon name="down" className="h-3.5 w-3.5" /></button>
                    <button type="button" className="p-1 hover:text-cyan" onClick={() => duplicate(l.id)} title={t('ทำสำเนา', 'Duplicate')} aria-label={t('ทำสำเนา', 'Duplicate')}><Icon name="copy" className="h-3.5 w-3.5" /></button>
                    <button type="button" className="p-1 hover:text-red" onClick={() => remove(l.id)} title={t('ลบ', 'Remove')} aria-label={t('ลบ', 'Remove')}><Icon name="trash" className="h-3.5 w-3.5" /></button>
                  </span>
                )}
              </div>
              {iss.length > 0 && (
                <div className="flex flex-col gap-1 border-t border-line/60 px-2.5 py-1.5">
                  {iss.map((x, k) => (
                    <div key={k} className={`flex items-start justify-between gap-2 text-[0.9375rem] ${validation.errors.includes(x) ? 'text-red' : 'text-orange'}`}>
                      <span>{t(x.msg)}</span>
                      {x.fix && <button type="button" className="shrink-0 underline" onClick={() => update(l.id, x.fix!.patch)}>{t(x.fix.label)}</button>}
                    </div>
                  ))}
                </div>
              )}
              {l.id === selectedId && <LayerEditor layer={l} level={level} store={store} />}
            </li>
            {i < layers.length - 1 && (
              <li className={`group/slot flex items-center gap-1 transition-all ${dragKind || dragFrom != null ? 'h-8' : 'h-3'}`}
                onDragOver={(e) => { e.preventDefault(); e.dataTransfer.dropEffect = dragKind ? 'copy' : 'move'; }}
                onDrop={(e) => {
                  e.preventDefault();
                  const kind = e.dataTransfer.getData('application/x-ml-layer') as LayerSpec['kind'];
                  if (BUILDABLE_LAYER_KINDS.includes(kind)) insert(kind, i + 1);
                  else if (dragFrom != null) move(dragFrom, dragFrom < i + 1 ? i : i + 1);
                  setDragKind(null); setDragFrom(null);
                }}>
                <span className="h-px flex-1 bg-line transition-colors group-hover/slot:bg-cyan/70" />
                <span className="grid h-4 w-4 shrink-0 place-items-center rounded-full border border-line bg-bg font-mono text-[0.8125rem] text-muted transition-colors group-hover/slot:border-cyan group-hover/slot:text-cyan">+</span>
                {(dragKind || dragFrom != null) && <span className="whitespace-nowrap text-[0.84375rem] text-cyan">{t('วางเพื่อต่อที่นี่', 'Drop to connect here')}</span>}
                <span className="h-px flex-1 bg-line transition-colors group-hover/slot:bg-cyan/70" />
              </li>
            )}
            </Fragment>
          );
        })}
      </ol>

      <div className="rounded-lg border border-dashed border-line px-2.5 py-2">
        <div className="label-xs mb-1">{universal ? t('โหมด Blueprint', 'Blueprint mode') : t('โหมด Neural Network', 'Neural Network mode')}</div>
        <p className="text-[0.875rem] leading-relaxed text-muted">{universal
          ? t('ประกอบ Conv, Sequence, Attention และ Merge เพื่อบันทึกและส่งออกสถาปัตยกรรมโมเดล', 'Compose Conv, sequence, attention and merge components into an exportable model architecture.')
          : t('หน้านี้ใช้สร้างและฝึก Neural Network แบบ Dense/MLP เท่านั้น', 'This page builds and trains Dense/MLP neural networks only.')}</p>
        <p className="mt-1.5 text-[0.90625rem] text-muted">{t('คลิก Layer เพื่อเพิ่มหลังชั้นที่เลือก หรือลากไปยังเส้นเชื่อม ลาก Layer ในรายการเพื่อจัดลำดับใหม่', 'Click a layer to add it after the selection, drag it onto a connector, or drag an existing layer to reorder it.')}</p>
      </div>

      {validation.errors.filter((e) => !e.layerId).concat(validation.warnings.filter((w) => !w.layerId)).map((x, k) => (
        <p key={k} className={`text-[0.96875rem] ${validation.errors.includes(x) ? 'text-red' : 'text-orange'}`}>{t(x.msg)}</p>
      ))}
      <div className="flex justify-between rounded-lg bg-bg px-2.5 py-2 font-mono text-[0.9375rem] text-muted">
        <span>{t('พารามิเตอร์', 'params')} <b className="text-ink">{validation.totalParams.toLocaleString()}</b></span>
        <span>{t('หน่วยความจำ', 'memory')} ≈ <b className="text-ink">{(validation.memoryBytes / 1024).toFixed(1)} KB</b></span>
      </div>
      {sel == null && <p className="text-[0.9375rem] text-muted">{t('คลิกชื่อชั้นเพื่อแก้พารามิเตอร์', 'Click a layer name to edit its parameters')}</p>}
    </div>
  );
}

function LayerEditor({ layer: l, level, store }: { layer: LayerSpec; level: number; store: ArchStore }) {
  const t = useT();
  const update = store((s) => s.update);
  const u = (patch: Partial<LayerSpec>) => update(l.id, patch);
  return (
    <div className="flex flex-col gap-3 border-t border-line/60 px-2.5 py-3">
      <label className="flex flex-col gap-1 text-[1rem]">
        <span>{t('ชื่อชั้น', 'Layer name')}</span>
        <input value={l.name} maxLength={32} onChange={(e) => u({ name: e.target.value })} className="rounded-md border border-line bg-bg px-2 py-1 text-[1rem]" />
      </label>
      {l.kind === 'input' && <p className="text-[0.9375rem] text-muted">{t('จำนวนอินพุตต้องเท่ากับจำนวนฟีเจอร์ของชุดข้อมูล ระบบตั้งให้อัตโนมัติเมื่อเปลี่ยนชุดข้อมูล', 'Input size must equal the dataset’s feature count; it is set automatically when the dataset changes.')}</p>}
      {(l.kind === 'dense' || l.kind === 'output') && (
        <>
          <NumberField label={t('จำนวนนิวรอน (units)', 'Units')} value={l.units ?? 1} min={1} max={MAX_UNITS} onChange={(v) => u({ units: Math.round(v) })} />
          {l.kind === 'dense' && <Slider label={t('ปรับด้วยแถบ (1–128)', 'Slider (1–128)')} value={Math.min(l.units ?? 1, 128)} min={1} max={128} onChange={(v) => u({ units: v })} />}
          <Select label="Activation" value={l.activation ?? 'linear'} onChange={(v) => u({ activation: v })} options={ACTIVATIONS.map((a) => ({ value: a, label: a }))} />
          <Toggle label={t('ใช้ bias', 'Use bias')} checked={l.useBias !== false} onChange={(v) => u({ useBias: v })} />
          {level >= 2 && (
            <>
              <Select label={t('การกำหนดน้ำหนักเริ่มต้น', 'Kernel initializer')} value={l.kernelInit ?? 'glorot_uniform'} onChange={(v) => u({ kernelInit: v })} options={INITS.map((a) => ({ value: a, label: a }))} />
              <Select label={t('ค่า bias เริ่มต้น', 'Bias initializer')} value={l.biasInit ?? 'zeros'} onChange={(v) => u({ biasInit: v })} options={[{ value: 'zeros', label: 'zeros' }, { value: 'small', label: '0.1' }, { value: 'ones', label: 'ones' }]} />
              <NumberField label="L2 (weight decay)" value={l.l2 ?? 0} min={0} max={1} step={0.0005} onChange={(v) => u({ l2: Math.max(0, v) })} />
              <NumberField label="L1" value={l.l1 ?? 0} min={0} max={1} step={0.0005} onChange={(v) => u({ l1: Math.max(0, v) })} />
            </>
          )}
        </>
      )}
      {l.kind === 'dropout' && <Slider label={t('อัตรา dropout', 'Dropout rate')} value={l.rate ?? 0} min={0} max={0.9} step={0.05} format={(v) => v.toFixed(2)} onChange={(v) => u({ rate: v })} />}
      {l.kind === 'activation' && <Select label="Activation" value={l.activation ?? 'relu'} onChange={(v) => u({ activation: v })} options={ACTIVATIONS.map((a) => ({ value: a, label: a }))} />}
      {(l.kind === 'conv1d' || l.kind === 'conv2d') && (
        <>
          <NumberField label={t('จำนวนฟิลเตอร์', 'Filters')} value={l.filters ?? 32} min={1} max={1024} onChange={(filters) => u({ filters: Math.round(filters) })} />
          <NumberField label={t('ขนาด Kernel', 'Kernel size')} value={l.kernelSize ?? 3} min={1} max={15} onChange={(kernelSize) => u({ kernelSize: Math.round(kernelSize) })} />
          <NumberField label={t('ระยะก้าว', 'Stride')} value={l.stride ?? 1} min={1} max={8} onChange={(stride) => u({ stride: Math.round(stride) })} />
          <Select label="Padding" value={l.padding ?? 'same'} onChange={(padding) => u({ padding })} options={[{ value: 'same', label: 'same' }, { value: 'valid', label: 'valid' }]} />
          <Select label="Activation" value={l.activation ?? 'relu'} onChange={(activation) => u({ activation })} options={ACTIVATIONS.map((a) => ({ value: a, label: a }))} />
        </>
      )}
      {(l.kind === 'maxpool2d' || l.kind === 'avgpool2d') && (
        <>
          <NumberField label={t('ขนาด Pool', 'Pool size')} value={l.poolSize ?? 2} min={1} max={8} onChange={(poolSize) => u({ poolSize: Math.round(poolSize) })} />
          <NumberField label={t('ระยะก้าว', 'Stride')} value={l.stride ?? 2} min={1} max={8} onChange={(stride) => u({ stride: Math.round(stride) })} />
          <Select label="Padding" value={l.padding ?? 'valid'} onChange={(padding) => u({ padding })} options={[{ value: 'same', label: 'same' }, { value: 'valid', label: 'valid' }]} />
        </>
      )}
      {l.kind === 'embedding' && (
        <>
          <NumberField label={t('ขนาดคำศัพท์', 'Vocabulary size')} value={l.vocabularySize ?? 10000} min={2} max={1000000} onChange={(vocabularySize) => u({ vocabularySize: Math.round(vocabularySize) })} />
          <NumberField label={t('มิติ Embedding', 'Embedding dimension')} value={l.embeddingDim ?? 64} min={1} max={4096} onChange={(embeddingDim) => u({ embeddingDim: Math.round(embeddingDim) })} />
        </>
      )}
      {(l.kind === 'rnn' || l.kind === 'lstm' || l.kind === 'gru') && (
        <>
          <NumberField label={t('จำนวนหน่วยซ่อน', 'Hidden units')} value={l.units ?? 64} min={1} max={2048} onChange={(units) => u({ units: Math.round(units) })} />
          <Select label="Activation" value={l.activation ?? 'tanh'} onChange={(activation) => u({ activation })} options={ACTIVATIONS.map((a) => ({ value: a, label: a }))} />
          <Toggle label={t('คืนค่าทุก Time step', 'Return sequences')} checked={l.returnSequences === true} onChange={(returnSequences) => u({ returnSequences })} />
        </>
      )}
      {l.kind === 'attention' && (
        <>
          <NumberField label={t('จำนวน Attention heads', 'Attention heads')} value={l.heads ?? 4} min={1} max={64} onChange={(heads) => u({ heads: Math.round(heads) })} />
          <NumberField label={t('มิติของ Key', 'Key dimension')} value={l.keyDim ?? 32} min={1} max={1024} onChange={(keyDim) => u({ keyDim: Math.round(keyDim) })} />
        </>
      )}
      {(['flatten', 'globalavgpool2d', 'batchnorm', 'layernorm', 'add', 'concatenate'] as LayerSpec['kind'][]).includes(l.kind) && (
        <p className="text-[0.9375rem] text-muted">{t('Layer นี้ไม่มีค่าหลักที่ต้องกำหนดใน Blueprint ปัจจุบัน', 'This layer has no primary parameters in the current blueprint.')}</p>
      )}
    </div>
  );
}
