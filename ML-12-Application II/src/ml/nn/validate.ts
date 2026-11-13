import type { Text } from '../../i18n/i18n';
import { isRuntimeLayer, type LayerSpec, type Loss } from './types';

export interface Issue {
  layerId?: string;
  msg: Text;
  /** optional one-click fix the builder can offer */
  fix?: { label: Text; patch: Partial<LayerSpec> };
}

export interface LayerInfo {
  inShape: number;
  outShape: number;
  params: number;
  /** connections drawn between this layer and the previous one */
  edges: number;
}

export interface Validation {
  ok: boolean;
  /** true when every layer is implemented by the in-browser training engine */
  runnable: boolean;
  errors: Issue[];
  warnings: Issue[];
  info: Record<string, LayerInfo>;
  totalParams: number;
  /** float64 bytes for weights + optimiser state (Adam keeps 2 extra copies) */
  memoryBytes: number;
}

export const MAX_UNITS = 1024;
export const MAX_TRAINABLE_PARAMS = 150_000;

export function validateArchitecture(layers: LayerSpec[], task: { nFeatures: number; nClasses: number }, loss: Loss): Validation {
  const errors: Issue[] = [];
  const warnings: Issue[] = [];
  const info: Record<string, LayerInfo> = {};
  let total = 0;

  if (!layers.length || layers[0].kind !== 'input') errors.push({ msg: { th: 'ชั้นแรกต้องเป็น Input', en: 'The first layer must be Input' } });
  if (layers.filter((l) => l.kind === 'input').length > 1) errors.push({ msg: { th: 'มีชั้น Input ได้เพียงชั้นเดียว', en: 'Only one Input layer is allowed' } });
  const outs = layers.filter((l) => l.kind === 'output');
  if (outs.length !== 1 || layers[layers.length - 1]?.kind !== 'output')
    errors.push({ msg: { th: 'ชั้นสุดท้ายต้องเป็น Output และมีได้ชั้นเดียว', en: 'The last layer must be the single Output layer' } });

  let width = 0;
  let linearRun = 0;
  let sigmoidHidden = 0;
  layers.forEach((l, i) => {
    const inShape = width;
    let params = 0;
    let edges = 0;
    if (l.kind === 'input') {
      if (l.units !== task.nFeatures)
        errors.push({ layerId: l.id, msg: { th: `Input ต้องมี ${task.nFeatures} ค่าให้ตรงกับจำนวนฟีเจอร์ของชุดข้อมูล (ตอนนี้ ${l.units})`, en: `Input must have ${task.nFeatures} units to match the dataset features (now ${l.units})` }, fix: { label: { th: `ตั้งเป็น ${task.nFeatures}`, en: `Set to ${task.nFeatures}` }, patch: { units: task.nFeatures } } });
      width = l.units ?? 0;
    } else if (l.kind === 'dense' || l.kind === 'output') {
      const u = l.units ?? 0;
      if (!Number.isInteger(u) || u < 1 || u > MAX_UNITS)
        errors.push({ layerId: l.id, msg: { th: `จำนวนนิวรอนต้องอยู่ระหว่าง 1–${MAX_UNITS}`, en: `Units must be between 1 and ${MAX_UNITS}` } });
      params = u * width + (l.useBias === false ? 0 : u);
      edges = u * width;
      width = u;
      if (l.kind === 'dense') {
        if (l.activation === 'softmax') warnings.push({ layerId: l.id, msg: { th: 'softmax ในชั้นซ่อนทำให้ผลรวมของนิวรอนถูกบังคับเป็น 1 มักไม่ใช่สิ่งที่ต้องการ', en: 'Softmax in a hidden layer forces activations to sum to 1; rarely intended' } });
        if (l.activation === 'linear') {
          linearRun++;
          if (linearRun >= 2) warnings.push({ layerId: l.id, msg: { th: 'Dense linear สองชั้นติดกันยุบรวมเป็นชั้นเดียวได้ทางคณิตศาสตร์ ไม่ได้เพิ่มความสามารถ', en: 'Two consecutive linear Dense layers collapse into one; no extra capacity' } });
        } else linearRun = 0;
        if (l.activation === 'sigmoid') sigmoidHidden++;
      }
    } else if (l.kind === 'dropout') {
      const r = l.rate ?? 0;
      if (r < 0 || r > 0.9) errors.push({ layerId: l.id, msg: { th: 'อัตรา dropout ต้องอยู่ระหว่าง 0–0.9', en: 'Dropout rate must be 0–0.9' } });
      if (i === layers.length - 2) warnings.push({ layerId: l.id, msg: { th: 'Dropout ก่อนชั้น Output โดยตรงทำให้การฝึกแกว่งมาก', en: 'Dropout right before the output layer makes training noisy' } });
    } else if (l.kind === 'conv1d' || l.kind === 'conv2d') {
      const filters = l.filters ?? 0;
      const kernel = l.kernelSize ?? 0;
      if (filters < 1 || kernel < 1) errors.push({ layerId: l.id, msg: { th: 'Conv ต้องมี filters และ kernel size มากกว่า 0', en: 'Conv filters and kernel size must be greater than zero' } });
      params = Math.max(0, filters * (kernel ** (l.kind === 'conv2d' ? 2 : 1)) * Math.max(1, width) + filters);
      width = filters;
      edges = params;
    } else if (l.kind === 'maxpool2d' || l.kind === 'avgpool2d') {
      if ((l.poolSize ?? 0) < 1) errors.push({ layerId: l.id, msg: { th: 'Pool size ต้องมากกว่า 0', en: 'Pool size must be greater than zero' } });
    } else if (l.kind === 'embedding') {
      const vocab = l.vocabularySize ?? 0;
      const dim = l.embeddingDim ?? 0;
      if (vocab < 2 || dim < 1) errors.push({ layerId: l.id, msg: { th: 'Embedding ต้องมี vocabulary และ dimension ที่ถูกต้อง', en: 'Embedding needs a valid vocabulary size and dimension' } });
      params = Math.max(0, vocab * dim);
      width = dim;
    } else if (l.kind === 'rnn' || l.kind === 'lstm' || l.kind === 'gru') {
      const units = l.units ?? 0;
      if (units < 1) errors.push({ layerId: l.id, msg: { th: 'Recurrent layer ต้องมี units มากกว่า 0', en: 'A recurrent layer needs at least one unit' } });
      const gates = l.kind === 'lstm' ? 4 : l.kind === 'gru' ? 3 : 1;
      params = gates * units * (Math.max(1, width) + units + 1);
      width = units;
    } else if (l.kind === 'attention') {
      const heads = l.heads ?? 0;
      const keyDim = l.keyDim ?? 0;
      if (heads < 1 || keyDim < 1) errors.push({ layerId: l.id, msg: { th: 'Attention ต้องมี heads และ key dimension มากกว่า 0', en: 'Attention needs positive head and key dimensions' } });
      const modelDim = heads * keyDim;
      params = 4 * Math.max(1, width) * modelDim;
      width = modelDim;
    } else if (l.kind === 'concatenate') {
      width = Math.max(1, width) * 2;
    }
    if (i > 0 && l.kind === 'input') errors.push({ layerId: l.id, msg: { th: 'Input ต้องอยู่ตำแหน่งแรก', en: 'Input must be first' } });
    total += params;
    info[l.id] = { inShape, outShape: width, params, edges };
  });

  if (sigmoidHidden >= 4) warnings.push({ msg: { th: 'ชั้นซ่อน sigmoid หลายชั้นเสี่ยงต่อ vanishing gradient ลองใช้ ReLU หรือ tanh', en: 'Many sigmoid hidden layers risk vanishing gradients; try ReLU or tanh' } });

  const out = outs[0];
  if (out) {
    const u = out.units ?? 0;
    const act = out.activation;
    const K = task.nClasses;
    if (loss === 'cce') {
      if (act !== 'softmax') errors.push({ layerId: out.id, msg: { th: 'Categorical cross-entropy ต้องใช้ Output แบบ softmax', en: 'Categorical cross-entropy needs a softmax output' }, fix: { label: { th: 'ใช้ softmax', en: 'Use softmax' }, patch: { activation: 'softmax' } } });
      if (u !== K) errors.push({ layerId: out.id, msg: { th: `Output ต้องมี ${K} นิวรอน (หนึ่งต่อคลาส)`, en: `Output needs ${K} units (one per class)` }, fix: { label: { th: `ตั้งเป็น ${K}`, en: `Set to ${K}` }, patch: { units: K } } });
    } else if (loss === 'bce') {
      if (K !== 2) errors.push({ layerId: out.id, msg: { th: `Binary cross-entropy ใช้กับ 2 คลาสเท่านั้น ชุดข้อมูลนี้มี ${K} คลาส`, en: `Binary cross-entropy is for 2 classes; this dataset has ${K}` } });
      if (act !== 'sigmoid') errors.push({ layerId: out.id, msg: { th: 'Binary cross-entropy ต้องใช้ Output แบบ sigmoid', en: 'Binary cross-entropy needs a sigmoid output' }, fix: { label: { th: 'ใช้ sigmoid', en: 'Use sigmoid' }, patch: { activation: 'sigmoid' } } });
      if (u !== 1) errors.push({ layerId: out.id, msg: { th: 'Binary cross-entropy ใช้ Output 1 นิวรอน', en: 'Binary cross-entropy uses 1 output unit' }, fix: { label: { th: 'ตั้งเป็น 1', en: 'Set to 1' }, patch: { units: 1 } } });
    } else {
      if (u !== K && !(K === 2 && u === 1)) errors.push({ layerId: out.id, msg: { th: `MSE ต้องการ Output ${K} นิวรอน (one-hot) หรือ 1 นิวรอนเมื่อมี 2 คลาส`, en: `MSE needs ${K} output units (one-hot), or 1 unit for 2 classes` }, fix: { label: { th: `ตั้งเป็น ${K}`, en: `Set to ${K}` }, patch: { units: K } } });
      warnings.push({ msg: { th: 'MSE ใช้กับการจำแนกได้ แต่มักเรียนรู้ช้ากว่า cross-entropy', en: 'MSE works for classification but usually learns slower than cross-entropy' } });
    }
  }

  const runtimeOnly = layers.every(isRuntimeLayer);
  if (total > MAX_TRAINABLE_PARAMS && runtimeOnly) errors.push({ msg: { th: `พารามิเตอร์ ${total.toLocaleString()} ตัว เกินขีดจำกัดการฝึกในเบราว์เซอร์ (${MAX_TRAINABLE_PARAMS.toLocaleString()})`, en: `${total.toLocaleString()} parameters exceeds the in-browser training limit (${MAX_TRAINABLE_PARAMS.toLocaleString()})` } });
  else if (total > 20_000) warnings.push({ msg: { th: 'โมเดลใหญ่ การฝึกจะช้าลง และแสดงเส้นเชื่อมเพียงบางส่วน', en: 'Large model: training slows down and only a sample of connections is drawn' } });

  const ok = errors.length === 0;
  const runnable = ok && runtimeOnly;
  if (ok && !runnable) warnings.push({ msg: { th: 'Blueprint นี้สร้างและส่งออกได้ แต่มี Layer ที่เอนจินฝึกในเบราว์เซอร์ยังไม่รองรับ', en: 'This blueprint can be built and exported, but includes layers not yet supported by the in-browser training engine' } });
  return { ok, runnable, errors, warnings, info, totalParams: total, memoryBytes: total * 8 * 3 };
}
