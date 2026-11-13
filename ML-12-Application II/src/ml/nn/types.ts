import type { Text } from '../../i18n/i18n';

export type Activation = 'relu' | 'leakyrelu' | 'sigmoid' | 'tanh' | 'gelu' | 'softmax' | 'linear';
export type Init = 'glorot_uniform' | 'he_normal' | 'lecun_normal' | 'random_normal' | 'zeros';
export type BiasInit = 'zeros' | 'ones' | 'small';
export type Loss = 'bce' | 'cce' | 'mse';
export type OptimizerName = 'sgd' | 'momentum' | 'rmsprop' | 'adam';

export type LayerKind =
  | 'input' | 'dense' | 'dropout' | 'activation' | 'output'
  | 'conv1d' | 'conv2d' | 'maxpool2d' | 'avgpool2d' | 'globalavgpool2d' | 'flatten'
  | 'batchnorm' | 'layernorm'
  | 'embedding' | 'rnn' | 'lstm' | 'gru' | 'attention'
  | 'add' | 'concatenate';

export const RUNTIME_LAYER_KINDS: LayerKind[] = ['input', 'dense', 'dropout', 'activation', 'output'];
export const BUILDABLE_LAYER_KINDS: LayerKind[] = [
  'dense', 'dropout', 'activation',
  'conv1d', 'conv2d', 'maxpool2d', 'avgpool2d', 'globalavgpool2d', 'flatten',
  'batchnorm', 'layernorm',
  'embedding', 'rnn', 'lstm', 'gru', 'attention',
  'add', 'concatenate',
];
export const isRuntimeLayer = (l: LayerSpec) => RUNTIME_LAYER_KINDS.includes(l.kind);

export interface LayerSpec {
  id: string;
  kind: LayerKind;
  name: string;
  /** input: feature count; dense/output: neurons. */
  units?: number;
  activation?: Activation;
  useBias?: boolean;
  kernelInit?: Init;
  biasInit?: BiasInit;
  l1?: number;
  l2?: number;
  /** dropout rate 0–0.9 */
  rate?: number;
  /** convolution / pooling options */
  filters?: number;
  kernelSize?: number;
  stride?: number;
  padding?: 'valid' | 'same';
  poolSize?: number;
  /** sequence and attention options */
  vocabularySize?: number;
  embeddingDim?: number;
  returnSequences?: boolean;
  heads?: number;
  keyDim?: number;
}

export const ACTIVATIONS: Activation[] = ['relu', 'leakyrelu', 'sigmoid', 'tanh', 'gelu', 'softmax', 'linear'];
export const INITS: Init[] = ['glorot_uniform', 'he_normal', 'lecun_normal', 'random_normal', 'zeros'];

/** Descriptions used by the universal architecture toolbox. */
export const PLANNED_LAYERS: { name: string; phase: number; note: Text }[] = [
  { name: 'Conv1D', phase: 4, note: { th: 'ต้องใช้ข้อมูลลำดับ', en: 'Needs sequence data' } },
  { name: 'Conv2D', phase: 4, note: { th: 'อยู่ใน CNN Lab', en: 'Part of the CNN Lab' } },
  { name: 'DepthwiseConv2D', phase: 4, note: { th: 'อยู่ใน CNN Lab', en: 'Part of the CNN Lab' } },
  { name: 'SeparableConv2D', phase: 4, note: { th: 'อยู่ใน CNN Lab', en: 'Part of the CNN Lab' } },
  { name: 'MaxPooling2D', phase: 4, note: { th: 'อยู่ใน CNN Lab', en: 'Part of the CNN Lab' } },
  { name: 'AveragePooling2D', phase: 4, note: { th: 'อยู่ใน CNN Lab', en: 'Part of the CNN Lab' } },
  { name: 'GlobalAveragePooling2D', phase: 4, note: { th: 'อยู่ใน CNN Lab', en: 'Part of the CNN Lab' } },
  { name: 'Flatten', phase: 4, note: { th: 'จำเป็นเมื่อมีชั้น Conv', en: 'Needed once Conv layers exist' } },
  { name: 'BatchNormalization', phase: 3, note: { th: 'รอบถัดไปของ Phase 3', en: 'Next Phase 3 iteration' } },
  { name: 'LayerNormalization', phase: 3, note: { th: 'รอบถัดไปของ Phase 3', en: 'Next Phase 3 iteration' } },
  { name: 'Embedding', phase: 5, note: { th: 'ต้องใช้ข้อมูลข้อความ', en: 'Needs token data' } },
  { name: 'SimpleRNN', phase: 5, note: { th: 'ต้องใช้ข้อมูลลำดับ', en: 'Needs sequence data' } },
  { name: 'LSTM', phase: 5, note: { th: 'ต้องใช้ข้อมูลลำดับ', en: 'Needs sequence data' } },
  { name: 'GRU', phase: 5, note: { th: 'ต้องใช้ข้อมูลลำดับ', en: 'Needs sequence data' } },
  { name: 'MultiHeadAttention', phase: 5, note: { th: 'ต้องใช้ข้อมูลลำดับ', en: 'Needs sequence data' } },
  { name: 'Add / Concatenate', phase: 5, note: { th: 'ต้องรองรับกราฟแบบแตกกิ่ง', en: 'Needs non-sequential graphs' } },
];

let uid = 0;
export const newId = () => `L${Date.now().toString(36)}${(uid++).toString(36)}`;

export const isParamLayer = (l: LayerSpec) => l.kind === 'dense' || l.kind === 'output';
