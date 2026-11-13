import { DATASETS, type DatasetId } from '../../data/datasets';
import { ACTIVATIONS, BUILDABLE_LAYER_KINDS, INITS, RUNTIME_LAYER_KINDS, type LayerSpec, type Loss, type OptimizerName } from './types';

interface DatasetConfig { id: DatasetId; n: number; classes: number; noise: number; seed: number; testRatio: number; valRatio: number; standardize: boolean }
interface TrainingConfig { optimizer: OptimizerName; loss: Loss; lr: number; batchSize: number; momentum: number; beta1: number; beta2: number; epsilon: number; shuffle: boolean; seed: number; earlyStopping: boolean; patience: number; maxEpochs: number }
type Obj = Record<string, unknown>;
const object = (value: unknown): Obj => { if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Expected a configuration object'); return value as Obj; };
function number(value: unknown, key: string, min: number, max: number, integer = false): number {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < min || value > max || (integer && !Number.isInteger(value))) throw new Error(`Invalid ${key}: expected ${integer ? 'an integer' : 'a number'} in ${min}–${max}`);
  return value;
}
function choice(value: unknown, key: string, allowed: readonly string[]): string { if (typeof value !== 'string' || !allowed.includes(value)) throw new Error(`Invalid ${key}`); return value; }

/** Validate before applying any React/store state: malformed imports must be atomic. */
export function parseBuilderConfig(text: string, mode: 'neural' | 'model') {
  const root = object(JSON.parse(text)), allowed = mode === 'model' ? ['input', ...BUILDABLE_LAYER_KINDS, 'output'] : RUNTIME_LAYER_KINDS;
  if (!Array.isArray(root.architecture) || !root.architecture.length || root.architecture.length > 128) throw new Error('Architecture must contain 1–128 layers');
  const ids = new Set<string>();
  const architecture = root.architecture.map((raw): LayerSpec => {
    const layer = object(raw);
    if (typeof layer.id !== 'string' || !layer.id.trim() || ids.has(layer.id)) throw new Error('Layer IDs must be nonempty and unique');
    ids.add(layer.id);
    const kind = choice(layer.kind, 'layer kind', allowed) as LayerSpec['kind'];
    const out: Obj = { id: layer.id, kind, name: typeof layer.name === 'string' ? layer.name : layer.id };
    for (const key of ['units', 'filters', 'kernelSize', 'stride', 'poolSize', 'embeddingDim', 'heads', 'keyDim']) if (layer[key] !== undefined) out[key] = number(layer[key], key, 1, 1024, true);
    if (['input', 'dense', 'output'].includes(kind) && out.units == null) throw new Error('Input, Dense and Output need units');
    for (const key of ['l1', 'l2']) if (layer[key] !== undefined) out[key] = number(layer[key], key, 0, 10);
    if (layer.rate !== undefined) out.rate = number(layer.rate, 'dropout rate', 0, 0.9);
    if (layer.vocabularySize !== undefined) out.vocabularySize = number(layer.vocabularySize, 'vocabularySize', 2, 100000, true);
    for (const [key, options] of [['activation', ACTIVATIONS], ['kernelInit', INITS], ['biasInit', ['zeros', 'ones', 'small']], ['padding', ['same', 'valid']]] as const) if (layer[key] !== undefined) out[key] = choice(layer[key], key, options);
    for (const key of ['useBias', 'returnSequences']) if (layer[key] !== undefined) { if (typeof layer[key] !== 'boolean') throw new Error(`Invalid ${key}`); out[key] = layer[key]; }
    return out as unknown as LayerSpec;
  });
  let dataset: Partial<DatasetConfig> | undefined, training: Partial<TrainingConfig> | undefined;
  if (root.dataset !== undefined) {
    const input = object(root.dataset), out: Obj = {};
    if (input.id !== undefined) out.id = choice(input.id, 'dataset ID', DATASETS.map((d) => d.id));
    for (const [key, min, max, int] of [['n', 40, 800, true], ['classes', 2, 6, true], ['noise', 0, 0.8, false], ['seed', 1, 99, true], ['testRatio', 0.1, 0.5, false], ['valRatio', 0, 0.4, false]] as const) if (input[key] !== undefined) out[key] = number(input[key], `dataset.${key}`, min, max, int);
    if (input.standardize !== undefined) { if (typeof input.standardize !== 'boolean') throw new Error('Invalid standardize'); out.standardize = input.standardize; }
    dataset = out;
  }
  if (root.training !== undefined) {
    const input = object(root.training), out: Obj = {};
    for (const [key, min, max, int] of [['lr', 1e-8, 1, false], ['batchSize', 1, 100000, true], ['momentum', 0, 0.9999, false], ['beta1', 0, 0.9999, false], ['beta2', 0, 0.99999, false], ['epsilon', 1e-12, 1, false], ['seed', 1, 99, true], ['patience', 1, 5000, true], ['maxEpochs', 1, 5000, true]] as const) if (input[key] !== undefined) out[key] = number(input[key], `training.${key}`, min, max, int);
    if (input.optimizer !== undefined) out.optimizer = choice(input.optimizer, 'optimizer', ['sgd', 'momentum', 'rmsprop', 'adam']);
    if (input.loss !== undefined) out.loss = choice(input.loss, 'loss', ['bce', 'cce', 'mse']);
    for (const key of ['shuffle', 'earlyStopping']) if (input[key] !== undefined) { if (typeof input[key] !== 'boolean') throw new Error(`Invalid ${key}`); out[key] = input[key]; }
    training = out;
  }
  return { architecture, dataset, training };
}
