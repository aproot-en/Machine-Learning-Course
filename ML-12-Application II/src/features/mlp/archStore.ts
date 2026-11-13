import { create } from 'zustand';
import { newId, type Activation, type LayerKind, type LayerSpec, type Loss } from '../../ml/nn/types';

export type PresetName = 'perceptron' | 'mlp2' | 'mlp5' | 'dropout' | 'cnn' | 'sequence' | 'transformer';

export function outputFor(nClasses: number, loss: Loss): Pick<LayerSpec, 'units' | 'activation'> {
  if (loss === 'bce') return { units: 1, activation: 'sigmoid' };
  if (loss === 'cce') return { units: nClasses, activation: 'softmax' };
  return { units: nClasses === 2 ? 1 : nClasses, activation: nClasses === 2 ? 'sigmoid' : 'softmax' };
}

const dense = (units: number, activation: Activation, name = 'Dense'): LayerSpec => ({ id: newId(), kind: 'dense', name, units, activation, useBias: true, kernelInit: activation === 'relu' || activation === 'leakyrelu' ? 'he_normal' : 'glorot_uniform', biasInit: 'zeros', l1: 0, l2: 0 });

export function presetLayers(p: PresetName, nFeatures: number, nClasses: number, loss: Loss): LayerSpec[] {
  const input: LayerSpec = { id: newId(), kind: 'input', name: 'Input', units: nFeatures };
  const output: LayerSpec = { id: newId(), kind: 'output', name: 'Output', useBias: true, kernelInit: 'glorot_uniform', biasInit: 'zeros', l1: 0, l2: 0, ...outputFor(nClasses, loss) };
  const hidden: LayerSpec[] =
    p === 'perceptron' ? []
      : p === 'mlp2' ? [dense(8, 'tanh', 'Hidden 1'), dense(8, 'tanh', 'Hidden 2')]
        : p === 'mlp5' ? [1, 2, 3, 4, 5].map((i) => dense(12, 'relu', `Hidden ${i}`))
          : p === 'dropout' ? [dense(32, 'relu', 'Hidden 1'), { id: newId(), kind: 'dropout', name: 'Dropout', rate: 0.2 }, dense(32, 'relu', 'Hidden 2')]
            : p === 'cnn' ? [
              createLayer('conv2d', []), createLayer('batchnorm', []), createLayer('activation', []),
              createLayer('maxpool2d', []), createLayer('conv2d', [{ kind: 'conv2d' } as LayerSpec]),
              createLayer('globalavgpool2d', []), dense(32, 'relu', 'Classifier'),
            ]
              : p === 'sequence' ? [
                createLayer('embedding', []), createLayer('lstm', []), createLayer('dropout', []), dense(32, 'relu', 'Classifier'),
              ]
                : [
                  createLayer('embedding', []), createLayer('attention', []), createLayer('add', []),
                  createLayer('layernorm', []), dense(64, 'gelu', 'Feed Forward'), createLayer('layernorm', [{ kind: 'layernorm' } as LayerSpec]),
                ];
  return [input, ...hidden, output];
}

interface ArchState {
  layers: LayerSpec[];
  past: LayerSpec[][];
  future: LayerSpec[][];
  selectedId: string | null;
  set: (layers: LayerSpec[], record?: boolean) => void;
  undo: () => void;
  redo: () => void;
  select: (id: string | null) => void;
  update: (id: string, patch: Partial<LayerSpec>) => void;
  add: (kind: LayerKind) => void;
  insert: (kind: LayerKind, at: number) => void;
  remove: (id: string) => void;
  duplicate: (id: string) => void;
  move: (from: number, to: number) => void;
}

const LIMIT = 100;

function createLayer(kind: LayerKind, layers: LayerSpec[]): LayerSpec {
  const n = layers.filter((l) => l.kind === kind).length + 1;
  const id = newId();
  switch (kind) {
    case 'dense': return dense(8, 'relu', `Dense ${n}`);
    case 'dropout': return { id, kind, name: `Dropout ${n}`, rate: 0.2 };
    case 'activation': return { id, kind, name: `Activation ${n}`, activation: 'relu' };
    case 'conv1d': return { id, kind, name: `Conv1D ${n}`, filters: 32, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' };
    case 'conv2d': return { id, kind, name: `Conv2D ${n}`, filters: 32, kernelSize: 3, stride: 1, padding: 'same', activation: 'relu' };
    case 'maxpool2d': return { id, kind, name: `MaxPool2D ${n}`, poolSize: 2, stride: 2, padding: 'valid' };
    case 'avgpool2d': return { id, kind, name: `AveragePool2D ${n}`, poolSize: 2, stride: 2, padding: 'valid' };
    case 'globalavgpool2d': return { id, kind, name: `GlobalAveragePool ${n}` };
    case 'flatten': return { id, kind, name: `Flatten ${n}` };
    case 'batchnorm': return { id, kind, name: `BatchNorm ${n}` };
    case 'layernorm': return { id, kind, name: `LayerNorm ${n}` };
    case 'embedding': return { id, kind, name: `Embedding ${n}`, vocabularySize: 10000, embeddingDim: 64 };
    case 'rnn': return { id, kind, name: `SimpleRNN ${n}`, units: 64, activation: 'tanh', returnSequences: false };
    case 'lstm': return { id, kind, name: `LSTM ${n}`, units: 64, activation: 'tanh', returnSequences: false };
    case 'gru': return { id, kind, name: `GRU ${n}`, units: 64, activation: 'tanh', returnSequences: false };
    case 'attention': return { id, kind, name: `Multi-Head Attention ${n}`, heads: 4, keyDim: 32 };
    case 'add': return { id, kind, name: `Add ${n}` };
    case 'concatenate': return { id, kind, name: `Concatenate ${n}` };
    default: return { id, kind: 'activation', name: `Layer ${n}`, activation: 'linear' };
  }
}

function makeArchStore(initial: LayerSpec[]) {
  return create<ArchState>((setState, get) => {
  const commit = (layers: LayerSpec[]) => setState((s) => ({ layers, past: [...s.past, s.layers].slice(-LIMIT), future: [] }));
  return {
    layers: initial,
    past: [],
    future: [],
    selectedId: null,
    set: (layers, record = true) => (record ? commit(layers) : setState({ layers })),
    undo: () => setState((s) => (s.past.length ? { layers: s.past[s.past.length - 1], past: s.past.slice(0, -1), future: [s.layers, ...s.future] } : s)),
    redo: () => setState((s) => (s.future.length ? { layers: s.future[0], future: s.future.slice(1), past: [...s.past, s.layers] } : s)),
    select: (selectedId) => setState({ selectedId }),
    update: (id, patch) => commit(get().layers.map((l) => (l.id === id ? { ...l, ...patch } : l))),
    add: (kind) => {
      const { layers, selectedId } = get();
      const sel = layers.findIndex((l) => l.id === selectedId);
      // insert after the selected layer, but never before Input or after Output
      const at = Math.min(Math.max(sel + 1, 1), layers.length - 1);
      const layer = createLayer(kind, layers);
      commit([...layers.slice(0, at), layer, ...layers.slice(at)]);
      setState({ selectedId: layer.id });
    },
    insert: (kind, requestedAt) => {
      if (kind === 'input' || kind === 'output') return;
      const layers = get().layers;
      const at = Math.min(Math.max(requestedAt, 1), layers.length - 1);
      const layer = createLayer(kind, layers);
      commit([...layers.slice(0, at), layer, ...layers.slice(at)]);
      setState({ selectedId: layer.id });
    },
    remove: (id) => {
      const l = get().layers.find((x) => x.id === id);
      if (!l || l.kind === 'input' || l.kind === 'output') return;
      commit(get().layers.filter((x) => x.id !== id));
      setState({ selectedId: null });
    },
    duplicate: (id) => {
      const layers = get().layers;
      const i = layers.findIndex((l) => l.id === id);
      const l = layers[i];
      if (!l || l.kind === 'input' || l.kind === 'output') return;
      const copy = { ...l, id: newId(), name: `${l.name} copy` };
      commit([...layers.slice(0, i + 1), copy, ...layers.slice(i + 1)]);
      setState({ selectedId: copy.id });
    },
    move: (from, to) => {
      const layers = [...get().layers];
      // hidden layers only move between Input (0) and Output (last)
      if (from <= 0 || from >= layers.length - 1 || to <= 0 || to >= layers.length - 1 || from === to) return;
      const [l] = layers.splice(from, 1);
      layers.splice(to, 0, l);
      commit(layers);
    },
    };
  });
}

/** Trainable MLP architecture, kept separate from universal model blueprints. */
export const useArch = makeArchStore(presetLayers('mlp2', 2, 2, 'bce'));
/** Independent state for Conv/sequence/attention architecture blueprints. */
export const useModelArch = makeArchStore(presetLayers('cnn', 2, 2, 'bce'));
export type ArchStore = typeof useArch;
