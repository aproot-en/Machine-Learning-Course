import type { VisionBackbone, VisionTrace, VisionTrainer, VTensor } from './vision';

export interface VisionNetworkLayer {
  nodeIndex: number | null; name: string; tensor: VTensor; inputs: number[];
  role: 'input' | 'feature' | 'scaling' | 'logits' | 'output'; weights?: Float64Array;
}

/** Keep image, full model path, and classifier output visible while expanding the inspected block. */
export function visionNetwork(backbone: VisionBackbone, trace: VisionTrace, trainer: VisionTrainer, selected: number): VisionNetworkLayer[] {
  const group = backbone.nodes[selected].group;
  const indices = [...new Set([0, ...backbone.overview, ...visionGroupIndices(backbone, group), backbone.nodes.length - 1])].sort((a, b) => a - b);
  const visible = new Map(indices.map((node, i) => [node, i])), memo = new Map<number, number[]>();
  const ancestors = (node: number): number[] => {
    if (visible.has(node)) return [visible.get(node)!];
    if (!memo.has(node)) memo.set(node, [...new Set(backbone.nodes[node].inputs.flatMap(ancestors))]);
    return memo.get(node)!;
  };
  const layers: VisionNetworkLayer[] = indices.map((nodeIndex) => {
    const node = backbone.nodes[nodeIndex];
    return { nodeIndex, name: nodeIndex === 0 ? 'Input image' : `${node.group} · ${node.name}`, tensor: trace.tensors[nodeIndex], inputs: [...new Set(node.inputs.flatMap(ancestors))], role: nodeIndex === 0 ? 'input' : 'feature' };
  });
  const vector = (a: Float64Array): VTensor => ({ h: 1, w: 1, c: a.length, a });
  layers.push({ nodeIndex: null, name: 'Feature scaling', tensor: vector(trainer.normalize(trace.features)), inputs: [layers.length - 1], role: 'scaling' });
  layers.push({ nodeIndex: null, name: 'Classifier', tensor: vector(trainer.logits(trace.features)), inputs: [layers.length - 1], role: 'logits', weights: trainer.W });
  layers.push({ nodeIndex: null, name: 'Output · softmax', tensor: vector(trainer.predict(trace.features)), inputs: [layers.length - 1], role: 'output' });
  return layers;
}

export interface VisionTerm { label: string; value: number; weight: number; parent: number; index: number }
export interface VisionCalculation {
  terms: VisionTerm[]; bias: number; aggregate: number; result: number; actual: number;
  formula: string; statistics?: string;
}

export function visionGroupIndices(backbone: VisionBackbone, group: string): number[] {
  if (group === 'overview') return backbone.overview;
  const inside = backbone.nodes.flatMap((node, i) => node.group === group ? [i] : []);
  const members = new Set(inside);
  const dependencies = inside.flatMap((i) => backbone.nodes[i].inputs.filter((parent) => !members.has(parent)));
  return [...new Set([...dependencies, ...inside])].sort((a, b) => a - b);
}

/** Explain one CHW output cell using the same parameters and tensor trace as inference. */
export function visionCalculation(backbone: VisionBackbone, trace: VisionTrace, nodeIndex: number, outputIndex: number): VisionCalculation {
  const node = backbone.nodes[nodeIndex], output = trace.tensors[nodeIndex];
  const index = Math.max(0, Math.min(output.a.length - 1, Math.trunc(outputIndex)));
  const spatial = output.h * output.w, channel = Math.floor(index / spatial), position = index % spatial;
  const row = Math.floor(position / output.w), col = position % output.w;
  const parent = node.inputs[0], x = trace.tensors[parent], terms: VisionTerm[] = [];
  let bias = 0, formula = 'y = x', statistics: string | undefined;
  const term = (p: number, i: number, weight = 1, label = `#${p}[${i}]`) => terms.push({ parent: p, index: i, value: i < 0 ? 0 : trace.tensors[p].a[i], weight, label });
  if (node.op === 'input') term(nodeIndex, index);
  else if (node.op === 'conv' || node.op === 'depthwise') {
    const kh = node.kh!, kw = node.kw!, stride = node.stride!, depthwise = node.op === 'depthwise';
    bias = node.b![channel]; formula = 'y = Σ(input × kernel) + bias';
    for (let c = depthwise ? channel : 0; c < (depthwise ? channel + 1 : x.c); c++) for (let r = 0; r < kh; r++) for (let k = 0; k < kw; k++) {
      const ir = row * stride + r - Math.floor(kh / 2), ic = col * stride + k - Math.floor(kw / 2);
      const source = ir >= 0 && ir < x.h && ic >= 0 && ic < x.w ? (c * x.h + ir) * x.w + ic : -1;
      term(parent, source, node.W![((channel * (depthwise ? 1 : x.c) + (depthwise ? 0 : c)) * kh + r) * kw + k], `c${c} (${ir},${ic})${source < 0 ? ' padding' : ''}`);
    }
  } else if (node.op === 'pool') {
    const k = node.kh!, stride = node.stride!, pad = stride === 1 ? 1 : 0;
    formula = node.mode === 'max' ? 'y = max(valid window)' : 'y = mean(valid window)';
    for (let r = 0; r < k; r++) for (let c = 0; c < k; c++) {
      const ir = row * stride + r - pad, ic = col * stride + c - pad;
      if (ir >= 0 && ir < x.h && ic >= 0 && ic < x.w) term(parent, (channel * x.h + ir) * x.w + ic, 1, `c${channel} (${ir},${ic})`);
    }
  } else if (node.op === 'add') {
    formula = 'y = F(x) + shortcut(x)'; node.inputs.forEach((p) => term(p, index));
  } else if (node.op === 'concat') {
    formula = 'y = concat(branches, channels)'; let offset = 0;
    for (const p of node.inputs) { const length = trace.tensors[p].a.length; if (index < offset + length) { term(p, index - offset); break; } offset += length; }
  } else if (node.op === 'gap') {
    formula = 'y[c] = Σ spatial x[c] / (H × W)';
    for (let p = 0; p < x.h * x.w; p++) term(parent, channel * x.h * x.w + p, 1 / (x.h * x.w));
  } else if (node.op === 'dense') {
    const positions = output.h, nin = positions > 1 ? x.c : x.a.length;
    bias = node.b![channel]; formula = 'y = Σ(input × W) + bias';
    for (let c = 0; c < nin; c++) term(parent, c * positions + position, node.W![channel * nin + c]);
  } else if (node.op === 'patch') {
    const p = node.patch!, tokens = output.h, token = position;
    bias = node.b![channel * tokens + token]; formula = token === 0 ? 'CLS = learned token + position' : 'token = Σ(patch pixel × W) + position';
    if (token > 0) for (let r = 0; r < p; r++) for (let c = 0; c < p; c++) term(parent, (Math.floor((token - 1) / (16 / p)) * p + r) * 16 + ((token - 1) % (16 / p)) * p + c, node.W![channel * p * p + r * p + c]);
  } else if (node.op === 'attention') {
    const d = x.c, tokens = x.h, dk = d / node.heads!;
    bias = node.b![3 * d + channel]; formula = 'A = softmax(QKᵀ / √dₖ); y = concat(A V) Wₒ + bₒ';
    for (let j = 0; j < d; j++) for (let key = 0; key < tokens; key++) {
      let value = node.b![2 * d + j];
      for (let c = 0; c < d; c++) value += node.W![(2 * d + j) * d + c] * x.a[c * tokens + key];
      const attention = trace.attention.get(nodeIndex)![Math.floor(j / dk)][position * tokens + key];
      terms.push({ label: `V[c${j}, token${key}]`, value, weight: attention * node.W![(3 * d + channel) * d + j], parent, index: -1 });
    }
    statistics = `Each coefficient = attention(query ${position}, key) × output projection weight`;
  } else if (node.op === 'ln') {
    let mean = 0, variance = 0;
    for (let c = 0; c < x.c; c++) mean += x.a[c * spatial + position] / x.c;
    for (let c = 0; c < x.c; c++) variance += (x.a[c * spatial + position] - mean) ** 2 / x.c;
    const scale = 1 / Math.sqrt(variance + 1e-6);
    term(parent, index, scale); bias = -mean * scale; formula = 'y = (x − mean) / √(variance + 10⁻⁶)'; statistics = `mean=${mean.toPrecision(6)}, variance=${variance.toPrecision(6)}`;
  } else if (node.op === 'cls') { formula = 'y[c] = tokens[CLS, c]'; term(parent, channel * x.h * x.w); }
  else { term(parent, index, node.op === 'bn' ? 1 / Math.sqrt(1 + 1e-5) : 1); formula = node.op === 'bn' ? 'y = x / √(1 + 10⁻⁵) (frozen initial BatchNorm)' : node.op === 'flatten' ? 'y = reshape(x)' : node.op === 'relu' ? 'y = max(0, x)' : node.op === 'relu6' ? 'y = min(6, max(0, x))' : 'y = GELU(x)'; }
  let aggregate = terms.reduce((sum, t) => sum + t.value * t.weight, bias);
  if (node.op === 'pool') aggregate = node.mode === 'max' ? (terms.length ? Math.max(...terms.map((t) => t.value)) : 0) : aggregate / Math.max(1, terms.length);
  const result = node.op === 'relu' ? Math.max(0, aggregate) : node.op === 'relu6' ? Math.min(6, Math.max(0, aggregate)) : node.op === 'gelu' ? 0.5 * aggregate * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (aggregate + 0.044715 * aggregate ** 3))) : aggregate;
  return { terms, bias, aggregate, result, actual: output.a[index], formula, statistics };
}
