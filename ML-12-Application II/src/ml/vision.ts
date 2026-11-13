import { Rng } from '../lib/random';
import { categoricalCrossEntropy } from './losses';
import { classificationReport, type ClassificationReport } from './metrics';
import { Optimizer } from './nn/optimizer';
import type { VisionId } from '../content/visionModels';

export interface VShape { h: number; w: number; c: number }
export interface VTensor extends VShape { a: Float64Array }
export type VOp = 'input' | 'conv' | 'depthwise' | 'relu' | 'relu6' | 'gelu' | 'bn' | 'ln' | 'pool' | 'add' | 'concat' | 'gap' | 'dense' | 'flatten' | 'patch' | 'attention' | 'cls';
export interface VNode {
  name: string; group: string; op: VOp; inputs: number[]; shape: VShape;
  W?: Float64Array; b?: Float64Array; kh?: number; kw?: number; stride?: number;
  mode?: 'max' | 'avg'; heads?: number; patch?: number;
}
export interface VisionTrace { tensors: VTensor[]; attention: Map<number, Float64Array[]>; features: Float64Array }
const size = (s: VShape) => s.h * s.w * s.c;
export function softmax(values: ArrayLike<number>): Float64Array {
  const max = Math.max(...Array.from(values));
  const out = Float64Array.from(values, (v) => Math.exp(v - max));
  const sum = out.reduce((a, b) => a + b, 0);
  return out.map((v) => v / sum);
}

/** Reduced channel widths; architectural block counts and branching remain explicit.
 * Backbones are seeded and frozen. Only the separate softmax head is optimized. */
export class VisionBackbone {
  nodes: VNode[] = [{ name: 'Input 16×16', group: 'input', op: 'input', inputs: [], shape: { h: 16, w: 16, c: 1 } }];
  overview: number[] = [0];
  private rng: Rng;
  constructor(readonly id: VisionId, readonly seed = 7, readonly width = 1, readonly patchSize = 4) {
    if (![1, 2].includes(width) || ![4, 8].includes(patchSize)) throw new Error('Unsupported teaching width or patch size');
    this.rng = new Rng(seed);
    if (id === 'vgg16') this.vgg();
    else if (id === 'resnet50') this.resnet();
    else if (id === 'densenet121') this.densenet();
    else if (id === 'inceptionv3') this.inception();
    else if (id === 'mobilenetv2') this.mobile();
    else this.vit();
  }
  get parameterCount() { return this.nodes.reduce((n, node) => n + (node.W?.length ?? 0) + (node.b?.length ?? 0), 0); }
  get featureCount() { return size(this.nodes[this.nodes.length - 1].shape); }
  private add(name: string, group: string, op: VOp, inputs: number[], shape: VShape, extra: Partial<VNode> = {}) {
    this.nodes.push({ name, group, op, inputs, shape, ...extra }); return this.nodes.length - 1;
  }
  private random(n: number, scale: number) { return Float64Array.from({ length: n }, () => this.rng.normal() * scale); }
  private simple(x: number, group: string, op: 'relu' | 'relu6' | 'bn' | 'ln' | 'gelu', name = op) { return this.add(name, group, op, [x], this.nodes[x].shape); }
  private conv(x: number, c: number, kh: number, group: string, stride = 1, kw = kh, depthwise = false) {
    const s = this.nodes[x].shape, out = depthwise ? s.c : c;
    const W = this.random(out * (depthwise ? 1 : s.c) * kh * kw, Math.sqrt(2 / ((depthwise ? 1 : s.c) * kh * kw)));
    return this.add(`${depthwise ? 'DW' : 'Conv'} ${kh}×${kw} → ${out}`, group, depthwise ? 'depthwise' : 'conv', [x], { h: Math.ceil(s.h / stride), w: Math.ceil(s.w / stride), c: out }, { W, b: new Float64Array(out), kh, kw, stride });
  }
  private cbr(x: number, c: number, k: number, group: string, stride = 1, kw = k, relu6 = false) {
    return this.simple(this.simple(this.conv(x, c, k, group, stride, kw), group, 'bn'), group, relu6 ? 'relu6' : 'relu');
  }
  private pool(x: number, group: string, stride = 2, mode: 'max' | 'avg' = 'max') {
    const s = this.nodes[x].shape;
    return this.add(`${mode} pool s${stride}`, group, 'pool', [x], { h: Math.max(1, Math.floor(s.h / stride)), w: Math.max(1, Math.floor(s.w / stride)), c: s.c }, { stride, kh: stride === 1 ? 3 : 2, mode });
  }
  private sum(a: number, b: number, group: string) {
    if (JSON.stringify(this.nodes[a].shape) !== JSON.stringify(this.nodes[b].shape)) throw new Error(`Residual shape mismatch: ${group}`);
    return this.add('Residual add', group, 'add', [a, b], this.nodes[a].shape);
  }
  private concat(xs: number[], group: string) {
    const s = this.nodes[xs[0]].shape;
    if (xs.some((x) => this.nodes[x].shape.h !== s.h || this.nodes[x].shape.w !== s.w)) throw new Error(`Concat shape mismatch: ${group}`);
    return this.add('Channel concat', group, 'concat', xs, { ...s, c: xs.reduce((c, x) => c + this.nodes[x].shape.c, 0) });
  }
  private dense(x: number, units: number, group: string, tokenwise = false) {
    const s = this.nodes[x].shape, nin = tokenwise ? s.c : size(s);
    return this.add(`Linear ${nin}→${units}`, group, 'dense', [x], { h: tokenwise ? s.h : 1, w: 1, c: units }, { W: this.random(nin * units, Math.sqrt(2 / (nin + units))), b: new Float64Array(units) });
  }
  private gap(x: number) { const s = this.nodes[x].shape; const out = this.add('Global average pool', 'readout', 'gap', [x], { h: 1, w: 1, c: s.c }); this.overview.push(out); return out; }
  private vgg() {
    let x = 0;
    [2, 2, 3, 3, 3].forEach((n, stage) => {
      const group = `Block ${stage + 1}`;
      for (let j = 0; j < n; j++) x = this.simple(this.conv(x, [4, 6, 8, 10, 12][stage] * this.width, 3, group), group, 'relu');
      x = this.pool(x, group); this.overview.push(x);
    });
    x = this.add('Flatten', 'Dense head', 'flatten', [x], { h: 1, w: 1, c: size(this.nodes[x].shape) });
    x = this.simple(this.dense(x, 32 * this.width, 'Dense head'), 'Dense head', 'relu');
    x = this.simple(this.dense(x, 32 * this.width, 'Dense head'), 'Dense head', 'relu'); this.overview.push(x);
  }
  private resnet() {
    let x = this.pool(this.cbr(0, 4 * this.width, 7, 'Stem', 2), 'Stem'); this.overview.push(x);
    [3, 4, 6, 3].forEach((count, stage) => {
      for (let unit = 0; unit < count; unit++) {
        const group = `Stage ${stage + 2} / bottleneck ${unit + 1}`, c = [2, 3, 4, 6][stage] * this.width;
        const stride = stage > 0 && unit === 0 ? 2 : 1;
        let shortcut = x;
        let y = this.cbr(x, c, 1, group, stride); y = this.cbr(y, c, 3, group);
        y = this.simple(this.conv(y, c * 4, 1, group), group, 'bn');
        if (this.nodes[x].shape.c !== c * 4 || stride !== 1) shortcut = this.simple(this.conv(x, c * 4, 1, group, stride), group, 'bn');
        x = this.simple(this.sum(y, shortcut, group), group, 'relu');
      }
      this.overview.push(x);
    });
    this.gap(x);
  }
  private densenet() {
    const k = 2 * this.width;
    let x = this.pool(this.simple(this.simple(this.conv(0, 4 * this.width, 7, 'Stem', 2), 'Stem', 'bn'), 'Stem', 'relu'), 'Stem'); this.overview.push(x);
    [6, 12, 24, 16].forEach((count, stage) => {
      for (let unit = 0; unit < count; unit++) {
        const group = `Dense ${stage + 1} / unit ${unit + 1}`;
        let y = this.conv(this.simple(this.simple(x, group, 'bn'), group, 'relu'), k * 4, 1, group);
        y = this.conv(this.simple(this.simple(y, group, 'bn'), group, 'relu'), k, 3, group);
        x = this.concat([x, y], group);
      }
      this.overview.push(x);
      if (stage < 3) {
        const group = `Transition ${stage + 1}`;
        x = this.pool(this.conv(this.simple(this.simple(x, group, 'bn'), group, 'relu'), Math.floor(this.nodes[x].shape.c / 2), 1, group), group, 2, 'avg'); this.overview.push(x);
      }
    });
    this.gap(this.simple(this.simple(x, 'Final norm', 'bn'), 'Final norm', 'relu'));
  }
  private inception() {
    const c = 2 * this.width;
    let x = this.cbr(0, c * 2, 3, 'Stem', 2); x = this.cbr(x, c * 2, 3, 'Stem'); x = this.cbr(x, c * 3, 3, 'Stem'); x = this.pool(x, 'Stem');
    x = this.cbr(x, c * 3, 1, 'Stem'); x = this.cbr(x, c * 4, 3, 'Stem'); x = this.pool(x, 'Stem'); this.overview.push(x);
    for (let unit = 0; unit < 3; unit++) {
      const g = `Inception-A ${unit + 1}`;
      const a = this.cbr(x, c, 1, g), b = this.cbr(this.cbr(x, c, 1, g), c, 5, g);
      const d = this.cbr(this.cbr(this.cbr(x, c, 1, g), c, 3, g), c, 3, g), e = this.cbr(this.pool(x, g, 1, 'avg'), c, 1, g);
      x = this.concat([a, b, d, e], g);
    }
    this.overview.push(x);
    let g = 'Reduction-A';
    x = this.concat([this.cbr(x, c * 2, 3, g, 2), this.cbr(this.cbr(this.cbr(x, c, 1, g), c, 3, g), c * 2, 3, g, 2), this.pool(x, g)], g); this.overview.push(x);
    for (let unit = 0; unit < 4; unit++) {
      g = `Inception-B ${unit + 1}`;
      const a = this.cbr(x, c * 2, 1, g);
      const b = this.cbr(this.cbr(this.cbr(x, c, 1, g), c, 1, g, 1, 7), c * 2, 7, g, 1, 1);
      let d = this.cbr(x, c, 1, g);
      for (const [kh, kw] of [[7, 1], [1, 7], [7, 1], [1, 7]]) d = this.cbr(d, c * 2, kh, g, 1, kw);
      x = this.concat([a, b, d, this.cbr(this.pool(x, g, 1, 'avg'), c * 2, 1, g)], g);
    }
    this.overview.push(x); g = 'Reduction-B';
    let b = this.cbr(x, c * 2, 1, g); b = this.cbr(b, c * 2, 1, g, 1, 7); b = this.cbr(b, c * 2, 7, g, 1, 1); b = this.cbr(b, c * 3, 3, g, 2);
    x = this.concat([this.cbr(this.cbr(x, c * 2, 1, g), c * 3, 3, g, 2), b, this.pool(x, g)], g); this.overview.push(x);
    for (let unit = 0; unit < 2; unit++) {
      g = `Inception-C ${unit + 1}`;
      const a = this.cbr(x, c * 2, 1, g), branch = this.cbr(x, c * 3, 1, g);
      const bs = this.concat([this.cbr(branch, c * 2, 1, g, 1, 3), this.cbr(branch, c * 2, 3, g, 1, 1)], g);
      const deep = this.cbr(this.cbr(x, c * 3, 1, g), c * 3, 3, g);
      const ds = this.concat([this.cbr(deep, c * 2, 1, g, 1, 3), this.cbr(deep, c * 2, 3, g, 1, 1)], g);
      x = this.concat([a, bs, ds, this.cbr(this.pool(x, g, 1, 'avg'), c * 2, 1, g)], g);
    }
    this.overview.push(x); this.gap(x);
  }
  private mobile() {
    let x = this.cbr(0, 4 * this.width, 3, 'Stem', 2, 3, true); this.overview.push(x);
    [[1, 2, 1, 1], [6, 3, 2, 2], [6, 4, 3, 2], [6, 6, 4, 2], [6, 8, 3, 1], [6, 10, 3, 2], [6, 12, 1, 1]].forEach(([expansion, channels, count, stride], stage) => {
      for (let unit = 0; unit < count; unit++) {
        const g = `Group ${stage + 1} / inverted ${unit + 1}`, c = channels * this.width, s = unit === 0 ? stride : 1, input = x;
        if (expansion !== 1) x = this.cbr(x, this.nodes[x].shape.c * expansion, 1, g, 1, 1, true);
        x = this.simple(this.simple(this.conv(x, 0, 3, g, s, 3, true), g, 'bn'), g, 'relu6');
        x = this.simple(this.conv(x, c, 1, g), g, 'bn');
        if (s === 1 && this.nodes[input].shape.c === c) x = this.sum(x, input, g);
      }
      this.overview.push(x);
    });
    x = this.cbr(x, 32 * this.width, 1, 'Final pointwise', 1, 1, true); this.overview.push(x); this.gap(x);
  }
  private vit() {
    const d = 16 * this.width, p = this.patchSize, tokens = (16 / p) ** 2 + 1;
    let x = this.add(`Patch${p} + CLS + position`, 'Patch embedding', 'patch', [0], { h: tokens, w: 1, c: d }, { patch: p, W: this.random(d * p * p, 1 / p), b: this.random(d * tokens, 0.02) }); this.overview.push(x);
    for (let block = 0; block < 12; block++) {
      const g = `Encoder ${block + 1}`, shortcut = x;
      let y = this.simple(x, g, 'ln');
      y = this.add('Multi-head self-attention', g, 'attention', [y], this.nodes[y].shape, { heads: 4, W: this.random(4 * d * d, 1 / Math.sqrt(d)), b: new Float64Array(4 * d) });
      x = this.sum(y, shortcut, g);
      y = this.simple(this.dense(this.simple(x, g, 'ln'), d * 4, g, true), g, 'gelu');
      y = this.dense(y, d, g, true); x = this.sum(y, x, g);
      if ([2, 5, 8, 11].includes(block)) this.overview.push(x);
    }
    x = this.simple(x, 'Final norm / CLS', 'ln');
    x = this.add('CLS readout', 'Final norm / CLS', 'cls', [x], { h: 1, w: 1, c: d }); this.overview.push(x);
  }

  forward(input: ArrayLike<number>, keepTrace = true): VisionTrace {
    if (input.length !== 256) throw new Error('Teaching images must contain 16×16 pixels');
    const tensors: VTensor[] = [{ ...this.nodes[0].shape, a: Float64Array.from(input) }];
    const attention = new Map<number, Float64Array[]>();
    for (let index = 1; index < this.nodes.length; index++) {
      const node = this.nodes[index], xs = node.inputs.map((i) => tensors[i]), x = xs[0], out = new Float64Array(size(node.shape));
      const spatial = x.h * x.w;
      if (node.op === 'conv' || node.op === 'depthwise') {
        const { kh = 1, kw = 1, stride = 1 } = node;
        const ph = Math.floor(kh / 2), pw = Math.floor(kw / 2);
        for (let oc = 0; oc < node.shape.c; oc++) for (let r = 0; r < node.shape.h; r++) for (let col = 0; col < node.shape.w; col++) {
          let sum = node.b![oc];
          for (let ic = node.op === 'depthwise' ? oc : 0; ic < (node.op === 'depthwise' ? oc + 1 : x.c); ic++) for (let kr = 0; kr < kh; kr++) for (let kc = 0; kc < kw; kc++) {
            const ir = r * stride + kr - ph, jc = col * stride + kc - pw;
            if (ir >= 0 && ir < x.h && jc >= 0 && jc < x.w) sum += x.a[(ic * x.h + ir) * x.w + jc] * node.W![((oc * (node.op === 'depthwise' ? 1 : x.c) + (node.op === 'depthwise' ? 0 : ic)) * kh + kr) * kw + kc];
          }
          out[(oc * node.shape.h + r) * node.shape.w + col] = sum;
        }
      } else if (node.op === 'relu' || node.op === 'relu6' || node.op === 'gelu' || node.op === 'bn') {
        x.a.forEach((v, i) => { out[i] = node.op === 'relu' ? Math.max(0, v) : node.op === 'relu6' ? Math.min(6, Math.max(0, v)) : node.op === 'bn' ? v / Math.sqrt(1 + 1e-5) : 0.5 * v * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (v + 0.044715 * v ** 3))); });
      } else if (node.op === 'ln') {
        for (let p = 0; p < spatial; p++) {
          let mean = 0, variance = 0;
          for (let c = 0; c < x.c; c++) mean += x.a[c * spatial + p] / x.c;
          for (let c = 0; c < x.c; c++) variance += (x.a[c * spatial + p] - mean) ** 2 / x.c;
          for (let c = 0; c < x.c; c++) out[c * spatial + p] = (x.a[c * spatial + p] - mean) / Math.sqrt(variance + 1e-6);
        }
      } else if (node.op === 'pool') {
        const k = node.kh!, s = node.stride!, pad = s === 1 ? 1 : 0;
        for (let c = 0; c < x.c; c++) for (let r = 0; r < node.shape.h; r++) for (let col = 0; col < node.shape.w; col++) {
          let result = node.mode === 'max' ? -Infinity : 0, n = 0;
          for (let kr = 0; kr < k; kr++) for (let kc = 0; kc < k; kc++) {
            const ir = r * s + kr - pad, jc = col * s + kc - pad;
            if (ir >= 0 && ir < x.h && jc >= 0 && jc < x.w) { const v = x.a[(c * x.h + ir) * x.w + jc]; result = node.mode === 'max' ? Math.max(result, v) : result + v; n++; }
          }
          out[(c * node.shape.h + r) * node.shape.w + col] = node.mode === 'avg' ? result / Math.max(1, n) : Number.isFinite(result) ? result : 0;
        }
      } else if (node.op === 'add') x.a.forEach((v, i) => { out[i] = v + xs[1].a[i]; });
      else if (node.op === 'concat') { let offset = 0; xs.forEach((tensor) => { out.set(tensor.a, offset); offset += tensor.a.length; }); }
      else if (node.op === 'flatten') out.set(x.a);
      else if (node.op === 'gap') for (let c = 0; c < x.c; c++) { let sum = 0; for (let p = 0; p < spatial; p++) sum += x.a[c * spatial + p]; out[c] = sum / spatial; }
      else if (node.op === 'dense') {
        const positions = node.shape.h, nin = positions > 1 ? x.c : x.a.length;
        for (let oc = 0; oc < node.shape.c; oc++) for (let p = 0; p < positions; p++) {
          let sum = node.b![oc]; for (let ic = 0; ic < nin; ic++) sum += node.W![oc * nin + ic] * x.a[ic * positions + p]; out[oc * positions + p] = sum;
        }
      } else if (node.op === 'patch') {
        const p = node.patch!, t = node.shape.h, patchesPerRow = 16 / p;
        for (let c = 0; c < node.shape.c; c++) for (let token = 0; token < t; token++) {
          let sum = node.b![c * t + token];
          if (token > 0) for (let r = 0; r < p; r++) for (let col = 0; col < p; col++) sum += x.a[(Math.floor((token - 1) / patchesPerRow) * p + r) * 16 + ((token - 1) % patchesPerRow) * p + col] * node.W![c * p * p + r * p + col];
          out[c * t + token] = sum;
        }
      } else if (node.op === 'attention') {
        const d = x.c, t = x.h, heads = node.heads!, dk = d / heads;
        const project = (offset: number) => {
          const matrix = new Float64Array(d * t);
          for (let c = 0; c < d; c++) for (let token = 0; token < t; token++) { let sum = node.b![offset * d + c]; for (let j = 0; j < d; j++) sum += node.W![(offset * d + c) * d + j] * x.a[j * t + token]; matrix[c * t + token] = sum; }
          return matrix;
        };
        const Q = project(0), K = project(1), V = project(2), context = new Float64Array(d * t), maps: Float64Array[] = [];
        for (let h = 0; h < heads; h++) {
          const map = new Float64Array(t * t);
          for (let qi = 0; qi < t; qi++) {
            const scores = new Float64Array(t);
            for (let kj = 0; kj < t; kj++) for (let j = 0; j < dk; j++) scores[kj] += Q[(h * dk + j) * t + qi] * K[(h * dk + j) * t + kj] / Math.sqrt(dk);
            const probs = softmax(scores); map.set(probs, qi * t);
            for (let j = 0; j < dk; j++) for (let kj = 0; kj < t; kj++) context[(h * dk + j) * t + qi] += probs[kj] * V[(h * dk + j) * t + kj];
          }
          maps.push(map);
        }
        for (let c = 0; c < d; c++) for (let token = 0; token < t; token++) { let sum = node.b![3 * d + c]; for (let j = 0; j < d; j++) sum += node.W![(3 * d + c) * d + j] * context[j * t + token]; out[c * t + token] = sum; }
        if (keepTrace) attention.set(index, maps);
      } else if (node.op === 'cls') for (let c = 0; c < x.c; c++) out[c] = x.a[c * spatial];
      tensors.push({ ...node.shape, a: out });
    }
    return { tensors: keepTrace ? tensors : [], attention, features: tensors[tensors.length - 1].a };
  }
}

export interface VisionData { Xtr: Float32Array[]; ytr: number[]; Xte: Float32Array[]; yte: number[] }
export interface VisionEpoch { epoch: number; loss: number; trainAcc: number; testAcc: number; testLoss: number }
export class VisionTrainer {
  W: Float64Array; b = new Float64Array(10); opt: Optimizer;
  features: Float64Array[] = []; mean: Float64Array; std: Float64Array;
  prepared = 0; ready = false; epoch = 0; steps = 0; cursor = 0; history: VisionEpoch[] = [];
  private order: number[]; private rng: Rng;
  constructor(readonly backbone: VisionBackbone, readonly data: VisionData, lr = 0.01, readonly seed = 7) {
    this.rng = new Rng(seed + 100); this.order = this.rng.shuffle(data.Xtr.map((_, i) => i));
    this.mean = new Float64Array(backbone.featureCount); this.std = new Float64Array(backbone.featureCount).fill(1);
    this.W = Float64Array.from({ length: 10 * backbone.featureCount }, () => this.rng.normal() * 0.01);
    this.opt = new Optimizer({ name: 'adam', lr, momentum: 0.9, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 });
  }
  get totalSamples() { return this.data.Xtr.length + this.data.Xte.length; }
  prepareOne() {
    if (this.ready) return;
    const image = this.prepared < this.data.Xtr.length ? this.data.Xtr[this.prepared] : this.data.Xte[this.prepared - this.data.Xtr.length];
    this.features.push(this.backbone.forward(image, false).features); this.prepared++;
    if (this.prepared === this.totalSamples) {
      const n = this.data.Xtr.length;
      for (let j = 0; j < this.mean.length; j++) {
        let mean = 0, variance = 0; for (let i = 0; i < n; i++) mean += this.features[i][j] / n;
        for (let i = 0; i < n; i++) variance += (this.features[i][j] - mean) ** 2 / n;
        this.mean[j] = mean; this.std[j] = Math.max(1e-12, Math.sqrt(variance));
      }
      this.ready = true;
    }
  }
  normalize(features: ArrayLike<number>) { return Float64Array.from(features, (v, j) => (v - this.mean[j]) / this.std[j]); }
  logits(features: ArrayLike<number>) {
    const x = this.normalize(features), d = x.length;
    return Float64Array.from({ length: 10 }, (_, c) => { let sum = this.b[c]; for (let j = 0; j < d; j++) sum += this.W[c * d + j] * x[j]; return sum; });
  }
  predict(features: ArrayLike<number>) { return softmax(this.logits(features)); }
  gradients(features: ArrayLike<number>[], labels: number[], l2 = 0) {
    const d = this.mean.length, gW = new Float64Array(this.W.length), gb = new Float64Array(10); let loss = 0;
    features.forEach((f, i) => {
      const x = this.normalize(f), p = this.predict(f); loss += categoricalCrossEntropy(this.logits(f), labels[i]) / labels.length;
      p.forEach((value, c) => { const delta = (value - (labels[i] === c ? 1 : 0)) / labels.length; gb[c] += delta; for (let j = 0; j < d; j++) gW[c * d + j] += delta * x[j]; });
    });
    this.W.forEach((w, i) => { loss += l2 * w * w / 2; gW[i] += l2 * w; });
    return { loss, gW, gb };
  }
  trainBatch(batchSize: number, l2 = 0) {
    if (!this.ready) throw new Error('Prepare backbone features before training the head');
    const batch = this.order.slice(this.cursor, this.cursor + batchSize);
    const g = this.gradients(batch.map((i) => this.features[i]), batch.map((i) => this.data.ytr[i]), l2);
    this.opt.begin(); this.opt.step('head.W', this.W, g.gW); this.opt.step('head.b', this.b, g.gb);
    this.cursor += batch.length; this.steps++;
    if (this.cursor >= this.order.length) {
      this.epoch++; this.cursor = 0; this.rng.shuffle(this.order);
      const train = this.evaluate(false), test = this.evaluate(true);
      this.history.push({ epoch: this.epoch, loss: train.loss, trainAcc: train.report.accuracy, testAcc: test.report.accuracy, testLoss: test.loss });
    }
    return g.loss;
  }
  evaluate(test = true): { report: ClassificationReport; loss: number } {
    const labels = test ? this.data.yte : this.data.ytr, offset = test ? this.data.Xtr.length : 0; let loss = 0;
    const predictions = labels.map((label, i) => { const features = this.features[offset + i], p = this.predict(features); loss += categoricalCrossEntropy(this.logits(features), label) / labels.length; return p.indexOf(Math.max(...p)); });
    return { report: classificationReport(labels, predictions, 10), loss };
  }
}
