import { Rng } from '../lib/random';
import { categoricalCrossEntropy } from './losses';
import { Optimizer, type OptimizerConfig } from './nn/optimizer';
import type { Text } from '../i18n/i18n';

export type CnnAct = 'relu' | 'tanh' | 'linear';
export type CnnLayer =
  | { kind: 'conv'; filters: number; k: number; stride: number; pad: 'same' | 'valid'; act: CnnAct }
  | { kind: 'pool'; mode: 'max' | 'avg'; size: number; stride: number }
  | { kind: 'flatten' }
  | { kind: 'dropout'; rate: number }
  | { kind: 'dense'; units: number; act: CnnAct }
  | { kind: 'output'; units: number };

export interface Shape { c: number; h: number; w: number }
const size = (s: Shape) => s.c * s.h * s.w;

export interface CnnParam { W: Float64Array; b: Float64Array }

export interface CnnTrace {
  /** a[0] is the input; a[l+1] is the output of layer l (after activation) */
  a: Float32Array[];
  /** pre-activation of conv/dense/output layers */
  z: (Float32Array | null)[];
  argmax: (Int32Array | null)[];
  mask: (Float32Array | null)[];
}

export interface CnnValidation { shapes: Shape[]; errors: Text[]; params: number[] }

export function cnnShapes(layers: CnnLayer[], input: Shape): CnnValidation {
  const shapes = [input];
  const errors: Text[] = [];
  const params: number[] = [];
  let s = input, flat = false;
  const positive = (value: number) => Number.isInteger(value) && value > 0;
  if (![input.c, input.h, input.w].every(positive)) errors.push({ th: 'ขนาดภาพอินพุตต้องเป็นจำนวนเต็มบวก', en: 'Input dimensions must be positive integers' });
  layers.forEach((l, i) => {
    let p = 0;
    if (l.kind === 'conv' || l.kind === 'pool') {
      if (flat) errors.push({ th: `ชั้นที่ ${i + 1}: ${l.kind} ต้องอยู่ก่อน Flatten`, en: `Layer ${i + 1}: ${l.kind} must come before Flatten` });
      const rawK = l.kind === 'conv' ? l.k : l.size;
      if (!positive(rawK) || !positive(l.stride) || (l.kind === 'conv' && !positive(l.filters))) errors.push({ th: `ชั้นที่ ${i + 1}: kernel, stride และ filters ต้องเป็นจำนวนเต็มบวก`, en: `Layer ${i + 1}: kernel, stride and filters must be positive integers` });
      const k = positive(rawK) ? rawK : 1, stride = positive(l.stride) ? l.stride : 1;
      const pad = l.kind === 'conv' && l.pad === 'same' ? Math.floor(k / 2) : 0;
      const h = Math.floor((s.h + 2 * pad - k) / stride) + 1, w = Math.floor((s.w + 2 * pad - k) / stride) + 1;
      if (h < 1 || w < 1) errors.push({ th: `ชั้นที่ ${i + 1}: ขนาด ${s.h}×${s.w} เล็กเกินไปสำหรับหน้าต่าง ${k}×${k}`, en: `Layer ${i + 1}: ${s.h}×${s.w} is too small for a ${k}×${k} window` });
      if (l.kind === 'conv') { p = l.filters * s.c * k * k + l.filters; s = { c: l.filters, h: Math.max(1, h), w: Math.max(1, w) }; }
      else s = { c: s.c, h: Math.max(1, h), w: Math.max(1, w) };
    } else if (l.kind === 'flatten') { s = { c: size(s), h: 1, w: 1 }; flat = true; }
    else if (l.kind === 'dense' || l.kind === 'output') {
      if (!positive(l.units)) errors.push({ th: `ชั้นที่ ${i + 1}: units ต้องเป็นจำนวนเต็มบวก`, en: `Layer ${i + 1}: units must be positive integers` });
      if (!flat) errors.push({ th: `ชั้นที่ ${i + 1}: ต้องมี Flatten ก่อนชั้น ${l.kind}`, en: `Layer ${i + 1}: add Flatten before ${l.kind}` });
      p = l.units * size(s) + l.units;
      s = { c: l.units, h: 1, w: 1 };
    } else if (l.kind === 'dropout' && (!Number.isFinite(l.rate) || l.rate < 0 || l.rate > 0.9)) errors.push({ th: 'Dropout rate ต้องอยู่ระหว่าง 0–0.9', en: 'Dropout rate must be between 0 and 0.9' });
    params.push(p);
    shapes.push(s);
  });
  const last = layers[layers.length - 1];
  if (!last || last.kind !== 'output') errors.push({ th: 'ชั้นสุดท้ายต้องเป็น Output', en: 'The last layer must be Output' });
  if (params.reduce((sum, n) => sum + n, 0) > 500000 || layers.length > 64) errors.push({ th: 'โมเดลเกินขีดจำกัด 500,000 พารามิเตอร์ หรือ 64 ชั้นของ CNN Lab', en: 'CNN Lab supports at most 500,000 parameters and 64 layers' });
  return { shapes, errors, params };
}

const act = (a: CnnAct, v: number) => (a === 'relu' ? (v > 0 ? v : 0) : a === 'tanh' ? Math.tanh(v) : v);
const dact = (a: CnnAct, z: number, y: number) => (a === 'relu' ? (z > 0 ? 1 : 0) : a === 'tanh' ? 1 - y * y : 1);

export class CNN {
  layers: CnnLayer[];
  shapes: Shape[];
  params: (CnnParam | null)[];

  constructor(layers: CnnLayer[], input: Shape, seed: number) {
    const validation = cnnShapes(layers, input);
    if (validation.errors.length) throw new Error(validation.errors.map((error) => error.en).join('; '));
    this.layers = layers;
    this.shapes = validation.shapes;
    const rng = new Rng(seed);
    this.params = layers.map((l, i) => {
      const s = this.shapes[i];
      if (l.kind === 'conv') {
        const fan = s.c * l.k * l.k;
        const W = new Float64Array(l.filters * fan).map(() => rng.normal() * Math.sqrt(2 / fan));
        return { W, b: new Float64Array(l.filters) };
      }
      if (l.kind === 'dense' || l.kind === 'output') {
        const nin = size(s);
        const W = new Float64Array(l.units * nin).map(() => rng.normal() * Math.sqrt(2 / (nin + l.units)));
        return { W, b: new Float64Array(l.units) };
      }
      return null;
    });
  }

  forward(x: Float32Array, training = false, rng?: Rng): CnnTrace {
    const a: Float32Array[] = [x];
    const z: (Float32Array | null)[] = [];
    const argmax: (Int32Array | null)[] = [];
    const mask: (Float32Array | null)[] = [];
    this.layers.forEach((l, li) => {
      const inp = a[li], si = this.shapes[li], so = this.shapes[li + 1];
      const p = this.params[li];
      let out: Float32Array, zz: Float32Array | null = null, am: Int32Array | null = null, mk: Float32Array | null = null;
      if (l.kind === 'conv') {
        const pad = l.pad === 'same' ? Math.floor(l.k / 2) : 0, K = l.k;
        zz = new Float32Array(size(so)); out = new Float32Array(size(so));
        for (let f = 0; f < l.filters; f++) for (let oy = 0; oy < so.h; oy++) for (let ox = 0; ox < so.w; ox++) {
          let s = p!.b[f];
          for (let c = 0; c < si.c; c++) for (let ky = 0; ky < K; ky++) {
            const iy = oy * l.stride + ky - pad;
            if (iy < 0 || iy >= si.h) continue;
            for (let kx = 0; kx < K; kx++) {
              const ix = ox * l.stride + kx - pad;
              if (ix < 0 || ix >= si.w) continue;
              s += p!.W[((f * si.c + c) * K + ky) * K + kx] * inp[(c * si.h + iy) * si.w + ix];
            }
          }
          const o = (f * so.h + oy) * so.w + ox;
          zz[o] = s; out[o] = act(l.act, s);
        }
      } else if (l.kind === 'pool') {
        out = new Float32Array(size(so)); am = new Int32Array(size(so));
        for (let c = 0; c < si.c; c++) for (let oy = 0; oy < so.h; oy++) for (let ox = 0; ox < so.w; ox++) {
          let best = -Infinity, bi = 0, sum = 0;
          for (let ky = 0; ky < l.size; ky++) for (let kx = 0; kx < l.size; kx++) {
            const idx = (c * si.h + oy * l.stride + ky) * si.w + ox * l.stride + kx;
            sum += inp[idx];
            if (inp[idx] > best) { best = inp[idx]; bi = idx; }
          }
          const o = (c * so.h + oy) * so.w + ox;
          out[o] = l.mode === 'max' ? best : sum / (l.size * l.size);
          am[o] = bi;
        }
      } else if (l.kind === 'flatten') out = inp;
      else if (l.kind === 'dropout') {
        if (training && rng && l.rate > 0) { mk = new Float32Array(inp.length).map(() => (rng.next() >= l.rate ? 1 / (1 - l.rate) : 0)); out = inp.map((v, i) => v * mk![i]); }
        else out = inp;
      } else {
        const nin = inp.length, n = l.units;
        zz = new Float32Array(n); out = new Float32Array(n);
        for (let j = 0; j < n; j++) { let s = p!.b[j]; const r = j * nin; for (let i = 0; i < nin; i++) s += p!.W[r + i] * inp[i]; zz[j] = s; }
        if (l.kind === 'output') { const m = Math.max(...zz); let t = 0; for (let j = 0; j < n; j++) { out[j] = Math.exp(zz[j] - m); t += out[j]; } for (let j = 0; j < n; j++) out[j] /= t; }
        else for (let j = 0; j < n; j++) out[j] = act(l.act, zz[j]);
      }
      a.push(out); z.push(zz); argmax.push(am); mask.push(mk);
    });
    return { a, z, argmax, mask };
  }

  /** Cross-entropy backward pass. Returns per-layer parameter gradients and the loss. */
  backward(t: CnnTrace, label: number): { gW: (Float64Array | null)[]; gb: (Float64Array | null)[]; loss: number; delta: (Float32Array | null)[] } {
    const L = this.layers.length;
    const gW: (Float64Array | null)[] = this.layers.map(() => null);
    const gb: (Float64Array | null)[] = this.layers.map(() => null);
    const delta: (Float32Array | null)[] = this.layers.map(() => null);
    const probs = t.a[L];
    const loss = categoricalCrossEntropy(t.z[L - 1]!, label);
    let g = new Float32Array(probs.length);
    for (let i = 0; i < g.length; i++) g[i] = probs[i] - (i === label ? 1 : 0); // dL/dz for softmax+CE
    for (let li = L - 1; li >= 0; li--) {
      const l = this.layers[li], p = this.params[li], inp = t.a[li], si = this.shapes[li], so = this.shapes[li + 1];
      if (l.kind === 'output' || l.kind === 'dense') {
        const dz = l.kind === 'output' ? g : g.map((v, j) => v * dact(l.act, t.z[li]![j], t.a[li + 1][j]));
        delta[li] = dz;
        const nin = inp.length;
        const W = new Float64Array(p!.W.length), B = new Float64Array(l.units), gi = new Float32Array(nin);
        for (let j = 0; j < l.units; j++) { B[j] = dz[j]; const r = j * nin; for (let i = 0; i < nin; i++) { W[r + i] = dz[j] * inp[i]; gi[i] += p!.W[r + i] * dz[j]; } }
        gW[li] = W; gb[li] = B; g = gi;
      } else if (l.kind === 'dropout') { const m = t.mask[li]; if (m) g = g.map((v, i) => v * m[i]); }
      else if (l.kind === 'pool') {
        const gi = new Float32Array(inp.length);
        for (let c = 0; c < si.c; c++) for (let oy = 0; oy < so.h; oy++) for (let ox = 0; ox < so.w; ox++) {
          const o = (c * so.h + oy) * so.w + ox;
          if (l.mode === 'max') gi[t.argmax[li]![o]] += g[o];
          else for (let ky = 0; ky < l.size; ky++) for (let kx = 0; kx < l.size; kx++) gi[(c * si.h + oy * l.stride + ky) * si.w + ox * l.stride + kx] += g[o] / (l.size * l.size);
        }
        g = gi;
      } else if (l.kind === 'conv') {
        const dz = g.map((v, j) => v * dact(l.act, t.z[li]![j], t.a[li + 1][j]));
        delta[li] = dz;
        const pad = l.pad === 'same' ? Math.floor(l.k / 2) : 0, K = l.k;
        const W = new Float64Array(p!.W.length), B = new Float64Array(l.filters), gi = new Float32Array(inp.length);
        for (let f = 0; f < l.filters; f++) for (let oy = 0; oy < so.h; oy++) for (let ox = 0; ox < so.w; ox++) {
          const d = dz[(f * so.h + oy) * so.w + ox];
          if (d === 0) continue;
          B[f] += d;
          for (let c = 0; c < si.c; c++) for (let ky = 0; ky < K; ky++) {
            const iy = oy * l.stride + ky - pad;
            if (iy < 0 || iy >= si.h) continue;
            for (let kx = 0; kx < K; kx++) {
              const ix = ox * l.stride + kx - pad;
              if (ix < 0 || ix >= si.w) continue;
              const wi = ((f * si.c + c) * K + ky) * K + kx, ii = (c * si.h + iy) * si.w + ix;
              W[wi] += d * inp[ii];
              gi[ii] += p!.W[wi] * d;
            }
          }
        }
        gW[li] = W; gb[li] = B; g = gi;
      }
    }
    return { gW, gb, loss, delta };
  }

  paramCount() { return this.params.reduce((s, p) => s + (p ? p.W.length + p.b.length : 0), 0); }
}

/** Real handwritten digits (8×8, 0–16) upsampled bilinearly to 16×16 and scaled to 0–1. */
export function upsample8to16(px: ArrayLike<number>): Float32Array {
  const out = new Float32Array(256);
  for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
    const sy = Math.min(7, Math.max(0, (y + 0.5) / 2 - 0.5)), sx = Math.min(7, Math.max(0, (x + 0.5) / 2 - 0.5));
    const y0 = Math.floor(sy), x0 = Math.floor(sx), y1 = Math.min(7, y0 + 1), x1 = Math.min(7, x0 + 1), fy = sy - y0, fx = sx - x0;
    const v = px[y0 * 8 + x0] * (1 - fx) * (1 - fy) + px[y0 * 8 + x1] * fx * (1 - fy) + px[y1 * 8 + x0] * (1 - fx) * fy + px[y1 * 8 + x1] * fx * fy;
    out[y * 16 + x] = v / 16;
  }
  return out;
}

export function loadDigits(pixels: string, labels: string): { X: Float32Array[]; y: number[] } {
  const X: Float32Array[] = [], y: number[] = [];
  for (let i = 0; i < labels.length; i++) {
    const px = Array.from(pixels.slice(i * 64, i * 64 + 64), (ch) => parseInt(ch, 17));
    X.push(upsample8to16(px));
    y.push(Number(labels[i]));
  }
  return { X, y };
}

export const DEFAULT_CNN: CnnLayer[] = [
  { kind: 'conv', filters: 6, k: 3, stride: 1, pad: 'same', act: 'relu' },
  { kind: 'pool', mode: 'max', size: 2, stride: 2 },
  { kind: 'conv', filters: 8, k: 3, stride: 1, pad: 'valid', act: 'relu' },
  { kind: 'pool', mode: 'max', size: 2, stride: 2 },
  { kind: 'flatten' },
  { kind: 'dense', units: 24, act: 'relu' },
  { kind: 'output', units: 10 },
];


/** Mini-batch training helper shared by the CNN Lab and the pre-training script. */
export class CnnTrainer {
  net: CNN;
  opt: Optimizer;
  rng: Rng;
  epoch = 0;
  steps = 0;
  cursor = 0;
  order: number[];
  history: { epoch: number; loss: number }[] = [];
  private epochLoss = 0;
  private epochSamples = 0;
  constructor(net: CNN, public X: Float32Array[], public y: number[], cfg: OptimizerConfig, public batch: number, seed: number) {
    this.net = net;
    this.opt = new Optimizer(cfg);
    this.rng = new Rng(seed);
    this.order = this.rng.shuffle(X.map((_, i) => i));
  }
  /** returns mean batch loss */
  trainBatch(): number {
    if (!this.X.length || this.X.length !== this.y.length || !Number.isInteger(this.batch) || this.batch < 1) throw new Error('Invalid CNN training data or batch size');
    const idx = this.order.slice(this.cursor, this.cursor + this.batch);
    const acc = this.net.params.map((p) => (p ? { W: new Float64Array(p.W.length), b: new Float64Array(p.b.length) } : null));
    let loss = 0;
    for (const i of idx) {
      const g = this.net.backward(this.net.forward(this.X[i], true, this.rng), this.y[i]);
      loss += g.loss;
      acc.forEach((a, l) => { if (!a) return; const gw = g.gW[l]!, gb = g.gb[l]!; for (let k = 0; k < gw.length; k++) a.W[k] += gw[k] / idx.length; for (let k = 0; k < gb.length; k++) a.b[k] += gb[k] / idx.length; });
    }
    if (!Number.isFinite(loss) || acc.some((a) => a && (!a.W.every(Number.isFinite) || !a.b.every(Number.isFinite)))) throw new Error('Non-finite CNN loss or gradient; lower the learning rate and reset');
    this.opt.begin();
    acc.forEach((a, l) => { if (!a) return; const p = this.net.params[l]!; this.opt.step(`W${l}`, p.W, a.W); this.opt.step(`b${l}`, p.b, a.b); });
    this.steps++;
    this.cursor += idx.length;
    this.epochLoss += loss; this.epochSamples += idx.length;
    if (this.cursor >= this.order.length) {
      this.cursor = 0; this.epoch++; this.rng.shuffle(this.order);
      this.history.push({ epoch: this.epoch, loss: this.epochLoss / this.epochSamples });
      this.epochLoss = 0; this.epochSamples = 0;
    }
    return loss / idx.length;
  }
}

export function cnnAccuracy(net: CNN, X: Float32Array[], y: number[]) {
  let ok = 0;
  const pred = X.map((x, i) => { const p = net.forward(x).a[net.layers.length]; const c = p.indexOf(Math.max(...p)); if (c === y[i]) ok++; return c; });
  return { acc: X.length ? ok / X.length : 0, pred };
}
