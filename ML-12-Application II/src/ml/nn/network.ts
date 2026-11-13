import { Rng } from '../../lib/random';
import { binaryCrossEntropy, softTargetCrossEntropy } from '../losses';
import { activate, activationBackward } from './activations';
import type { LayerSpec, Loss } from './types';

export interface DenseParams {
  nin: number;
  nout: number;
  /** row-major [out][in] */
  W: Float64Array;
  b: Float64Array;
}

/** Everything one forward pass produced, kept so the inspector and backward pass read the same numbers. */
export interface Trace {
  /** a[l] = output of layer l (a[0] = the input vector). */
  a: number[][];
  /** z[l] = pre-activation of layer l, for dense/output/activation layers. */
  z: (number[] | null)[];
  /** dropout keep-mask (already scaled by 1/(1-rate)), or null. */
  mask: (number[] | null)[];
}

export interface Gradients {
  gW: (Float64Array | null)[];
  gb: (Float64Array | null)[];
  /** dL/dz per layer (the "local gradient" δ of each neuron). */
  delta: (number[] | null)[];
  loss: number;
}

export function initParams(l: LayerSpec, nin: number, rng: Rng): DenseParams {
  const nout = l.units!;
  const W = new Float64Array(nout * nin);
  const init = l.kernelInit ?? 'glorot_uniform';
  for (let i = 0; i < W.length; i++) {
    switch (init) {
      case 'glorot_uniform': { const lim = Math.sqrt(6 / (nin + nout)); W[i] = (rng.next() * 2 - 1) * lim; break; }
      case 'he_normal': W[i] = rng.normal() * Math.sqrt(2 / nin); break;
      case 'lecun_normal': W[i] = rng.normal() * Math.sqrt(1 / nin); break;
      case 'random_normal': W[i] = rng.normal() * 0.05; break;
      case 'zeros': W[i] = 0; break;
    }
  }
  const b = new Float64Array(nout);
  if (l.useBias !== false) b.fill(l.biasInit === 'ones' ? 1 : l.biasInit === 'small' ? 0.1 : 0);
  return { nin, nout, W, b };
}

export class Network {
  readonly layers: LayerSpec[];
  params: (DenseParams | null)[];
  /** output width of each layer */
  readonly widths: number[];

  constructor(layers: LayerSpec[], seed: number) {
    this.layers = layers;
    const rng = new Rng(seed);
    this.widths = [];
    this.params = layers.map((l, i) => {
      if (l.kind === 'input') { this.widths.push(l.units!); return null; }
      const nin = this.widths[i - 1];
      if (l.kind === 'dense' || l.kind === 'output') { this.widths.push(l.units!); return initParams(l, nin, rng); }
      this.widths.push(nin);
      return null;
    });
  }

  forward(x: number[], training = false, rng?: Rng): Trace {
    const a: number[][] = [x.slice()];
    const z: (number[] | null)[] = [null];
    const mask: (number[] | null)[] = [null];
    for (let i = 1; i < this.layers.length; i++) {
      const l = this.layers[i];
      const prev = a[i - 1];
      const p = this.params[i];
      if (p) {
        const zi = new Array(p.nout);
        for (let j = 0; j < p.nout; j++) {
          let s = l.useBias === false ? 0 : p.b[j];
          const row = j * p.nin;
          for (let k = 0; k < p.nin; k++) s += p.W[row + k] * prev[k];
          zi[j] = s;
        }
        z.push(zi); a.push(activate(l.activation ?? 'linear', zi)); mask.push(null);
      } else if (l.kind === 'activation') {
        z.push(prev.slice()); a.push(activate(l.activation ?? 'relu', prev)); mask.push(null);
      } else if (l.kind === 'dropout') {
        const rate = l.rate ?? 0;
        if (training && rate > 0 && rng) {
          const m = prev.map(() => (rng.next() >= rate ? 1 / (1 - rate) : 0));
          mask.push(m); a.push(prev.map((v, k) => v * m[k]));
        } else { mask.push(null); a.push(prev.slice()); }
        z.push(null);
      }
    }
    return { a, z, mask };
  }

  output(x: number[]): number[] {
    const t = this.forward(x);
    return t.a[t.a.length - 1];
  }

  backward(t: Trace, target: number[], loss: Loss): Gradients {
    const L = this.layers.length - 1;
    const out = t.a[L];
    const outAct = this.layers[L].activation ?? 'linear';
    const eps = 1e-12;
    let lossVal = 0;
    let gA: number[];
    let gZ: number[] | null = null;
    if (loss === 'bce') {
      out.forEach((p, i) => { lossVal += outAct === 'sigmoid' ? binaryCrossEntropy(t.z[L]![i], target[i]) : -target[i] * Math.log(Math.max(eps, p)) - (1 - target[i]) * Math.log(Math.max(eps, 1 - p)); });
      if (outAct === 'sigmoid') gZ = out.map((p, i) => p - target[i]);
      gA = out.map((p, i) => (p - target[i]) / Math.max(p * (1 - p), eps));
    } else if (loss === 'cce') {
      lossVal = outAct === 'softmax' ? softTargetCrossEntropy(t.z[L]!, target) : out.reduce((sum, p, i) => sum - target[i] * Math.log(Math.max(p, eps)), 0);
      if (outAct === 'softmax') gZ = out.map((p, i) => p * target.reduce((sum, value) => sum + value, 0) - target[i]);
      gA = out.map((p, i) => -target[i] / Math.max(p, eps));
    } else {
      out.forEach((p, i) => (lossVal += (p - target[i]) ** 2 / out.length));
      gA = out.map((p, i) => (2 * (p - target[i])) / out.length);
    }

    const gW: (Float64Array | null)[] = this.layers.map(() => null);
    const gb: (Float64Array | null)[] = this.layers.map(() => null);
    const delta: (number[] | null)[] = this.layers.map(() => null);

    for (let i = L; i >= 1; i--) {
      const l = this.layers[i];
      const p = this.params[i];
      if (p) {
        const dz = i === L && gZ ? gZ : activationBackward(l.activation ?? 'linear', t.z[i]!, t.a[i], gA);
        delta[i] = dz;
        const prev = t.a[i - 1];
        const W = new Float64Array(p.W.length);
        const B = new Float64Array(p.nout);
        const gPrev = new Array(p.nin).fill(0);
        for (let j = 0; j < p.nout; j++) {
          const row = j * p.nin;
          B[j] = l.useBias === false ? 0 : dz[j];
          for (let k = 0; k < p.nin; k++) {
            W[row + k] = dz[j] * prev[k];
            gPrev[k] += p.W[row + k] * dz[j];
          }
        }
        gW[i] = W; gb[i] = B; gA = gPrev;
      } else if (l.kind === 'activation') {
        const dz = activationBackward(l.activation ?? 'relu', t.z[i]!, t.a[i], gA);
        delta[i] = dz; gA = dz;
      } else if (l.kind === 'dropout') {
        const m = t.mask[i];
        gA = m ? gA.map((v, k) => v * m[k]) : gA;
      }
    }
    return { gW, gb, delta, loss: lossVal };
  }

  regularizationLoss(): number {
    let s = 0;
    this.layers.forEach((l, i) => {
      const p = this.params[i];
      if (!p) return;
      for (const w of p.W) s += (l.l2 ?? 0) * 0.5 * w * w + (l.l1 ?? 0) * Math.abs(w);
    });
    return s;
  }

  paramCount(): number {
    return this.params.reduce((s, p, i) => s + (p ? p.W.length + (this.layers[i].useBias === false ? 0 : p.b.length) : 0), 0);
  }

  cloneParams(): (DenseParams | null)[] {
    return this.params.map((p) => (p ? { ...p, W: p.W.slice(), b: p.b.slice() } : null));
  }
}

/** Class index predicted from an output vector. */
export const argClass = (out: number[]) => (out.length === 1 ? (out[0] > 0.5 ? 1 : 0) : out.indexOf(Math.max(...out)));
