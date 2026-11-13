import type { Activation } from './types';

const SQ2PI = Math.sqrt(2 / Math.PI);

export function activate(act: Activation, z: number[]): number[] {
  switch (act) {
    case 'relu': return z.map((v) => (v > 0 ? v : 0));
    case 'leakyrelu': return z.map((v) => (v > 0 ? v : 0.01 * v));
    case 'sigmoid': return z.map((v) => 1 / (1 + Math.exp(-v)));
    case 'tanh': return z.map((v) => Math.tanh(v));
    case 'gelu': return z.map((v) => 0.5 * v * (1 + Math.tanh(SQ2PI * (v + 0.044715 * v ** 3))));
    case 'linear': return z.slice();
    case 'softmax': {
      const m = Math.max(...z);
      const e = z.map((v) => Math.exp(v - m));
      const s = e.reduce((a, b) => a + b, 0);
      return e.map((v) => v / s);
    }
  }
}

/** Given dL/da, return dL/dz. */
export function activationBackward(act: Activation, z: number[], a: number[], g: number[]): number[] {
  switch (act) {
    case 'relu': return g.map((v, i) => (z[i] > 0 ? v : 0));
    case 'leakyrelu': return g.map((v, i) => (z[i] > 0 ? v : 0.01 * v));
    case 'sigmoid': return g.map((v, i) => v * a[i] * (1 - a[i]));
    case 'tanh': return g.map((v, i) => v * (1 - a[i] * a[i]));
    case 'linear': return g.slice();
    case 'gelu':
      return g.map((v, i) => {
        const x = z[i];
        const u = SQ2PI * (x + 0.044715 * x ** 3);
        const t = Math.tanh(u);
        return v * (0.5 * (1 + t) + 0.5 * x * (1 - t * t) * SQ2PI * (1 + 3 * 0.044715 * x * x));
      });
    case 'softmax': {
      const dot = g.reduce((s, v, i) => s + v * a[i], 0);
      return a.map((ai, i) => ai * (g[i] - dot));
    }
  }
}

export const ACTIVATION_TEX: Record<Activation, string> = {
  relu: '\\max(0, z)',
  leakyrelu: '\\max(0.01z,\\, z)',
  sigmoid: '\\dfrac{1}{1+e^{-z}}',
  tanh: '\\tanh(z)',
  gelu: '\\tfrac12 z\\,\\big(1+\\tanh(\\sqrt{2/\\pi}\\,(z+0.044715z^3))\\big)',
  softmax: '\\dfrac{e^{z_i}}{\\sum_j e^{z_j}}',
  linear: 'z',
};
