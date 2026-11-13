import { Rng } from '../lib/random';
import { binaryCrossEntropy } from './losses';

export type Cell = 'rnn' | 'lstm';
export type SeqTask = 'sine' | 'memory';

export interface SeqConfig { cell: Cell; hidden: number; layers: number; inDim: number; outDim: number; seed: number }

interface LayerP { Wx: Float64Array; Wh: Float64Array; b: Float64Array; nin: number }

/** Values computed at one time step of one recurrent layer. LSTM gates are i, f, g, o. */
export interface StepTrace {
  x: number[];
  hPrev: number[];
  cPrev: number[];
  z: number[];
  h: number[];
  c: number[];
  i?: number[]; f?: number[]; g?: number[]; o?: number[];
}

export interface SeqTrace {
  /** steps[layer][t] */
  steps: StepTrace[][];
  /** network output per step (linear for sine, sigmoid for memory) */
  y: number[][];
}

const sig = (v: number) => 1 / (1 + Math.exp(-v));

export class SeqModel {
  cfg: SeqConfig;
  P: LayerP[];
  Wy: Float64Array;
  by: Float64Array;

  constructor(cfg: SeqConfig) {
    this.cfg = cfg;
    const rng = new Rng(cfg.seed);
    const G = cfg.cell === 'lstm' ? 4 : 1, H = cfg.hidden;
    this.P = Array.from({ length: cfg.layers }, (_, l) => {
      const nin = l === 0 ? cfg.inDim : H;
      const sx = Math.sqrt(1 / nin), sh = Math.sqrt(1 / H);
      const b = new Float64Array(G * H);
      if (cfg.cell === 'lstm') for (let j = H; j < 2 * H; j++) b[j] = 1; // forget-gate bias 1
      return { nin, Wx: new Float64Array(G * H * nin).map(() => rng.normal() * sx), Wh: new Float64Array(G * H * H).map(() => rng.normal() * sh), b };
    });
    this.Wy = new Float64Array(cfg.outDim * H).map(() => rng.normal() * Math.sqrt(1 / H));
    this.by = new Float64Array(cfg.outDim);
  }

  forward(xs: number[][], task: SeqTask): SeqTrace {
    const H = this.cfg.hidden, lstm = this.cfg.cell === 'lstm';
    const steps: StepTrace[][] = [];
    let inputs = xs;
    this.P.forEach((p) => {
      const tr: StepTrace[] = [];
      let h = Array(H).fill(0), c = Array(H).fill(0);
      inputs.forEach((x) => {
        const G = lstm ? 4 : 1;
        const z = Array(G * H).fill(0);
        for (let j = 0; j < G * H; j++) {
          let s = p.b[j];
          for (let k = 0; k < p.nin; k++) s += p.Wx[j * p.nin + k] * x[k];
          for (let k = 0; k < H; k++) s += p.Wh[j * H + k] * h[k];
          z[j] = s;
        }
        if (lstm) {
          const i = z.slice(0, H).map(sig), f = z.slice(H, 2 * H).map(sig), g = z.slice(2 * H, 3 * H).map(Math.tanh), o = z.slice(3 * H).map(sig);
          const cn = c.map((cp, k) => f[k] * cp + i[k] * g[k]);
          const hn = cn.map((cv, k) => o[k] * Math.tanh(cv));
          tr.push({ x, hPrev: h, cPrev: c, z, h: hn, c: cn, i, f, g, o });
          h = hn; c = cn;
        } else {
          const hn = z.map(Math.tanh);
          tr.push({ x, hPrev: h, cPrev: c, z, h: hn, c: hn });
          h = hn;
        }
      });
      steps.push(tr);
      inputs = tr.map((s) => s.h);
    });
    const top = steps[steps.length - 1];
    const y = top.map((s) => {
      const o = Array.from({ length: this.cfg.outDim }, (_, j) => { let v = this.by[j]; for (let k = 0; k < H; k++) v += this.Wy[j * H + k] * s.h[k]; return v; });
      return task === 'memory' ? o.map(sig) : o;
    });
    return { steps, y };
  }

  /** Loss: sine = MSE over every step; memory = binary cross-entropy at the last step only. */
  loss(tr: SeqTrace, targets: number[][], task: SeqTask): number {
    if (task === 'memory') { const top = tr.steps[tr.steps.length - 1], h = top[top.length - 1].h; let logit = this.by[0]; for (let j = 0; j < h.length; j++) logit += this.Wy[j] * h[j]; return binaryCrossEntropy(logit, targets[targets.length - 1][0]); }
    let s = 0;
    tr.y.forEach((o, t) => o.forEach((v, j) => (s += (v - targets[t][j]) ** 2)));
    return s / (tr.y.length * this.cfg.outDim);
  }

  /** Backpropagation through time. Returns gradients in the same layout as the parameters. */
  backward(tr: SeqTrace, targets: number[][], task: SeqTask) {
    const H = this.cfg.hidden, T = tr.y.length, lstm = this.cfg.cell === 'lstm', G = lstm ? 4 : 1;
    const gWy = new Float64Array(this.Wy.length), gby = new Float64Array(this.by.length);
    const top = tr.steps[tr.steps.length - 1];
    // dL/dh for the top layer at each step
    let dhs: number[][] = top.map(() => Array(H).fill(0));
    for (let t = 0; t < T; t++) {
      let dy: number[];
      if (task === 'memory') dy = t === T - 1 ? [tr.y[t][0] - targets[t][0]] : [0];
      else dy = tr.y[t].map((v, j) => (2 * (v - targets[t][j])) / (T * this.cfg.outDim));
      dy.forEach((d, j) => {
        if (!d) return;
        gby[j] += d;
        for (let k = 0; k < H; k++) { gWy[j * H + k] += d * top[t].h[k]; dhs[t][k] += this.Wy[j * H + k] * d; }
      });
    }
    const gP = this.P.map((p) => ({ Wx: new Float64Array(p.Wx.length), Wh: new Float64Array(p.Wh.length), b: new Float64Array(p.b.length) }));
    const gradNorms: number[] = Array(T).fill(0);
    for (let l = this.P.length - 1; l >= 0; l--) {
      const p = this.P[l], st = tr.steps[l], gp = gP[l];
      const dxs: number[][] = st.map(() => Array(p.nin).fill(0));
      let dhNext = Array(H).fill(0), dcNext = Array(H).fill(0);
      for (let t = T - 1; t >= 0; t--) {
        const s = st[t];
        const dh = dhs[t].map((v, k) => v + dhNext[k]);
        if (l === 0) gradNorms[t] = Math.sqrt(dh.reduce((a, v) => a + v * v, 0));
        const dz = Array(G * H).fill(0);
        if (lstm) {
          const tc = s.c.map(Math.tanh);
          const dc = dh.map((v, k) => v * s.o![k] * (1 - tc[k] * tc[k]) + dcNext[k]);
          for (let k = 0; k < H; k++) {
            dz[k] = dc[k] * s.g![k] * s.i![k] * (1 - s.i![k]);
            dz[H + k] = dc[k] * s.cPrev[k] * s.f![k] * (1 - s.f![k]);
            dz[2 * H + k] = dc[k] * s.i![k] * (1 - s.g![k] * s.g![k]);
            dz[3 * H + k] = dh[k] * tc[k] * s.o![k] * (1 - s.o![k]);
          }
          dcNext = dc.map((v, k) => v * s.f![k]);
        } else for (let k = 0; k < H; k++) dz[k] = dh[k] * (1 - s.h[k] * s.h[k]);
        const dhp = Array(H).fill(0);
        for (let j = 0; j < G * H; j++) {
          if (!dz[j]) continue;
          gp.b[j] += dz[j];
          for (let k = 0; k < p.nin; k++) { gp.Wx[j * p.nin + k] += dz[j] * s.x[k]; dxs[t][k] += p.Wx[j * p.nin + k] * dz[j]; }
          for (let k = 0; k < H; k++) { gp.Wh[j * H + k] += dz[j] * s.hPrev[k]; dhp[k] += p.Wh[j * H + k] * dz[j]; }
        }
        dhNext = dhp;
      }
      dhs = dxs;
    }
    return { gP, gWy, gby, gradNorms };
  }
}

export function makeSeqBatch(task: SeqTask, n: number, T: number, rng: Rng) {
  const X: number[][][] = [], Y: number[][][] = [];
  for (let s = 0; s < n; s++) {
    if (task === 'sine') {
      const w = 0.25 + rng.next() * 0.35, ph = rng.next() * Math.PI * 2;
      const v = Array.from({ length: T + 1 }, (_, t) => Math.sin(w * t + ph));
      X.push(v.slice(0, T).map((a) => [a]));
      Y.push(v.slice(1).map((a) => [a]));
    } else {
      const bits = Array.from({ length: T }, () => (rng.next() < 0.5 ? -1 : 1));
      X.push(bits.map((b) => [b]));
      Y.push(bits.map(() => [bits[0] > 0 ? 1 : 0]));
    }
  }
  return { X, Y };
}
