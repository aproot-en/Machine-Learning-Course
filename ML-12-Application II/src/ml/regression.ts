import { Rng } from '../lib/random';
import { binaryCrossEntropy } from './losses';
import { Optimizer, type OptimizerConfig } from './nn/optimizer';

export type RegKind = 'linear' | 'logistic';

export interface RegConfig {
  kind: RegKind;
  optimizer: OptimizerConfig;
  /** 0 = full batch */
  batchSize: number;
  l2: number;
  l1: number;
  seed: number;
  init: 'zeros' | 'random';
}

export interface RegStep {
  iter: number;
  w: number[];
  b: number;
  loss: number;
  gw: number[];
  gb: number;
}

const sigmoid = (z: number) => 1 / (1 + Math.exp(-z));

/** Linear (MSE) or logistic (cross-entropy) regression trained by gradient descent, step by step. */
export class GDRegression {
  X: number[][];
  y: number[];
  cfg: RegConfig;
  w: Float64Array;
  b: Float64Array;
  opt: Optimizer;
  iter = 0;
  history: RegStep[] = [];
  private rng: Rng;

  constructor(X: number[][], y: number[], cfg: RegConfig, start?: { w: number[]; b: number }) {
    this.X = X; this.y = y; this.cfg = cfg;
    this.rng = new Rng(cfg.seed);
    const d = X[0].length;
    this.w = new Float64Array(d).map(() => (cfg.init === 'random' ? this.rng.normal() * 1.5 : 0));
    this.b = new Float64Array([cfg.init === 'random' ? this.rng.normal() : 0]);
    if (start) { this.w = Float64Array.from(start.w); this.b[0] = start.b; }
    this.opt = new Optimizer({ ...cfg.optimizer });
    this.history.push({ iter: 0, w: [...this.w], b: this.b[0], loss: this.loss(), gw: Array(d).fill(0), gb: 0 });
  }

  predict(x: number[]): number {
    let z = this.b[0];
    for (let k = 0; k < x.length; k++) z += this.w[k] * x[k];
    return this.cfg.kind === 'logistic' ? sigmoid(z) : z;
  }

  /** Full-data loss including regularisation. */
  loss(w: ArrayLike<number> = this.w, b = this.b[0]): number {
    let s = 0;
    for (let i = 0; i < this.X.length; i++) {
      let z = b;
      for (let k = 0; k < w.length; k++) z += w[k] * this.X[i][k];
      if (this.cfg.kind === 'linear') s += (z - this.y[i]) ** 2;
      else s += binaryCrossEntropy(z, this.y[i]);
    }
    let r = 0;
    for (let k = 0; k < w.length; k++) r += 0.5 * this.cfg.l2 * w[k] * w[k] + this.cfg.l1 * Math.abs(w[k]);
    return s / this.X.length + r;
  }

  step(): RegStep {
    const n = this.X.length, d = this.w.length;
    const bs = this.cfg.batchSize > 0 ? Math.min(this.cfg.batchSize, n) : n;
    const idx = bs === n ? this.X.map((_, i) => i) : Array.from({ length: bs }, () => this.rng.int(n));
    const gw = new Float64Array(d), gb = new Float64Array(1);
    for (const i of idx) {
      const p = this.predict(this.X[i]);
      // d/dz of MSE is 2(p−y); of sigmoid+BCE is (p−y)
      const e = this.cfg.kind === 'linear' ? 2 * (p - this.y[i]) : p - this.y[i];
      for (let k = 0; k < d; k++) gw[k] += (e * this.X[i][k]) / idx.length;
      gb[0] += e / idx.length;
    }
    for (let k = 0; k < d; k++) gw[k] += this.cfg.l2 * this.w[k] + this.cfg.l1 * Math.sign(this.w[k]);
    this.opt.cfg = { ...this.cfg.optimizer };
    this.opt.begin();
    this.opt.step('w', this.w, gw);
    this.opt.step('b', this.b, gb);
    this.iter++;
    const rec = { iter: this.iter, w: [...this.w], b: this.b[0], loss: this.loss(), gw: [...gw], gb: gb[0] };
    this.history.push(rec);
    return rec;
  }
}

/** Ordinary least squares via the normal equation (d ≤ 2), for comparison with gradient descent. */
export function closedForm(X: number[][], y: number[]): { w: number[]; b: number } {
  const d = X[0].length;
  const A = Array.from({ length: d + 1 }, () => Array(d + 1).fill(0));
  const v = Array(d + 1).fill(0);
  X.forEach((x, i) => {
    const r = [...x, 1];
    for (let a = 0; a <= d; a++) { v[a] += r[a] * y[i]; for (let c = 0; c <= d; c++) A[a][c] += r[a] * r[c]; }
  });
  // Gaussian elimination
  const n = d + 1;
  for (let c = 0; c < n; c++) {
    let piv = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[piv][c])) piv = r;
    [A[c], A[piv]] = [A[piv], A[c]]; [v[c], v[piv]] = [v[piv], v[c]];
    for (let r = 0; r < n; r++) {
      if (r === c || Math.abs(A[c][c]) < 1e-12) continue;
      const f = A[r][c] / A[c][c];
      for (let k = c; k < n; k++) A[r][k] -= f * A[c][k];
      v[r] -= f * v[c];
    }
  }
  const sol = v.map((x, i) => x / (A[i][i] || 1));
  return { w: sol.slice(0, d), b: sol[d] };
}

export function makeRegressionData(n: number, d: 1 | 2, noise: number, seed: number, nonlinear = false) {
  const rng = new Rng(seed);
  const trueW = d === 1 ? [1.6] : [1.2, -0.8];
  const trueB = 0.5;
  const X: number[][] = [], y: number[] = [];
  for (let i = 0; i < n; i++) {
    const x = Array.from({ length: d }, () => rng.next() * 4 - 2);
    let t = trueB + x.reduce((s, v, k) => s + v * trueW[k], 0);
    if (nonlinear) t += 0.6 * x[0] * x[0] - 1;
    X.push(x);
    y.push(t + rng.normal() * noise);
  }
  return { X, y, trueW, trueB };
}

export function r2(yTrue: number[], yPred: number[]) {
  const m = yTrue.reduce((a, b) => a + b, 0) / yTrue.length;
  const ssr = yTrue.reduce((s, v, i) => s + (v - yPred[i]) ** 2, 0);
  const sst = yTrue.reduce((s, v) => s + (v - m) ** 2, 0);
  return 1 - ssr / (sst || 1);
}
