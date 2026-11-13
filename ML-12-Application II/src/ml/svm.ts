import { Rng } from '../lib/random';

export type Kernel = 'linear' | 'rbf' | 'poly' | 'sigmoid';

export interface SvmParams {
  kernel: Kernel;
  C: number;
  gamma: number;
  degree: number;
  coef0: number;
  tol: number;
  /** multiplies C for class +1 / class −1 */
  classWeight: [number, number];
}

export function kernel(a: number[], b: number[], p: SvmParams): number {
  let dot = 0;
  if (p.kernel === 'rbf') {
    let s = 0;
    for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
    return Math.exp(-p.gamma * s);
  }
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  if (p.kernel === 'linear') return dot;
  if (p.kernel === 'poly') return (p.gamma * dot + p.coef0) ** p.degree;
  return Math.tanh(p.gamma * dot + p.coef0);
}

/**
 * Soft-margin SVM trained with (simplified) Sequential Minimal Optimisation on the dual problem.
 * `pass()` sweeps every multiplier once so the lab can animate training pass by pass.
 */
export class SMO {
  readonly X: number[][];
  readonly y: number[];
  readonly p: SvmParams;
  alpha: Float64Array;
  b = 0;
  passes = 0;
  iterations = 0;
  lastChanged = 0;
  done = false;
  private K: Float64Array;
  private rng: Rng;
  readonly maxPasses = 6;
  readonly maxIterations = 400;

  /** y must be ±1 */
  constructor(X: number[][], y: number[], p: SvmParams, seed = 1) {
    this.X = X; this.y = y; this.p = p;
    const n = X.length;
    this.alpha = new Float64Array(n);
    this.K = new Float64Array(n * n);
    for (let i = 0; i < n; i++) for (let j = i; j < n; j++) { const k = kernel(X[i], X[j], p); this.K[i * n + j] = k; this.K[j * n + i] = k; }
    this.rng = new Rng(seed);
  }

  Ci(i: number) { return this.p.C * (this.y[i] > 0 ? this.p.classWeight[0] : this.p.classWeight[1]); }

  private f(i: number) {
    const n = this.X.length;
    let s = this.b;
    for (let k = 0; k < n; k++) if (this.alpha[k] > 0) s += this.alpha[k] * this.y[k] * this.K[k * n + i];
    return s;
  }

  pass(): number {
    if (this.done) return 0;
    const n = this.X.length, { tol } = this.p, y = this.y, a = this.alpha, K = this.K;
    let changed = 0;
    for (let i = 0; i < n; i++) {
      const Ei = this.f(i) - y[i];
      const Ci = this.Ci(i);
      if ((y[i] * Ei < -tol && a[i] < Ci) || (y[i] * Ei > tol && a[i] > 0)) {
        let j = this.rng.int(n - 1);
        if (j >= i) j++;
        const Ej = this.f(j) - y[j];
        const Cj = this.Ci(j);
        const ai = a[i], aj = a[j];
        let L: number, H: number;
        if (y[i] !== y[j]) { L = Math.max(0, aj - ai); H = Math.min(Cj, Ci + aj - ai); }
        else { L = Math.max(0, ai + aj - Ci); H = Math.min(Cj, ai + aj); }
        if (H - L < 1e-12) continue;
        const eta = 2 * K[i * n + j] - K[i * n + i] - K[j * n + j];
        if (eta >= 0) continue;
        let ajNew = aj - (y[j] * (Ei - Ej)) / eta;
        ajNew = Math.min(H, Math.max(L, ajNew));
        if (Math.abs(ajNew - aj) < 1e-7) continue;
        const aiNew = ai + y[i] * y[j] * (aj - ajNew);
        a[i] = aiNew; a[j] = ajNew;
        const b1 = this.b - Ei - y[i] * (aiNew - ai) * K[i * n + i] - y[j] * (ajNew - aj) * K[i * n + j];
        const b2 = this.b - Ej - y[i] * (aiNew - ai) * K[i * n + j] - y[j] * (ajNew - aj) * K[j * n + j];
        this.b = aiNew > 0 && aiNew < Ci ? b1 : ajNew > 0 && ajNew < Cj ? b2 : (b1 + b2) / 2;
        changed++;
      }
    }
    this.iterations++;
    this.lastChanged = changed;
    this.passes = changed === 0 ? this.passes + 1 : 0;
    if (this.passes >= this.maxPasses || this.iterations >= this.maxIterations) this.done = true;
    return changed;
  }

  decision(x: number[]): number {
    let s = this.b;
    for (let k = 0; k < this.X.length; k++) if (this.alpha[k] > 1e-9) s += this.alpha[k] * this.y[k] * kernel(this.X[k], x, this.p);
    return s;
  }

  supportVectors(): number[] {
    const out: number[] = [];
    this.alpha.forEach((v, i) => { if (v > 1e-6) out.push(i); });
    return out;
  }

  /** Primal weights; only meaningful for the linear kernel. */
  weights(): number[] {
    const d = this.X[0].length, w = Array(d).fill(0);
    this.alpha.forEach((a, i) => { if (a > 0) for (let k = 0; k < d; k++) w[k] += a * this.y[i] * this.X[i][k]; });
    return w;
  }

  /** Dual objective Σα − ½ΣΣ αᵢαⱼyᵢyⱼK — increases during training. */
  dualObjective(): number {
    const n = this.X.length;
    let s = 0, q = 0;
    for (let i = 0; i < n; i++) {
      if (this.alpha[i] <= 0) continue;
      s += this.alpha[i];
      for (let j = 0; j < n; j++) if (this.alpha[j] > 0) q += this.alpha[i] * this.alpha[j] * this.y[i] * this.y[j] * this.K[i * n + j];
    }
    return s - 0.5 * q;
  }
}
