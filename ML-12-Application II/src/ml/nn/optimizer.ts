import type { OptimizerName } from './types';

export interface OptimizerConfig {
  name: OptimizerName;
  lr: number;
  momentum: number;
  beta1: number;
  beta2: number;
  epsilon: number;
}

interface Slot { m: Float64Array; v: Float64Array }

/**
 * One optimiser instance owns per-parameter state. `step` mutates the parameter array in place
 * and returns the applied update Δθ so the weight inspector can show old → new values.
 */
export class Optimizer {
  cfg: OptimizerConfig;
  t = 0;
  private slots = new Map<string, Slot>();

  constructor(cfg: OptimizerConfig) {
    this.cfg = cfg;
  }

  begin() {
    this.t++;
  }

  step(key: string, theta: Float64Array, grad: Float64Array): Float64Array {
    const { name, lr, momentum, beta1, beta2, epsilon } = this.cfg;
    let s = this.slots.get(key);
    if (!s) { s = { m: new Float64Array(theta.length), v: new Float64Array(theta.length) }; this.slots.set(key, s); }
    const d = new Float64Array(theta.length);
    for (let i = 0; i < theta.length; i++) {
      const g = grad[i];
      let u: number;
      if (name === 'sgd') u = -lr * g;
      else if (name === 'momentum') { s.m[i] = momentum * s.m[i] - lr * g; u = s.m[i]; }
      else if (name === 'rmsprop') { s.v[i] = beta2 * s.v[i] + (1 - beta2) * g * g; u = (-lr * g) / (Math.sqrt(s.v[i]) + epsilon); }
      else {
        s.m[i] = beta1 * s.m[i] + (1 - beta1) * g;
        s.v[i] = beta2 * s.v[i] + (1 - beta2) * g * g;
        const mh = s.m[i] / (1 - beta1 ** this.t);
        const vh = s.v[i] / (1 - beta2 ** this.t);
        u = (-lr * mh) / (Math.sqrt(vh) + epsilon);
      }
      theta[i] += u;
      d[i] = u;
    }
    return d;
  }

  snapshot() {
    const slots = new Map<string, Slot>();
    this.slots.forEach((s, k) => slots.set(k, { m: s.m.slice(), v: s.v.slice() }));
    return { t: this.t, slots };
  }
  restore(snap: { t: number; slots: Map<string, Slot> }) {
    this.t = snap.t;
    this.slots = new Map();
    snap.slots.forEach((s, k) => this.slots.set(k, { m: s.m.slice(), v: s.v.slice() }));
  }
}
