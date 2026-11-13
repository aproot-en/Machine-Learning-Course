import { Rng } from '../../lib/random';
import { classificationReport, type ClassificationReport } from '../metrics';
import { argClass, Network, type DenseParams } from './network';
import { Optimizer, type OptimizerConfig } from './optimizer';
import type { LayerSpec, Loss } from './types';

export interface TrainConfig {
  optimizer: OptimizerConfig;
  batchSize: number;
  loss: Loss;
  shuffle: boolean;
  seed: number;
  earlyStopping: boolean;
  patience: number;
}

export interface PreparedData {
  Xtr: number[][]; ytr: number[];
  Xval: number[][]; yval: number[];
  Xte: number[][]; yte: number[];
  nClasses: number;
}

export interface EpochRecord {
  epoch: number;
  loss: number;
  acc: number;
  valLoss: number;
  valAcc: number;
  testAcc: number;
}

interface Snapshot {
  epoch: number;
  steps: number;
  params: (DenseParams | null)[];
  opt: ReturnType<Optimizer['snapshot']>;
  rng: number;
  order: number[];
  bestVal: number;
  bad: number;
}

const MAX_SNAPSHOTS = 400;

export class Trainer {
  readonly net: Network;
  readonly opt: Optimizer;
  cfg: TrainConfig;
  readonly data: PreparedData;
  epoch = 0;
  steps = 0;
  stopped: string | null = null;
  history: EpochRecord[] = [];
  snapshots: Snapshot[] = [];
  /** batch-averaged gradient, previous value and applied update of every weight from the most recent step */
  lastGradW: (Float64Array | null)[];
  lastPrevW: (Float64Array | null)[];
  lastDeltaW: (Float64Array | null)[];
  lastBatchLoss = 0;
  private rng: Rng;
  private order: number[];
  private cursor = 0;
  private bestVal = Infinity;
  private bad = 0;

  constructor(layers: LayerSpec[], data: PreparedData, cfg: TrainConfig) {
    if (!data.Xtr.length || data.Xtr.length !== data.ytr.length) throw new Error('Training data must be nonempty and have one label per row');
    this.cfg = cfg;
    this.data = data;
    this.net = new Network(layers, cfg.seed);
    this.opt = new Optimizer(cfg.optimizer);
    this.rng = new Rng(cfg.seed * 31 + 7);
    this.order = data.Xtr.map((_, i) => i);
    if (cfg.shuffle) this.rng.shuffle(this.order);
    this.lastGradW = layers.map(() => null);
    this.lastPrevW = layers.map(() => null);
    this.lastDeltaW = layers.map(() => null);
    this.snapshots.push(this.snap());
  }

  target(y: number): number[] {
    const units = this.net.widths[this.net.widths.length - 1];
    if (units === 1) return [y];
    const t = Array(units).fill(0);
    t[y] = 1;
    return t;
  }

  /** One mini-batch of gradient descent. Returns true when it completed an epoch. */
  trainBatch(): boolean {
    if (this.stopped) return false;
    const { Xtr, ytr } = this.data;
    const bs = Math.max(1, Math.min(this.cfg.batchSize, Xtr.length));
    const idx = this.order.slice(this.cursor, this.cursor + bs);
    const net = this.net;
    const acc = net.params.map((p) => (p ? { W: new Float64Array(p.W.length), b: new Float64Array(p.b.length) } : null));
    let loss = 0;
    for (const i of idx) {
      const t = net.forward(Xtr[i], true, this.rng);
      const g = net.backward(t, this.target(ytr[i]), this.cfg.loss);
      loss += g.loss;
      acc.forEach((a, l) => {
        if (!a) return;
        const gw = g.gW[l]!, gb = g.gb[l]!;
        for (let k = 0; k < gw.length; k++) a.W[k] += gw[k];
        for (let k = 0; k < gb.length; k++) a.b[k] += gb[k];
      });
    }
    if (!Number.isFinite(loss) || acc.some((a) => a && (!a.W.every(Number.isFinite) || !a.b.every(Number.isFinite)))) {
      this.stopped = 'non-finite-gradient';
      return false;
    }
    this.opt.begin();
    acc.forEach((a, l) => {
      if (!a) return;
      const p = net.params[l]!;
      const spec = net.layers[l];
      const l1 = spec.l1 ?? 0, l2 = spec.l2 ?? 0;
      for (let k = 0; k < a.W.length; k++) a.W[k] = a.W[k] / idx.length + l2 * p.W[k] + l1 * Math.sign(p.W[k]);
      for (let k = 0; k < a.b.length; k++) a.b[k] /= idx.length;
      this.lastGradW[l] = a.W;
      this.lastPrevW[l] = p.W.slice();
      this.lastDeltaW[l] = this.opt.step(`W${l}`, p.W, a.W);
      if (spec.useBias !== false) this.opt.step(`b${l}`, p.b, a.b);
    });
    this.lastBatchLoss = loss / idx.length + net.regularizationLoss();
    this.steps++;
    this.cursor += bs;
    if (this.cursor >= this.order.length) {
      this.cursor = 0;
      if (this.cfg.shuffle) this.rng.shuffle(this.order);
      this.endEpoch();
      return true;
    }
    return false;
  }

  trainEpoch() {
    while (!this.stopped && !this.trainBatch());
  }

  evaluate(X: number[][], y: number[]): { loss: number; acc: number; preds: number[] } {
    if (!X.length) return { loss: 0, acc: 0, preds: [] };
    let loss = 0, ok = 0;
    const preds = X.map((x, i) => {
      const t = this.net.forward(x);
      loss += this.net.backward(t, this.target(y[i]), this.cfg.loss).loss;
      const p = argClass(t.a[t.a.length - 1]);
      if (p === y[i]) ok++;
      return p;
    });
    return { loss: loss / X.length + this.net.regularizationLoss(), acc: ok / X.length, preds };
  }

  report(which: 'train' | 'val' | 'test'): ClassificationReport {
    const d = this.data;
    const [X, y] = which === 'train' ? [d.Xtr, d.ytr] : which === 'val' ? [d.Xval, d.yval] : [d.Xte, d.yte];
    return classificationReport(y, this.evaluate(X, y).preds, d.nClasses);
  }

  private endEpoch() {
    this.epoch++;
    const tr = this.evaluate(this.data.Xtr, this.data.ytr);
    const va = this.data.Xval.length ? this.evaluate(this.data.Xval, this.data.yval) : tr;
    const te = this.evaluate(this.data.Xte, this.data.yte);
    this.history.push({ epoch: this.epoch, loss: tr.loss, acc: tr.acc, valLoss: va.loss, valAcc: va.acc, testAcc: te.acc });
    if (this.cfg.earlyStopping && this.data.Xval.length) {
      if (va.loss < this.bestVal - 1e-4) { this.bestVal = va.loss; this.bad = 0; }
      else if (++this.bad >= this.cfg.patience) this.stopped = `early-stop@${this.epoch}`;
    }
    this.snapshots.push(this.snap());
    const bytesPerSnapshot = this.net.params.reduce((bytes, p, i) => bytes + (p ? (p.W.length * 3 + p.b.length * (this.net.layers[i].useBias === false ? 1 : 3)) * 8 : 0), 0) + this.order.length * 8;
    const limit = Math.max(2, Math.min(MAX_SNAPSHOTS, Math.floor(32 * 1024 * 1024 / Math.max(1, bytesPerSnapshot))));
    while (this.snapshots.length > limit) this.snapshots.splice(1, 1);
  }

  private snap(): Snapshot {
    return { epoch: this.epoch, steps: this.steps, params: this.net.cloneParams(), opt: this.opt.snapshot(), rng: this.rng.snapshot(), order: this.order.slice(), bestVal: this.bestVal, bad: this.bad };
  }

  snapshotAt(epoch: number) { return [...this.snapshots].reverse().find((s) => s.epoch <= epoch); }

  /** Jump back to a recorded epoch; later history is discarded so training resumes from there. */
  restore(epoch: number) {
    const s = this.snapshotAt(epoch);
    if (!s) return;
    this.net.params = s.params.map((p) => (p ? { ...p, W: p.W.slice(), b: p.b.slice() } : null));
    this.opt.restore(s.opt);
    this.rng.restore(s.rng);
    this.order = s.order.slice();
    this.epoch = s.epoch;
    this.steps = s.steps;
    this.cursor = 0;
    this.history = this.history.filter((h) => h.epoch <= s.epoch);
    this.snapshots = this.snapshots.filter((x) => x.epoch <= s.epoch);
    this.stopped = null;
    this.bestVal = s.bestVal;
    this.bad = s.bad;
    this.lastGradW = this.lastGradW.map(() => null);
    this.lastPrevW = this.lastPrevW.map(() => null);
    this.lastDeltaW = this.lastDeltaW.map(() => null);
  }
}
