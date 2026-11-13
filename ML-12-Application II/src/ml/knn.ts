import { classificationReport, type ClassificationReport } from './metrics';

export type Metric = 'euclidean' | 'manhattan' | 'minkowski';
export type Weighting = 'uniform' | 'distance';

export interface KnnOptions {
  k: number;
  metric: Metric;
  /** Minkowski order; only used when metric === 'minkowski'. */
  p: number;
  weighting: Weighting;
}

export function distance(a: number[], b: number[], metric: Metric, p = 3): number {
  let s = 0;
  if (metric === 'euclidean') {
    for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
    return Math.sqrt(s);
  }
  if (metric === 'manhattan') {
    for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]);
    return s;
  }
  for (let i = 0; i < a.length; i++) s += Math.abs(a[i] - b[i]) ** p;
  return Math.pow(s, 1 / p);
}

export interface Neighbor {
  index: number;
  distance: number;
  label: number;
  weight: number;
}

export interface KnnResult {
  /** Every training point, sorted nearest first. */
  ranked: { index: number; distance: number }[];
  neighbors: Neighbor[];
  votes: number[];
  predicted: number;
  tie: boolean;
}

/** Classify one query. X/y are the training rows only. Ties go to the tied class with the nearest member. */
export function knnPredict(X: number[][], y: number[], nClasses: number, q: number[], o: KnnOptions, skip = -1): KnnResult {
  const ranked = X.map((x, index) => ({ index, distance: distance(x, q, o.metric, o.p) }))
    .filter((r) => r.index !== skip)
    .sort((a, b) => a.distance - b.distance);
  const k = Math.max(1, Math.min(o.k, ranked.length));
  const neighbors = ranked.slice(0, k).map((r) => ({
    ...r,
    label: y[r.index],
    weight: o.weighting === 'distance' ? 1 / (r.distance + 1e-9) : 1,
  }));
  const votes = Array(nClasses).fill(0);
  neighbors.forEach((n) => (votes[n.label] += n.weight));
  const max = Math.max(...votes);
  const tied = votes.map((v, c) => (Math.abs(v - max) < 1e-12 ? c : -1)).filter((c) => c >= 0);
  const predicted = tied.length === 1 ? tied[0] : neighbors.find((n) => tied.includes(n.label))!.label;
  return { ranked, neighbors, votes, predicted, tie: tied.length > 1 };
}

export function knnEvaluate(
  Xtr: number[][], ytr: number[], Xte: number[][], yte: number[], nClasses: number, o: KnnOptions,
): ClassificationReport {
  const pred = Xte.map((q) => knnPredict(Xtr, ytr, nClasses, q, o).predicted);
  return classificationReport(yte, pred, nClasses);
}

/**
 * Accuracy for K = 1..maxK. Training accuracy is computed with each point included in its own
 * neighbourhood (as a fitted model would), which is why K = 1 always scores 100% on train.
 */
export function accuracyByK(
  Xtr: number[][], ytr: number[], Xte: number[][], yte: number[], nClasses: number, o: KnnOptions, maxK: number,
): { k: number; train: number; test: number }[] {
  const rank = (rows: number[][]) => rows.map((q) => Xtr.map((x, i) => ({ i, d: distance(x, q, o.metric, o.p) })).sort((a, b) => a.d - b.d));
  const rTr = rank(Xtr);
  const rTe = rank(Xte);
  const acc = (ranks: { i: number; d: number }[][], truth: number[], k: number) => {
    let ok = 0;
    ranks.forEach((r, qi) => {
      const votes = Array(nClasses).fill(0);
      const nb = r.slice(0, k);
      nb.forEach((n) => (votes[ytr[n.i]] += o.weighting === 'distance' ? 1 / (n.d + 1e-9) : 1));
      const max = Math.max(...votes);
      const tied = votes.map((v, c) => (Math.abs(v - max) < 1e-12 ? c : -1)).filter((c) => c >= 0);
      const pred = tied.length === 1 ? tied[0] : ytr[nb.find((n) => tied.includes(ytr[n.i]))!.i];
      if (pred === truth[qi]) ok++;
    });
    return ranks.length ? ok / ranks.length : 0;
  };
  const out = [];
  for (let k = 1; k <= Math.min(maxK, Xtr.length); k++) out.push({ k, train: acc(rTr, ytr, k), test: acc(rTe, yte, k) });
  return out;
}
