import { Rng } from '../lib/random';

const d2 = (a: number[], b: number[]) => { let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2; return s; };

export interface KMeansState {
  iter: number;
  phase: 'assign' | 'update';
  centroids: number[][];
  assign: number[];
  inertia: number | null;
  changed: number;
  displacement: number;
  converged: boolean;
}

export function initCentroids(X: number[][], k: number, method: 'random' | 'kmeans++', seed: number): number[][] {
  const rng = new Rng(seed);
  const out = [X[rng.int(X.length)].slice()];
  while (out.length < k) {
    if (method === 'random') { out.push(X[rng.int(X.length)].slice()); continue; }
    const w = X.map((x) => Math.min(...out.map((c) => d2(x, c))));
    let r = rng.next() * w.reduce((a, b) => a + b, 0), i = 0;
    while (i < X.length - 1 && r > w[i]) { r -= w[i]; i++; }
    out.push(X[i].slice());
  }
  return out;
}

/**
 * K-Means as an explicit sequence of states. Each call to `next` performs exactly one half-step
 * (assignment or centroid update), so the lab can step forward and back through recorded states.
 */
export function kmeansNext(X: number[][], s: KMeansState, tol: number, maxIter: number): KMeansState {
  if (s.converged) return s;
  if (s.phase === 'assign') {
    let changed = 0;
    const assign = X.map((x, i) => {
      let best = 0, bd = Infinity;
      s.centroids.forEach((c, j) => { const d = d2(x, c); if (d < bd) { bd = d; best = j; } });
      if (best !== s.assign[i]) changed++;
      return best;
    });
    const inertia = X.reduce((t, x, i) => t + d2(x, s.centroids[assign[i]]), 0);
    return { ...s, assign, changed, inertia, phase: 'update', converged: s.iter > 0 && changed === 0 };
  }
  const dim = X[0].length;
  let disp = 0;
  const centroids = s.centroids.map((c, j) => {
    const sum = Array(dim).fill(0); let n = 0;
    X.forEach((x, i) => { if (s.assign[i] === j) { n++; for (let k = 0; k < dim; k++) sum[k] += x[k]; } });
    const nc = n ? sum.map((v) => v / n) : c.slice();
    disp = Math.max(disp, Math.sqrt(d2(nc, c)));
    return nc;
  });
  const iter = s.iter + 1;
  const inertia = X.reduce((t, x, i) => t + d2(x, centroids[s.assign[i]]), 0);
  return { ...s, centroids, iter, displacement: disp, inertia, phase: 'assign', converged: disp < tol || iter >= maxIter };
}

/** Mean silhouette coefficient; labels < 0 (noise) are ignored. */
export function silhouette(X: number[][], labels: number[]): number | null {
  const idx = labels.map((l, i) => (l >= 0 ? i : -1)).filter((i) => i >= 0);
  const ks = [...new Set(idx.map((i) => labels[i]))];
  if (ks.length < 2) return null;
  let total = 0;
  for (const i of idx) {
    const byC = new Map<number, { s: number; n: number }>();
    for (const j of idx) { if (j === i) continue; const e = byC.get(labels[j]) ?? { s: 0, n: 0 }; e.s += Math.sqrt(d2(X[i], X[j])); e.n++; byC.set(labels[j], e); }
    const own = byC.get(labels[i]);
    const a = own && own.n ? own.s / own.n : 0;
    let b = Infinity;
    byC.forEach((e, c) => { if (c !== labels[i] && e.n) b = Math.min(b, e.s / e.n); });
    total += own && own.n ? (b - a) / Math.max(a, b) : 0;
  }
  return total / idx.length;
}

export type DbscanEvent =
  | { type: 'visit'; p: number; neighbors: number[] }
  | { type: 'noise'; p: number }
  | { type: 'cluster'; p: number; c: number }
  | { type: 'core'; p: number; c: number }
  | { type: 'border'; p: number; c: number };

export interface DbscanState {
  labels: number[]; // -2 unvisited, -1 noise, ≥0 cluster
  kind: ('unvisited' | 'core' | 'border' | 'noise')[];
  current: number | null;
  neighbors: number[];
  clusters: number;
}

export function regionQuery(X: number[][], i: number, eps: number, metric: 'euclidean' | 'manhattan'): number[] {
  const out: number[] = [];
  for (let j = 0; j < X.length; j++) {
    let d = 0;
    if (metric === 'euclidean') d = Math.sqrt(d2(X[i], X[j]));
    else for (let k = 0; k < X[i].length; k++) d += Math.abs(X[i][k] - X[j][k]);
    if (d <= eps) out.push(j);
  }
  return out;
}

/** Runs DBSCAN and records every decision as an event list that can be replayed to any step. */
export function dbscanEvents(X: number[][], eps: number, minPts: number, metric: 'euclidean' | 'manhattan'): DbscanEvent[] {
  const n = X.length, label = new Array(n).fill(-2), ev: DbscanEvent[] = [];
  let c = -1;
  for (let p = 0; p < n; p++) {
    if (label[p] !== -2) continue;
    const nb = regionQuery(X, p, eps, metric);
    ev.push({ type: 'visit', p, neighbors: nb });
    if (nb.length < minPts) { label[p] = -1; ev.push({ type: 'noise', p }); continue; }
    c++;
    label[p] = c;
    ev.push({ type: 'cluster', p, c });
    ev.push({ type: 'core', p, c });
    const seeds = nb.filter((q) => q !== p);
    for (let s = 0; s < seeds.length; s++) {
      const q = seeds[s];
      if (label[q] === -1) { label[q] = c; ev.push({ type: 'border', p: q, c }); }
      if (label[q] !== -2) continue;
      label[q] = c;
      const nq = regionQuery(X, q, eps, metric);
      ev.push({ type: 'visit', p: q, neighbors: nq });
      if (nq.length >= minPts) {
        ev.push({ type: 'core', p: q, c });
        nq.forEach((r) => { if (!seeds.includes(r) && r !== p) seeds.push(r); });
      } else ev.push({ type: 'border', p: q, c });
    }
  }
  return ev;
}

export function dbscanReplay(n: number, events: DbscanEvent[], upto: number): DbscanState {
  const labels = new Array(n).fill(-2);
  const kind: DbscanState['kind'] = new Array(n).fill('unvisited');
  let current: number | null = null, neighbors: number[] = [], clusters = 0;
  for (let k = 0; k < Math.min(upto, events.length); k++) {
    const e = events[k];
    if (e.type === 'visit') { current = e.p; neighbors = e.neighbors; }
    else if (e.type === 'noise') { labels[e.p] = -1; kind[e.p] = 'noise'; }
    else if (e.type === 'cluster') { labels[e.p] = e.c; clusters = e.c + 1; }
    else if (e.type === 'core') { labels[e.p] = e.c; kind[e.p] = 'core'; }
    else { labels[e.p] = e.c; kind[e.p] = 'border'; }
  }
  return { labels, kind, current, neighbors, clusters };
}
