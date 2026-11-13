import { Rng } from '../lib/random';

export type Criterion = 'gini' | 'entropy';

export interface TreeParams {
  maxDepth: number;
  minSamplesSplit: number;
  minSamplesLeaf: number;
  criterion: Criterion;
  /** features tried per split: all, sqrt(d), or a fixed count */
  maxFeatures: 'all' | 'sqrt' | number;
  /** cost-complexity pruning α: drop splits whose weighted impurity gain is below it */
  ccpAlpha: number;
}

export interface TreeNode {
  id: number;
  /** creation order, used to animate growth */
  order: number;
  depth: number;
  samples: number[];
  counts: number[];
  impurity: number;
  prediction: number;
  feature?: number;
  threshold?: number;
  gain?: number;
  left?: TreeNode;
  right?: TreeNode;
}

export function impurity(counts: number[], c: Criterion): number {
  const n = counts.reduce((a, b) => a + b, 0);
  if (!n) return 0;
  if (c === 'gini') return 1 - counts.reduce((s, k) => s + (k / n) ** 2, 0);
  return -counts.reduce((s, k) => (k ? s + (k / n) * Math.log2(k / n) : s), 0);
}

const countsOf = (idx: number[], y: number[], K: number) => { const c = Array(K).fill(0); idx.forEach((i) => c[y[i]]++); return c; };
const argmax = (a: number[]) => a.indexOf(Math.max(...a));

/** CART classification tree (binary splits on thresholds halfway between sorted values). */
export function buildTree(X: number[][], y: number[], K: number, p: TreeParams, idx: number[] = X.map((_, i) => i), seed = 1): TreeNode {
  const rng = new Rng(seed);
  const d = X[0].length;
  let nextId = 0;
  const nF = p.maxFeatures === 'all' ? d : p.maxFeatures === 'sqrt' ? Math.max(1, Math.round(Math.sqrt(d))) : Math.min(d, p.maxFeatures);
  const N = idx.length;
  // breadth-first growth so `order` matches a level-by-level animation
  const root: TreeNode = mk(idx, 0);
  const queue = [root];
  while (queue.length) {
    const node = queue.shift()!;
    if (node.depth >= p.maxDepth || node.samples.length < p.minSamplesSplit || node.impurity === 0) continue;
    const feats = rng.shuffle([...Array(d).keys()]).slice(0, nF);
    let best: { f: number; t: number; gain: number; L: number[]; R: number[] } | null = null;
    for (const f of feats) {
      const sorted = [...node.samples].sort((a, b) => X[a][f] - X[b][f]);
      const left = Array(K).fill(0), right = countsOf(sorted, y, K);
      for (let k = 0; k < sorted.length - 1; k++) {
        left[y[sorted[k]]]++; right[y[sorted[k]]]--;
        const v0 = X[sorted[k]][f], v1 = X[sorted[k + 1]][f];
        if (v0 === v1) continue;
        const nl = k + 1, nr = sorted.length - nl;
        if (nl < p.minSamplesLeaf || nr < p.minSamplesLeaf) continue;
        const w = (nl * impurity(left, p.criterion) + nr * impurity(right, p.criterion)) / sorted.length;
        const gain = node.impurity - w;
        if (!best || gain > best.gain + 1e-12) best = { f, t: (v0 + v1) / 2, gain, L: sorted.slice(0, nl), R: sorted.slice(nl) };
      }
    }
    if (!best || best.gain <= 1e-12 || (best.gain * node.samples.length) / N < p.ccpAlpha) continue;
    node.feature = best.f; node.threshold = best.t; node.gain = best.gain;
    node.left = mk(best.L, node.depth + 1);
    node.right = mk(best.R, node.depth + 1);
    queue.push(node.left, node.right);
  }
  return root;

  function mk(s: number[], depth: number): TreeNode {
    const counts = countsOf(s, y, K);
    const id = nextId++;
    return { id, order: id, depth, samples: s, counts, impurity: impurity(counts, p.criterion), prediction: argmax(counts) };
  }
}

export function predictTree(node: TreeNode, x: number[], maxOrder = Infinity): { cls: number; path: number[]; probs: number[] } {
  const path: number[] = [];
  let n = node;
  for (;;) {
    path.push(n.id);
    if (n.feature == null || !n.left || !n.right || n.right.order >= maxOrder) break;
    n = x[n.feature] <= n.threshold! ? n.left : n.right;
  }
  const tot = n.counts.reduce((a, b) => a + b, 0) || 1;
  return { cls: n.prediction, path, probs: n.counts.map((c) => c / tot) };
}

export function treeNodes(root: TreeNode): TreeNode[] {
  const out: TreeNode[] = [];
  const walk = (n: TreeNode) => { out.push(n); if (n.left) walk(n.left); if (n.right) walk(n.right); };
  walk(root);
  return out.sort((a, b) => a.order - b.order);
}

/** Mean-decrease-in-impurity importance, normalised to sum to 1. */
export function featureImportance(root: TreeNode, d: number, N = root.samples.length): number[] {
  const imp = Array(d).fill(0);
  treeNodes(root).forEach((n) => { if (n.feature != null && n.gain != null) imp[n.feature] += (n.gain * n.samples.length) / N; });
  const s = imp.reduce((a, b) => a + b, 0) || 1;
  return imp.map((v) => v / s);
}

export interface Forest {
  trees: TreeNode[];
  bootstraps: number[][];
  oob: number | null;
}

export function buildForest(X: number[][], y: number[], K: number, p: TreeParams, nTrees: number, bootstrap: boolean, seed: number): Forest {
  const rng = new Rng(seed);
  const trees: TreeNode[] = [], bootstraps: number[][] = [];
  for (let t = 0; t < nTrees; t++) {
    const idx = bootstrap ? Array.from({ length: X.length }, () => rng.int(X.length)) : X.map((_, i) => i);
    bootstraps.push(idx);
    trees.push(buildTree(X, y, K, p, idx, seed * 1000 + t));
  }
  let oob: number | null = null;
  if (bootstrap) {
    let ok = 0, n = 0;
    X.forEach((x, i) => {
      const votes = Array(K).fill(0); let any = false;
      trees.forEach((tr, t) => { if (!bootstraps[t].includes(i)) { votes[predictTree(tr, x).cls]++; any = true; } });
      if (any) { n++; if (argmax(votes) === y[i]) ok++; }
    });
    oob = n ? ok / n : null;
  }
  return { trees, bootstraps, oob };
}

export function predictForest(f: Forest, x: number[], K: number, nUsed = f.trees.length) {
  const votes = Array(K).fill(0);
  const each = f.trees.slice(0, nUsed).map((t) => predictTree(t, x).cls);
  each.forEach((c) => votes[c]++);
  return { cls: argmax(votes), votes, each };
}
