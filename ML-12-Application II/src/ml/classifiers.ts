import { knnPredict } from './knn';
import { SMO, type Kernel } from './svm';
import { GDRegression } from './regression';
import { buildForest, buildTree, predictForest, predictTree, treeNodes } from './tree';
import { Trainer } from './nn/trainer';
import type { LayerSpec } from './nn/types';

export type ModelKind = 'knn' | 'svm' | 'logistic' | 'tree' | 'forest' | 'mlp';

export interface ModelParams {
  kind: ModelKind;
  k: number;
  kernel: Kernel;
  C: number;
  gamma: number;
  maxDepth: number;
  nTrees: number;
  hidden: number;
  epochs: number;
  seed: number;
}

export interface Fitted {
  /** class probabilities (or normalised scores for SVM) */
  proba: (x: number[]) => number[];
  /** a short, honest description of model size */
  complexity: string;
}

/** One-vs-rest wrapper used for the binary SVM and logistic models. */
function ovr(K: number, fitOne: (yb: number[]) => (x: number[]) => number, y: number[]): (x: number[]) => number[] {
  if (K === 2) { const f = fitOne(y.map((c) => (c === 1 ? 1 : 0))); return (x) => { const s = f(x); return [1 - s, s]; }; }
  const fs = Array.from({ length: K }, (_, c) => fitOne(y.map((v) => (v === c ? 1 : 0))));
  return (x) => { const s = fs.map((f) => f(x)); const t = s.reduce((a, b) => a + b, 0) || 1; return s.map((v) => v / t); };
}

/** Fit any of the lab models behind one interface so the Compare page can treat them identically. */
export function fitModel(p: ModelParams, X: number[][], y: number[], K: number): Fitted {
  switch (p.kind) {
    case 'knn': {
      const o = { k: p.k, metric: 'euclidean' as const, p: 2, weighting: 'uniform' as const };
      return { proba: (x) => { const r = knnPredict(X, y, K, x, o); const t = r.votes.reduce((a, b) => a + b, 0); return r.votes.map((v) => v / t); }, complexity: `${X.length} stored points` };
    }
    case 'svm': {
      let sv = 0;
      const proba = ovr(K, (yb) => {
        const m = new SMO(X, yb.map((v) => (v ? 1 : -1)), { kernel: p.kernel, C: p.C, gamma: p.gamma, degree: 3, coef0: 1, tol: 1e-3, classWeight: [1, 1] }, p.seed);
        while (!m.done) m.pass();
        sv += m.supportVectors().length;
        return (x) => 1 / (1 + Math.exp(-2 * m.decision(x)));
      }, y);
      return { proba, complexity: `${sv} support vectors` };
    }
    case 'logistic': {
      const proba = ovr(K, (yb) => {
        const m = new GDRegression(X, yb, { kind: 'logistic', optimizer: { name: 'adam', lr: 0.05, momentum: 0.9, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }, batchSize: 0, l2: 0.001, l1: 0, seed: p.seed, init: 'zeros' });
        for (let i = 0; i < 400; i++) m.step();
        return (x) => m.predict(x);
      }, y);
      return { proba, complexity: `${(X[0].length + 1) * (K === 2 ? 1 : K)} weights` };
    }
    case 'tree': {
      const t = buildTree(X, y, K, { maxDepth: p.maxDepth, minSamplesSplit: 2, minSamplesLeaf: 1, criterion: 'gini', maxFeatures: 'all', ccpAlpha: 0 }, undefined, p.seed);
      return { proba: (x) => predictTree(t, x).probs, complexity: `${treeNodes(t).length} nodes` };
    }
    case 'forest': {
      const f = buildForest(X, y, K, { maxDepth: p.maxDepth, minSamplesSplit: 2, minSamplesLeaf: 1, criterion: 'gini', maxFeatures: 'sqrt', ccpAlpha: 0 }, p.nTrees, true, p.seed);
      const nodes = f.trees.reduce((s, t) => s + treeNodes(t).length, 0);
      return { proba: (x) => predictForest(f, x, K).votes.map((v) => v / p.nTrees), complexity: `${p.nTrees} trees · ${nodes} nodes` };
    }
    case 'mlp': {
      const out = K === 2 ? { units: 1, activation: 'sigmoid' as const } : { units: K, activation: 'softmax' as const };
      const layers: LayerSpec[] = [
        { id: 'i', kind: 'input', name: 'Input', units: X[0].length },
        { id: 'h1', kind: 'dense', name: 'H1', units: p.hidden, activation: 'tanh' },
        { id: 'h2', kind: 'dense', name: 'H2', units: p.hidden, activation: 'tanh' },
        { id: 'o', kind: 'output', name: 'Output', ...out },
      ];
      const tr = new Trainer(layers, { Xtr: X, ytr: y, Xval: [], yval: [], Xte: [], yte: [], nClasses: K }, {
        optimizer: { name: 'adam', lr: 0.02, momentum: 0.9, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }, batchSize: 16, loss: K === 2 ? 'bce' : 'cce', shuffle: true, seed: p.seed, earlyStopping: false, patience: 10,
      });
      for (let e = 0; e < p.epochs; e++) tr.trainEpoch();
      return { proba: (x) => { const o = tr.net.output(x); return o.length === 1 ? [1 - o[0], o[0]] : o; }, complexity: `${tr.net.paramCount()} parameters` };
    }
  }
}

export const argmax = (a: number[]) => a.indexOf(Math.max(...a));
