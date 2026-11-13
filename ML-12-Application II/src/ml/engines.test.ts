import { describe, expect, it } from 'vitest';
import { SMO } from './svm';
import { GDRegression, closedForm, makeRegressionData } from './regression';
import { dbscanEvents, dbscanReplay, initCentroids, kmeansNext, silhouette, type KMeansState } from './clustering';
import { buildForest, buildTree, impurity, predictForest, predictTree } from './tree';
import { compileExpression, runOptimizer, SURFACES } from './optim';
import { CNN, type CnnLayer } from './cnn';
import { SeqModel, type SeqTask } from './sequence';
import { encoder } from './transformer';
import { Rng } from '../lib/random';
import { makeDataset } from '../data/datasets';

const opt = { name: 'sgd' as const, lr: 0.1, momentum: 0.9, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 };

describe('SVM (SMO)', () => {
  it('separates linearly separable data with a maximum margin', () => {
    const X = [[0, 0], [0, 1], [1, 0], [3, 3], [3, 4], [4, 3]];
    const y = [-1, -1, -1, 1, 1, 1];
    const m = new SMO(X, y, { kernel: 'linear', C: 100, gamma: 1, degree: 3, coef0: 1, tol: 1e-4, classWeight: [1, 1] });
    while (!m.done) m.pass();
    X.forEach((x, i) => expect(Math.sign(m.decision(x))).toBe(y[i]));
    // support vectors sit on the margin: y·f(x) ≈ 1
    m.supportVectors().forEach((i) => expect(y[i] * m.decision(X[i])).toBeCloseTo(1, 1));
  });
  it('RBF kernel solves concentric circles', () => {
    const ds = makeDataset('circles', { n: 120, classes: 2, noise: 0.05, seed: 2 });
    const y = ds.y.map((c) => (c ? 1 : -1));
    const m = new SMO(ds.X, y, { kernel: 'rbf', C: 5, gamma: 1, degree: 3, coef0: 1, tol: 1e-3, classWeight: [1, 1] });
    while (!m.done) m.pass();
    const acc = ds.X.filter((x, i) => Math.sign(m.decision(x)) === y[i]).length / ds.X.length;
    expect(acc).toBeGreaterThan(0.95);
  });
});

describe('regression', () => {
  it('gradient descent converges to the normal-equation solution', () => {
    const d = makeRegressionData(80, 2, 0.3, 3);
    const m = new GDRegression(d.X, d.y, { kind: 'linear', optimizer: opt, batchSize: 0, l2: 0, l1: 0, seed: 1, init: 'zeros' });
    for (let i = 0; i < 800; i++) m.step();
    const cf = closedForm(d.X, d.y);
    expect(m.w[0]).toBeCloseTo(cf.w[0], 2);
    expect(m.w[1]).toBeCloseTo(cf.w[1], 2);
    expect(m.b[0]).toBeCloseTo(cf.b, 2);
  });
  it('logistic regression loss decreases', () => {
    const ds = makeDataset('blobs2d', { n: 100, classes: 2, noise: 0.2, seed: 4 });
    const m = new GDRegression(ds.X, ds.y, { kind: 'logistic', optimizer: opt, batchSize: 0, l2: 0, l1: 0, seed: 1, init: 'zeros' });
    const l0 = m.loss();
    for (let i = 0; i < 100; i++) m.step();
    expect(m.loss()).toBeLessThan(l0 * 0.5);
  });
});

describe('clustering', () => {
  const ds = makeDataset('blobs2d', { n: 90, classes: 3, noise: 0.05, seed: 5 });
  it('k-means converges with non-increasing inertia', () => {
    let s: KMeansState = { iter: 0, phase: 'assign', centroids: initCentroids(ds.X, 3, 'kmeans++', 1), assign: ds.X.map(() => -1), inertia: null, changed: 0, displacement: 0, converged: false };
    const inertias: number[] = [];
    for (let k = 0; k < 60 && !s.converged; k++) { s = kmeansNext(ds.X, s, 1e-6, 100); if (s.phase === 'assign') inertias.push(s.inertia!); }
    expect(s.converged).toBe(true);
    for (let i = 1; i < inertias.length; i++) expect(inertias[i]).toBeLessThanOrEqual(inertias[i - 1] + 1e-9);
    expect(silhouette(ds.X, s.assign)!).toBeGreaterThan(0.6);
  });
  it('dbscan finds the blobs and replays to the final state', () => {
    const ev = dbscanEvents(ds.X, 0.9, 4, 'euclidean');
    const st = dbscanReplay(ds.X.length, ev, ev.length);
    expect(st.clusters).toBe(3);
    expect(st.labels.every((l) => l !== -2)).toBe(true);
  });
});

describe('trees', () => {
  it('gini impurity matches hand values', () => {
    expect(impurity([5, 5], 'gini')).toBeCloseTo(0.5);
    expect(impurity([10, 0], 'entropy')).toBeCloseTo(0);
    expect(impurity([5, 5], 'entropy')).toBeCloseTo(1);
  });
  it('a deep tree fits training data; a forest predicts well', () => {
    const ds = makeDataset('moons', { n: 160, classes: 2, noise: 0.1, seed: 6 });
    const p = { maxDepth: 20, minSamplesSplit: 2, minSamplesLeaf: 1, criterion: 'gini' as const, maxFeatures: 'all' as const, ccpAlpha: 0 };
    const t = buildTree(ds.X, ds.y, 2, p);
    expect(ds.X.every((x, i) => predictTree(t, x).cls === ds.y[i])).toBe(true);
    const f = buildForest(ds.X, ds.y, 2, { ...p, maxDepth: 6, maxFeatures: 'sqrt' }, 15, true, 3);
    const acc = ds.X.filter((x, i) => predictForest(f, x, 2).cls === ds.y[i]).length / ds.X.length;
    expect(acc).toBeGreaterThan(0.95);
    expect(f.oob!).toBeGreaterThan(0.85);
  });
});

describe('optimisation', () => {
  it('safe expression parser', () => {
    const f = compileExpression('x^2 + 3*sin(y) - -2');
    expect(f(2, 0)).toBeCloseTo(6);
    expect(() => compileExpression('alert(1)')).toThrow();
  });
  it('adam reaches the minimum of the convex quadratic; huge lr diverges', () => {
    const s = SURFACES[0];
    const { path } = runOptimizer(s, { ...opt, name: 'adam', lr: 0.1, noise: 0, seed: 1 }, s.start, 400);
    const last = path[path.length - 1];
    expect(Math.hypot(last.x, last.y)).toBeLessThan(0.05);
    expect(runOptimizer(s, { ...opt, name: 'gd', lr: 1.5, noise: 0, seed: 1 }, s.start, 200).diverged).toBe(true);
  });
});

describe('CNN backprop matches numerical gradients', () => {
  const layers: CnnLayer[] = [
    { kind: 'conv', filters: 2, k: 3, stride: 1, pad: 'same', act: 'tanh' },
    { kind: 'pool', mode: 'avg', size: 2, stride: 2 },
    { kind: 'conv', filters: 2, k: 3, stride: 2, pad: 'valid', act: 'tanh' },
    { kind: 'flatten' },
    { kind: 'dense', units: 4, act: 'tanh' },
    { kind: 'output', units: 3 },
  ];
  it('conv / pool / dense weights', () => {
    const net = new CNN(layers, { c: 1, h: 8, w: 8 }, 4);
    const rng = new Rng(9);
    const x = new Float32Array(64).map(() => rng.next());
    const g = net.backward(net.forward(x), 1);
    const h = 1e-3;
    let worst = 0;
    net.params.forEach((p, l) => {
      if (!p) return;
      for (let k = 0; k < p.W.length; k += 3) {
        const w0 = p.W[k];
        p.W[k] = w0 + h; const lp = net.backward(net.forward(x), 1).loss;
        p.W[k] = w0 - h; const lm = net.backward(net.forward(x), 1).loss;
        p.W[k] = w0;
        worst = Math.max(worst, Math.abs((lp - lm) / (2 * h) - g.gW[l]![k]));
      }
    });
    expect(worst).toBeLessThan(2e-3);
  });
});

describe('RNN / LSTM backprop through time matches numerical gradients', () => {
  for (const cell of ['rnn', 'lstm'] as const) for (const task of ['sine', 'memory'] as SeqTask[]) {
    it(`${cell} · ${task}`, () => {
      const m = new SeqModel({ cell, hidden: 3, layers: 2, inDim: 1, outDim: 1, seed: 2 });
      const xs = [[0.3], [-0.5], [0.8], [0.1]];
      const ys = task === 'sine' ? [[0.2], [0.4], [-0.3], [0.5]] : [[1], [1], [1], [1]];
      const g = m.backward(m.forward(xs, task), ys, task);
      const h = 1e-5;
      let worst = 0;
      m.P.forEach((p, l) => (['Wx', 'Wh', 'b'] as const).forEach((key) => {
        for (let k = 0; k < p[key].length; k++) {
          const w0 = p[key][k];
          p[key][k] = w0 + h; const lp = m.loss(m.forward(xs, task), ys, task);
          p[key][k] = w0 - h; const lm = m.loss(m.forward(xs, task), ys, task);
          p[key][k] = w0;
          worst = Math.max(worst, Math.abs((lp - lm) / (2 * h) - g.gP[l][key][k]));
        }
      }));
      expect(worst).toBeLessThan(1e-6);
    });
  }
});

describe('transformer', () => {
  it('attention rows are probability distributions; causal mask hides the future', () => {
    const tr = encoder(['the', 'cat', 'sat', 'down'], { dModel: 8, heads: 2, ffDim: 16, blocks: 1, causal: true, seed: 1 });
    tr.blocks[0].heads.forEach((h) => h.attn.forEach((row, i) => {
      expect(row.reduce((a, b) => a + b, 0)).toBeCloseTo(1);
      row.forEach((v, j) => { if (j > i) expect(v).toBe(0); });
    }));
  });
});

import { argmax, fitModel, type ModelKind } from './classifiers';
describe('compare: every model behind one interface', () => {
  const ds = makeDataset('blobs2d', { n: 150, classes: 3, noise: 0.1, seed: 8 });
  for (const kind of ['knn', 'svm', 'logistic', 'tree', 'forest', 'mlp'] as ModelKind[]) {
    it(kind, () => {
      const m = fitModel({ kind, k: 5, kernel: 'rbf', C: 1, gamma: 0.5, maxDepth: 5, nTrees: 10, hidden: 8, epochs: 60, seed: 1 }, ds.X, ds.y, 3);
      const acc = ds.X.filter((x, i) => argmax(m.proba(x)) === ds.y[i]).length / ds.X.length;
      expect(acc).toBeGreaterThan(0.9);
      const p = m.proba(ds.X[0]);
      expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 5);
    });
  }
});

import { builtin, handleMissing, parseCSV, pca, tableFromCSV } from '../data/explorer';
describe('dataset explorer', () => {
  it('PCA recovers the dominant direction and explained variance', () => {
    const rng = new Rng(3);
    const X = Array.from({ length: 400 }, () => { const t = rng.normal() * 3, e = rng.normal() * 0.3; return [t + e, t - e, rng.normal() * 0.1]; });
    const r = pca(X, 3);
    expect(Math.abs(r.components[0][0])).toBeCloseTo(Math.SQRT1_2, 1);
    expect(Math.abs(r.components[0][1])).toBeCloseTo(Math.SQRT1_2, 1);
    expect(r.ratio[0]).toBeGreaterThan(0.95);
    const dot = r.components[0].reduce((s, v, i) => s + v * r.components[1][i], 0);
    expect(Math.abs(dot)).toBeLessThan(1e-6);
  });
  it('iris PCA: first component explains about 92% (known value)', () => {
    const t = builtin('iris', { n: 0, noise: 0, seed: 1, classes: 3 });
    expect(pca(t.X as number[][], 2).ratio[0]).toBeCloseTo(0.9246, 3);
  });
  it('parses CSV with quotes, missing values and a text label', () => {
    const p = parseCSV('a,b,"name, full",label\n1,2,"x ""y""",cat\n3,,z,dog\n5,6,w,cat\n');
    expect(p.header).toEqual(['a', 'b', 'name, full', 'label']);
    expect(p.rows[0][2]).toBe('x "y"');
    const t = tableFromCSV('t', p, 3);
    expect(t.columns).toEqual(['a', 'b']);
    expect(t.X[1][1]).toBeNull();
    expect(t.classNames).toEqual(['cat', 'dog']);
    expect(handleMissing(t, 'mean').X[1][1]).toBeCloseTo(4);
    expect(handleMissing(t, 'drop').dropped).toBe(1);
  });
});
