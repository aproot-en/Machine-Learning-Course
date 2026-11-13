import { describe, expect, it } from 'vitest';
import { distance, knnPredict, accuracyByK } from './knn';
import { classificationReport } from './metrics';
import { Network } from './nn/network';
import { Trainer } from './nn/trainer';
import { validateArchitecture } from './nn/validate';
import type { LayerSpec, Loss } from './nn/types';
import { makeDataset, trainTestSplit } from '../data/datasets';

describe('distance', () => {
  it('matches hand calculations', () => {
    expect(distance([0, 0], [3, 4], 'euclidean')).toBeCloseTo(5);
    expect(distance([0, 0], [3, 4], 'manhattan')).toBeCloseTo(7);
    expect(distance([0, 0], [3, 4], 'minkowski', 2)).toBeCloseTo(5);
    expect(distance([0, 0], [3, 4], 'minkowski', 1)).toBeCloseTo(7);
  });
});

describe('knn', () => {
  const X = [[0, 0], [0, 1], [1, 0], [5, 5], [5, 6]];
  const y = [0, 0, 0, 1, 1];
  it('votes by majority', () => {
    const r = knnPredict(X, y, 2, [0.2, 0.2], { k: 3, metric: 'euclidean', p: 2, weighting: 'uniform' });
    expect(r.predicted).toBe(0);
    expect(r.votes).toEqual([3, 0]);
    expect(r.neighbors[0].index).toBe(0);
  });
  it('breaks ties toward the nearest class', () => {
    const r = knnPredict(X, y, 2, [4.9, 5.1], { k: 4, metric: 'euclidean', p: 2, weighting: 'uniform' });
    expect(r.tie).toBe(true);
    expect(r.predicted).toBe(1);
  });
  it('K=1 is perfect on its own training data', () => {
    const curve = accuracyByK(X, y, X, y, 2, { k: 1, metric: 'euclidean', p: 2, weighting: 'uniform' }, 3);
    expect(curve[0].train).toBe(1);
  });
});

describe('metrics', () => {
  it('computes macro precision/recall/F1', () => {
    const r = classificationReport([0, 0, 1, 1], [0, 1, 1, 1], 2);
    expect(r.accuracy).toBeCloseTo(0.75);
    expect(r.confusion).toEqual([[1, 1], [0, 2]]);
    expect(r.precision).toBeCloseTo((1 + 2 / 3) / 2);
    expect(r.recall).toBeCloseTo((0.5 + 1) / 2);
  });
});

function gradCheck(layers: LayerSpec[], loss: Loss, target: number[]) {
  const net = new Network(layers, 3);
  const x = [0.3, -0.7, 0.5];
  const g = net.backward(net.forward(x), target, loss);
  const h = 1e-6;
  let worst = 0;
  net.params.forEach((p, l) => {
    if (!p) return;
    for (let k = 0; k < p.W.length; k++) {
      const w0 = p.W[k];
      p.W[k] = w0 + h; const lp = net.backward(net.forward(x), target, loss).loss;
      p.W[k] = w0 - h; const lm = net.backward(net.forward(x), target, loss).loss;
      p.W[k] = w0;
      const num = (lp - lm) / (2 * h);
      worst = Math.max(worst, Math.abs(num - g.gW[l]![k]));
    }
  });
  return worst;
}

describe('backpropagation matches numerical gradients', () => {
  const base = (out: LayerSpec, hiddenAct: LayerSpec['activation']): LayerSpec[] => [
    { id: 'i', kind: 'input', name: 'in', units: 3 },
    { id: 'h1', kind: 'dense', name: 'h1', units: 4, activation: hiddenAct },
    { id: 'a', kind: 'activation', name: 'act', activation: 'tanh' },
    { id: 'h2', kind: 'dense', name: 'h2', units: 3, activation: 'gelu' },
    out,
  ];
  it('softmax + categorical cross-entropy', () => {
    expect(gradCheck(base({ id: 'o', kind: 'output', name: 'o', units: 3, activation: 'softmax' }, 'leakyrelu'), 'cce', [0, 1, 0])).toBeLessThan(1e-6);
  });
  it('sigmoid + binary cross-entropy', () => {
    expect(gradCheck(base({ id: 'o', kind: 'output', name: 'o', units: 1, activation: 'sigmoid' }, 'tanh'), 'bce', [1])).toBeLessThan(1e-6);
  });
  it('linear + MSE', () => {
    expect(gradCheck(base({ id: 'o', kind: 'output', name: 'o', units: 2, activation: 'linear' }, 'sigmoid'), 'mse', [1, 0])).toBeLessThan(1e-6);
  });
});

describe('training', () => {
  it('learns XOR with a small MLP', () => {
    const ds = makeDataset('xor', { n: 200, classes: 2, noise: 0.05, seed: 1 });
    const sp = trainTestSplit(ds.y, 0.25, 1);
    const pick = (ix: number[]) => [ix.map((i) => ds.X[i]), ix.map((i) => ds.y[i])] as const;
    const [Xtr, ytr] = pick(sp.train), [Xte, yte] = pick(sp.test);
    const layers: LayerSpec[] = [
      { id: 'i', kind: 'input', name: 'in', units: 2 },
      { id: 'h', kind: 'dense', name: 'h', units: 8, activation: 'tanh' },
      { id: 'o', kind: 'output', name: 'o', units: 1, activation: 'sigmoid' },
    ];
    const tr = new Trainer(layers, { Xtr, ytr, Xval: [], yval: [], Xte, yte, nClasses: 2 }, {
      optimizer: { name: 'adam', lr: 0.03, momentum: 0.9, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 },
      batchSize: 16, loss: 'bce', shuffle: true, seed: 2, earlyStopping: false, patience: 10,
    });
    for (let e = 0; e < 150; e++) tr.trainEpoch();
    expect(tr.history.at(-1)!.testAcc).toBeGreaterThan(0.9);
    const lossStart = tr.history[0].loss;
    tr.restore(1);
    expect(tr.epoch).toBe(1);
    expect(tr.history.length).toBe(1);
    expect(tr.history[0].loss).toBe(lossStart);
  });
});

describe('architecture validation', () => {
  const ok: LayerSpec[] = [
    { id: 'i', kind: 'input', name: 'in', units: 2 },
    { id: 'h', kind: 'dense', name: 'h', units: 4, activation: 'relu' },
    { id: 'o', kind: 'output', name: 'o', units: 3, activation: 'softmax' },
  ];
  it('accepts a valid network and counts parameters', () => {
    const v = validateArchitecture(ok, { nFeatures: 2, nClasses: 3 }, 'cce');
    expect(v.ok).toBe(true);
    expect(v.totalParams).toBe(2 * 4 + 4 + 4 * 3 + 3);
  });
  it('rejects mismatched input, output and loss', () => {
    expect(validateArchitecture(ok, { nFeatures: 4, nClasses: 3 }, 'cce').ok).toBe(false);
    expect(validateArchitecture(ok, { nFeatures: 2, nClasses: 2 }, 'cce').ok).toBe(false);
    expect(validateArchitecture(ok, { nFeatures: 2, nClasses: 3 }, 'bce').ok).toBe(false);
  });
});
