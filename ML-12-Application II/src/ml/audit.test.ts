import { describe, expect, it } from 'vitest';
import { binaryCrossEntropy, categoricalCrossEntropy } from './losses';
import { Network } from './nn/network';
import { Trainer } from './nn/trainer';
import { CNN, CnnTrainer, cnnAccuracy, cnnShapes, loadDigits, DEFAULT_CNN, type CnnLayer } from './cnn';
import pretrained from '../data/cnnPretrained.json';
import { DIGIT_PIXELS, DIGIT_LABELS } from '../data/digits';
import { trainTestSplit } from '../data/datasets';
import { SeqModel } from './sequence';
import { VisionBackbone, VisionTrainer } from './vision';
import { parseBuilderConfig } from './nn/config';
import { customSurface, runOptimizer, SURFACES } from './optim';
import { classificationReport } from './metrics';
import { columnStats, parseCSV } from '../data/explorer';
import { grayscaleImage, digitInput } from '../data/imageInput';
import type { LayerSpec } from './nn/types';

const optimizer = { name: 'adam' as const, lr: 0.01, momentum: 0.9, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 };
const layers: LayerSpec[] = [{ id: 'i', kind: 'input', name: 'Input', units: 1 }, { id: 'h', kind: 'dense', name: 'Hidden', units: 4, activation: 'tanh' }, { id: 'd', kind: 'dropout', name: 'Dropout', rate: 0.5 }, { id: 'o', kind: 'output', name: 'Output', units: 1, activation: 'sigmoid' }];
const training = { optimizer, batchSize: 2, loss: 'bce' as const, shuffle: true, seed: 9, earlyStopping: false, patience: 2 };
const data = { Xtr: [[-1], [1], [0.2]], ytr: [0, 1, 1], Xval: [[0.4]], yval: [1], Xte: [[-0.4]], yte: [0], nClasses: 2 };

describe('audit: stable loss and training state', () => {
  it('reproduces the published pretrained CNN test accuracy on the actual held-out split', () => {
    const net = new CNN(DEFAULT_CNN, { c: 1, h: 16, w: 16 }, 7), digits = loadDigits(DIGIT_PIXELS, DIGIT_LABELS);
    pretrained.params.forEach((p, i) => { if (p && net.params[i]) { net.params[i]!.W.set(p.W); net.params[i]!.b.set(p.b); } });
    const test = trainTestSplit(digits.y, 0.2, 42).test;
    expect(cnnAccuracy(net, test.map((i) => digits.X[i]), test.map((i) => digits.y[i])).acc).toBeCloseTo(pretrained.testAcc, 12);
  });
  it('does not cap confident mistakes or produce negative perfect-prediction loss', () => {
    expect(binaryCrossEntropy(1000, 0)).toBe(1000);
    expect(binaryCrossEntropy(-1000, 1)).toBe(1000);
    expect(binaryCrossEntropy(1000, 1)).toBe(0);
    expect(categoricalCrossEntropy([1000, -1000], 1)).toBe(2000);
    const net = new Network([layers[0], layers[3]], 1); net.params[1]!.W[0] = 1000; net.params[1]!.b[0] = 0;
    expect(net.backward(net.forward([1]), [0], 'bce').loss).toBe(1000);
    const seq = new SeqModel({ cell: 'rnn', hidden: 2, layers: 1, inDim: 1, outDim: 1, seed: 1 }); seq.Wy.fill(0); seq.by[0] = 1000;
    expect(seq.loss(seq.forward([[1]], 'memory'), [[0]], 'memory')).toBe(1000);
  });
  it('computes extreme CNN and vision-head loss from logits', () => {
    const cnn = new CNN([{ kind: 'flatten' }, { kind: 'output', units: 2 }], { c: 1, h: 1, w: 1 }, 1);
    cnn.params[1]!.W.fill(0); cnn.params[1]!.b.set([1000, -1000]);
    expect(cnn.backward(cnn.forward(new Float32Array([0])), 1).loss).toBe(2000);
    const trainer = new VisionTrainer(new VisionBackbone('vgg16'), { Xtr: [new Float32Array(256)], ytr: [1], Xte: [new Float32Array(256)], yte: [1] });
    trainer.W.fill(0); trainer.b[0] = 1000; trainer.b[1] = -1000;
    expect(trainer.gradients([new Float64Array(trainer.mean.length)], [1]).loss).toBeGreaterThan(1999);
  });
  it('replays shuffled dropout training exactly after restoring an epoch', () => {
    const trainer = new Trainer(layers, data, training);
    for (let i = 0; i < 5; i++) trainer.trainEpoch();
    const expected = trainer.net.cloneParams(), loss = trainer.history.at(-1)!.loss;
    trainer.restore(2);
    for (let i = 0; i < 3; i++) trainer.trainEpoch();
    expect(trainer.net.cloneParams()).toEqual(expected);
    expect(trainer.history.at(-1)!.loss).toBe(loss);
  });
  it('restores early-stopping patience instead of silently restarting its counter', () => {
    const trainer = new Trainer(layers, data, { ...training, earlyStopping: true, optimizer: { ...optimizer, lr: 0 } });
    trainer.trainEpoch(); trainer.trainEpoch(); trainer.restore(2); trainer.trainEpoch();
    expect(trainer.stopped).toBe('early-stop@3');
    const steps = trainer.steps; trainer.trainEpoch(); expect(trainer.steps).toBe(steps);
  });
  it('stops non-finite gradients before corrupting weights', () => {
    const trainer = new Trainer(layers, { ...data, Xtr: [[Infinity]], ytr: [0] }, training);
    const before = trainer.net.cloneParams(); trainer.trainBatch();
    expect(trainer.stopped).toBe('non-finite-gradient'); expect(trainer.net.cloneParams()).toEqual(before);
  });
  it('exposes the actual available epoch when old snapshots were discarded', () => {
    const trainer = new Trainer([layers[0], layers[3]], data, { ...training, batchSize: 3 });
    for (let i = 0; i < 405; i++) trainer.trainEpoch();
    expect(trainer.snapshots.length).toBeLessThanOrEqual(400);
    expect(trainer.snapshotAt(1)!.epoch).toBe(0);
    trainer.restore(1); expect(trainer.epoch).toBe(0);
  });
  it('records sample-weighted CNN epoch loss for partial final batches', () => {
    const net = new CNN([{ kind: 'flatten' }, { kind: 'output', units: 2 }], { c: 1, h: 1, w: 1 }, 1);
    net.params[1]!.W.fill(0); net.params[1]!.b.set([2, 0]);
    const trainer = new CnnTrainer(net, [0, 0, 0].map(() => new Float32Array([0])), [0, 0, 1], { ...optimizer, lr: 0 }, 2, 1);
    trainer.trainBatch(); trainer.trainBatch();
    const expected = (2 * categoricalCrossEntropy([2, 0], 0) + categoricalCrossEntropy([2, 0], 1)) / 3;
    expect(trainer.history.length).toBe(1); expect(trainer.history[0].loss).toBeCloseTo(expected, 10);
  });
});

describe('audit: imported data and browser boundary conditions', () => {
  it('rejects invalid model settings before applying state', () => {
    const valid = { architecture: layers, dataset: { id: 'circles', n: 300, testRatio: 0.2 }, training: { optimizer: 'adam', maxEpochs: 500, batchSize: 100000 } };
    expect(parseBuilderConfig(JSON.stringify(valid), 'neural').architecture).toEqual(layers);
    for (const dataset of [{ id: 'missing' }, { n: '300' }, { testRatio: 1 }, { valRatio: 2 }]) expect(() => parseBuilderConfig(JSON.stringify({ ...valid, dataset }), 'neural')).toThrow();
    for (const cfg of [{ beta1: 1 }, { optimizer: 'unknown' }, { lr: -1 }, { batchSize: 0 }, { maxEpochs: Infinity }]) expect(() => parseBuilderConfig(JSON.stringify({ ...valid, training: cfg }), 'neural')).toThrow();
  });
  it('rejects duplicate layer IDs and invalid activations, permits supported blueprints', () => {
    expect(() => parseBuilderConfig(JSON.stringify({ architecture: [layers[0], layers[0]] }), 'neural')).toThrow();
    expect(() => parseBuilderConfig(JSON.stringify({ architecture: [{ ...layers[0], activation: 'unknown' }] }), 'neural')).toThrow();
    const arch = [{ ...layers[0] }, { id: 'conv', kind: 'conv2d', name: 'Conv', filters: 4, kernelSize: 3 }, layers[3]];
    expect(parseBuilderConfig(JSON.stringify({ architecture: arch }), 'model').architecture[1].kind).toBe('conv2d');
    expect(() => parseBuilderConfig(JSON.stringify({ architecture: arch }), 'neural')).toThrow();
  });
  it('rejects invalid CNN shapes, dropout rates and excessive allocations', () => {
    const arch: CnnLayer[] = [{ kind: 'conv', filters: 2, k: 3, stride: 0, pad: 'same', act: 'relu' }, { kind: 'flatten' }, { kind: 'output', units: 2 }];
    expect(cnnShapes(arch, { c: 1, h: 4, w: 4 }).errors.length).toBeGreaterThan(0);
    expect(() => new CNN(arch, { c: 1, h: 4, w: 4 }, 1)).toThrow();
    expect(cnnShapes([{ kind: 'dropout', rate: NaN }, { kind: 'flatten' }, { kind: 'output', units: 2 }], { c: 1, h: 4, w: 4 }).errors.length).toBeGreaterThan(0);
    expect(() => new CNN([{ kind: 'flatten' }, { kind: 'dense', units: 1000000, act: 'relu' }, { kind: 'output', units: 2 }], { c: 1, h: 16, w: 16 }, 1)).toThrow();
  });
  it('detects CSV delimiters outside quoted fields and strips BOMs', () => {
    const parsed = parseCSV('\uFEFF"feature,a,b,c";value;label\n1;2;a\n3;4;b');
    expect(parsed.header).toEqual(['feature,a,b,c', 'value', 'label']); expect(parsed.rows[1]).toEqual(['3', '4', 'b']);
    expect(() => parseCSV('a,b\n1,"unterminated')).toThrow();
  });
  it('supports large uploaded columns without spread-argument overflow', () => {
    const stats = columnStats(Array.from({ length: 150000 }, (_, i) => [i]));
    expect(stats[0].min).toBe(0); expect(stats[0].max).toBe(149999);
  });
  it('rejects undefined custom surfaces and stops invalid gradients', () => {
    expect(() => customSurface('sqrt(-1)', [-5, 5, -5, 5])).toThrow();
    expect(() => customSurface('1/x', [-5, 5, -5, 5])).toThrow();
    const run = runOptimizer({ ...SURFACES[0], grad: () => [NaN, 1] }, { ...optimizer, name: 'gd', noise: 0, seed: 1 }, [0, 0], 5);
    expect(run.diverged).toBe(true); expect(run.path.length).toBe(0);
  });
  it('does not silently lose invalid classification predictions', () => {
    expect(() => classificationReport([0], [-1], 2)).toThrow(); expect(() => classificationReport([0, 1], [0], 2)).toThrow();
  });
  it('normalizes black/white and transparent uploads consistently', () => {
    const rgba = (whiteInk: boolean, transparent = false) => Uint8ClampedArray.from({ length: 5 * 5 * 4 }, (_, i) => { const pixel = Math.floor(i / 4), ink = pixel === 12; return i % 4 === 3 ? transparent && !ink ? 0 : 255 : ink === whiteInk ? 255 : 0; });
    const reference = digitInput(grayscaleImage(rgba(true), 5, 5), 5, 5);
    expect(digitInput(grayscaleImage(rgba(false), 5, 5), 5, 5)).toEqual(reference);
    expect(digitInput(grayscaleImage(rgba(true, true), 5, 5), 5, 5)).toEqual(reference);
    expect(digitInput(grayscaleImage(rgba(false, true), 5, 5), 5, 5)).toEqual(reference);
    expect(digitInput(new Float32Array(25), 5, 5).every((v) => v === 0)).toBe(true);
  });
});
