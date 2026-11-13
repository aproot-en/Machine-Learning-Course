import { describe, expect, it } from 'vitest';
import { VISION_IDS } from '../content/visionModels';
import { VisionBackbone, VisionTrainer, softmax, type VisionData } from './vision';

const image = Float32Array.from({ length: 256 }, (_, i) => (i % 17) / 16);
const data: VisionData = { Xtr: [image, image.map((v) => 1 - v)], ytr: [0, 1], Xte: [image.map((v) => v * 0.2)], yte: [0] };

describe('vision architecture execution', () => {
  for (const id of VISION_IDS) it(`${id}: all tensors have valid shapes and finite values`, () => {
    const model = new VisionBackbone(id, 7), trace = model.forward(image);
    expect(trace.tensors.length).toBe(model.nodes.length);
    trace.tensors.forEach((tensor) => {
      expect(tensor.a.length).toBe(tensor.h * tensor.w * tensor.c);
      expect(Array.from(tensor.a).every(Number.isFinite)).toBe(true);
    });
    expect(model.forward(image, false).features).toEqual(trace.features);
  });
  it('preserves each family’s defining block counts', () => {
    expect(new VisionBackbone('vgg16').nodes.filter((n) => n.op === 'conv').length).toBe(13);
    expect(new VisionBackbone('resnet50').nodes.filter((n) => n.op === 'add').length).toBe(16);
    expect(new VisionBackbone('densenet121').nodes.filter((n) => n.op === 'concat').length).toBe(58);
    expect(new VisionBackbone('mobilenetv2').nodes.filter((n) => n.op === 'depthwise').length).toBe(17);
    expect(new VisionBackbone('vit').nodes.filter((n) => n.op === 'attention').length).toBe(12);
    expect(new VisionBackbone('inceptionv3').nodes.filter((n) => n.op === 'concat' && n.group.startsWith('Inception-')).length).toBe(13); // 9 module outputs + 4 internal split concats
  });
  it('adds residual tensors element by element and concatenates dense channels', () => {
    const res = new VisionBackbone('resnet50'), rt = res.forward(image);
    res.nodes.forEach((node, i) => { if (node.op === 'add') rt.tensors[i].a.forEach((v, j) => expect(v).toBeCloseTo(rt.tensors[node.inputs[0]].a[j] + rt.tensors[node.inputs[1]].a[j], 12)); });
    const dense = new VisionBackbone('densenet121'), dt = dense.forward(image);
    dense.nodes.forEach((node, i) => { if (node.op === 'concat') expect(Array.from(dt.tensors[i].a)).toEqual(node.inputs.flatMap((parent) => Array.from(dt.tensors[parent].a))); });
  });
  it('keeps depthwise channels independent', () => {
    const model = new VisionBackbone('mobilenetv2'), index = model.nodes.findIndex((n) => n.op === 'depthwise'), before = model.forward(image).tensors[index];
    model.nodes[index].W!.fill(0, 9, 18);
    const after = model.forward(image).tensors[index], spatial = before.h * before.w;
    expect(after.a.slice(0, spatial)).toEqual(before.a.slice(0, spatial));
    expect(Array.from(after.a.slice(spatial, 2 * spatial)).some((v, i) => v !== before.a[spatial + i])).toBe(true);
  });
  it('creates image patches and probability-normalized attention at both patch sizes', () => {
    for (const patch of [4, 8]) {
      const model = new VisionBackbone('vit', 7, 1, patch), trace = model.forward(image), tokens = (16 / patch) ** 2 + 1;
      expect(model.nodes[1].shape.h).toBe(tokens);
      trace.attention.forEach((heads) => heads.forEach((map) => {
        expect(map.length).toBe(tokens * tokens);
        for (let row = 0; row < tokens; row++) expect(map.slice(row * tokens, (row + 1) * tokens).reduce((a, b) => a + b, 0)).toBeCloseTo(1, 12);
      }));
    }
  });
  it('is reproducible for the same seed and changes with another seed', () => {
    const first = new VisionBackbone('vit', 3).forward(image).features;
    expect(new VisionBackbone('vit', 3).forward(image).features).toEqual(first);
    expect(Array.from(new VisionBackbone('vit', 4).forward(image).features).some((v, i) => v !== first[i])).toBe(true);
  });
});

describe('frozen feature classifier training', () => {
  it('fits the scaler on training features only', () => {
    const trainer = new VisionTrainer(new VisionBackbone('vgg16'), data);
    while (!trainer.ready) trainer.prepareOne();
    trainer.mean.forEach((mean, j) => expect(mean).toBeCloseTo((trainer.features[0][j] + trainer.features[1][j]) / 2, 12));
  });
  it('matches numerical gradients for softmax cross-entropy with L2', () => {
    const trainer = new VisionTrainer(new VisionBackbone('vgg16'), data), d = trainer.mean.length;
    const features = [Float64Array.from({ length: d }, (_, i) => Math.sin(i)), Float64Array.from({ length: d }, (_, i) => Math.cos(i))], labels = [2, 7];
    const gradient = trainer.gradients(features, labels, 0.01), h = 1e-5;
    for (let i = 0; i < trainer.W.length; i += 13) {
      const initial = trainer.W[i]; trainer.W[i] = initial + h; const plus = trainer.gradients(features, labels, 0.01).loss;
      trainer.W[i] = initial - h; const minus = trainer.gradients(features, labels, 0.01).loss; trainer.W[i] = initial;
      expect(Math.abs((plus - minus) / (2 * h) - gradient.gW[i])).toBeLessThan(1e-7);
    }
  });
  it('learns a separable feature task without modifying backbone weights', () => {
    const backbone = new VisionBackbone('vgg16'), trainer = new VisionTrainer(backbone, data, 0.05), d = backbone.featureCount;
    const original = backbone.nodes.find((n) => n.W)!.W!.slice();
    trainer.features = [new Float64Array(d).fill(-1), new Float64Array(d).fill(1), new Float64Array(d).fill(-1)]; trainer.ready = true;
    const initial = trainer.evaluate(false).loss;
    for (let i = 0; i < 40; i++) trainer.trainBatch(2);
    expect(trainer.evaluate(false).loss).toBeLessThan(initial * 0.1);
    expect(trainer.evaluate(true).report.accuracy).toBe(1);
    expect(backbone.nodes.find((n) => n.W)!.W).toEqual(original);
  });
  it('keeps softmax stable for large logits', () => {
    const p = softmax([1000, 1001, 1002]);
    expect(p.reduce((a, b) => a + b, 0)).toBeCloseTo(1, 14);
    expect(p.every(Number.isFinite)).toBe(true);
  });
});
