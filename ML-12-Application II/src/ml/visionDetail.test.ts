import { describe, expect, it } from 'vitest';
import { VISION_IDS } from '../content/visionModels';
import { VisionBackbone, VisionTrainer } from './vision';
import { visionCalculation, visionGroupIndices, visionNetwork } from './visionDetail';

describe('live vision algorithm detail', () => {
  for (const id of VISION_IDS) it(`${id}: always shows the image, complete network path and all output classes together`, () => {
    const backbone = new VisionBackbone(id), image = Float32Array.from({ length: 256 }, (_, i) => (i % 17) / 16);
    const trace = backbone.forward(image), trainer = new VisionTrainer(backbone, { Xtr: [image], ytr: [3], Xte: [image], yte: [3] });
    const selections = [...new Set([0, 1, ...backbone.nodes.flatMap((node, i) => ['add', 'concat', 'attention', 'depthwise'].includes(node.op) ? [i] : []), backbone.nodes.length - 1])];
    for (const selected of selections) {
      const layers = visionNetwork(backbone, trace, trainer, selected);
      expect(layers[0].role).toBe('input');
      expect(Array.from(layers[0].tensor.a)).toEqual(Array.from(image));
      expect(layers.some((layer) => layer.nodeIndex === selected)).toBe(true);
      expect(backbone.overview.every((index) => layers.some((layer) => layer.nodeIndex === index))).toBe(true);
      expect(layers[layers.length - 1].role).toBe('output');
      expect(layers[layers.length - 1].tensor.c).toBe(10);
      expect(layers[layers.length - 1].tensor.a).toEqual(trainer.predict(trace.features));
      expect(layers[layers.length - 2].tensor.a).toEqual(trainer.logits(trace.features));
      const reachable = new Set([0]);
      layers.forEach((layer, i) => {
        expect(layer.inputs.every((parent) => parent >= 0 && parent < i)).toBe(true);
        if (i > 0) expect(layer.inputs.some((parent) => reachable.has(parent))).toBe(true);
        reachable.add(i);
      });
      if (['add', 'concat'].includes(backbone.nodes[selected].op)) expect(layers.find((layer) => layer.nodeIndex === selected)!.inputs.length).toBeGreaterThan(1);
    }
  });

  for (const id of VISION_IDS) it(`${id}: every operation inspector reproduces actual forward values`, () => {
    const backbone = new VisionBackbone(id, 9);
    const trace = backbone.forward(Float32Array.from({ length: 256 }, (_, i) => (i % 23) / 22));
    backbone.nodes.forEach((_, nodeIndex) => {
      const tensor = trace.tensors[nodeIndex];
      const cells = [...new Set([0, Math.floor(tensor.a.length / 2), tensor.a.length - 1])];
      for (const cell of cells) {
        const calculation = visionCalculation(backbone, trace, nodeIndex, cell);
        expect(Number.isFinite(calculation.result)).toBe(true);
        expect(calculation.result).toBeCloseTo(tensor.a[cell], 8);
      }
    });
  });

  it('includes real branch dependencies and keeps them before their consumers', () => {
    for (const id of VISION_IDS) {
      const backbone = new VisionBackbone(id);
      for (const group of new Set(backbone.nodes.map((node) => node.group))) {
        const indices = visionGroupIndices(backbone, group);
        for (const index of indices.filter((i) => backbone.nodes[i].group === group)) {
          for (const parent of backbone.nodes[index].inputs) {
            expect(indices.includes(parent)).toBe(true);
            expect(indices.indexOf(parent)).toBeLessThan(indices.indexOf(index));
          }
        }
      }
    }
  });

  it('shows zero padding and channel-specific depthwise kernels', () => {
    const backbone = new VisionBackbone('mobilenetv2');
    const trace = backbone.forward(new Float32Array(256).fill(1));
    const conv = visionCalculation(backbone, trace, 1, 0);
    expect(conv.terms.some((term) => term.index === -1 && term.value === 0)).toBe(true);
    const depthwise = backbone.nodes.findIndex((node) => node.op === 'depthwise');
    const tensor = trace.tensors[depthwise];
    const calc = visionCalculation(backbone, trace, depthwise, tensor.h * tensor.w);
    expect(calc.terms.every((term) => term.label.startsWith('c1 '))).toBe(true);
  });

  it('distinguishes CLS from image patch projection after a patch-size change', () => {
    const backbone = new VisionBackbone('vit', 7, 2, 8), trace = backbone.forward(new Float32Array(256).fill(0.5));
    expect(trace.tensors[1].h).toBe(5);
    expect(visionCalculation(backbone, trace, 1, 0).terms.length).toBe(0);
    const patch = visionCalculation(backbone, trace, 1, 1);
    expect(patch.terms.length).toBe(64);
    expect(patch.result).toBeCloseTo(trace.tensors[1].a[1], 10);
  });
});
