import * as THREE from 'three';
import { Rng } from '../../lib/random';
import type { LayerSpec } from '../../ml/nn/types';

/**
 * Neuron positions per layer. Layers sit along x; small layers are a vertical column,
 * larger ones a square grid in the y–z plane (like a stack of neuron sheets).
 */
export function layoutNetwork(widths: number[], spacing: number, gap: number): THREE.Vector3[][] {
  const L = widths.length;
  return widths.map((n, l) => {
    const x = (l - (L - 1) / 2) * spacing;
    if (n <= 12) return Array.from({ length: n }, (_, j) => new THREE.Vector3(x, ((n - 1) / 2 - j) * gap, 0));
    const cols = Math.ceil(Math.sqrt(n));
    const rows = Math.ceil(n / cols);
    const g = gap * Math.min(1, 9 / cols);
    return Array.from({ length: n }, (_, j) => {
      const r = Math.floor(j / cols), c = j % cols;
      return new THREE.Vector3(x, ((rows - 1) / 2 - r) * g, (c - (cols - 1) / 2) * g);
    });
  });
}

export interface Edge {
  /** layer index of the target */
  l: number;
  /** target neuron */
  j: number;
  /** source neuron (in layer l-1) */
  i: number;
  /** element-wise (dropout/activation) link rather than a weight */
  elementwise: boolean;
}

/**
 * Edges that will be drawn. Dense layers are fully connected; when the total exceeds `max`
 * a reproducible random sample is drawn and the caller reports the rendered percentage.
 */
export function buildEdges(layers: LayerSpec[], widths: number[], max: number): { edges: Edge[]; total: number; drawn: number } {
  let total = 0;
  layers.forEach((l, k) => {
    if (k === 0) return;
    total += l.kind === 'dense' || l.kind === 'output' ? widths[k] * widths[k - 1] : widths[k];
  });
  const keep = Math.min(1, max / Math.max(1, total));
  const rng = new Rng(12345);
  const edges: Edge[] = [];
  layers.forEach((l, k) => {
    if (k === 0) return;
    if (l.kind === 'dense' || l.kind === 'output') {
      for (let j = 0; j < widths[k]; j++) for (let i = 0; i < widths[k - 1]; i++) if (keep >= 1 || rng.next() < keep) edges.push({ l: k, j, i, elementwise: false });
    } else {
      for (let j = 0; j < widths[k]; j++) edges.push({ l: k, j, i: j, elementwise: true });
    }
  });
  return { edges, total, drawn: edges.length };
}
