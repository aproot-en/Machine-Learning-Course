export interface ExperimentStage {
  name: string;
  values: ArrayLike<number>;
  kind?: 'vector' | 'tensor' | 'points' | 'tree';
  /** Tensor layout is rows × columns × channels, with channel-major values. */
  shape?: [number, number, number];
  points?: number[][];
  labels?: string[];
  classes?: number[];
  highlight?: number[];
  /** Actual tree edges, indexed into values/points. */
  edges?: [number, number][];
  /** Actual row-major dense weights from the previous stage, when applicable. */
  weights?: ArrayLike<number>;
  /** Indices of actual input stages; absent means the previous stage. */
  inputs?: number[];
}

export function stageConnections(stages: ExperimentStage[]): { from: number; to: number }[] {
  return stages.flatMap((stage, to) => (stage.inputs ?? (to > 0 ? [to - 1] : [])).filter((from) => Number.isInteger(from) && from >= 0 && from < to).map((from) => ({ from, to })));
}

export interface ModelExperiment {
  model: string;
  source: string;
  output: string;
  note?: string;
  stages: ExperimentStage[];
  metrics: { label: string; value: string }[];
  history?: { label: string; values: number[]; color?: string }[];
  confusion?: number[][];
  sample?: { index: number; count: number; onChange: (index: number) => void };
}

export function flowVector(values: ArrayLike<number> | undefined, limit = 6): string {
  if (!values?.length) return '—';
  const shown = Array.from({ length: Math.min(values.length, limit) }, (_, i) =>
    Number.isFinite(values[i]) ? Number(values[i].toFixed(3)).toString() : '—');
  return `[${shown.join(', ')}${values.length > limit ? ', …' : ''}]`;
}

/** Sample large tensors deterministically while retaining their original indices. */
export function stageIndices(stage: ExperimentStage, cap = 256): number[] {
  const size = stage.kind === 'points' || stage.kind === 'tree' ? stage.points?.length ?? stage.values.length : stage.values.length;
  if (cap <= 0 || size <= 0) return [];
  if (cap === 1) return [stage.highlight?.find((index) => index >= 0 && index < size) ?? 0];
  if (size <= cap) return Array.from({ length: size }, (_, i) => i);
  const important = new Set((stage.highlight ?? []).filter((i) => i >= 0 && i < size).slice(0, cap));
  for (let i = 0; important.size < cap && i < cap; i++) important.add(Math.round(i * (size - 1) / (cap - 1)));
  return [...important].sort((a, b) => a - b);
}

export function stagePosition(stage: ExperimentStage, index: number, shownIndex: number, shownCount: number): [number, number, number] {
  if (stage.points?.[index]) {
    const point = stage.points[index];
    const max = Math.max(1e-9, ...stage.points.flat().filter(Number.isFinite).map(Math.abs));
    return [(point[0] ?? 0) / max * 2.2, (point[1] ?? 0) / max * 2.4, (point[2] ?? 0) / max * 1.5];
  }
  if (stage.shape && stage.kind === 'tensor') {
    const [rows, cols, channels] = stage.shape;
    const col = index % cols, row = Math.floor(index / cols) % rows, channel = Math.floor(index / (rows * cols));
    const cell = Math.min(0.38, 4.5 / Math.max(rows, cols));
    return [(col - (cols - 1) / 2) * cell, ((rows - 1) / 2 - row) * cell, channel / Math.max(1, channels - 1) * Math.min(1.4, channels * 0.2)];
  }
  const columns = Math.max(1, Math.ceil(shownCount / 12));
  const rows = Math.min(12, shownCount);
  return [(Math.floor(shownIndex / 12) - (columns - 1) / 2) * 0.45, ((rows - 1) / 2 - shownIndex % rows) * 0.4, 0];
}
