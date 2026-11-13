import { describe, expect, it } from 'vitest';
import { stageConnections, stageIndices, stagePosition, type ExperimentStage } from './experiment';

describe('3D experiment sampling and geometry', () => {
  it('preserves parallel branches and residual inputs rather than inventing sequential edges', () => {
    const stages: ExperimentStage[] = [
      { name: 'input', values: [1] }, { name: 'branch A', values: [2], inputs: [0] },
      { name: 'branch B', values: [3], inputs: [0] }, { name: 'merge', values: [5], inputs: [1, 2] },
      { name: 'output', values: [5] },
    ];
    expect(stageConnections(stages)).toEqual([{ from: 0, to: 1 }, { from: 0, to: 2 }, { from: 1, to: 3 }, { from: 2, to: 3 }, { from: 3, to: 4 }]);
  });
  it('keeps highlighted neighbors within the rendering budget', () => {
    const indices = stageIndices({ name: 'neighbors', values: new Float64Array(1000), highlight: [3, 501, 999] }, 64);
    expect(indices.length).toBe(64);
    expect(new Set(indices).size).toBe(64);
    expect(indices.includes(3) && indices.includes(501) && indices.includes(999)).toBe(true);
    expect(indices.every((i) => Number.isInteger(i) && i >= 0 && i < 1000)).toBe(true);
  });
  it('handles empty and single-value budgets without invalid indices', () => {
    expect(stageIndices({ name: 'empty', values: [] })).toEqual([]);
    expect(stageIndices({ name: 'one', values: new Float64Array(100), highlight: [70] }, 1)).toEqual([70]);
    expect(stageIndices({ name: 'none', values: [1, 2] }, 0)).toEqual([]);
  });
  it('keeps tensor rows and columns in the visible plane and channels at distinct depths', () => {
    const stage: ExperimentStage = { name: 'tensor', kind: 'tensor', values: new Float32Array(24), shape: [2, 3, 4] };
    const first = stagePosition(stage, 0, 0, 24), col = stagePosition(stage, 1, 1, 24), row = stagePosition(stage, 3, 3, 24), channel = stagePosition(stage, 6, 6, 24);
    expect(col[0]).toBeGreaterThan(first[0]);
    expect(row[1]).toBeLessThan(first[1]);
    expect(channel[0]).toBe(first[0]);
    expect(channel[1]).toBe(first[1]);
    expect(channel[2]).toBeGreaterThan(first[2]);
  });
  it('uses true point positions and fits sampled vectors within a layer', () => {
    const stage: ExperimentStage = { name: 'points', kind: 'points', values: [1, 1], points: [[-2, 1, 0], [2, -1, 0]] };
    expect(stagePosition(stage, 0, 0, 2)[0]).toBeLessThan(0);
    expect(stagePosition(stage, 1, 1, 2)[0]).toBeGreaterThan(0);
    expect(Math.abs(stagePosition({ name: 'vector', values: new Float64Array(1000) }, 999, 63, 64)[0])).toBeLessThan(2);
  });
});
