/** Avoid function-argument limits when reducing user-uploaded columns. */
export function extent(values: Iterable<number>): [number, number] {
  let min = Infinity, max = -Infinity;
  for (const value of values) if (Number.isFinite(value)) { min = Math.min(min, value); max = Math.max(max, value); }
  return min === Infinity ? [NaN, NaN] : [min, max];
}
