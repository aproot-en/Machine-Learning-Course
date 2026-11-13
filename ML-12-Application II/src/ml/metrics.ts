export interface ClassificationReport {
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  confusion: number[][];
  support: number;
}

/** confusion[actual][predicted]; precision/recall/F1 are macro-averaged over classes that appear. */
export function classificationReport(actual: number[], predicted: number[], nClasses: number): ClassificationReport {
  if (!Number.isInteger(nClasses) || nClasses < 1 || actual.length !== predicted.length || [...actual, ...predicted].some((label) => !Number.isInteger(label) || label < 0 || label >= nClasses)) throw new Error('Classification labels must be aligned integer class indices');
  const confusion = Array.from({ length: nClasses }, () => Array(nClasses).fill(0));
  actual.forEach((a, i) => confusion[a][predicted[i]]++);
  let correct = 0;
  for (let c = 0; c < nClasses; c++) correct += confusion[c][c];
  let p = 0, r = 0, f = 0, k = 0;
  for (let c = 0; c < nClasses; c++) {
    const tp = confusion[c][c];
    const fn = confusion[c].reduce((s, v) => s + v, 0) - tp;
    const fp = confusion.reduce((s, row) => s + row[c], 0) - tp;
    if (tp + fn === 0 && fp === 0) continue;
    const pc = tp + fp ? tp / (tp + fp) : 0;
    const rc = tp + fn ? tp / (tp + fn) : 0;
    p += pc;
    r += rc;
    f += pc + rc ? (2 * pc * rc) / (pc + rc) : 0;
    k++;
  }
  return { accuracy: actual.length ? correct / actual.length : 0, precision: k ? p / k : 0, recall: k ? r / k : 0, f1: k ? f / k : 0, confusion, support: actual.length };
}
