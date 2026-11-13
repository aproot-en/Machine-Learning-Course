import { extent } from '../lib/numbers';
import { IRIS_ROWS } from './iris';
import { WINE, BREAST_CANCER } from './realDatasets';
import { DIGIT_LABELS, DIGIT_PIXELS } from './digits';
import { makeDataset } from './datasets';
import { Rng } from '../lib/random';

/** A table of numeric features (null = missing) with an optional class label. */
export interface Table {
  name: string;
  columns: string[];
  X: (number | null)[][];
  y: number[] | null;
  classNames: string[];
  source: string;
}

export type BuiltinId = 'iris' | 'wine' | 'breast' | 'digits' | 'moons' | 'circles' | 'blobs3d' | 'spiral' | 'xor';

export const BUILTINS: { id: BuiltinId; name: string; real: boolean }[] = [
  { id: 'iris', name: 'Iris', real: true },
  { id: 'wine', name: 'Wine', real: true },
  { id: 'breast', name: 'Breast Cancer Wisconsin', real: true },
  { id: 'digits', name: 'Handwritten digits (8×8)', real: true },
  { id: 'moons', name: 'Two Moons', real: false },
  { id: 'circles', name: 'Circles', real: false },
  { id: 'blobs3d', name: 'Blobs 3D', real: false },
  { id: 'spiral', name: 'Spiral', real: false },
  { id: 'xor', name: 'XOR', real: false },
];

export function builtin(id: BuiltinId, opts: { n: number; noise: number; seed: number; classes: number }): Table {
  const split = (rows: number[][]) => ({ X: rows.map((r) => r.slice(0, -1)), y: rows.map((r) => r[r.length - 1]) });
  if (id === 'iris') return { name: 'Iris', columns: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'], ...split(IRIS_ROWS), classNames: ['setosa', 'versicolor', 'virginica'], source: 'Fisher (1936), scikit-learn load_iris' };
  if (id === 'wine') return { name: 'Wine', columns: WINE.features, ...split(WINE.rows), classNames: WINE.classes, source: 'UCI Wine, scikit-learn load_wine' };
  if (id === 'breast') return { name: 'Breast Cancer Wisconsin', columns: BREAST_CANCER.features, ...split(BREAST_CANCER.rows), classNames: BREAST_CANCER.classes, source: 'Wisconsin Diagnostic, scikit-learn load_breast_cancer' };
  if (id === 'digits') {
    const X: number[][] = [], y: number[] = [];
    for (let i = 0; i < DIGIT_LABELS.length; i++) { X.push(Array.from(DIGIT_PIXELS.slice(i * 64, i * 64 + 64), (c) => parseInt(c, 17))); y.push(Number(DIGIT_LABELS[i])); }
    return { name: 'Handwritten digits', columns: Array.from({ length: 64 }, (_, k) => `px_${Math.floor(k / 8)}_${k % 8}`), X, y, classNames: [...Array(10).keys()].map(String), source: 'UCI optical digits, scikit-learn load_digits' };
  }
  const d = makeDataset(id, { n: opts.n, classes: opts.classes, noise: opts.noise, seed: opts.seed });
  return { name: id, columns: d.featureNames, X: d.X, y: d.y, classNames: d.classNames, source: `synthetic, seed ${opts.seed}` };
}

/* ---------------- CSV ---------------- */

/** RFC-4180-ish parser: quoted fields, escaped quotes, comma / semicolon / tab delimiter detected from the header line. */
export function parseCSV(text: string): { header: string[]; rows: string[][] } {
  text = text.replace(/^\uFEFF/, '');
  const counts = new Map([[',', 0], [';', 0], ['\t', 0]]); let quoted = false;
  for (let i = 0; i < text.length; i++) {
    if (text[i] === '"') { if (quoted && text[i + 1] === '"') i++; else quoted = !quoted; }
    else if (!quoted) { if (text[i] === '\r' || text[i] === '\n') break; if (counts.has(text[i])) counts.set(text[i], counts.get(text[i])! + 1); }
  }
  const delim = [...counts].sort((a, b) => b[1] - a[1])[0][0];
  const out: string[][] = [];
  let row: string[] = [], cell = '', q = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (q) {
      if (ch === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (ch === '"') q = false;
      else cell += ch;
    } else if (ch === '"') q = true;
    else if (ch === delim) { row.push(cell); cell = ''; }
    else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && text[i + 1] === '\n') i++;
      row.push(cell); cell = '';
      if (row.some((c) => c.trim() !== '')) out.push(row);
      row = [];
    } else cell += ch;
  }
  if (q) throw new Error('Unterminated quoted CSV field');
  row.push(cell);
  if (row.some((c) => c.trim() !== '')) out.push(row);
  const header = (out.shift() ?? []).map((h, i) => h.trim() || `col_${i + 1}`);
  return { header, rows: out };
}

const MISSING = new Set(['', 'na', 'nan', 'null', 'none', '?', 'n/a']);
export const isMissing = (s: string) => MISSING.has(s.trim().toLowerCase());

export function columnKinds(p: { header: string[]; rows: string[][] }): ('numeric' | 'text')[] {
  return p.header.map((_, c) => {
    const vals = p.rows.map((r) => r[c] ?? '').filter((v) => !isMissing(v));
    const num = vals.filter((v) => Number.isFinite(Number(v))).length;
    return vals.length && num / vals.length >= 0.9 ? 'numeric' : 'text';
  });
}

/** Build a table: numeric columns become features, `labelCol` (any kind) becomes the class. */
export function tableFromCSV(name: string, p: { header: string[]; rows: string[][] }, labelCol: number | null): Table {
  const kinds = columnKinds(p);
  const feats = p.header.map((_, i) => i).filter((i) => i !== labelCol && kinds[i] === 'numeric');
  const X = p.rows.map((r) => feats.map((c) => { const v = r[c] ?? ''; return isMissing(v) || !Number.isFinite(Number(v)) ? null : Number(v); }));
  let y: number[] | null = null, classNames: string[] = [];
  if (labelCol != null) {
    const labels = p.rows.map((r) => (r[labelCol] ?? '').trim());
    classNames = [...new Set(labels)].sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
    const idx = new Map(classNames.map((c, i) => [c, i]));
    y = labels.map((l) => idx.get(l)!);
  }
  return { name, columns: feats.map((c) => p.header[c]), X, y, classNames, source: `uploaded CSV, ${p.rows.length} rows` };
}

/* ---------------- statistics & preprocessing ---------------- */

export interface ColStats { min: number; max: number; mean: number; std: number; missing: number }

export function columnStats(X: (number | null)[][]): ColStats[] {
  const d = X[0]?.length ?? 0;
  return Array.from({ length: d }, (_, j) => {
    const v = X.map((r) => r[j]).filter((x): x is number => x != null);
    const mean = v.reduce((a, b) => a + b, 0) / (v.length || 1);
    const std = Math.sqrt(v.reduce((a, b) => a + (b - mean) ** 2, 0) / (v.length || 1));
    const [min, max] = extent(v);
    return { min, max, mean, std, missing: X.length - v.length };
  });
}

export function handleMissing(t: Table, mode: 'mean' | 'drop'): { X: number[][]; y: number[] | null; dropped: number } {
  if (mode === 'drop') {
    const keep = t.X.map((r, i) => (r.every((v) => v != null) ? i : -1)).filter((i) => i >= 0);
    return { X: keep.map((i) => t.X[i] as number[]), y: t.y ? keep.map((i) => t.y![i]) : null, dropped: t.X.length - keep.length };
  }
  const st = columnStats(t.X);
  return { X: t.X.map((r) => r.map((v, j) => v ?? st[j].mean)), y: t.y, dropped: 0 };
}

export function scaleColumns(X: number[][], mode: 'none' | 'standardize' | 'minmax'): number[][] {
  if (mode === 'none' || !X.length) return X;
  const st = columnStats(X);
  return X.map((r) => r.map((v, j) => (mode === 'standardize' ? (v - st[j].mean) / (st[j].std || 1) : (v - st[j].min) / ((st[j].max - st[j].min) || 1))));
}

export interface PCAResult { components: number[][]; variances: number[]; ratio: number[]; projected: number[][]; mean: number[] }

/** PCA by power iteration with deflation on the covariance matrix (top k components). */
export function pca(X: number[][], k = 3, seed = 1): PCAResult {
  const n = X.length, d = X[0].length;
  const mean = Array.from({ length: d }, (_, j) => X.reduce((s, r) => s + r[j], 0) / n);
  const Xc = X.map((r) => r.map((v, j) => v - mean[j]));
  const C = Array.from({ length: d }, () => new Float64Array(d));
  Xc.forEach((r) => { for (let a = 0; a < d; a++) { const ra = r[a]; if (!ra) continue; for (let b = a; b < d; b++) C[a][b] += (ra * r[b]) / (n - 1 || 1); } });
  for (let a = 0; a < d; a++) for (let b = 0; b < a; b++) C[a][b] = C[b][a];
  const total = C.reduce((s, row, i) => s + row[i], 0);
  const rng = new Rng(seed);
  const components: number[][] = [], variances: number[] = [];
  for (let c = 0; c < Math.min(k, d); c++) {
    let v = Array.from({ length: d }, () => rng.normal());
    let lambda = 0;
    for (let it = 0; it < 300; it++) {
      const w = Array.from({ length: d }, (_, a) => { let s = 0; for (let b = 0; b < d; b++) s += C[a][b] * v[b]; return s; });
      const norm = Math.hypot(...w) || 1;
      const next = w.map((x) => x / norm);
      const diff = next.reduce((s, x, i) => s + Math.abs(x - v[i]), 0);
      v = next; lambda = norm;
      if (diff < 1e-10) break;
    }
    // deterministic sign: largest-magnitude loading positive
    const big = v.reduce((m, x, i) => (Math.abs(x) > Math.abs(v[m]) ? i : m), 0);
    if (v[big] < 0) v = v.map((x) => -x);
    components.push(v); variances.push(lambda);
    for (let a = 0; a < d; a++) for (let b = 0; b < d; b++) C[a][b] -= lambda * v[a] * v[b];
  }
  const projected = Xc.map((r) => components.map((v) => r.reduce((s, x, j) => s + x * v[j], 0)));
  return { components, variances, ratio: variances.map((l) => l / (total || 1)), projected, mean };
}
