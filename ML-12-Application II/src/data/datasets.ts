import { Rng } from '../lib/random';
import { IRIS_ROWS } from './iris';
import type { Text } from '../i18n/i18n';

export interface Dataset {
  id: DatasetId;
  X: number[][];
  y: number[];
  featureNames: string[];
  classNames: string[];
  nClasses: number;
}

export type DatasetId = 'blobs3d' | 'blobs2d' | 'moons' | 'circles' | 'xor' | 'spiral' | 'iris';

export interface DatasetParams {
  n: number;
  classes: number;
  noise: number;
  seed: number;
}

export interface DatasetInfo {
  id: DatasetId;
  name: Text;
  features: number;
  /** Fixed class count, or null when the user can choose it. */
  fixedClasses: number | null;
  synthetic: boolean;
  note: Text;
}

export const DATASETS: DatasetInfo[] = [
  { id: 'blobs3d', name: { th: 'กลุ่มก้อน 3 มิติ (Blobs)', en: 'Blobs (3D)' }, features: 3, fixedClasses: null, synthetic: true, note: { th: 'กลุ่มข้อมูลแบบ Gaussian ใน 3 ฟีเจอร์', en: 'Gaussian clusters in 3 features' } },
  { id: 'blobs2d', name: { th: 'กลุ่มก้อน 2 มิติ (Blobs)', en: 'Blobs (2D)' }, features: 2, fixedClasses: null, synthetic: true, note: { th: 'กลุ่มข้อมูลแบบ Gaussian ใน 2 ฟีเจอร์', en: 'Gaussian clusters in 2 features' } },
  { id: 'moons', name: { th: 'พระจันทร์สองเสี้ยว (Two Moons)', en: 'Two Moons' }, features: 2, fixedClasses: 2, synthetic: true, note: { th: 'แยกด้วยเส้นตรงไม่ได้', en: 'Not linearly separable' } },
  { id: 'circles', name: { th: 'วงกลมซ้อน (Circles)', en: 'Circles' }, features: 2, fixedClasses: 2, synthetic: true, note: { th: 'วงในกับวงนอก', en: 'Inner vs outer ring' } },
  { id: 'xor', name: { th: 'XOR', en: 'XOR' }, features: 2, fixedClasses: 2, synthetic: true, note: { th: 'ปัญหาคลาสสิกที่ perceptron ชั้นเดียวแก้ไม่ได้', en: 'Classic problem a single perceptron cannot solve' } },
  { id: 'spiral', name: { th: 'เกลียว (Spiral)', en: 'Spiral' }, features: 2, fixedClasses: null, synthetic: true, note: { th: 'ยาก ต้องใช้โมเดลที่ไม่เป็นเชิงเส้นสูง', en: 'Hard; needs a strongly non-linear model' } },
  { id: 'iris', name: { th: 'Iris (ข้อมูลจริง)', en: 'Iris (real data)' }, features: 4, fixedClasses: 3, synthetic: false, note: { th: 'ดอกไอริส 150 ดอก 4 ฟีเจอร์ 3 สายพันธุ์ (Fisher, 1936)', en: '150 iris flowers, 4 features, 3 species (Fisher, 1936)' } },
];

export const datasetInfo = (id: DatasetId) => DATASETS.find((d) => d.id === id)!;

const axisNames = (n: number) => Array.from({ length: n }, (_, i) => `x${i + 1}`);
const classNames = (n: number) => Array.from({ length: n }, (_, i) => String.fromCharCode(65 + i));

export function makeDataset(id: DatasetId, p: DatasetParams): Dataset {
  const rng = new Rng(p.seed);
  const info = datasetInfo(id);
  const K = info.fixedClasses ?? Math.max(2, Math.min(6, p.classes));
  const X: number[][] = [];
  const y: number[] = [];
  const n = Math.max(K * 4, p.n);

  if (id === 'iris') {
    IRIS_ROWS.forEach((r) => {
      X.push(r.slice(0, 4));
      y.push(r[4]);
    });
    return { id, X, y, nClasses: 3, featureNames: ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'], classNames: ['setosa', 'versicolor', 'virginica'] };
  }

  if (id === 'blobs3d' || id === 'blobs2d') {
    const d = id === 'blobs3d' ? 3 : 2;
    // class centres spread evenly on a circle/sphere so every class count stays readable
    const centers = Array.from({ length: K }, (_, c) => {
      const a = (c / K) * Math.PI * 2 + 0.4;
      return d === 3 ? [2.6 * Math.cos(a), 1.6 * Math.sin(a * 2), 2.6 * Math.sin(a)] : [2.6 * Math.cos(a), 2.6 * Math.sin(a)];
    });
    const sd = 0.5 + p.noise * 2;
    for (let i = 0; i < n; i++) {
      const c = i % K;
      X.push(centers[c].map((v) => v + rng.normal() * sd));
      y.push(c);
    }
  } else if (id === 'moons') {
    for (let i = 0; i < n; i++) {
      const c = i % 2;
      const t = rng.next() * Math.PI;
      const pt = c === 0 ? [Math.cos(t), Math.sin(t)] : [1 - Math.cos(t), 0.5 - Math.sin(t)];
      X.push([(pt[0] - 0.5) * 2.4 + rng.normal() * p.noise, (pt[1] - 0.25) * 2.4 + rng.normal() * p.noise]);
      y.push(c);
    }
  } else if (id === 'circles') {
    for (let i = 0; i < n; i++) {
      const c = i % 2;
      const a = rng.next() * Math.PI * 2;
      const r = c === 0 ? 1.2 : 2.6;
      X.push([r * Math.cos(a) + rng.normal() * p.noise * 1.5, r * Math.sin(a) + rng.normal() * p.noise * 1.5]);
      y.push(c);
    }
  } else if (id === 'xor') {
    for (let i = 0; i < n; i++) {
      const sx = rng.next() < 0.5 ? -1 : 1;
      const sy = rng.next() < 0.5 ? -1 : 1;
      X.push([sx * 1.4 + rng.normal() * (0.35 + p.noise), sy * 1.4 + rng.normal() * (0.35 + p.noise)]);
      y.push(sx * sy > 0 ? 0 : 1);
    }
  } else if (id === 'spiral') {
    const per = Math.floor(n / K);
    for (let c = 0; c < K; c++) {
      for (let i = 0; i < per; i++) {
        const r = (i / per) * 3 + 0.15;
        const t = (i / per) * 3.5 + (c * 2 * Math.PI) / K;
        X.push([r * Math.sin(t) + rng.normal() * p.noise * 0.6, r * Math.cos(t) + rng.normal() * p.noise * 0.6]);
        y.push(c);
      }
    }
  }
  return { id, X, y, nClasses: K, featureNames: axisNames(X[0].length), classNames: classNames(K) };
}

export interface Split {
  train: number[];
  test: number[];
}

/** Stratified train/test split by index. */
export function trainTestSplit(y: number[], testRatio: number, seed: number): Split {
  const rng = new Rng(seed * 7919 + 13);
  const byClass = new Map<number, number[]>();
  y.forEach((c, i) => { let indices = byClass.get(c); if (!indices) { indices = []; byClass.set(c, indices); } indices.push(i); });
  const train: number[] = [];
  const test: number[] = [];
  byClass.forEach((idx) => {
    rng.shuffle(idx);
    const nt = Math.round(idx.length * testRatio);
    test.push(...idx.slice(0, nt));
    train.push(...idx.slice(nt));
  });
  return { train: train.sort((a, b) => a - b), test: test.sort((a, b) => a - b) };
}

export interface Scaler {
  mean: number[];
  std: number[];
}

/** Fit a standardiser on the given rows only (the training split), so test data never leaks into it. */
export function fitScaler(rows: number[][]): Scaler {
  const d = rows[0].length;
  const mean = Array(d).fill(0);
  const std = Array(d).fill(0);
  rows.forEach((r) => r.forEach((v, j) => (mean[j] += v / rows.length)));
  rows.forEach((r) => r.forEach((v, j) => (std[j] += (v - mean[j]) ** 2 / rows.length)));
  return { mean, std: std.map((s) => Math.sqrt(s) || 1) };
}
export const applyScaler = (s: Scaler, r: number[]) => r.map((v, j) => (v - s.mean[j]) / s.std[j]);
