// Pre-trains the default CNN architecture on the real handwritten digits and writes src/data/cnnPretrained.json.
// Run: npx vite-node scripts/pretrain-cnn.ts
import { writeFileSync } from 'node:fs';
import { CNN, CnnTrainer, DEFAULT_CNN, cnnAccuracy, loadDigits } from '../src/ml/cnn';
import { DIGIT_LABELS, DIGIT_PIXELS } from '../src/data/digits';
import { trainTestSplit } from '../src/data/datasets';

const { X, y } = loadDigits(DIGIT_PIXELS, DIGIT_LABELS);
const split = trainTestSplit(y, 0.2, 42);
const Xtr = split.train.map((i) => X[i]), ytr = split.train.map((i) => y[i]);
const Xte = split.test.map((i) => X[i]), yte = split.test.map((i) => y[i]);
const net = new CNN(DEFAULT_CNN, { c: 1, h: 16, w: 16 }, 7);
const tr = new CnnTrainer(net, Xtr, ytr, { name: 'adam', lr: 0.003, momentum: 0.9, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }, 16, 3);
const EPOCHS = 8;
while (tr.epoch < EPOCHS) {
  tr.trainBatch();
  if (tr.cursor === 0) console.log(`epoch ${tr.epoch}  test acc ${(cnnAccuracy(net, Xte, yte).acc * 100).toFixed(1)}%`);
}
const testAcc = cnnAccuracy(net, Xte, yte).acc;
const round = (a: Float64Array) => Array.from(a, (v) => Math.round(v * 1e5) / 1e5);
writeFileSync('src/data/cnnPretrained.json', JSON.stringify({
  note: 'Default CNN trained by scripts/pretrain-cnn.ts on the UCI handwritten digits (80/20 stratified split, seed 42).',
  layers: DEFAULT_CNN, epochs: EPOCHS, trainSize: Xtr.length, testSize: Xte.length, testAcc,
  params: net.params.map((p) => (p ? { W: round(p.W), b: round(p.b) } : null)),
}));
console.log('saved, test accuracy', testAcc);
