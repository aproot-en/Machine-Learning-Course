/** Cross-entropies computed from logits so confident mistakes do not get clipped
 * to -log(epsilon) while their gradients continue to grow. */
export function binaryCrossEntropy(logit: number, target: number): number {
  return Math.max(logit, 0) - target * logit + Math.log1p(Math.exp(-Math.abs(logit)));
}
export function categoricalCrossEntropy(logits: ArrayLike<number>, label: number): number {
  if (!Number.isInteger(label) || label < 0 || label >= logits.length) throw new Error('Invalid target class');
  const max = Math.max(...Array.from(logits));
  let sum = 0; for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - max);
  return Math.log(sum) - (logits[label] - max);
}
export function softTargetCrossEntropy(logits: ArrayLike<number>, targets: ArrayLike<number>): number {
  const max = Math.max(...Array.from(logits));
  let sum = 0; for (let i = 0; i < logits.length; i++) sum += Math.exp(logits[i] - max);
  const norm = Math.log(sum);
  let loss = 0; for (let i = 0; i < logits.length; i++) loss -= targets[i] * (logits[i] - max - norm);
  return loss;
}
