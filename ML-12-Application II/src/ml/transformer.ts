import { Rng } from '../lib/random';

export interface TfConfig { dModel: number; heads: number; ffDim: number; blocks: number; causal: boolean; seed: number }

export interface HeadTrace { Q: number[][]; K: number[][]; V: number[][]; scores: number[][]; attn: number[][]; out: number[][] }
export interface BlockTrace { input: number[][]; heads: HeadTrace[]; attnOut: number[][]; res1: number[][]; norm1: number[][]; ff: number[][]; res2: number[][]; output: number[][] }
export interface TfTrace { tokens: string[]; embed: number[][]; pos: number[][]; x0: number[][]; blocks: BlockTrace[] }

const matmul = (A: number[][], W: number[][]) => A.map((r) => W[0].map((_, j) => r.reduce((s, v, k) => s + v * W[k][j], 0)));
const randMat = (r: number, c: number, rng: Rng) => Array.from({ length: r }, () => Array.from({ length: c }, () => rng.normal() * Math.sqrt(1 / r)));
const layerNorm = (X: number[][]) => X.map((r) => { const m = r.reduce((a, b) => a + b, 0) / r.length; const v = r.reduce((a, b) => a + (b - m) ** 2, 0) / r.length; return r.map((x) => (x - m) / Math.sqrt(v + 1e-5)); });

/** Deterministic embedding per token string (hash-seeded), so the same word always maps to the same vector. */
export function tokenEmbedding(tok: string, d: number, seed: number): number[] {
  let h = 2166136261 ^ seed;
  for (const ch of tok.toLowerCase()) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
  const rng = new Rng(h >>> 0);
  return Array.from({ length: d }, () => rng.normal() * 0.8);
}

export function positionalEncoding(T: number, d: number): number[][] {
  return Array.from({ length: T }, (_, p) => Array.from({ length: d }, (_, i) => {
    const a = p / Math.pow(10000, (2 * Math.floor(i / 2)) / d);
    return i % 2 === 0 ? Math.sin(a) : Math.cos(a);
  }));
}

/** Weights are seeded random (untrained): the encoder demonstrates the computation, not a learned language model. */
export function encoder(tokens: string[], cfg: TfConfig, temperature = 1): TfTrace {
  const rng = new Rng(cfg.seed);
  const d = cfg.dModel, dk = d / cfg.heads;
  const embed = tokens.map((t) => tokenEmbedding(t, d, cfg.seed));
  const pos = positionalEncoding(tokens.length, d);
  const x0 = embed.map((e, i) => e.map((v, k) => v + pos[i][k]));
  let X = x0;
  const blocks: BlockTrace[] = [];
  for (let b = 0; b < cfg.blocks; b++) {
    const heads: HeadTrace[] = [];
    for (let h = 0; h < cfg.heads; h++) {
      const Wq = randMat(d, dk, rng), Wk = randMat(d, dk, rng), Wv = randMat(d, dk, rng);
      const Q = matmul(X, Wq), K = matmul(X, Wk), V = matmul(X, Wv);
      const scores = Q.map((q, i) => K.map((k, j) => (cfg.causal && j > i ? -Infinity : q.reduce((s, v, n) => s + v * k[n], 0) / Math.sqrt(dk))));
      const attn = scores.map((r) => { const m = Math.max(...r.filter(Number.isFinite)); const e = r.map((s) => (Number.isFinite(s) ? Math.exp((s - m) / temperature) : 0)); const t = e.reduce((a, c) => a + c, 0); return e.map((v) => v / t); });
      const out = attn.map((r) => V[0].map((_, n) => r.reduce((s, a, j) => s + a * V[j][n], 0)));
      heads.push({ Q, K, V, scores, attn, out });
    }
    const concat = X.map((_, i) => heads.flatMap((hd) => hd.out[i]));
    const Wo = randMat(d, d, rng);
    const attnOut = matmul(concat, Wo);
    const res1 = X.map((r, i) => r.map((v, k) => v + attnOut[i][k]));
    const norm1 = layerNorm(res1);
    const W1 = randMat(d, cfg.ffDim, rng), W2 = randMat(cfg.ffDim, d, rng);
    const ff = matmul(matmul(norm1, W1).map((r) => r.map((v) => Math.max(0, v))), W2);
    const res2 = norm1.map((r, i) => r.map((v, k) => v + ff[i][k]));
    const output = layerNorm(res2);
    blocks.push({ input: X, heads, attnOut, res1, norm1, ff, res2, output });
    X = output;
  }
  return { tokens, embed, pos, x0, blocks };
}
