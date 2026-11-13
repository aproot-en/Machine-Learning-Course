import { Rng } from '../lib/random';

export interface Surface {
  id: string;
  name: string;
  tex: string;
  f: (x: number, y: number) => number;
  grad: (x: number, y: number) => [number, number];
  range: [number, number, number, number];
  minimum?: [number, number];
  start: [number, number];
}

const numGrad = (f: (x: number, y: number) => number) => (x: number, y: number): [number, number] => {
  const h = 1e-5;
  return [(f(x + h, y) - f(x - h, y)) / (2 * h), (f(x, y + h) - f(x, y - h)) / (2 * h)];
};

export const SURFACES: Surface[] = [
  { id: 'quadratic', name: 'Convex quadratic', tex: 'f = 0.5x^2 + 2.5y^2', f: (x, y) => 0.5 * x * x + 2.5 * y * y, grad: (x, y) => [x, 5 * y], range: [-4, 4, -4, 4], minimum: [0, 0], start: [-3.5, 2.5] },
  { id: 'rosenbrock', name: 'Rosenbrock', tex: 'f = (1-x)^2 + 10(y-x^2)^2', f: (x, y) => (1 - x) ** 2 + 10 * (y - x * x) ** 2, grad: (x, y) => [-2 * (1 - x) - 40 * x * (y - x * x), 20 * (y - x * x)], range: [-2, 2, -1, 3], minimum: [1, 1], start: [-1.5, 2.5] },
  { id: 'himmelblau', name: 'Himmelblau (4 minima)', tex: 'f = (x^2+y-11)^2 + (x+y^2-7)^2', f: (x, y) => (x * x + y - 11) ** 2 + (x + y * y - 7) ** 2, grad: (x, y) => [4 * x * (x * x + y - 11) + 2 * (x + y * y - 7), 2 * (x * x + y - 11) + 4 * y * (x + y * y - 7)], range: [-5, 5, -5, 5], start: [-0.5, -0.5] },
  { id: 'saddle', name: 'Saddle', tex: 'f = x^2 - y^2 + 0.05y^4', f: (x, y) => x * x - y * y + 0.05 * y ** 4, grad: (x, y) => [2 * x, -2 * y + 0.2 * y ** 3], range: [-4, 4, -4, 4], start: [-3, 0.05] },
  { id: 'bumpy', name: 'Non-convex (bumpy bowl)', tex: 'f = 0.1(x^2+y^2) - \\cos(1.5x)\\cos(1.5y)', f: (x, y) => 0.1 * (x * x + y * y) - Math.cos(1.5 * x) * Math.cos(1.5 * y), grad: (x, y) => [0.2 * x + 1.5 * Math.sin(1.5 * x) * Math.cos(1.5 * y), 0.2 * y + 1.5 * Math.cos(1.5 * x) * Math.sin(1.5 * y)], range: [-5, 5, -5, 5], minimum: [0, 0], start: [3.8, 3.2] },
];

/* ---------- safe expression parser (no eval): numbers, x, y, + - * / ^, parentheses, a few functions ---------- */
type Tok = { t: 'num'; v: number } | { t: 'id'; v: string } | { t: 'op'; v: string };
const FUNCS: Record<string, (a: number) => number> = { sin: Math.sin, cos: Math.cos, tan: Math.tan, exp: Math.exp, log: Math.log, sqrt: Math.sqrt, abs: Math.abs, tanh: Math.tanh };

export function compileExpression(src: string): ((x: number, y: number) => number) {
  const toks: Tok[] = [];
  const re = /\s*(?:(\d+\.?\d*(?:e[+-]?\d+)?|\.\d+)|([a-zA-Z]+)|(\*\*|[-+*/^(),]))/y;
  let pos = 0;
  while (pos < src.length) {
    if (/^\s*$/.test(src.slice(pos))) break;
    re.lastIndex = pos;
    const m = re.exec(src);
    if (!m) throw new Error(`Unexpected character at ${pos + 1}`);
    if (m[1]) toks.push({ t: 'num', v: parseFloat(m[1]) });
    else if (m[2]) toks.push({ t: 'id', v: m[2].toLowerCase() });
    else toks.push({ t: 'op', v: m[3] === '**' ? '^' : m[3] });
    pos = re.lastIndex;
  }
  let i = 0;
  type Node = (x: number, y: number) => number;
  const peek = () => toks[i];
  const eat = (v?: string) => { const t = toks[i++]; if (!t || (v && t.v !== v)) throw new Error(`Expected ${v ?? 'token'}`); return t; };
  const expr = (): Node => { let l = term(); while (peek()?.t === 'op' && (peek().v === '+' || peek().v === '-')) { const op = eat().v; const r = term(); const a = l; l = op === '+' ? (x, y) => a(x, y) + r(x, y) : (x, y) => a(x, y) - r(x, y); } return l; };
  const term = (): Node => { let l = unary(); while (peek()?.t === 'op' && (peek().v === '*' || peek().v === '/')) { const op = eat().v; const r = unary(); const a = l; l = op === '*' ? (x, y) => a(x, y) * r(x, y) : (x, y) => a(x, y) / r(x, y); } return l; };
  const unary = (): Node => { if (peek()?.t === 'op' && peek().v === '-') { eat(); const a = unary(); return (x, y) => -a(x, y); } return power(); };
  const power = (): Node => { const b = atom(); if (peek()?.t === 'op' && peek().v === '^') { eat(); const e = unary(); return (x, y) => Math.pow(b(x, y), e(x, y)); } return b; };
  const atom = (): Node => {
    const t = eat();
    if (t.t === 'num') { const v = t.v; return () => v; }
    if (t.t === 'id') {
      if (t.v === 'x') return (x) => x;
      if (t.v === 'y') return (_, y) => y;
      if (t.v === 'pi') return () => Math.PI;
      const fn = FUNCS[t.v];
      if (!fn) throw new Error(`Unknown name "${t.v}"`);
      eat('('); const a = expr(); eat(')');
      return (x, y) => fn(a(x, y));
    }
    if (t.v === '(') { const a = expr(); eat(')'); return a; }
    throw new Error(`Unexpected "${t.v}"`);
  };
  const root = expr();
  if (i < toks.length) throw new Error(`Unexpected "${toks[i].v}"`);
  return root;
}

export function customSurface(src: string, range: [number, number, number, number]): Surface {
  const f = compileExpression(src);
  // Match the lab's 65×65 display grid: invalid domains must be reported before
  // NaN/Infinity can reach WebGL vertex buffers.
  for (let row = 0; row <= 64; row++) for (let col = 0; col <= 64; col++) {
    const value = f(range[0] + col / 64 * (range[1] - range[0]), range[2] + row / 64 * (range[3] - range[2]));
    if (!Number.isFinite(Math.fround(value))) throw new Error('Function must be finite over the displayed range');
  }
  return { id: 'custom', name: 'Custom', tex: `f = ${src}`, f, grad: numGrad(f), range, start: [range[0] * 0.8, range[3] * 0.7] };
}

export type OptName = 'gd' | 'sgd' | 'momentum' | 'rmsprop' | 'adam';
export interface OptParams { name: OptName; lr: number; momentum: number; beta1: number; beta2: number; epsilon: number; noise: number; seed: number }
export interface OptPoint { x: number; y: number; f: number; gx: number; gy: number; ux: number; uy: number }

/** Runs an optimiser on a surface; SGD is modelled as the true gradient plus Gaussian noise of std `noise`. */
export function runOptimizer(s: Surface, p: OptParams, start: [number, number], iters: number): { path: OptPoint[]; diverged: boolean } {
  const rng = new Rng(p.seed);
  let [x, y] = start;
  let mx = 0, my = 0, vx = 0, vy = 0;
  const path: OptPoint[] = [];
  let diverged = false;
  for (let t = 0; t <= iters; t++) {
    const f = s.f(x, y);
    let [gx, gy] = s.grad(x, y);
    if (!Number.isFinite(f) || !Number.isFinite(gx) || !Number.isFinite(gy) || Math.abs(f) > 1e8 || Math.abs(x) > 1e4 || Math.abs(y) > 1e4) { diverged = true; break; }
    if (p.name === 'sgd') { gx += rng.normal() * p.noise; gy += rng.normal() * p.noise; }
    let ux: number, uy: number;
    if (p.name === 'gd' || p.name === 'sgd') { ux = -p.lr * gx; uy = -p.lr * gy; }
    else if (p.name === 'momentum') { mx = p.momentum * mx - p.lr * gx; my = p.momentum * my - p.lr * gy; ux = mx; uy = my; }
    else if (p.name === 'rmsprop') { vx = p.beta2 * vx + (1 - p.beta2) * gx * gx; vy = p.beta2 * vy + (1 - p.beta2) * gy * gy; ux = (-p.lr * gx) / (Math.sqrt(vx) + p.epsilon); uy = (-p.lr * gy) / (Math.sqrt(vy) + p.epsilon); }
    else {
      const k = t + 1;
      mx = p.beta1 * mx + (1 - p.beta1) * gx; my = p.beta1 * my + (1 - p.beta1) * gy;
      vx = p.beta2 * vx + (1 - p.beta2) * gx * gx; vy = p.beta2 * vy + (1 - p.beta2) * gy * gy;
      const mhx = mx / (1 - p.beta1 ** k), mhy = my / (1 - p.beta1 ** k), vhx = vx / (1 - p.beta2 ** k), vhy = vy / (1 - p.beta2 ** k);
      ux = (-p.lr * mhx) / (Math.sqrt(vhx) + p.epsilon); uy = (-p.lr * mhy) / (Math.sqrt(vhy) + p.epsilon);
    }
    path.push({ x, y, f, gx, gy, ux, uy });
    x += ux; y += uy;
  }
  return { path, diverged };
}
