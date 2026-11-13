import { useFrame } from '@react-three/fiber';
import { useMemo, useRef, useState, type MutableRefObject } from 'react';
import * as THREE from 'three';
import type { Gradients, Network, Trace } from '../../ml/nn/network';
import type { LayerSpec } from '../../ml/nn/types';
import { commitInstances, Instances, Label, Lines, setInstance, useLineBuffer } from '../../engine3d/primitives';
import { CLASS_COLORS, GRAD, NEG_W, POS_W } from '../../engine3d/palette';
import { buildEdges, layoutNetwork } from './layout';

export type Selection = { type: 'neuron'; l: number; j: number } | { type: 'edge'; l: number; j: number; i: number };

export interface SampleTrace {
  x: number[];
  label: number;
  target: number[];
  trace: Trace;
  grads: Gradients;
  custom: boolean;
}

export interface FlowState {
  /** forward: signal moves input→output; backward: gradients move output→input */
  mode: 'forward' | 'backward' | 'static';
  /** position measured in layers (0 … L-1) */
  pos: number;
  loop: boolean;
  speed: number;
}

export interface ViewSettings {
  spacing: number;
  neuronSize: number;
  showConnections: boolean;
  maxEdges: number;
  opacity: number;
  weightColors: boolean;
  activationColors: boolean;
  labels: boolean;
  isolate: boolean;
}

const _c = new THREE.Color();
const DIM = new THREE.Color('#111a2c');
const ACT_POS = '#00D9FF';
const ACT_NEG = '#F59E0B';

export function NetworkScene({ net, layers, sample, flow, view, selected, onSelect, classNames, onEdgeStats }: {
  net: Network; layers: LayerSpec[]; sample: SampleTrace | null; flow: MutableRefObject<FlowState>; view: ViewSettings;
  selected: Selection | null; onSelect: (s: Selection | null) => void; classNames: string[]; onEdgeStats?: (drawn: number, total: number) => void;
}) {
  const widths = net.widths;
  const L = widths.length;
  const gap = Math.max(0.5, view.neuronSize * 3.2);
  const pos = useMemo(() => layoutNetwork(widths, view.spacing, gap), [widths, view.spacing, gap]);
  const { edges, total, drawn } = useMemo(() => buildEdges(layers, widths, view.maxEdges), [layers, widths, view.maxEdges]);
  const reported = useRef('');
  if (onEdgeStats && reported.current !== `${drawn}/${total}`) { reported.current = `${drawn}/${total}`; queueMicrotask(() => onEdgeStats(drawn, total)); }
  const offsets = useMemo(() => { const o = [0]; widths.forEach((w) => o.push(o[o.length - 1] + w)); return o; }, [widths]);
  const nodeCount = offsets[L];
  const lines = useLineBuffer(edges.length);
  const nodes = useRef<THREE.InstancedMesh>(null);
  const pulses = useRef<THREE.InstancedMesh>(null);
  const halo = useRef<THREE.Mesh>(null);
  const [hover, setHover] = useState<number | null>(null);
  const maxPulse = 1500;

  useFrame((_, rawDt) => {
    const dt = Math.min(rawDt, 0.05);
    const f = flow.current;
    if (sample && f.mode !== 'static') {
      f.pos += dt * 0.9 * f.speed * (f.mode === 'forward' ? 1 : -1);
      if (f.mode === 'forward' && f.pos > L - 1 + 0.35) { f.mode = 'backward'; f.pos = L - 1; }
      else if (f.mode === 'backward' && f.pos < -0.35) { if (f.loop) { f.mode = 'forward'; f.pos = 0; } else { f.mode = 'static'; f.pos = L - 1; } }
    }
    const A = sample?.trace.a;
    const D = sample?.grads.delta;
    const mask = sample?.trace.mask;
    const fwd = f.mode === 'forward', bwd = f.mode === 'backward';
    const selL = selected?.l ?? -1;

    // ---- neurons ----
    const m = nodes.current;
    if (m) {
      let k = 0;
      for (let l = 0; l < L; l++) {
        const a = A?.[l];
        const maxA = a ? Math.max(1e-9, ...a.map(Math.abs)) : 1;
        const d = D?.[l];
        const maxD = d ? Math.max(1e-12, ...d.map(Math.abs)) : 1;
        const lit = !A || f.mode === 'static' || (fwd ? l <= f.pos : true);
        const iso = view.isolate && selL >= 0 && l !== selL ? 0.2 : 1;
        for (let j = 0; j < widths[l]; j++) {
          const p = pos[l][j];
          if (bwd && d && l >= f.pos) {
            _c.set(GRAD).lerp(DIM, 1 - (0.25 + 0.75 * Math.abs(d[j]) / maxD));
          } else if (a && view.activationColors) {
            if (l === L - 1) {
              const out = a;
              const cls = out.length === 1 ? (out[0] > 0.5 ? 1 : 0) : j;
              const pr = out.length === 1 ? out[0] : out[j];
              _c.set(CLASS_COLORS[cls]).lerp(DIM, lit ? 1 - (0.15 + 0.85 * pr) : 0.85);
            } else {
              const v = a[j];
              const dropped = mask?.[l] && mask[l]![j] === 0;
              if (dropped) _c.set('#3a1d24');
              else _c.set(v >= 0 ? ACT_POS : ACT_NEG).lerp(DIM, lit ? 1 - (0.15 + 0.85 * Math.abs(v) / maxA) : 0.82);
            }
          } else _c.set(layers[l].kind === 'input' ? '#E5E7EB' : layers[l].kind === 'dropout' ? '#64748B' : layers[l].kind === 'activation' ? '#A855F7' : '#00D9FF').lerp(DIM, 0.35);
          _c.multiplyScalar(iso);
          const sz = view.neuronSize * (layers[l].kind === 'dropout' || layers[l].kind === 'activation' ? 0.7 : l === 0 || l === L - 1 ? 1.25 : 1);
          setInstance(m, k++, p.x, p.y, p.z, sz, _c);
        }
      }
      commitInstances(m, nodeCount);
    }

    // ---- connections: colour = weight sign, brightness = |w| (normalised per layer) ----
    lines.clear();
    const maxW: number[] = layers.map((_, l) => { const pr = net.params[l]; if (!pr) return 1; let mx = 1e-9; for (const w of pr.W) mx = Math.max(mx, Math.abs(w)); return mx; });
    const pl = pulses.current;
    let pc = 0;
    const segF = Math.floor(f.pos) + 1;
    const fracF = f.pos - Math.floor(f.pos);
    const segB = Math.ceil(f.pos);
    const fracB = segB - f.pos;
    let maxG = 1e-12;
    if (bwd && D && A) for (const e of edges) if (e.l === segB && !e.elementwise && D[e.l]) maxG = Math.max(maxG, Math.abs(D[e.l]![e.j] * A[e.l - 1][e.i]));
    for (const e of edges) {
      const a = pos[e.l - 1][e.i], b = pos[e.l][e.j];
      const iso = view.isolate && selL >= 0 && e.l !== selL && e.l !== selL + 1 ? 0.15 : 1;
      let col: THREE.ColorRepresentation = '#4b5d85';
      let inten = 0.12;
      let w = 1;
      if (!e.elementwise) {
        const pr = net.params[e.l]!;
        w = pr.W[e.j * pr.nin + e.i];
        if (view.weightColors) { col = w >= 0 ? POS_W : NEG_W; inten = 0.04 + 0.6 * Math.abs(w) / maxW[e.l]; }
      } else if (mask?.[e.l] && mask[e.l]![e.j] === 0) { col = '#EF4444'; inten = 0.12; }
      if (bwd && D && A && e.l === segB && !e.elementwise && D[e.l]) { col = GRAD; inten = 0.08 + 0.9 * Math.abs(D[e.l]![e.j] * A[e.l - 1][e.i]) / maxG; }
      if (selected?.type === 'edge' && selected.l === e.l && selected.j === e.j && selected.i === e.i) { col = '#ffffff'; inten = 1; }
      else if (selected?.type === 'neuron' && selected.l === e.l && selected.j === e.j) inten = Math.max(inten, 0.5);
      lines.add(a, b, col, col, view.showConnections ? inten * view.opacity * iso : 0);

      // pulses carry a real quantity: |w·a| forward, |δ·a| = |∂L/∂w| backward
      if (pl && A && pc < maxPulse && f.mode !== 'static') {
        if (fwd && e.l === segF) {
          const s = Math.min(1, Math.abs(w * A[e.l - 1][e.i]) / (e.elementwise ? 1 : maxW[e.l]));
          if (s > 0.04) setInstance(pl, pc++, a.x + (b.x - a.x) * fracF, a.y + (b.y - a.y) * fracF, a.z + (b.z - a.z) * fracF, view.neuronSize * (0.2 + 0.35 * s), e.elementwise ? '#cbd5e1' : w >= 0 ? POS_W : NEG_W);
        } else if (bwd && D && e.l === segB && !e.elementwise && D[e.l]) {
          const s = Math.abs(D[e.l]![e.j] * A[e.l - 1][e.i]) / maxG;
          if (s > 0.04) setInstance(pl, pc++, b.x + (a.x - b.x) * fracB, b.y + (a.y - b.y) * fracB, b.z + (a.z - b.z) * fracB, view.neuronSize * (0.2 + 0.35 * s), GRAD);
        }
      }
    }
    lines.commit();
    if (pl) commitInstances(pl, pc);

    if (halo.current) {
      if (selected?.type === 'neuron' && pos[selected.l]?.[selected.j]) { halo.current.visible = true; halo.current.position.copy(pos[selected.l][selected.j]); halo.current.scale.setScalar(view.neuronSize * 1.9); }
      else halo.current.visible = false;
    }
  });

  const neuronAt = (k: number): [number, number] => { let l = 0; while (offsets[l + 1] <= k) l++; return [l, k - offsets[l]]; };
  const hoverEdge = hover != null ? edges[hover] : null;
  const out = sample?.trace.a[L - 1];

  return (
    <group>
      <Lines buffer={lines} opacity={1}
        onPointerMove={view.showConnections ? (seg) => setHover(seg) : undefined} onPointerOut={() => setHover(null)}
        onClick={view.showConnections ? (seg) => { const e = edges[seg]; if (e && !e.elementwise) onSelect({ type: 'edge', l: e.l, j: e.j, i: e.i }); } : undefined} />
      <Instances ref={nodes} capacity={Math.max(1, nodeCount)} glow={0.5} onClick={(k) => { const [l, j] = neuronAt(k); onSelect({ type: 'neuron', l, j }); }} />
      <Instances ref={pulses} capacity={maxPulse} basic opacity={0.95} />
      <mesh ref={halo} visible={false}><sphereGeometry args={[1, 20, 14]} /><meshBasicMaterial color="#ffffff" wireframe transparent opacity={0.6} /></mesh>
      {view.labels && layers.map((l, k) => {
        const top = Math.max(...pos[k].map((p) => p.y));
        const desc = l.kind === 'input' ? `${l.units}` : l.kind === 'dropout' ? `p=${l.rate}` : l.kind === 'activation' ? l.activation : `${l.units} · ${l.activation}`;
        return <Label key={l.id} position={[pos[k][0].x, top + gap * 0.9 + 0.4, 0]} className={selected?.l === k ? 'text-cyan' : ''}>{l.name} <span className="text-muted">{desc}</span></Label>;
      })}
      {view.labels && out && pos[L - 1].map((p, j) => (
        <Label key={j} position={[p.x + gap * 1.2 + 0.5, p.y, p.z]}>
          <span style={{ color: CLASS_COLORS[out.length === 1 ? 1 : j] }}>{out.length === 1 ? `P(${classNames[1]})` : classNames[j]}</span> {out[j].toFixed(3)}
        </Label>
      ))}
      {hoverEdge && !hoverEdge.elementwise && (() => {
        const a = pos[hoverEdge.l - 1][hoverEdge.i], b = pos[hoverEdge.l][hoverEdge.j];
        const pr = net.params[hoverEdge.l]!;
        return <Label position={[(a.x + b.x) / 2, (a.y + b.y) / 2 + 0.3, (a.z + b.z) / 2]}>w[{hoverEdge.l}]<sub>{hoverEdge.j},{hoverEdge.i}</sub> = {pr.W[hoverEdge.j * pr.nin + hoverEdge.i].toFixed(4)}</Label>;
      })()}
    </group>
  );
}
