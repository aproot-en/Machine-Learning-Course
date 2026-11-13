import { useFrame } from '@react-three/fiber';
import { useMemo, useRef } from 'react';
import * as THREE from 'three';
import { Network } from '../ml/nn/network';
import type { LayerSpec } from '../ml/nn/types';
import { Rng } from '../lib/random';
import { buildEdges, layoutNetwork } from '../features/mlp/layout';
import { commitInstances, Instances, Lines, setInstance, useLineBuffer } from '../engine3d/primitives';
import { CLASS_COLORS, NEG_W, POS_W } from '../engine3d/palette';

const LAYERS: LayerSpec[] = [
  { id: 'i', kind: 'input', name: 'Input', units: 4 },
  { id: 'a', kind: 'dense', name: 'Dense', units: 16, activation: 'tanh' },
  { id: 'b', kind: 'dense', name: 'Dense', units: 16, activation: 'tanh' },
  { id: 'c', kind: 'dense', name: 'Dense', units: 9, activation: 'tanh' },
  { id: 'o', kind: 'output', name: 'Output', units: 3, activation: 'softmax' },
];
const CYCLE = 3.2;
const _c = new THREE.Color();
const _d = new THREE.Color('#0d1424');

/**
 * Hero: a real 4-16-16-9-3 network with random (untrained) weights. Each cycle feeds a new random
 * input through `Network.forward`; neuron brightness and pulse strength are the computed |a| and |w·a|.
 */
export function HeroNetwork({ onCycle }: { onCycle?: (input: number[], out: number[]) => void }) {
  const net = useMemo(() => new Network(LAYERS, 7), []);
  const pos = useMemo(() => layoutNetwork(net.widths, 3.4, 1.15), [net]);
  const { edges } = useMemo(() => buildEdges(LAYERS, net.widths, 2000), [net]);
  const nodes = useRef<THREE.InstancedMesh>(null);
  const pulses = useRef<THREE.InstancedMesh>(null);
  const lines = useLineBuffer(edges.length);
  const total = net.widths.reduce((a, b) => a + b, 0);
  const st = useRef({ t: CYCLE, trace: net.forward([0, 0, 0, 0]), rng: new Rng(3) });

  useFrame((_, dt) => {
    const s = st.current;
    s.t += Math.min(dt, 0.05);
    if (s.t >= CYCLE) {
      s.t = 0;
      const x = [0, 0, 0, 0].map(() => s.rng.normal());
      s.trace = net.forward(x);
      onCycle?.(x, s.trace.a[s.trace.a.length - 1]);
    }
    const phase = (s.t / (CYCLE * 0.8)) * (pos.length - 1);
    const A = s.trace.a;
    if (nodes.current) {
      let k = 0;
      pos.forEach((layer, l) => {
        const mx = Math.max(1e-6, ...A[l].map(Math.abs));
        layer.forEach((p, j) => {
          const lit = phase >= l;
          const v = Math.abs(A[l][j]) / mx;
          if (l === pos.length - 1) _c.set(CLASS_COLORS[j]).lerp(_d, lit ? 1 - (0.25 + 0.75 * A[l][j]) : 0.85);
          else _c.set(A[l][j] >= 0 ? '#00D9FF' : '#F59E0B').lerp(_d, lit ? 1 - (0.2 + 0.8 * v) : 0.8);
          setInstance(nodes.current!, k++, p.x, p.y, p.z, l === pos.length - 1 ? 0.34 : 0.22, _c);
        });
      });
      commitInstances(nodes.current, total);
    }
    lines.clear();
    let pc = 0;
    const seg = Math.floor(phase) + 1;
    const frac = phase - Math.floor(phase);
    edges.forEach((e) => {
      const p = net.params[e.l]!;
      const w = p.W[e.j * p.nin + e.i];
      const c = w >= 0 ? POS_W : NEG_W;
      lines.add(pos[e.l - 1][e.i], pos[e.l][e.j], c, c, 0.05 + 0.25 * Math.min(1, Math.abs(w)));
      if (e.l === seg && pulses.current && pc < 400) {
        const a = pos[e.l - 1][e.i], b = pos[e.l][e.j];
        const str = Math.min(1, Math.abs(w * A[e.l - 1][e.i]));
        if (str > 0.08) setInstance(pulses.current, pc++, a.x + (b.x - a.x) * frac, a.y + (b.y - a.y) * frac, a.z + (b.z - a.z) * frac, 0.05 + 0.08 * str, c);
      }
    });
    lines.commit();
    if (pulses.current) commitInstances(pulses.current, pc);
  });

  return (
    <group rotation={[0, -0.5, 0]}>
      <Lines buffer={lines} opacity={0.9} />
      <Instances ref={nodes} capacity={total} glow={0.55} />
      <Instances ref={pulses} capacity={400} basic opacity={0.95} />
    </group>
  );
}
