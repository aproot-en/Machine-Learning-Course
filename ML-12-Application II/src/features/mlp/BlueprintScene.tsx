import { Line } from '@react-three/drei';
import { useMemo } from 'react';
import type { LayerKind, LayerSpec } from '../../ml/nn/types';
import { Label } from '../../engine3d/primitives';

const TONES: Partial<Record<LayerKind, string>> = {
  input: '#E5E7EB', output: '#22C55E', dense: '#00D9FF', dropout: '#64748B', activation: '#A855F7',
  conv1d: '#F59E0B', conv2d: '#F59E0B', maxpool2d: '#FB923C', avgpool2d: '#FB923C', globalavgpool2d: '#FB923C', flatten: '#FDBA74',
  batchnorm: '#60A5FA', layernorm: '#60A5FA', embedding: '#34D399', rnn: '#C084FC', lstm: '#C084FC', gru: '#C084FC',
  attention: '#F472B6', add: '#FACC15', concatenate: '#FACC15',
};

function detail(l: LayerSpec) {
  if (l.kind === 'conv1d' || l.kind === 'conv2d') return `${l.filters} filters · k${l.kernelSize}`;
  if (l.kind === 'maxpool2d' || l.kind === 'avgpool2d') return `pool ${l.poolSize}`;
  if (l.kind === 'embedding') return `${l.vocabularySize} × ${l.embeddingDim}`;
  if (l.kind === 'rnn' || l.kind === 'lstm' || l.kind === 'gru') return `${l.units} units`;
  if (l.kind === 'attention') return `${l.heads} heads · d${l.keyDim}`;
  if (l.kind === 'dense' || l.kind === 'output') return `${l.units} · ${l.activation}`;
  if (l.kind === 'dropout') return `rate ${l.rate}`;
  if (l.kind === 'activation') return l.activation ?? '';
  return l.kind;
}

/** Generic 3D architecture preview for blueprints that are not executable by the MLP runtime. */
export function BlueprintScene({ layers, selectedId, onSelect }: { layers: LayerSpec[]; selectedId: string | null; onSelect: (id: string) => void }) {
  const positions = useMemo(() => {
    const offset = (layers.length - 1) * 1.65;
    return layers.map((_, i) => [i * 3.3 - offset, Math.sin(i * 0.9) * 0.45, (i % 2) * 0.35] as [number, number, number]);
  }, [layers]);
  return (
    <group>
      {positions.slice(1).map((p, i) => <Line key={`edge-${i}`} points={[positions[i], p]} color="#3B82F6" lineWidth={1.5} transparent opacity={0.55} />)}
      {layers.map((l, i) => {
        const selected = l.id === selectedId;
        const color = TONES[l.kind] ?? '#94A3B8';
        return (
          <group key={l.id} position={positions[i]}>
            <mesh onClick={(e) => { e.stopPropagation(); onSelect(l.id); }} scale={selected ? 1.18 : 1}>
              <boxGeometry args={[2.2, 1.25, 1]} />
              <meshStandardMaterial color={color} emissive={color} emissiveIntensity={selected ? 0.55 : 0.18} roughness={0.35} metalness={0.15} />
            </mesh>
            <Label position={[0, 1.05, 0]} className={selected ? 'text-cyan' : ''}>{l.name}</Label>
            <Label position={[0, -1.0, 0]}><span className="text-muted">{detail(l)}</span></Label>
          </group>
        );
      })}
    </group>
  );
}
