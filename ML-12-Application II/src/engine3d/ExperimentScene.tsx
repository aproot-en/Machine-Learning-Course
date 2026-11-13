import { useMemo, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { BOX, Label, SPHERE } from './primitives';
import { PointCloud, Polyline, Segments } from './plots';
import { CLASS_COLORS } from './palette';
import { stageConnections, stageIndices, stagePosition, type ExperimentStage, type ModelExperiment } from './experiment';
import { useT } from '../i18n/i18n';

type V3 = [number, number, number];
const GAP = 5.8;

function ValueLayer({ stage, index, x, selected, onSelect }: { stage: ExperimentStage; index: number; x: number; selected: boolean; onSelect: () => void }) {
  const t = useT();
  const geometry = useMemo(() => {
    const indices = stageIndices(stage, stage.kind === 'tensor' ? 256 : stage.kind === 'points' ? 128 : 64);
    const max = Math.max(1e-9, ...indices.map((i) => Math.abs(stage.values[i])).filter(Number.isFinite));
    const points = indices.map((index, j) => stagePosition(stage, index, j, indices.length));
    const colors = indices.map((index) => {
      if (stage.highlight?.includes(index)) return '#F59E0B';
      if (stage.classes) return stage.classes[index] >= 0 ? CLASS_COLORS[stage.classes[index] % CLASS_COLORS.length] : stage.classes[index] === -1 ? '#7f1d1d' : '#475569';
      if (!Number.isFinite(stage.values[index])) return '#475569';
      return new THREE.Color(stage.values[index] < 0 ? '#A855F7' : '#00D9FF').multiplyScalar(0.06 + 0.94 * Math.abs(stage.values[index]) / max);
    });
    const sizes: (number | V3)[] = indices.map((index) => {
      const v = Number.isFinite(stage.values[index]) ? Math.abs(stage.values[index]) / max : 0;
      if (stage.kind === 'tensor') {
        const cell = Math.min(0.38, 4.5 / Math.max(stage.shape?.[0] ?? 8, stage.shape?.[1] ?? 8));
        return [cell * 0.86, cell * 0.86, 0.06 + v * 0.24];
      }
      if (stage.kind === 'points') return stage.highlight?.includes(index) ? 0.19 : 0.12;
      return (stage.kind === 'tree' ? 0.075 : 0.09) + 0.12 * Math.sqrt(v);
    });
    const indexMap = new Map(indices.map((index, i) => [index, i]));
    const edges = (stage.edges ?? []).flatMap(([a, b]) => indexMap.has(a) && indexMap.has(b) ? [{ a: points[indexMap.get(a)!], b: points[indexMap.get(b)!], c: '#536a94', i: 0.7 }] : []);
    return { indices, points, colors, sizes, edges };
  }, [stage]);

  return <group position={[x, 0, 0]}>
    <mesh position={[0, -3.05, 0]} onClick={(e) => { e.stopPropagation(); onSelect(); }}>
      <boxGeometry args={[3.9, 0.12, 5.6]} /><meshStandardMaterial color={selected ? '#164e63' : '#101827'} metalness={0.5} roughness={0.4} />
    </mesh>
    <PointCloud points={geometry.points} colors={geometry.colors} sizes={geometry.sizes} geometry={stage.kind === 'tensor' ? BOX : SPHERE} glow={selected ? 0.6 : 0.25} onClick={onSelect} />
    <Segments segs={geometry.edges} />
    <Label position={[0, 3.2, 0]} className="!whitespace-normal text-center sm:w-28"><span className={selected ? 'text-cyan' : ''}><span className="sm:hidden">{index + 1}</span><span className="hidden sm:inline">{stage.name}</span></span></Label>
    <Label position={[0, -3.55, 0]} className="hidden !whitespace-normal text-center sm:block sm:w-28" muted>{stage.shape ? stage.shape.join(' × ') : `${stage.values.length} ${t('ค่า', 'values')}`}{geometry.indices.length < stage.values.length ? ` · ${t('แสดง', 'shown')} ${geometry.indices.length}` : ''}</Label>
    {geometry.indices.length === 0 && <Label position={[0, 0, 0]} muted>{t('ยังไม่มีค่า', 'No values yet')}</Label>}
  </group>;
}

function signalPosition(from: number, to: number, count: number, strand: number, u: number): V3 {
  const skip = to - from > 1 ? Math.sin(Math.PI * u) : 0;
  return [((from + (to - from) * u) - (count - 1) / 2) * GAP, Math.sin(strand * 1.8) * 1.4 + skip * 2.3, Math.cos(strand * 1.8) * 1.15 + skip * 1.8];
}

function Packets({ edges, count, playing, speed }: { edges: { from: number; to: number }[]; count: number; playing: boolean; speed: number }) {
  const group = useRef<THREE.Group>(null);
  const time = useRef(0);
  const reduced = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  useFrame((_, dt) => {
    if (!playing || reduced || !group.current) return;
    time.current += Math.min(dt, 0.1) * speed * 0.65;
    group.current.children.forEach((mesh, i) => {
      const edge = Math.floor(i / 5), strand = i % 5;
      const u = (time.current + strand / 5) % 1;
      const { from, to } = edges[edge];
      mesh.position.set(...signalPosition(from, to, count, strand, u));
    });
  });
  return <group ref={group} visible={!reduced}>
    {Array.from({ length: edges.length * 5 }, (_, i) => <mesh key={i} position={signalPosition(edges[Math.floor(i / 5)].from, edges[Math.floor(i / 5)].to, count, i % 5, 0)}>
      <sphereGeometry args={[0.075, 10, 8]} /><meshBasicMaterial color={i % 5 === 0 ? '#F59E0B' : '#8beaff'} toneMapped={false} />
    </mesh>)}
  </group>;
}

/** Frame the full computation horizontally, including narrow mobile viewports. */
export function ExperimentCameraFit({ count }: { count: number }) {
  const { camera, size, controls } = useThree();
  const fitted = useRef('');
  useFrame(() => {
    const key = `${count}:${size.width}:${size.height}:${camera.uuid}`;
    if (fitted.current === key) return;
    fitted.current = key;
    const aspect = size.width / Math.max(1, size.height);
    const distance = Math.max(18, ((count - 1) * GAP + 7) / (2 * Math.tan(Math.PI / 8) * aspect) * 1.15);
    camera.position.set(distance * 0.12, distance * 0.2, distance);
    if (camera instanceof THREE.OrthographicCamera) { camera.zoom = Math.min(size.width / (((count - 1) * GAP + 7) * 1.15), size.height / 12); camera.updateProjectionMatrix(); }
    camera.lookAt(0, 0, 0);
    const orbit = controls as unknown as { target?: THREE.Vector3; update?: () => void } | null;
    orbit?.target?.set(0, 0, 0); orbit?.update?.();
  });
  return null;
}

export function ExperimentScene({ data, playing, speed, selected, onSelect }: { data: ModelExperiment; playing: boolean; speed: number; selected: number; onSelect: (stage: number) => void }) {
  const t = useT();
  const count = data.stages.length;
  const edges = useMemo(() => stageConnections(data.stages), [data.stages]);
  const links = useMemo(() => edges.flatMap(({ from, to }) => Array.from({ length: 5 }, (_, strand) => Array.from({ length: 12 }, (_, step) => ({
    a: signalPosition(from, to, count, strand, step / 12), b: signalPosition(from, to, count, strand, (step + 1) / 12),
    c: to - from > 1 ? '#627f9f' : '#315476', i: 0.65,
  }))).flat()), [count, edges]);
  const weightLinks = useMemo(() => data.stages.flatMap((stage, i) => {
    if (!stage.weights || i === 0) return [];
    const previous = data.stages[i - 1];
    const source = stageIndices(previous, previous.kind === 'tensor' ? 256 : 64);
    const target = stageIndices(stage, stage.kind === 'tensor' ? 256 : 64);
    const sampled = (indices: number[], n: number) => indices.filter((_, j) => j % Math.max(1, Math.ceil(indices.length / n)) === 0);
    const maxWeight = Math.max(1e-9, ...sampled(target, 8).flatMap((to) => sampled(source, 8).map((from) => Math.abs(stage.weights![to * previous.values.length + from]))).filter(Number.isFinite));
    return sampled(target, 8).flatMap((to) => sampled(source, 8).flatMap((from) => {
      const weight = stage.weights![to * previous.values.length + from];
      if (!Number.isFinite(weight)) return [];
      const a = stagePosition(previous, from, source.indexOf(from), source.length), b = stagePosition(stage, to, target.indexOf(to), target.length);
      return [{ a: [a[0] + (i - 1 - (count - 1) / 2) * GAP, a[1], a[2]] as V3, b: [b[0] + (i - (count - 1) / 2) * GAP, b[1], b[2]] as V3, c: weight < 0 ? '#A855F7' : '#00D9FF', i: 0.1 + Math.abs(weight) / maxWeight * 0.55 }];
    }));
  }), [data.stages, count]);
  return <group>
    <ExperimentCameraFit count={count} />
    <gridHelper args={[Math.max(24, count * GAP), count * 5, '#1d3850', '#101d30']} position={[0, -3.2, 0]} />
    <Segments segs={links} opacity={0.65} />
    <Segments segs={weightLinks} opacity={0.7} />
    {data.stages.map((stage, i) => <ValueLayer key={`${data.model}:${i}`} stage={stage} index={i} x={(i - (count - 1) / 2) * GAP} selected={selected === i} onSelect={() => onSelect(i)} />)}
    <Packets edges={edges} count={count} playing={playing} speed={speed} />
    <Label position={[-(count - 1) / 2 * GAP, 4.2, 0]}>{t('ข้อมูลเข้า', 'INPUT')}</Label>
    <Label position={[(count - 1) / 2 * GAP, 4.2, 0]}><span className="text-orange">{t('ผลลัพธ์', 'OUTPUT')}</span></Label>
    <Polyline points={data.stages.map((_, i) => [(i - (count - 1) / 2) * GAP, -4.2, 0])} color="#00D9FF" colorEnd="#F59E0B" opacity={0.5} />
  </group>;
}
