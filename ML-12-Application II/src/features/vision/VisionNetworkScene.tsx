import { useMemo, useRef } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { BOX, Label, SPHERE } from '../../engine3d/primitives';
import { PointCloud, Segments } from '../../engine3d/plots';
import type { VisionNetworkLayer } from '../../ml/visionDetail';
import type { CameraState } from '../../engine3d/Stage';
import { useT } from '../../i18n/i18n';

type V3 = [number, number, number];
export interface VisionNetworkSceneProps {
  layers: VisionNetworkLayer[]; modelName: string; vit: boolean; selected: number;
  cell: number; footprint: { parent: number; index: number }[];
  onSelect: (node: number) => void; playing: boolean; speed: number; cam: CameraState;
}

/** CNN-style stacks of feature maps, neuron columns, real image input and digit outputs. */
export function VisionNetworkScene(p: VisionNetworkSceneProps) {
  const t = useT();
  const geometry = useMemo(() => {
    let cursor = 0;
    const layers = p.layers.map((layer) => {
      const s = layer.tensor, tokens = p.vit && s.h > 1 && s.w === 1 && layer.role !== 'input';
      const image = !tokens && (s.h > 1 || s.w > 1), spatial = s.h * s.w;
      const count = image ? Math.min(5, s.c) : 1;
      const selectedChannel = layer.nodeIndex === p.selected ? Math.floor(p.cell / spatial) : 0;
      const channels = image ? [...new Set([selectedChannel, ...Array.from({ length: count }, (_, i) => Math.round(i * (s.c - 1) / Math.max(1, count - 1)))])].sort((a, b) => a - b) : [0];
      const depth = image ? Math.max(0.3, channels.length * 0.3) : 0.4;
      const x = cursor + depth / 2; cursor += depth + (layer.role === 'input' ? 3 : 2.6);
      const sampled = tokens ? [...new Set([selectedChannel, ...Array.from({ length: Math.min(16, s.c) }, (_, i) => Math.round(i * (s.c - 1) / Math.max(1, Math.min(16, s.c) - 1)))])].sort((a, b) => a - b) : [];
      const indices = image ? channels.flatMap((c) => Array.from({ length: spatial }, (_, i) => c * spatial + i)) : tokens ? sampled.flatMap((c) => Array.from({ length: s.h }, (_, token) => c * s.h + token)) : Array.from({ length: Math.min(64, s.a.length) }, (_, i) => Math.round(i * (s.a.length - 1) / Math.max(1, Math.min(64, s.a.length) - 1)));
      const gridCell = image ? Math.min(0.36, 5.5 / Math.max(s.h, s.w)) : tokens ? Math.min(0.35, 5.5 / Math.max(s.h, sampled.length)) : 0.4;
      const point = (index: number, shown: number): V3 => {
        if (image) return [x + (channels.indexOf(Math.floor(index / spatial)) - (channels.length - 1) / 2) * 0.3, ((s.h - 1) / 2 - Math.floor(index % spatial / s.w)) * gridCell, (index % s.w - (s.w - 1) / 2) * gridCell];
        if (tokens) return [x, ((s.h - 1) / 2 - index % s.h) * gridCell, (sampled.indexOf(Math.floor(index / s.h)) - (sampled.length - 1) / 2) * gridCell];
        const columns = indices.length > 32 ? Math.ceil(Math.sqrt(indices.length)) : 1, rows = Math.ceil(indices.length / columns);
        return [x, ((rows - 1) / 2 - Math.floor(shown / columns)) * Math.min(0.4, 6 / rows), (shown % columns - (columns - 1) / 2) * 0.4];
      };
      return { x, image, tokens, channels, indices, gridCell, points: indices.map(point), depth };
    });
    const shift = (cursor - 2.6) / 2;
    layers.forEach((layer) => { layer.x -= shift; layer.points.forEach((point) => { point[0] -= shift; }); });
    return { layers, width: cursor + 6 };
  }, [p.layers, p.vit, p.selected, p.cell]);
  const footprint = useMemo(() => new Set(p.footprint.map((term) => `${term.parent}:${term.index}`)), [p.footprint]);
  const links = useMemo(() => p.layers.flatMap((layer, to) => layer.inputs.flatMap((from) => {
    const a = geometry.layers[from], b = geometry.layers[to];
    const y = to - from > 1 ? 4.2 : 0, mid = (a.x + b.x) / 2;
    return [{ a: [a.x + a.depth / 2, 0, 0] as V3, b: [mid, y, 0] as V3, c: to - from > 1 ? '#A855F7' : '#315476', i: 0.8 }, { a: [mid, y, 0] as V3, b: [b.x - b.depth / 2, 0, 0] as V3, c: '#00D9FF', i: 0.8 }];
  })), [p.layers, geometry]);
  const weights = useMemo(() => p.layers.flatMap((layer, i) => {
    if (!layer.weights || !i) return [];
    const source = geometry.layers[i - 1], target = geometry.layers[i], input = p.layers[i - 1].tensor.a.length;
    return target.indices.flatMap((out, j) => source.indices.filter((_, k) => k % Math.max(1, Math.ceil(source.indices.length / 8)) === 0).map((index) => {
      const w = layer.weights![out * input + index];
      return { a: source.points[source.indices.indexOf(index)], b: target.points[j], c: w < 0 ? '#A855F7' : '#00D9FF', i: Math.min(0.7, 0.12 + Math.abs(w)) };
    }));
  }), [p.layers, geometry]);
  const probs = p.layers[p.layers.length - 1].tensor.a, prediction = probs.indexOf(Math.max(...probs));
  return <group>
    <NetworkCameraFit width={geometry.width} cam={p.cam} />
    <Label position={[0, 7.5, 0]}>{p.modelName}</Label>
    <gridHelper args={[geometry.width, Math.max(20, p.layers.length * 4), '#203c55', '#101d30']} position={[0, -4.2, 0]} />
    <Segments segs={links} /><Segments segs={weights} opacity={0.65} />
    {p.layers.map((layer, i) => {
      const geo = geometry.layers[i], s = layer.tensor;
      let max = 1e-9; for (const index of geo.indices) max = Math.max(max, Math.abs(s.a[index]));
      const colors = geo.indices.map((index) => {
        const value = s.a[index], magnitude = Math.abs(value) / max;
        const selected = layer.nodeIndex === p.selected && index === p.cell || footprint.has(`${layer.nodeIndex}:${index}`);
        if (selected) return '#FFFFFF';
        if (layer.role === 'input') return new THREE.Color().setRGB(value, value, value);
        const color = layer.role === 'output' ? index === prediction ? '#F59E0B' : '#22C55E' : value < 0 ? '#A855F7' : geo.image || geo.tokens ? '#00D9FF' : '#22C55E';
        return new THREE.Color(color).multiplyScalar(0.08 + magnitude * 0.92);
      });
      const sizes = geo.indices.map((index): number | V3 => geo.image || geo.tokens ? [0.07 + 0.08 * Math.abs(s.a[index]) / max, geo.gridCell * 0.9, geo.gridCell * 0.9] : layer.role === 'output' ? 0.18 + 0.25 * s.a[index] : 0.11);
      return <group key={`${layer.nodeIndex}:${layer.role}`}>
        <PointCloud points={geo.points} colors={colors} sizes={sizes} geometry={geo.image || geo.tokens ? BOX : SPHERE} basic={layer.role === 'input'} glow={0.4} onClick={layer.nodeIndex == null ? undefined : () => p.onSelect(layer.nodeIndex!)} />
        <Label position={[geo.x, 5, 0]} className="!whitespace-normal text-center w-28"><span className={layer.nodeIndex === p.selected ? 'text-cyan' : ''}>{layer.name}</span></Label>
        <Label position={[geo.x, -4.8, 0]} muted>{geo.tokens ? `${s.h} tokens × ${s.c}` : `${s.h}×${s.w}×${s.c}`}{geo.image && geo.channels.length < s.c ? ` · ${geo.channels.length}/${s.c} maps` : ''}</Label>
        {layer.role === 'input' && <Label position={[geo.x, 6.5, 0]}>{t('ภาพอินพุต', 'INPUT IMAGE')}</Label>}
        {layer.role === 'output' && geo.points.map((point, digit) => <Label key={digit} position={[point[0] + 1.1, point[1], point[2]]}><span className={digit === prediction ? 'text-orange' : ''}>{digit} · {(probs[digit] * 100).toFixed(1)}%</span></Label>)}
      </group>;
    })}
    <Label position={[geometry.layers.at(-1)!.x, 6.5, 0]}><span className="text-orange">{t('ผลทำนาย', 'PREDICTION')}: {prediction}</span></Label>
    <FlowPackets links={links} playing={p.playing} speed={p.speed} />
  </group>;
}

function NetworkCameraFit({ width, cam }: { width: number; cam: CameraState }) {
  const { camera, size, controls } = useThree(), fitted = useRef('');
  useFrame(() => {
    const key = `${width}:${size.width}:${size.height}:${camera.uuid}:${cam.preset}:${cam.nonce}`;
    if (fitted.current === key) return;
    fitted.current = key;
    const distance = Math.max(22, width / (2 * Math.tan(Math.PI / 8) * Math.max(0.1, size.width / size.height)) * 1.3);
    const direction = cam.preset === 'top' ? new THREE.Vector3(0, 1, 0.001) : cam.preset === 'side' ? new THREE.Vector3(1, 0.1, 0.01) : cam.preset === 'front' ? new THREE.Vector3(0.35, 0.1, 1) : new THREE.Vector3(0.7, 0.35, 1);
    camera.position.copy(direction.normalize().multiplyScalar(distance));
    if (camera instanceof THREE.OrthographicCamera) { camera.zoom = Math.min(size.width / (width * 1.2), size.height / 16); camera.updateProjectionMatrix(); }
    camera.lookAt(0, 0, 0);
    const orbit = controls as unknown as { target?: THREE.Vector3; update?: () => void } | null;
    orbit?.target?.set(0, 0, 0); orbit?.update?.();
  });
  return null;
}

function FlowPackets({ links, playing, speed }: { links: { a: V3; b: V3 }[]; playing: boolean; speed: number }) {
  const group = useRef<THREE.Group>(null), time = useRef(0);
  const reduced = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  useFrame((_, delta) => {
    if (!playing || reduced || !group.current) return;
    time.current += Math.min(delta, 0.1) * speed;
    group.current.children.forEach((mesh, i) => { const link = links[i], u = (time.current * 0.6) % 1; mesh.position.set(link.a[0] + (link.b[0] - link.a[0]) * u, link.a[1] + (link.b[1] - link.a[1]) * u, 0); });
  });
  return <group ref={group} visible={!reduced}>{links.map((link, i) => <mesh key={i} position={link.a}><sphereGeometry args={[0.075, 8, 8]} /><meshBasicMaterial color="#F59E0B" /></mesh>)}</group>;
}
