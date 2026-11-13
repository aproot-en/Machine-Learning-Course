import { useFrame, useThree, type ThreeEvent } from '@react-three/fiber';
import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import * as THREE from 'three';
import type { KnnResult, Metric } from '../../ml/knn';
import { BOX, commitInstances, FrameBox, Instances, Label, Lines, setInstance, useLineBuffer } from '../../engine3d/primitives';
import { CLASS_COLORS } from '../../engine3d/palette';

export const R = 5;

export interface KnnSceneProps {
  dims: 2 | 3;
  /** scene units per model unit */
  scale: number;
  train: number[][];
  ytr: number[];
  test: number[][];
  yte: number[];
  testWrong: boolean[];
  showTest: boolean;
  query: number[];
  result: KnnResult;
  stage: number;
  metric: Metric;
  p: number;
  boundary: { grid: number; classes: Uint8Array; conf: Float32Array } | null;
  showLabels: boolean;
  axisLabels: string[];
  classNames: string[];
  selected: number | null;
  onSelect: (i: number | null) => void;
  onDragQuery: (q: number[]) => void;
  onDragPoint: (i: number, q: number[]) => void;
  setDragging: (v: boolean) => void;
  allowPointDrag: boolean;
}

const toScene = (v: number[], s: number): [number, number, number] => [v[0] * s, v[1] * s, (v[2] ?? 0) * s];

/** Unit L_p ball: each direction u scaled by 1/‖u‖_p, so its surface is exactly the points at distance 1. */
function lpBall(metric: Metric, p: number, dims: 2 | 3) {
  const q = metric === 'euclidean' ? 2 : metric === 'manhattan' ? 1 : p;
  const norm = (v: number[]) => Math.pow(v.reduce((s, x) => s + Math.abs(x) ** q, 0), 1 / q);
  if (dims === 2) {
    const pts: THREE.Vector3[] = [];
    for (let i = 0; i <= 160; i++) { const a = (i / 160) * Math.PI * 2; const v = [Math.cos(a), Math.sin(a)]; const n = norm(v); pts.push(new THREE.Vector3(v[0] / n, v[1] / n, 0)); }
    return new THREE.BufferGeometry().setFromPoints(pts);
  }
  const g = new THREE.SphereGeometry(1, 48, 32);
  const pos = g.attributes.position as THREE.BufferAttribute;
  for (let i = 0; i < pos.count; i++) {
    const v = [pos.getX(i), pos.getY(i), pos.getZ(i)];
    const n = norm(v) || 1;
    pos.setXYZ(i, v[0] / n, v[1] / n, v[2] / n);
  }
  g.computeVertexNormals();
  return g;
}

function DragPlane({ active, origin, onMove, onEnd }: { active: boolean; origin: THREE.Vector3; onMove: (p: THREE.Vector3) => void; onEnd: () => void }) {
  const ref = useRef<THREE.Mesh>(null);
  const { camera } = useThree();
  useFrame(() => { if (ref.current) { ref.current.quaternion.copy(camera.quaternion); ref.current.position.copy(origin); } });
  useEffect(() => {
    if (!active) return;
    window.addEventListener('pointerup', onEnd);
    return () => window.removeEventListener('pointerup', onEnd);
  }, [active, onEnd]);
  if (!active) return null;
  return (
    <mesh ref={ref} onPointerMove={(e) => { e.stopPropagation(); onMove(e.point); }} onPointerUp={onEnd}>
      <planeGeometry args={[400, 400]} />
      <meshBasicMaterial transparent opacity={0} depthWrite={false} colorWrite={false} side={THREE.DoubleSide} />
    </mesh>
  );
}

export function KnnScene(p: KnnSceneProps) {
  const s = p.scale;
  const pts = useRef<THREE.InstancedMesh>(null);
  const tests = useRef<THREE.InstancedMesh>(null);
  const wrong = useRef<THREE.InstancedMesh>(null);
  const voxels = useRef<THREE.InstancedMesh>(null);
  const lines = useLineBuffer(Math.max(64, p.train.length + 8));
  const qRef = useRef<THREE.Mesh>(null);
  const [drag, setDrag] = useState<{ kind: 'query' | 'point'; i: number; origin: THREE.Vector3 } | null>(null);
  const neighborSet = useMemo(() => new Set(p.result.neighbors.map((n) => n.index)), [p.result]);
  const rankOf = useMemo(() => { const m = new Map<number, number>(); p.result.ranked.forEach((r, k) => m.set(r.index, k)); return m; }, [p.result]);
  const qs = toScene(p.query, s);
  const predColor = CLASS_COLORS[p.result.predicted];
  const ballGeo = useMemo(() => lpBall(p.metric, p.p, p.dims), [p.metric, p.p, p.dims]);
  useEffect(() => () => ballGeo.dispose(), [ballGeo]);
  const dK = p.result.neighbors[p.result.neighbors.length - 1]?.distance ?? 0;

  // training points
  useLayoutEffect(() => {
    const m = pts.current; if (!m) return;
    p.train.forEach((x, i) => {
      const [a, b, c] = toScene(x, s);
      const nb = p.stage >= 3 && neighborSet.has(i);
      const dim = p.stage >= 3 && !nb ? 0.45 : 1;
      const col = new THREE.Color(CLASS_COLORS[p.ytr[i]]).multiplyScalar(dim);
      setInstance(m, i, a, b, c, (nb ? 0.24 : 0.16) * (i === p.selected ? 1.5 : 1), col);
    });
    commitInstances(m, p.train.length);
  }, [p.train, p.ytr, s, p.stage, neighborSet, p.selected]);

  // test points (cubes) and misclassification markers
  useLayoutEffect(() => {
    const m = tests.current, w = wrong.current; if (!m || !w) return;
    let k = 0;
    if (p.showTest) p.test.forEach((x, i) => {
      const [a, b, c] = toScene(x, s);
      setInstance(m, i, a, b, c, 0.2, new THREE.Color(CLASS_COLORS[p.yte[i]]).multiplyScalar(0.8));
      if (p.testWrong[i]) setInstance(w, k++, a, b, c, 0.42, '#EF4444');
    });
    commitInstances(m, p.showTest ? p.test.length : 0);
    commitInstances(w, k);
  }, [p.test, p.yte, p.testWrong, p.showTest, s]);

  // distance / neighbour lines per stage
  useLayoutEffect(() => {
    lines.clear();
    const n = p.result.ranked.length;
    if (p.stage === 1 || p.stage === 2) {
      p.result.ranked.forEach((r, k) => {
        const t = 1 - k / n;
        lines.add(qs, toScene(p.train[r.index], s), '#E5E7EB', CLASS_COLORS[p.ytr[r.index]], 0.05 + 0.5 * t ** 3, 0.05 + 0.6 * t ** 3);
      });
    } else if (p.stage >= 3) {
      p.result.neighbors.forEach((nb) => lines.add(qs, toScene(p.train[nb.index], s), p.stage >= 4 ? CLASS_COLORS[nb.label] : '#E5E7EB', CLASS_COLORS[nb.label], 0.9, 1));
    }
    lines.commit();
  }, [p.stage, p.result, p.train, p.ytr, s, qs[0], qs[1], qs[2]]); // eslint-disable-line react-hooks/exhaustive-deps

  // decision region
  const texture = useMemo(() => {
    if (!p.boundary || p.dims === 3) return null;
    const G = p.boundary.grid;
    const data = new Uint8Array(G * G * 4);
    const c = new THREE.Color();
    for (let i = 0; i < G * G; i++) {
      c.set(CLASS_COLORS[p.boundary.classes[i]]);
      data[i * 4] = c.r * 255; data[i * 4 + 1] = c.g * 255; data[i * 4 + 2] = c.b * 255; data[i * 4 + 3] = 40 + 90 * p.boundary.conf[i];
    }
    const t = new THREE.DataTexture(data, G, G);
    t.magFilter = THREE.NearestFilter;
    t.needsUpdate = true;
    return t;
  }, [p.boundary, p.dims]);
  useEffect(() => () => texture?.dispose(), [texture]);
  useLayoutEffect(() => {
    const m = voxels.current; if (!m) return;
    if (!p.boundary || p.dims === 2) { commitInstances(m, 0); return; }
    const G = p.boundary.grid, step = (2 * R) / G;
    let k = 0;
    for (let a = 0; a < G; a++) for (let b = 0; b < G; b++) for (let c = 0; c < G; c++) {
      setInstance(m, k, -R + step * (a + 0.5), -R + step * (b + 0.5), -R + step * (c + 0.5), step * (0.12 + 0.18 * p.boundary.conf[k]), CLASS_COLORS[p.boundary.classes[k]]);
      k++;
    }
    commitInstances(m, k);
  }, [p.boundary, p.dims]);

  useFrame(({ clock }) => {
    if (qRef.current) qRef.current.scale.setScalar(1 + 0.07 * Math.sin(clock.elapsedTime * 4));
  });

  const toModel = (v: THREE.Vector3) => {
    const c = (x: number) => Math.max(-R, Math.min(R, x)) / s;
    return p.dims === 2 ? [c(v.x), c(v.y)] : [c(v.x), c(v.y), c(v.z)];
  };
  const begin = (kind: 'query' | 'point', i: number, e: ThreeEvent<PointerEvent>) => {
    e.stopPropagation();
    setDrag({ kind, i, origin: e.point.clone() });
    p.setDragging(true);
    if (kind === 'point') p.onSelect(i);
  };
  const end = () => { setDrag(null); p.setDragging(false); };

  return (
    <group>
      <FrameBox size={R} labels={p.axisLabels} flat={p.dims === 2} />
      {texture && (
        <mesh position={[0, 0, -0.02]}>
          <planeGeometry args={[2 * R, 2 * R]} />
          <meshBasicMaterial map={texture} transparent depthWrite={false} toneMapped={false} />
        </mesh>
      )}
      <Instances ref={voxels} capacity={16 * 16 * 16} geometry={BOX} basic opacity={0.22} />
      <Instances ref={pts} capacity={Math.max(1, p.train.length)} glow={0.4}
        onPointerDown={(i, e) => (p.allowPointDrag ? begin('point', i, e) : (e.stopPropagation(), p.onSelect(i)))} />
      <Instances ref={tests} capacity={Math.max(1, p.test.length)} geometry={BOX} glow={0.2} />
      <Instances ref={wrong} capacity={Math.max(1, p.test.length)} geometry={BOX} basic wireframe opacity={0.9} />
      <Lines buffer={lines} opacity={0.95} />

      <mesh ref={qRef} position={qs} onPointerDown={(e) => begin('query', -1, e)}>
        <sphereGeometry args={[0.3, 32, 20]} />
        <meshStandardMaterial color={p.stage >= 5 ? predColor : '#ffffff'} emissive={p.stage >= 5 ? predColor : '#ffffff'} emissiveIntensity={0.9} />
      </mesh>
      {p.stage >= 3 && dK > 0 && (p.dims === 3 ? (
        <group position={qs} scale={dK * s}>
          <mesh geometry={ballGeo}><meshBasicMaterial color={p.stage >= 5 ? predColor : '#E5E7EB'} wireframe transparent opacity={0.12} depthWrite={false} /></mesh>
          <mesh geometry={ballGeo}><meshBasicMaterial color={p.stage >= 5 ? predColor : '#E5E7EB'} transparent opacity={0.05} depthWrite={false} side={THREE.DoubleSide} /></mesh>
        </group>
      ) : (
        <lineLoop geometry={ballGeo} position={qs} scale={dK * s}><lineBasicMaterial color={p.stage >= 5 ? predColor : '#E5E7EB'} transparent opacity={0.8} /></lineLoop>
      ))}

      {p.showLabels && p.stage === 2 && p.result.ranked.slice(0, Math.min(p.result.ranked.length, p.result.neighbors.length + 3)).map((r, k) => {
        const [a, b, c] = toScene(p.train[r.index], s);
        return <Label key={r.index} position={[a, b + 0.45, c]} className={k < p.result.neighbors.length ? '' : 'opacity-60'}>#{k + 1} · {r.distance.toFixed(2)}</Label>;
      })}
      {p.showLabels && p.stage >= 3 && p.result.neighbors.map((n) => {
        const [a, b, c] = toScene(p.train[n.index], s);
        return <Label key={n.index} position={[a, b + 0.45, c]}>{n.distance.toFixed(2)}</Label>;
      })}
      {p.stage >= 5 && <Label position={[qs[0], qs[1] + 0.75, qs[2]]}><span style={{ color: predColor }}>→ {p.classNames[p.result.predicted]}</span></Label>}
      {p.selected != null && p.train[p.selected] && (
        <Label position={(() => { const [a, b, c] = toScene(p.train[p.selected!], s); return [a, b - 0.5, c] as [number, number, number]; })()}>
          #{(rankOf.get(p.selected) ?? 0) + 1}
        </Label>
      )}
      <DragPlane active={!!drag} origin={drag?.origin ?? new THREE.Vector3()} onEnd={end}
        onMove={(v) => { if (!drag) return; const m = toModel(v); if (drag.kind === 'query') p.onDragQuery(m); else p.onDragPoint(drag.i, m); }} />
    </group>
  );
}
