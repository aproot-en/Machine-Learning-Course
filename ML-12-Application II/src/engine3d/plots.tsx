import type { ThreeEvent } from '@react-three/fiber';
import { useEffect, useLayoutEffect, useMemo, useRef } from 'react';
import * as THREE from 'three';
import { commitInstances, Instances, Lines, setInstance, SPHERE, useLineBuffer } from './primitives';

type V3 = [number, number, number];

/** Declarative instanced points. */
export function PointCloud({ points, colors, sizes = 0.16, geometry = SPHERE, glow = 0.35, opacity, basic, wire, onClick }: {
  points: V3[]; colors: (string | THREE.Color)[]; sizes?: number | (number | V3)[]; geometry?: THREE.BufferGeometry; glow?: number;
  opacity?: number; basic?: boolean; wire?: boolean; onClick?: (i: number, e: ThreeEvent<MouseEvent>) => void;
}) {
  const ref = useRef<THREE.InstancedMesh>(null);
  const cap = useMemo(() => Math.max(16, 2 ** Math.ceil(Math.log2(Math.max(1, points.length)))), [points.length]);
  useLayoutEffect(() => {
    const m = ref.current;
    if (!m) return;
    points.forEach((p, i) => setInstance(m, i, p[0], p[1], p[2], typeof sizes === 'number' ? sizes : sizes[i], colors[i] ?? '#ffffff'));
    commitInstances(m, points.length);
  });
  return <Instances key={cap} ref={ref} capacity={cap} geometry={geometry} glow={glow} opacity={opacity} basic={basic} wireframe={wire} onClick={onClick} />;
}

/** Polyline through points (as segments). */
export function Polyline({ points, color, colorEnd, opacity = 1 }: { points: V3[]; color: string; colorEnd?: string; opacity?: number }) {
  const buf = useLineBuffer(Math.max(1, points.length));
  useLayoutEffect(() => {
    buf.clear();
    const c0 = new THREE.Color(color), c1 = new THREE.Color(colorEnd ?? color);
    for (let i = 1; i < points.length; i++) {
      const t = i / Math.max(1, points.length - 1);
      buf.add(points[i - 1], points[i], c0.clone().lerp(c1, t), c0.clone().lerp(c1, t), 1);
    }
    buf.commit();
  });
  return <Lines buffer={buf} opacity={opacity} />;
}

/** Arbitrary segments with per-segment colour/intensity. */
export function Segments({ segs, opacity = 0.9, additive = true }: { segs: { a: V3; b: V3; c: string; c2?: string; i?: number }[]; opacity?: number; additive?: boolean }) {
  const buf = useLineBuffer(Math.max(1, segs.length));
  useLayoutEffect(() => {
    buf.clear();
    segs.forEach((s) => buf.add(s.a, s.b, s.c, s.c2 ?? s.c, s.i ?? 1));
    buf.commit();
  });
  return <Lines buffer={buf} opacity={opacity} additive={additive} />;
}

/**
 * Height field over a square: heights[(row)*(G+1)+col] with row along z and col along x.
 * Colours are per-vertex RGB triples.
 */
export function HeightSurface({ G, heights, colors, size, position = [0, 0, 0], opacity = 0.85, wire = 0.08, clip }: {
  G: number; heights: Float32Array; colors: Float32Array; size: number; position?: V3; opacity?: number; wire?: number; clip?: THREE.Plane[];
}) {
  const geo = useMemo(() => {
    const g = new THREE.BufferGeometry();
    const n = (G + 1) * (G + 1);
    const pos = new Float32Array(n * 3);
    for (let r = 0; r <= G; r++) for (let c = 0; c <= G; c++) { const v = r * (G + 1) + c; pos[v * 3] = (c / G - 0.5) * size; pos[v * 3 + 2] = (r / G - 0.5) * size; }
    const idx: number[] = [];
    for (let r = 0; r < G; r++) for (let c = 0; c < G; c++) { const a = r * (G + 1) + c, b = a + 1, d = a + G + 1, e = d + 1; idx.push(a, d, b, b, d, e); }
    g.setIndex(idx);
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    return g;
  }, [G, size]);
  useEffect(() => () => geo.dispose(), [geo]);
  useLayoutEffect(() => {
    const pos = geo.attributes.position as THREE.BufferAttribute;
    for (let v = 0; v < heights.length; v++) pos.setY(v, heights[v]);
    (geo.attributes.color as THREE.BufferAttribute).copyArray(colors);
    pos.needsUpdate = true; geo.attributes.color.needsUpdate = true;
    geo.computeVertexNormals();
    geo.computeBoundingSphere();
  }, [geo, heights, colors]);
  return (
    <group position={position}>
      <mesh geometry={geo}><meshLambertMaterial vertexColors side={THREE.DoubleSide} transparent opacity={opacity} depthWrite={opacity >= 1} clippingPlanes={clip} /></mesh>
      {wire > 0 && <mesh geometry={geo}><meshBasicMaterial color="#9fb4dd" wireframe transparent opacity={wire} depthWrite={false} clippingPlanes={clip} /></mesh>}
    </group>
  );
}

/** Plane n·x = d drawn as a square, optionally clipped to a box. */
export function OrientedPlane({ normal, d, color, opacity = 0.15, wire = 0.2, size = 40, clip }: { normal: V3; d: number; color: string; opacity?: number; wire?: number; size?: number; clip?: THREE.Plane[] }) {
  const n = new THREE.Vector3(...normal);
  const len = n.length();
  if (len < 1e-9) return null;
  n.divideScalar(len);
  const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), n);
  const p = n.clone().multiplyScalar(d / len);
  return (
    <group quaternion={q} position={p}>
      <mesh><planeGeometry args={[size, size]} /><meshBasicMaterial color={color} transparent opacity={opacity} side={THREE.DoubleSide} depthWrite={false} clippingPlanes={clip} /></mesh>
      {wire > 0 && <mesh><planeGeometry args={[size, size, size, size]} /><meshBasicMaterial color={color} wireframe transparent opacity={wire} depthWrite={false} clippingPlanes={clip} /></mesh>}
    </group>
  );
}

export const boxClip = (R: number, Ry = R) => [
  new THREE.Plane(new THREE.Vector3(1, 0, 0), R), new THREE.Plane(new THREE.Vector3(-1, 0, 0), R),
  new THREE.Plane(new THREE.Vector3(0, 1, 0), Ry), new THREE.Plane(new THREE.Vector3(0, -1, 0), Ry),
  new THREE.Plane(new THREE.Vector3(0, 0, 1), R), new THREE.Plane(new THREE.Vector3(0, 0, -1), R),
];

/** Maps a 2-feature dataset into scene x/z within ±R (keeps aspect). */
export function planeMapper(X: number[][], R: number, pad = 0.6) {
  const xs = X.map((r) => r[0]), zs = X.map((r) => r[1] ?? 0);
  const lo = [Math.min(...xs) - pad, Math.min(...zs) - pad], hi = [Math.max(...xs) + pad, Math.max(...zs) + pad];
  const span = Math.max(hi[0] - lo[0], hi[1] - lo[1]);
  const c = [(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2];
  const s = (2 * R) / span;
  return {
    toScene: (v: number[]): [number, number] => [(v[0] - c[0]) * s, -((v[1] ?? 0) - c[1]) * s],
    toData: (sx: number, sz: number): number[] => [sx / s + c[0], -sz / s + c[1]],
    scale: s,
  };
}

export function rgb(hex: string) { const c = new THREE.Color(hex); return [c.r, c.g, c.b]; }

/**
 * Marching squares on a (G+1)×(G+1) grid of values (row-major, row = z index).
 * Returns iso-line segments in grid coordinates (col, row) at the given level.
 */
export function contour(G: number, values: ArrayLike<number>, level: number): [number, number, number, number][] {
  const out: [number, number, number, number][] = [];
  const v = (r: number, c: number) => values[r * (G + 1) + c] - level;
  for (let r = 0; r < G; r++) for (let c = 0; c < G; c++) {
    const a = v(r, c), b = v(r, c + 1), d = v(r + 1, c + 1), e = v(r + 1, c);
    const pts: [number, number][] = [];
    const edge = (p: number, q: number, x0: number, y0: number, x1: number, y1: number) => { if ((p < 0) !== (q < 0)) { const t = p / (p - q); pts.push([x0 + (x1 - x0) * t, y0 + (y1 - y0) * t]); } };
    edge(a, b, c, r, c + 1, r); edge(b, d, c + 1, r, c + 1, r + 1); edge(d, e, c + 1, r + 1, c, r + 1); edge(e, a, c, r + 1, c, r);
    if (pts.length >= 2) out.push([pts[0][0], pts[0][1], pts[1][0], pts[1][1]]);
    if (pts.length === 4) out.push([pts[2][0], pts[2][1], pts[3][0], pts[3][1]]);
  }
  return out;
}
