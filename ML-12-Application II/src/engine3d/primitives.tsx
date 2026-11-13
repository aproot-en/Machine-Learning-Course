import { Html } from '@react-three/drei';
import type { ThreeEvent } from '@react-three/fiber';
import { forwardRef, useLayoutEffect, useMemo, useRef, type ReactNode } from 'react';
import * as THREE from 'three';

const _m = new THREE.Matrix4();
const _p = new THREE.Vector3();
const _s = new THREE.Vector3();
const _q = new THREE.Quaternion();
const _c = new THREE.Color();

export const SPHERE = new THREE.SphereGeometry(1, 24, 16);
export const BOX = new THREE.BoxGeometry(1, 1, 1);

/** Standard material whose instance colour also feeds emission, so bright neurons bloom in their own colour. */
export function glowMaterial(glow = 0.35, opacity = 1) {
  const m = new THREE.MeshStandardMaterial({ roughness: 0.3, metalness: 0.05, transparent: opacity < 1, opacity });
  m.onBeforeCompile = (sh) => {
    sh.fragmentShader = sh.fragmentShader.replace('#include <emissivemap_fragment>', `#include <emissivemap_fragment>\n#if defined(USE_INSTANCING_COLOR) || defined(USE_COLOR)\n totalEmissiveRadiance += vColor * ${glow.toFixed(3)};\n#endif`);
  };
  m.customProgramCacheKey = () => `glow${glow}`;
  return m;
}

export function setInstance(mesh: THREE.InstancedMesh, i: number, x: number, y: number, z: number, s: number | [number, number, number], color?: THREE.ColorRepresentation) {
  _p.set(x, y, z);
  if (typeof s === 'number') _s.set(s, s, s); else _s.set(s[0], s[1], s[2]);
  _m.compose(_p, _q.identity(), _s);
  mesh.setMatrixAt(i, _m);
  if (color !== undefined) mesh.setColorAt(i, _c.set(color));
}
export function commitInstances(mesh: THREE.InstancedMesh, count: number) {
  mesh.count = count;
  mesh.instanceMatrix.needsUpdate = true;
  if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  mesh.computeBoundingSphere();
}

export interface InstancedProps {
  capacity: number;
  geometry?: THREE.BufferGeometry;
  glow?: number;
  opacity?: number;
  basic?: boolean;
  wireframe?: boolean;
  onPointerDown?: (i: number, e: ThreeEvent<PointerEvent>) => void;
  onPointerOver?: (i: number, e: ThreeEvent<PointerEvent>) => void;
  onPointerOut?: () => void;
  onClick?: (i: number, e: ThreeEvent<MouseEvent>) => void;
}

/** Imperative instanced mesh: callers write instances through the forwarded ref (setInstance + commitInstances). */
export const Instances = forwardRef<THREE.InstancedMesh, InstancedProps>(function Instances(
  { capacity, geometry = SPHERE, glow = 0.3, opacity = 1, basic, wireframe, onPointerDown, onPointerOver, onPointerOut, onClick }, ref,
) {
  const material = useMemo(() => (basic
    ? new THREE.MeshBasicMaterial({ transparent: opacity < 1, opacity, wireframe, depthWrite: opacity >= 1, toneMapped: false })
    : glowMaterial(glow, opacity)), [basic, glow, opacity, wireframe]);
  const local = useRef<THREE.InstancedMesh>(null);
  useLayoutEffect(() => {
    const m = local.current!;
    // allocate the colour buffer up front so setColorAt works on the first frame
    for (let i = 0; i < capacity; i++) m.setColorAt(i, _c.set('#ffffff'));
    m.count = 0;
    if (typeof ref === 'function') ref(m); else if (ref) ref.current = m;
  }, [capacity, ref]);
  return (
    <instancedMesh ref={local} args={[geometry, material, capacity]} frustumCulled={false}
      onPointerDown={onPointerDown ? (e) => { e.stopPropagation(); onPointerDown(e.instanceId!, e); } : undefined}
      onPointerOver={onPointerOver ? (e) => { e.stopPropagation(); onPointerOver(e.instanceId!, e); } : undefined}
      onPointerOut={onPointerOut}
      onClick={onClick ? (e) => { e.stopPropagation(); onClick(e.instanceId!, e); } : undefined} />
  );
});

/** Growable line-segment buffer written from code (per frame or per state change). */
export class LineBuffer {
  geometry = new THREE.BufferGeometry();
  private pos: Float32Array;
  private col: Float32Array;
  n = 0;
  constructor(public capacity: number) {
    this.pos = new Float32Array(capacity * 6);
    this.col = new Float32Array(capacity * 6);
    this.geometry.setAttribute('position', new THREE.BufferAttribute(this.pos, 3).setUsage(THREE.DynamicDrawUsage));
    this.geometry.setAttribute('color', new THREE.BufferAttribute(this.col, 3).setUsage(THREE.DynamicDrawUsage));
    this.geometry.setDrawRange(0, 0);
  }
  clear() { this.n = 0; }
  add(a: ArrayLike<number> | THREE.Vector3, b: ArrayLike<number> | THREE.Vector3, ca: THREE.ColorRepresentation, cb: THREE.ColorRepresentation = ca, ia = 1, ib = ia) {
    if (this.n >= this.capacity) return;
    const k = this.n * 6;
    const A = a as THREE.Vector3 & ArrayLike<number>, B = b as THREE.Vector3 & ArrayLike<number>;
    this.pos[k] = A.x ?? A[0]; this.pos[k + 1] = A.y ?? A[1]; this.pos[k + 2] = A.z ?? A[2];
    this.pos[k + 3] = B.x ?? B[0]; this.pos[k + 4] = B.y ?? B[1]; this.pos[k + 5] = B.z ?? B[2];
    _c.set(ca); this.col[k] = _c.r * ia; this.col[k + 1] = _c.g * ia; this.col[k + 2] = _c.b * ia;
    _c.set(cb); this.col[k + 3] = _c.r * ib; this.col[k + 4] = _c.g * ib; this.col[k + 5] = _c.b * ib;
    this.n++;
  }
  commit() {
    this.geometry.setDrawRange(0, this.n * 2);
    this.geometry.attributes.position.needsUpdate = true;
    this.geometry.attributes.color.needsUpdate = true;
    this.geometry.computeBoundingSphere();
  }
  dispose() { this.geometry.dispose(); }
}

export function useLineBuffer(capacity: number) {
  const buf = useMemo(() => new LineBuffer(Math.max(1, capacity)), [capacity]);
  useLayoutEffect(() => () => buf.dispose(), [buf]);
  return buf;
}

export function Lines({ buffer, opacity = 0.85, additive = true, onPointerMove, onPointerOut, onClick }: {
  buffer: LineBuffer; opacity?: number; additive?: boolean;
  onPointerMove?: (segment: number, e: ThreeEvent<PointerEvent>) => void; onPointerOut?: () => void; onClick?: (segment: number, e: ThreeEvent<MouseEvent>) => void;
}) {
  return (
    <lineSegments geometry={buffer.geometry} frustumCulled={false}
      onPointerMove={onPointerMove ? (e) => { e.stopPropagation(); onPointerMove(Math.floor((e.index ?? 0) / 2), e); } : undefined}
      onPointerOut={onPointerOut}
      onClick={onClick ? (e) => { e.stopPropagation(); onClick(Math.floor((e.index ?? 0) / 2), e); } : undefined}>
      <lineBasicMaterial vertexColors transparent opacity={opacity} depthWrite={false} blending={additive ? THREE.AdditiveBlending : THREE.NormalBlending} toneMapped={false} />
    </lineSegments>
  );
}

export function Label({ position, children, className = '', muted }: { position: [number, number, number]; children: ReactNode; className?: string; muted?: boolean }) {
  return (
    <Html position={position} center zIndexRange={[20, 0]} style={{ pointerEvents: 'none' }}>
      <div className={`tag-3d ${muted ? 'text-muted' : ''} ${className}`}>{children}</div>
    </Html>
  );
}

/** Bounding cube with floor grid and axis names, for feature-space scenes. */
export function FrameBox({ size, labels, flat }: { size: number; labels: string[]; flat?: boolean }) {
  const edges = useMemo(() => new THREE.EdgesGeometry(flat ? new THREE.PlaneGeometry(2 * size, 2 * size) : new THREE.BoxGeometry(2 * size, 2 * size, 2 * size)), [size, flat]);
  return (
    <group>
      <lineSegments geometry={edges}><lineBasicMaterial color="#26355a" transparent opacity={0.8} /></lineSegments>
      {flat
        ? <gridHelper args={[2 * size, 12, '#1d2a4a', '#131c33']} rotation={[Math.PI / 2, 0, 0]} position={[0, 0, -0.01]} />
        : <gridHelper args={[2 * size, 12, '#1d2a4a', '#131c33']} position={[0, -size, 0]} />}
      <Label position={[size + 0.9, -size, flat ? 0 : size]} muted>{labels[0]}</Label>
      <Label position={[-size, size + 0.6, flat ? 0 : size]} muted>{labels[1]}</Label>
      {!flat && labels[2] && <Label position={[size, -size, -size - 0.9]} muted>{labels[2]}</Label>}
    </group>
  );
}
