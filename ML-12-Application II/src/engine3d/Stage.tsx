import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, OrthographicCamera, PerspectiveCamera } from '@react-three/drei';
import { Bloom, EffectComposer } from '@react-three/postprocessing';
import { useEffect, useRef, useState, type ReactNode, type RefObject } from 'react';
import * as THREE from 'three';
import type { OrbitControls as OrbitImpl } from 'three-stdlib';
import { useSettings, useT } from '../i18n/i18n';
import { Icon } from '../components/ui';

export type CameraPreset = 'iso' | 'front' | 'top' | 'side';
const DIRS: Record<CameraPreset, [number, number, number]> = {
  iso: [1, 0.7, 1.25],
  front: [0, 0, 1],
  top: [0, 1, 0.001],
  side: [1, 0.05, 0],
};

export interface CameraState {
  preset: CameraPreset;
  /** bump to re-apply the same preset */
  nonce: number;
  ortho: boolean;
  autoRotate: boolean;
}

export function useCameraState(initial: Partial<CameraState> = {}) {
  return useState<CameraState>({ preset: 'iso', nonce: 0, ortho: false, autoRotate: false, ...initial });
}

function CameraRig({ cam, distance, target }: { cam: CameraState; distance: number; target: [number, number, number] }) {
  const { camera, controls, size } = useThree() as unknown as { camera: THREE.Camera; controls: OrbitImpl | null; size: { height: number } };
  useEffect(() => {
    const d = new THREE.Vector3(...DIRS[cam.preset]).normalize().multiplyScalar(distance);
    camera.position.set(target[0] + d.x, target[1] + d.y, target[2] + d.z);
    if ((camera as THREE.OrthographicCamera).isOrthographicCamera) {
      (camera as THREE.OrthographicCamera).zoom = size.height / (distance * 0.75);
      (camera as THREE.OrthographicCamera).updateProjectionMatrix();
    }
    camera.lookAt(...target);
    if (controls) { controls.target.set(...target); controls.update(); }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cam.preset, cam.nonce, cam.ortho, controls, distance, camera, size.height, target[0], target[1], target[2]]);
  return null;
}

export interface StageProps {
  children: ReactNode;
  cam: CameraState;
  distance?: number;
  target?: [number, number, number];
  /** 2D mode locks rotation so the plane stays face-on */
  lockRotation?: boolean;
  controlsEnabled?: boolean;
  preserveDrawingBuffer?: boolean;
  onPointerMissed?: () => void;
  fogEnabled?: boolean;
}

export function Stage({ children, cam, distance = 18, target = [0, 0, 0], lockRotation, controlsEnabled = true, preserveDrawingBuffer, onPointerMissed, fogEnabled = true }: StageProps) {
  const quality = useSettings((s) => s.quality);
  const reduce = typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches;
  return (
    <Canvas dpr={quality === 'high' ? [1, 2] : [1, 1]} gl={{ antialias: true, preserveDrawingBuffer }} onPointerMissed={onPointerMissed}
      raycaster={{ params: { Line: { threshold: 0.06 } } as THREE.RaycasterParameters }}>
      <color attach="background" args={['#070B14']} />
      {fogEnabled && <fog attach="fog" args={['#070B14', distance * 1.6, distance * 4]} />}
      {cam.ortho
        ? <OrthographicCamera makeDefault near={-200} far={Math.max(400, distance * 8)} position={[0, 0, distance]} />
        : <PerspectiveCamera makeDefault fov={45} near={0.1} far={Math.max(500, distance * 8)} position={[0, 0, distance]} />}
      <OrbitControls makeDefault enableDamping dampingFactor={0.08} enabled={controlsEnabled} enableRotate={!lockRotation}
        autoRotate={cam.autoRotate && !reduce} autoRotateSpeed={0.6} />
      <CameraRig cam={cam} distance={distance} target={target} />
      <hemisphereLight args={['#bcd3ff', '#0b1020', 0.9]} />
      <directionalLight position={[10, 20, 14]} intensity={1.1} />
      <directionalLight position={[-14, 4, -10]} intensity={0.35} color="#00D9FF" />
      {children}
      {quality === 'high' && (
        <EffectComposer>
          <Bloom mipmapBlur intensity={0.65} luminanceThreshold={0.22} luminanceSmoothing={0.2} />
        </EffectComposer>
      )}
    </Canvas>
  );
}

/** Camera presets, projection, auto-rotate, fullscreen. Keys 1–4 select presets, F toggles fullscreen. */
export function StageToolbar({ cam, setCam, container, allowPresets = true, extra }: {
  cam: CameraState; setCam: (c: CameraState) => void; container: RefObject<HTMLElement>; allowPresets?: boolean; extra?: ReactNode;
}) {
  const t = useT();
  const fs = () => {
    const el = container.current;
    if (!el) return;
    if (document.fullscreenElement) document.exitFullscreen().catch(() => {});
    else el.requestFullscreen?.().catch(() => {});
  };
  const camRef = useRef(cam);
  camRef.current = cam;
  useEffect(() => {
    const keys: Record<string, CameraPreset> = { '1': 'iso', '2': 'front', '3': 'top', '4': 'side' };
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
      if (allowPresets && keys[e.key]) setCam({ ...camRef.current, preset: keys[e.key], nonce: camRef.current.nonce + 1 });
      if (e.key === 'f' || e.key === 'F') fs();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [allowPresets, setCam]);
  const b = 'rounded-md px-2 py-1 text-[0.9375rem] transition-colors';
  const presets: [CameraPreset, string][] = [['iso', t('มุมเฉียง', 'Iso')], ['front', t('หน้า', 'Front')], ['top', t('บน', 'Top')], ['side', t('ข้าง', 'Side')]];
  return (
    <div className="glass pointer-events-auto flex flex-wrap items-center gap-0.5 p-1">
      {allowPresets && presets.map(([p, label], i) => (
        <button key={p} type="button" title={`${label} (${i + 1})`} onClick={() => setCam({ ...cam, preset: p, nonce: cam.nonce + 1 })}
          className={`${b} ${cam.preset === p ? 'bg-cyan/15 text-cyan' : 'text-muted hover:text-ink'}`}>{label}</button>
      ))}
      <span className="mx-0.5 h-5 w-px bg-line" />
      <button type="button" onClick={() => setCam({ ...cam, ortho: !cam.ortho, nonce: cam.nonce + 1 })} aria-pressed={cam.ortho}
        className={`${b} ${cam.ortho ? 'text-cyan' : 'text-muted hover:text-ink'}`} title={t('สลับมุมมองแบบ perspective / orthographic', 'Toggle perspective / orthographic')}>
        {cam.ortho ? 'Ortho' : 'Persp'}
      </button>
      {allowPresets && (
        <button type="button" onClick={() => setCam({ ...cam, autoRotate: !cam.autoRotate })} aria-pressed={cam.autoRotate}
          className={`${b} ${cam.autoRotate ? 'text-cyan' : 'text-muted hover:text-ink'}`}>{t('หมุน', 'Spin')}</button>
      )}
      {extra}
      <button type="button" onClick={fs} className={`${b} flex items-center gap-1 text-muted hover:text-ink`} title={t('เต็มจอ (F)', 'Fullscreen (F)')} aria-label={t('เต็มจอ', 'Fullscreen')}>
        <Icon name="full" className="h-3.5 w-3.5" />
        {t('เต็มจอ', 'Full')}
      </button>
    </div>
  );
}
