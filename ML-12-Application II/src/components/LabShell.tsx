import { useRef, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { Stage, StageToolbar, type CameraState } from '../engine3d/Stage';
import { useT } from '../i18n/i18n';
import { PlaybackBar, type PlaybackProps } from './PlaybackBar';
import { FloatingPanel, Readout, Tabs } from './ui';
import type { ModelExperiment } from '../engine3d/experiment';
import { ExperimentLab } from './ExperimentLab';

export interface LabShellProps {
  crumb: string;
  title: string;
  intro: string;
  chips: string[];
  cam: CameraState;
  setCam: (c: CameraState) => void;
  distance: number;
  target?: [number, number, number];
  lockRotation?: boolean;
  controlsEnabled?: boolean;
  onPointerMissed?: () => void;
  preserve?: boolean;
  scene: ReactNode;
  readout: { title: string; rows: [string, ReactNode, string?][] };
  controls: ReactNode;
  playback?: PlaybackProps & { extra?: ReactNode };
  caption?: ReactNode;
  /** extra floating content under the readout (e.g. a 2D heat-map) */
  aside?: ReactNode;
  /** replaces the 3D stage, e.g. when the configuration is invalid */
  blocked?: ReactNode;
  tabs: { id: string; label: string; content: ReactNode }[];
  /** content shown in a band between the header and the scene */
  toolbar?: ReactNode;
  panelTitle?: string;
  experiment?: ModelExperiment;
}

/** Common layout for every algorithm lab: big scene, floating readout/controls, bottom playback bar, tabbed lesson below. */
export function LabShell(p: LabShellProps) {
  const t = useT();
  const box = useRef<HTMLDivElement>(null);
  const nativeStage = (<>
      <div ref={box} className="relative h-[calc(100svh-190px)] min-h-[600px] overflow-hidden border-y border-line bg-bg">
        <div className="absolute inset-0">
          {p.blocked ?? (
            <Stage cam={p.cam} distance={p.distance} target={p.target} lockRotation={p.lockRotation} controlsEnabled={p.controlsEnabled}
              onPointerMissed={p.onPointerMissed} preserveDrawingBuffer={p.preserve}>
              {p.scene}
            </Stage>
          )}
        </div>
        <div className="pointer-events-none absolute inset-0 z-30 flex flex-col p-3 sm:p-4">
          <div className="flex min-h-0 flex-1 items-start justify-between gap-3">
            <div className="flex min-h-0 max-w-[440px] flex-col gap-2">
              <Readout title={p.readout.title} rows={p.readout.rows} />
              <StageToolbar cam={p.cam} setCam={p.setCam} container={box} allowPresets={!p.lockRotation} />
              {p.aside}
            </div>
            <FloatingPanel title={p.panelTitle ?? t('ตัวควบคุม', 'Controls')} className="hidden max-h-full w-[340px] shrink-0 lg:flex">
              <div className="flex flex-col gap-0 pb-2">{p.controls}</div>
            </FloatingPanel>
          </div>
          <div className="mt-2 flex flex-col items-center gap-2">
            {p.caption && <div className="glass max-w-[720px] px-4 py-2 text-center text-[1.03125rem]">{p.caption}</div>}
            {p.playback && <PlaybackBar {...p.playback}>{p.playback.extra}</PlaybackBar>}
          </div>
        </div>
      </div>

      <div className="border-b border-line bg-surface/40 p-4 lg:hidden">{p.controls}</div>

  </>);
  return (
    <div>
      <header className="mx-auto flex max-w-[1600px] flex-col gap-2 px-4 pb-4 pt-6 sm:px-6 md:flex-row md:items-end md:justify-between">
        <div>
          <nav className="text-[1rem] text-muted"><Link to="/algorithms" className="hover:text-ink">{t('อัลกอริทึม', 'Algorithms')}</Link> / {p.crumb}</nav>
          <h1 className="mt-1 font-display text-3xl font-semibold">{p.title}</h1>
          <p className="mt-1 max-w-[78ch] text-[1.125rem] text-muted">{p.experiment ? t('เลือกข้อมูลเข้าและปรับพารามิเตอร์ ดูค่าที่ผ่านแต่ละขั้นในฉาก 3D และผลการทดลองด้านล่าง เปิด “สำรวจอัลกอริทึม” เพื่อดูมุมมองเฉพาะเทคนิค', 'Choose input data and parameters, inspect each computation stage in 3D, and see the experiment results below. Open Algorithm detail for the technique-specific view.') : p.intro}</p>
        </div>
        <div className="flex flex-wrap gap-2">{p.chips.map((c) => <span key={c} className="chip">{c}</span>)}</div>
      </header>
      {p.toolbar && <div className="mx-auto max-w-[1600px] px-4 pb-4 sm:px-6">{p.toolbar}</div>}

      {p.experiment ? <ExperimentLab data={p.experiment} controls={p.controls} playback={p.playback} detail={nativeStage} /> : nativeStage}

      <section className="mx-auto max-w-[1600px] px-4 py-10 sm:px-6">
        <Tabs tabs={p.tabs} />
      </section>
    </div>
  );
}

/** Small numeric metric tile row used in evaluation tabs. */
export function Metrics({ items }: { items: [string, string][] }) {
  return (
    <div className="grid grid-cols-2 gap-3 font-mono sm:grid-cols-4">
      {items.map(([k, v]) => (
        <div key={k} className="rounded-lg border border-line px-3 py-2"><div className="text-[0.875rem] text-muted">{k}</div><div className="num text-lg">{v}</div></div>
      ))}
    </div>
  );
}
