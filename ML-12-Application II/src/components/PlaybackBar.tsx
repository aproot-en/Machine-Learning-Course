import { useEffect, useState, type ReactNode } from 'react';
import { useT } from '../i18n/i18n';
import { Icon } from './ui';

export const SPEEDS = [0.25, 0.5, 1, 2, 4];

export interface PlaybackProps {
  playing: boolean;
  onToggle: () => void;
  onNext: () => void;
  onPrev?: () => void;
  onReset: () => void;
  canPrev?: boolean;
  canNext?: boolean;
  speed: number;
  onSpeed: (s: number) => void;
  status?: ReactNode;
  children?: ReactNode;
}

/** Bottom playback toolbar. Keyboard: Space play/pause · → next · ← previous · R reset. */
export function PlaybackBar(p: PlaybackProps) {
  const t = useT();
  const [collapsed, setCollapsed] = useState(false);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA' || (e.target as HTMLElement)?.isContentEditable) return;
      if (e.key === ' ') { e.preventDefault(); p.onToggle(); }
      else if (e.key === 'ArrowRight' && p.canNext !== false) p.onNext();
      else if (e.key === 'ArrowLeft' && p.onPrev && p.canPrev) p.onPrev();
      else if (e.key === 'r' || e.key === 'R') p.onReset();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [p]);
  const btn = 'grid h-9 w-9 place-items-center rounded-lg text-ink transition-colors hover:bg-white/5 disabled:opacity-35';
  if (collapsed) {
    return (
      <button type="button" onClick={() => setCollapsed(false)}
        className="glass pointer-events-auto flex items-center gap-2 px-3 py-2 text-[0.96875rem] text-cyan transition-colors hover:text-ink"
        title={t('แสดงแถบควบคุมการทำงาน', 'Show playback controls')} aria-label={t('แสดงแถบควบคุมการทำงาน', 'Show playback controls')}>
        <Icon name="chevron" className="h-3.5 w-3.5 rotate-180" />
        {t('แสดงแถบควบคุม', 'Show controls')}
      </button>
    );
  }
  return (
    <div className="glass pointer-events-auto flex flex-wrap items-center gap-1 px-2 py-1.5">
      <button type="button" className={btn} onClick={p.onReset} title={t('รีเซ็ต (R)', 'Reset (R)')} aria-label={t('รีเซ็ต', 'Reset')}><Icon name="reset" /></button>
      {p.onPrev && <button type="button" className={btn} onClick={p.onPrev} disabled={!p.canPrev} title={t('ย้อนสถานะที่บันทึก (←)', 'Previous recorded state (←)')} aria-label={t('ย้อนกลับ', 'Previous')}><Icon name="prev" /></button>}
      <button type="button" onClick={p.onToggle} title={t('เล่น/หยุด (Space)', 'Play/Pause (Space)')} aria-label={p.playing ? t('หยุด', 'Pause') : t('เล่น', 'Play')}
        className="grid h-10 w-10 place-items-center rounded-full bg-cyan text-bg transition-transform hover:scale-105">
        <Icon name={p.playing ? 'pause' : 'play'} className="h-4 w-4" />
      </button>
      <button type="button" className={btn} onClick={p.onNext} disabled={p.canNext === false} title={t('ขั้นถัดไป (→)', 'Next step (→)')} aria-label={t('ถัดไป', 'Next')}><Icon name="next" /></button>
      <div className="mx-1 h-6 w-px bg-line" />
      <label className="flex items-center gap-1.5 px-1 text-[0.9375rem] text-muted">
        {t('ความเร็ว', 'Speed')}
        <select value={p.speed} onChange={(e) => p.onSpeed(Number(e.target.value))} className="rounded-md border border-line bg-bg px-1.5 py-1 font-mono text-[0.9375rem] text-ink">
          {SPEEDS.map((s) => <option key={s} value={s}>{s}×</option>)}
        </select>
      </label>
      {p.children}
      {p.status && <div className="ml-1 min-w-0 px-2 font-mono text-[0.9375rem] text-muted">{p.status}</div>}
      <div className="mx-0.5 h-6 w-px bg-line" />
      <button type="button" className={btn} onClick={() => setCollapsed(true)}
        title={t('พับแถบควบคุมการทำงาน', 'Collapse playback controls')} aria-label={t('พับแถบควบคุมการทำงาน', 'Collapse playback controls')}>
        <Icon name="chevron" className="h-3.5 w-3.5" />
      </button>
    </div>
  );
}
