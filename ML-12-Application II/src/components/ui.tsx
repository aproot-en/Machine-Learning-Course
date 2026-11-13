import { useId, useState, type ReactNode } from 'react';

export function Icon({ name, className = 'h-4 w-4' }: { name: IconName; className?: string }) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden="true">
      {ICONS[name]}
    </svg>
  );
}
type IconName = keyof typeof ICONS;
const ICONS = {
  play: <path d="M7 5v14l11-7z" fill="currentColor" stroke="none" />,
  pause: <path d="M8 5v14M16 5v14" strokeWidth={3} />,
  next: <path d="M6 5v14l9-7zM18 5v14" />,
  prev: <path d="M18 5v14l-9-7zM6 5v14" />,
  reset: <path d="M4 12a8 8 0 1 0 2.3-5.6M4 4v5h5" />,
  stop: <rect x="6" y="6" width="12" height="12" rx="1.5" />,
  full: <path d="M4 9V4h5M20 9V4h-5M4 15v5h5M20 15v5h-5" />,
  chevron: <path d="m6 9 6 6 6-6" />,
  plus: <path d="M12 5v14M5 12h14" />,
  trash: <path d="M4 7h16M9 7V4h6v3M6 7l1 13h10l1-13" />,
  copy: <><rect x="8" y="8" width="12" height="12" rx="2" /><path d="M16 8V4H4v12h4" /></>,
  up: <path d="m6 15 6-6 6 6" />,
  down: <path d="m6 9 6 6 6-6" />,
  undo: <path d="M9 14 4 9l5-5M4 9h10a6 6 0 0 1 0 12h-3" />,
  redo: <path d="m15 14 5-5-5-5M20 9H10a6 6 0 0 0 0 12h3" />,
  grip: <path d="M9 6h.01M15 6h.01M9 12h.01M15 12h.01M9 18h.01M15 18h.01" strokeWidth={3} />,
  close: <path d="M6 6l12 12M18 6 6 18" />,
  download: <path d="M12 4v11m0 0-4-4m4 4 4-4M5 20h14" />,
  upload: <path d="M12 20V9m0 0-4 4m4-4 4 4M5 4h14" />,
  search: <><circle cx="11" cy="11" r="6" /><path d="m20 20-4.5-4.5" /></>,
  camera: <><path d="M4 8h3l2-3h6l2 3h3v11H4z" /><circle cx="12" cy="13" r="3.5" /></>,
  arrow: <path d="M5 12h14m-5-5 5 5-5 5" />,
};

export function Button({
  children, onClick, variant = 'ghost', size = 'md', disabled, title, className = '', type = 'button',
}: {
  children: ReactNode; onClick?: () => void; variant?: 'primary' | 'ghost' | 'danger' | 'solid'; size?: 'sm' | 'md';
  disabled?: boolean; title?: string; className?: string; type?: 'button' | 'submit';
}) {
  const v = {
    primary: 'bg-cyan text-bg border-cyan hover:bg-cyan/90 font-semibold',
    solid: 'bg-surface-2 border-line text-ink hover:border-cyan/60',
    ghost: 'bg-transparent border-line text-ink hover:border-cyan/60 hover:text-white',
    danger: 'bg-transparent border-red/40 text-red hover:bg-red/10',
  }[variant];
  const s = size === 'sm' ? 'min-h-9 px-3 py-1.5 text-[1rem]' : 'min-h-11 px-4 py-2 text-sm';
  return (
    <button type={type} title={title} aria-label={title} disabled={disabled} onClick={onClick}
      className={`inline-flex items-center justify-center gap-1.5 rounded-lg border transition-colors disabled:cursor-not-allowed disabled:opacity-40 ${v} ${s} ${className}`}>
      {children}
    </button>
  );
}

export function Slider({
  label, value, min, max, step = 1, onChange, format = (v) => String(v), hint, disabled,
}: {
  label: string; value: number; min: number; max: number; step?: number; onChange: (v: number) => void;
  format?: (v: number) => string; hint?: string; disabled?: boolean;
}) {
  const id = useId();
  return (
    <div className={`flex flex-col gap-1 ${disabled ? 'opacity-50' : ''}`} title={hint}>
      <label htmlFor={id} className="flex items-baseline justify-between gap-3 text-[1rem]">
        <span>{label}</span>
        <output htmlFor={id} className="num font-mono text-cyan">{format(value)}</output>
      </label>
      <input id={id} type="range" min={min} max={max} step={step} value={value} disabled={disabled}
        onChange={(e) => onChange(Number(e.target.value))} className="w-full" />
    </div>
  );
}

export function NumberField({ label, value, min, max, step = 1, onChange }: { label: string; value: number; min: number; max: number; step?: number; onChange: (v: number) => void }) {
  const id = useId();
  return (
    <label htmlFor={id} className="flex items-center justify-between gap-3 text-[1rem]">
      <span>{label}</span>
      <input id={id} type="number" min={min} max={max} step={step} value={value}
        onChange={(e) => { const v = Number(e.target.value); if (Number.isFinite(v)) onChange(Math.max(min, Math.min(max, v))); }}
        className="num w-24 rounded-md border border-line bg-bg px-2 py-1 text-right font-mono text-[1rem] text-ink" />
    </label>
  );
}

export function Select<T extends string | number>({
  label, value, options, onChange,
}: {
  label: string; value: T; options: { value: T; label: string; disabled?: boolean }[]; onChange: (v: T) => void;
}) {
  const id = useId();
  return (
    <label htmlFor={id} className="flex flex-col gap-1 text-[1rem]">
      <span>{label}</span>
      <select id={id} value={String(value)}
        onChange={(e) => { const o = options.find((x) => String(x.value) === e.target.value); if (o) onChange(o.value); }}
        className="rounded-lg border border-line bg-bg px-2.5 py-1.5 text-[1.03125rem] text-ink">
        {options.map((o) => <option key={String(o.value)} value={String(o.value)} disabled={o.disabled}>{o.label}</option>)}
      </select>
    </label>
  );
}

export function Toggle({ label, checked, onChange, hint }: { label: string; checked: boolean; onChange: (v: boolean) => void; hint?: string }) {
  return (
    <label className="flex cursor-pointer items-center justify-between gap-3 text-[1rem]" title={hint}>
      <span>{label}</span>
      <span className="relative inline-flex">
        <input type="checkbox" className="peer sr-only" checked={checked} onChange={(e) => onChange(e.target.checked)} />
        <span className="h-5 w-9 rounded-full border border-line bg-bg transition-colors peer-checked:border-cyan peer-checked:bg-cyan/25 peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:outline-cyan" />
        <span className="absolute left-0.5 top-0.5 h-4 w-4 rounded-full bg-muted transition-transform peer-checked:translate-x-4 peer-checked:bg-cyan" />
      </span>
    </label>
  );
}

export function Segmented<T extends string>({ value, options, onChange, label }: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void; label?: string }) {
  return (
    <div className="flex flex-col gap-1 text-[1rem]">
      {label && <span>{label}</span>}
      <div role="radiogroup" aria-label={label} className="inline-flex flex-wrap rounded-lg border border-line bg-bg p-0.5">
        {options.map((o) => (
          <button key={o.value} type="button" role="radio" aria-checked={value === o.value} onClick={() => onChange(o.value)}
            className={`flex-1 rounded-md px-2.5 py-1 text-[0.96875rem] transition-colors ${value === o.value ? 'bg-cyan/15 text-cyan' : 'text-muted hover:text-ink'}`}>
            {o.label}
          </button>
        ))}
      </div>
    </div>
  );
}

/** Collapsible group inside a floating panel. */
export function Section({ title, children, defaultOpen = true }: { title: string; children: ReactNode; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <section className="border-t border-line first:border-t-0">
      <button type="button" onClick={() => setOpen(!open)} aria-expanded={open}
        className="flex w-full items-center justify-between py-2.5 text-left">
        <span className="label-xs">{title}</span>
        <Icon name="chevron" className={`h-3.5 w-3.5 text-muted transition-transform ${open ? '' : '-rotate-90'}`} />
      </button>
      {open && <div className="flex flex-col gap-3 pb-4">{children}</div>}
    </section>
  );
}

/** Floating, collapsible control panel that sits over the 3D scene. */
export function FloatingPanel({ title, children, className = '', defaultOpen = true, actions }: { title: string; children: ReactNode; className?: string; defaultOpen?: boolean; actions?: ReactNode }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className={`glass pointer-events-auto flex max-h-full flex-col ${className}`}>
      <div className="flex items-center justify-between gap-2 px-3.5 py-2.5">
        <button type="button" onClick={() => setOpen(!open)} aria-expanded={open} className="flex min-w-0 items-center gap-2 text-left font-display text-[1.0625rem] font-semibold tracking-wide">
          <Icon name="chevron" className={`h-4 w-4 text-cyan transition-transform ${open ? '' : '-rotate-90'}`} />
          {title}
        </button>
        {actions}
      </div>
      {open && <div className="scroll-thin min-h-0 overflow-y-auto px-3.5 pb-2">{children}</div>}
    </div>
  );
}

export function Tabs({ tabs, initial }: { tabs: { id: string; label: string; content: ReactNode }[]; initial?: string }) {
  const [active, setActive] = useState(initial ?? tabs[0].id);
  const cur = tabs.find((t) => t.id === active) ?? tabs[0];
  return (
    <div>
      <div role="tablist" className="scroll-thin flex gap-1 overflow-x-auto border-b border-line">
        {tabs.map((t) => (
          <button key={t.id} role="tab" type="button" aria-selected={t.id === cur.id} onClick={() => setActive(t.id)}
            className={`-mb-px whitespace-nowrap border-b-2 px-3.5 py-2.5 text-[1.0625rem] transition-colors ${t.id === cur.id ? 'border-cyan text-ink' : 'border-transparent text-muted hover:text-ink'}`}>
            {t.label}
          </button>
        ))}
      </div>
      <div role="tabpanel" className="pt-6">{cur.content}</div>
    </div>
  );
}

export function Readout({ rows, title }: { title?: string; rows: [string, ReactNode, string?][] }) {
  return (
    <div className="glass pointer-events-none max-w-full px-3.5 py-2.5 font-mono text-[0.9375rem] leading-relaxed">
      {title && <div className="mb-1 font-display text-[1rem] font-semibold tracking-wide text-cyan">{title}</div>}
      {rows.map(([k, v, color]) => (
        <div key={k} className="flex flex-wrap justify-between gap-x-4 gap-y-0.5">
          <span className="text-muted">{k}</span>
          <b className="num min-w-0 break-words text-right font-medium" style={color ? { color } : undefined}>{v}</b>
        </div>
      ))}
    </div>
  );
}

export function Prose({ children }: { children: ReactNode }) {
  return <div className="max-w-[68ch] space-y-3 text-[1.125rem] leading-7 text-ink/90 [&_b]:text-ink [&_li]:ml-5 [&_ol]:list-decimal [&_ol]:space-y-1.5 [&_ul]:list-disc [&_ul]:space-y-1.5">{children}</div>;
}

export const fmt = (v: number, d = 3) => (Number.isFinite(v) ? (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : v.toFixed(d)) : '—');
export const pct = (v: number) => `${(v * 100).toFixed(1)}%`;
