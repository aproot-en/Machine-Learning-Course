import { lazy, Suspense, useEffect, useMemo, useState, type ReactNode } from 'react';
import katex from 'katex';
import { useT } from '../i18n/i18n';
import { Button, Icon } from './ui';

export function Tex({ children, block }: { children: string; block?: boolean }) {
  const html = useMemo(() => katex.renderToString(children, { displayMode: !!block, throwOnError: false }), [children, block]);
  return block
    ? <div className="scroll-thin overflow-x-auto py-1" dangerouslySetInnerHTML={{ __html: html }} />
    : <span dangerouslySetInnerHTML={{ __html: html }} />;
}

const PlotImpl = lazy(() => import('./PlotImpl'));
export interface Series { name: string; x: number[]; y: number[]; color: string; dash?: 'dot' | 'dash'; mode?: 'lines' | 'markers' | 'lines+markers'; size?: number; text?: string[] }
export function LinePlot(props: { series: Series[]; xTitle: string; yTitle: string; height?: number; yRange?: [number, number]; xRange?: [number, number]; markerX?: number; yLog?: boolean }) {
  return (
    <Suspense fallback={<div className="grid h-[260px] place-items-center text-sm text-muted">…</div>}>
      <PlotImpl {...props} />
    </Suspense>
  );
}

export function ConfusionMatrix({ m, classNames }: { m: number[][]; classNames: string[] }) {
  const t = useT();
  const max = Math.max(1, ...m.flat());
  return (
    <div className="scroll-thin overflow-x-auto">
      <table className="num border-separate border-spacing-1 font-mono text-[0.96875rem]">
        <thead>
          <tr>
            <th className="px-2 text-left text-[0.875rem] font-normal text-muted">{t('จริง ↓ / ทำนาย →', 'actual ↓ / predicted →')}</th>
            {classNames.map((c) => <th key={c} className="px-2 font-normal text-muted">{c}</th>)}
          </tr>
        </thead>
        <tbody>
          {m.map((row, i) => (
            <tr key={i}>
              <th className="px-2 text-left font-normal text-muted">{classNames[i]}</th>
              {row.map((v, j) => (
                <td key={j} className="h-9 min-w-[44px] rounded text-center"
                  style={{ background: i === j ? `rgba(34,197,94,${0.12 + 0.6 * v / max})` : v ? `rgba(239,68,68,${0.12 + 0.6 * v / max})` : 'rgba(31,42,68,0.5)' }}>
                  {v}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export interface QuizItem { q: string; options: string[]; answer: number; why: string }
/** Multiple-choice quiz with immediate feedback; answers persist locally per quiz id. */
export function Quiz(props: { id: string; items: QuizItem[] }) {
  return <QuizState key={props.id} {...props} />;
}
function QuizState({ id, items }: { id: string; items: QuizItem[] }) {
  const t = useT();
  const key = `mlvl-quiz-${id}`;
  const [answers, setAnswers] = useState<Record<number, number>>(() => {
    try { const stored = JSON.parse(localStorage.getItem(key) ?? '{}'); return stored && typeof stored === 'object' && !Array.isArray(stored) ? stored : {}; } catch { return {}; }
  });
  useEffect(() => { try { localStorage.setItem(key, JSON.stringify(answers)); } catch { /* storage unavailable */ } }, [answers, key]);
  const score = items.filter((it, i) => answers[i] === it.answer).length;
  return (
    <div className="flex flex-col gap-5">
      <div className="flex items-center justify-between">
        <span className="label-xs">{t('คะแนน', 'Score')} {score}/{items.length}</span>
        <Button size="sm" onClick={() => setAnswers({})}><Icon name="reset" />{t('ทำใหม่', 'Retry')}</Button>
      </div>
      {items.map((it, i) => {
        const a = answers[i];
        return (
          <fieldset key={i} className="rounded-xl border border-line bg-surface/60 p-4">
            <legend className="px-1 text-[1.125rem] font-medium">{i + 1}. {it.q}</legend>
            <div className="mt-2 flex flex-col gap-1.5">
              {it.options.map((o, j) => {
                const chosen = a === j, right = j === it.answer;
                const tone = a == null ? 'border-line hover:border-cyan/50' : right ? 'border-green/60 bg-green/10' : chosen ? 'border-red/60 bg-red/10' : 'border-line opacity-60';
                return (
                  <button key={j} type="button" disabled={a != null} onClick={() => setAnswers({ ...answers, [i]: j })}
                    className={`rounded-lg border px-3 py-2 text-left text-[1.0625rem] transition-colors ${tone}`}>
                    {o}
                  </button>
                );
              })}
            </div>
            {a != null && (
              <p className={`mt-3 text-[1.0625rem] ${a === it.answer ? 'text-green' : 'text-orange'}`}>
                {a === it.answer ? t('ถูกต้อง — ', 'Correct — ') : t('ยังไม่ใช่ — ', 'Not quite — ')}
                <span className="text-ink/85">{it.why}</span>
              </p>
            )}
          </fieldset>
        );
      })}
    </div>
  );
}

export function Callout({ children, tone = 'info' }: { children: ReactNode; tone?: 'info' | 'warn' }) {
  return <div className={`rounded-lg border-l-2 bg-surface/70 px-4 py-3 text-[1.0625rem] leading-6 ${tone === 'warn' ? 'border-orange' : 'border-cyan'}`}>{children}</div>;
}

/** Trigger a browser download of text or a data URL. */
export function download(name: string, data: string, mime = 'application/json') {
  const href = data.startsWith('data:') ? data : URL.createObjectURL(new Blob([data], { type: mime }));
  const a = document.createElement('a');
  a.href = href; a.download = name; a.click();
  if (!data.startsWith('data:')) setTimeout(() => URL.revokeObjectURL(href), 1000);
}
