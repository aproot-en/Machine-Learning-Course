import { useEffect } from 'react';
import { Link, NavLink, useNavigate, useParams } from 'react-router-dom';
import { useT } from '../../i18n/i18n';
import { LESSONS } from './lessons';

export default function LearnPage() {
  const t = useT();
  const { slug } = useParams();
  const nav = useNavigate();
  const idx = Math.max(0, LESSONS.findIndex((l) => l.slug === slug));
  const lesson = LESSONS[idx];
  useEffect(() => { if (slug && !LESSONS.some((l) => l.slug === slug)) nav('/learn', { replace: true }); }, [slug, nav]);
  useEffect(() => { window.scrollTo(0, 0); }, [idx]);
  const groups = [...new Set(LESSONS.map((l) => l.group.en))];
  const prev = LESSONS[idx - 1], next = LESSONS[idx + 1];

  return (
    <div className="mx-auto grid max-w-[1400px] gap-8 px-4 py-8 sm:px-6 lg:grid-cols-[260px_minmax(0,1fr)]">
      <aside className="lg:sticky lg:top-20 lg:h-[calc(100vh-6rem)] lg:overflow-y-auto scroll-thin">
        <p className="label-xs">{t('บทเรียน', 'Lessons')}</p>
        <h1 className="mt-2 font-display text-2xl font-semibold">{t('ทฤษฎี Machine Learning', 'Machine learning theory')}</h1>
        <nav className="mt-5 flex flex-col gap-5" aria-label={t('สารบัญบทเรียน', 'Lesson index')}>
          {groups.map((g) => (
            <div key={g}>
              <div className="label-xs mb-1.5">{t(LESSONS.find((l) => l.group.en === g)!.group)}</div>
              <ul className="flex flex-col gap-0.5">
                {LESSONS.filter((l) => l.group.en === g).map((l) => {
                  const n = LESSONS.indexOf(l) + 1;
                  return (
                    <li key={l.slug}>
                      <NavLink to={`/learn/${l.slug}`} className={({ isActive }) => `flex gap-2 rounded-md px-2 py-1.5 text-[1.0625rem] ${isActive || (!slug && n === 1) ? 'bg-cyan/10 text-cyan' : 'text-muted hover:text-ink'}`}>
                        <span className="w-5 shrink-0 font-mono text-[0.9375rem] opacity-70">{n}</span>{t(l.title)}
                      </NavLink>
                    </li>
                  );
                })}
              </ul>
            </div>
          ))}
        </nav>
      </aside>

      <article className="min-w-0">
        <div className="mb-6 border-b border-line pb-5">
          <p className="label-xs">{t(lesson.group)} · {t('บทที่', 'Lesson')} {idx + 1}/{LESSONS.length} · ~{lesson.minutes} {t('นาที', 'min')}</p>
          <h1 className="mt-2 font-display text-[clamp(1.8rem,3.5vw,2.6rem)] font-semibold leading-tight">{t(lesson.title)}</h1>
          <p className="mt-2 text-[1.0625rem] text-muted">{t('กราฟโต้ตอบใช้เอนจินเดียวกับห้องทดลอง ส่วนแผนผังอ้างอิงอธิบายโครงสร้างและแยกจากผลการทดลองจริง', 'Interactive plots use the lab engines. Reference diagrams explain architecture and are separate from measured experiment results.')}</p>
        </div>
        {lesson.render()}
        <div className="mt-12 flex flex-wrap justify-between gap-3 border-t border-line pt-6">
          {prev ? <Link to={`/learn/${prev.slug}`} className="rounded-lg border border-line px-4 py-2 text-[1.0625rem] hover:border-cyan/60">← {t(prev.title)}</Link> : <span />}
          {next && <Link to={`/learn/${next.slug}`} className="rounded-lg border border-cyan/60 px-4 py-2 text-[1.0625rem] text-cyan hover:bg-cyan/10">{t(next.title)} →</Link>}
        </div>
      </article>
    </div>
  );
}
