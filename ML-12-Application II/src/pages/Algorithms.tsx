import { useMemo, useState } from 'react';
import { Link, useSearchParams } from 'react-router-dom';
import { useT } from '../i18n/i18n';
import { ALGORITHMS, CATEGORY_NAME, DIFFICULTY, type Category } from '../content/algorithms';
import { Icon } from '../components/ui';

export default function Algorithms() {
  const t = useT();
  const [params, setParams] = useSearchParams();
  const [q, setQ] = useState('');
  const cat = (params.get('cat') as Category | null) ?? null;
  const list = useMemo(() => ALGORITHMS.filter((a) => (!cat || a.category === cat) && (`${a.name} ${a.summary.th} ${a.summary.en}`.toLowerCase().includes(q.toLowerCase()))), [cat, q]);
  return (
    <div className="mx-auto max-w-[1600px] px-4 py-12 sm:px-6">
      <p className="label-xs">{t('คลังอัลกอริทึม', 'Algorithm gallery')}</p>
      <h1 className="mt-3 font-display text-4xl font-semibold">{t('สำรวจอัลกอริทึม', 'Explore Algorithms')}</h1>
      <div className="mt-8 flex flex-col gap-4 md:flex-row md:items-center">
        <label className="relative flex-1 md:max-w-sm">
          <Icon name="search" className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder={t('ค้นหา…', 'Search…')} aria-label={t('ค้นหา', 'Search')}
            className="h-10 w-full rounded-lg border border-line bg-surface pl-9 pr-3 text-[1.0625rem]" />
        </label>
        <div className="flex flex-wrap gap-1.5">
          <button type="button" onClick={() => setParams({})} className={`chip ${!cat ? 'border-cyan text-cyan' : ''}`}>{t('ทั้งหมด', 'All')}</button>
          {(Object.keys(CATEGORY_NAME) as Category[]).map((c) => (
            <button key={c} type="button" onClick={() => setParams({ cat: c })} className={`chip ${cat === c ? 'border-cyan text-cyan' : ''}`}>{t(CATEGORY_NAME[c])}</button>
          ))}
        </div>
      </div>
      <div className="mt-8 grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
        {list.map((a) => {
          const body = (
            <>
              <div className="flex flex-wrap items-center gap-2">
                <span className="chip">{t(CATEGORY_NAME[a.category])}</span>
                <span className="chip">{t(DIFFICULTY[a.difficulty])}</span>
                {a.status === 'available'
                  ? <span className="chip border-green/50 text-green">{t('พร้อมใช้', 'Available')}</span>
                  : <span className="chip">{t(`Phase ${a.phase} · กำลังพัฒนา`, `Phase ${a.phase} · planned`)}</span>}
              </div>
              <h2 className="mt-4 font-display text-xl font-semibold">{a.name}</h2>
              <p className="mt-2 text-[1.09375rem] text-muted">{t(a.summary)}</p>
              {a.status === 'available' && <span className="mt-5 inline-flex items-center gap-2 text-[1.0625rem] text-cyan">{t('เปิดห้องทดลอง', 'Launch')}<Icon name="arrow" /></span>}
            </>
          );
          return a.status === 'available'
            ? <Link key={a.slug} to={a.route!} className="rounded-2xl border border-line bg-surface/60 p-5 transition-colors hover:border-cyan/60">{body}</Link>
            : <div key={a.slug} className="rounded-2xl border border-dashed border-line p-5 opacity-80">{body}</div>;
        })}
        {!list.length && <p className="text-muted">{t('ไม่พบอัลกอริทึมที่ตรงกับคำค้น', 'No algorithm matches that search.')}</p>}
      </div>
    </div>
  );
}
