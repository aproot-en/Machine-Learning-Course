import { useEffect, useState } from 'react';
import { Link, NavLink, Outlet, useLocation } from 'react-router-dom';
import { useSettings, useT, type Level } from '../i18n/i18n';
import { Icon } from './ui';

const NAV: { to: string; th: string; en: string }[] = [
  { to: '/', th: 'หน้าแรก', en: 'Home' },
  { to: '/algorithms', th: 'อัลกอริทึม', en: 'Algorithms' },
  { to: '/model-builder', th: 'สร้างโมเดล', en: 'Model Builder' },
  { to: '/optimization', th: 'การเพิ่มประสิทธิภาพ', en: 'Optimization' },
  { to: '/datasets', th: 'ชุดข้อมูล', en: 'Datasets' },
  { to: '/learn', th: 'บทเรียน', en: 'Learn' },
  { to: '/compare', th: 'เปรียบเทียบ', en: 'Compare' },
  { to: '/about', th: 'เกี่ยวกับ', en: 'About' },
];

export function Logo() {
  return (
    <svg viewBox="0 0 32 32" className="h-7 w-7" aria-hidden="true">
      <g stroke="#00D9FF" strokeWidth="1" opacity=".5">
        <path d="M5 9 16 5M5 9 16 16M5 9 16 27M5 23 16 5M5 23 16 16M5 23 16 27M16 5 27 16M16 16 27 16M16 27 27 16" />
      </g>
      <circle cx="5" cy="9" r="2.8" fill="#00D9FF" /><circle cx="5" cy="23" r="2.8" fill="#F59E0B" />
      <circle cx="16" cy="5" r="2.8" fill="#A855F7" /><circle cx="16" cy="16" r="2.8" fill="#22C55E" /><circle cx="16" cy="27" r="2.8" fill="#3B82F6" />
      <circle cx="27" cy="16" r="3.4" fill="#E5E7EB" />
    </svg>
  );
}

function Navbar() {
  const t = useT();
  const { lang, setLang, level, setLevel } = useSettings();
  const [open, setOpen] = useState(false);
  const loc = useLocation();
  useEffect(() => { setOpen(false); }, [loc.pathname]);
  const link = ({ isActive }: { isActive: boolean }) => `whitespace-nowrap rounded-md px-2.5 py-1.5 text-[1.03125rem] transition-colors ${isActive ? 'text-cyan' : 'text-muted hover:text-ink'}`;
  return (
    <header className="sticky top-0 z-40 border-b border-line bg-bg/85 backdrop-blur-md">
      <div className="mx-auto flex min-h-16 max-w-[1600px] items-center gap-4 px-4 sm:px-6">
        <Link to="/" className="flex shrink-0 items-center gap-2.5">
          <Logo />
          <span className="font-display text-[1.25rem] font-semibold tracking-wide">ML Visual Lab</span>
        </Link>
        <nav aria-label={t('เมนูหลัก', 'Main')} className="hidden min-w-0 flex-1 items-center gap-0.5 overflow-x-auto 2xl:flex">
          {NAV.map((n) => <NavLink key={n.to} to={n.to} end={n.to === '/'} className={link}>{lang === 'th' ? n.th : n.en}</NavLink>)}
        </nav>
        <div className="ml-auto flex items-center gap-2">
          <label className="hidden items-center gap-1.5 text-[0.9375rem] text-muted sm:flex">
            {t('ระดับ', 'Level')}
            <select value={level} onChange={(e) => setLevel(Number(e.target.value) as Level)} className="rounded-md border border-line bg-surface px-1.5 py-1 text-[0.96875rem] text-ink">
              <option value={1}>{t('พื้นฐาน', 'Beginner')}</option>
              <option value={2}>{t('ปานกลาง', 'Intermediate')}</option>
              <option value={3}>{t('ขั้นสูง', 'Advanced')}</option>
            </select>
          </label>
          <div role="radiogroup" aria-label="Language" className="flex rounded-md border border-line p-0.5 font-mono text-[0.9375rem]">
            {(['th', 'en'] as const).map((l) => (
              <button key={l} type="button" role="radio" aria-checked={lang === l} onClick={() => setLang(l)}
                className={`rounded px-2 py-0.5 uppercase ${lang === l ? 'bg-cyan/15 text-cyan' : 'text-muted'}`}>{l}</button>
            ))}
          </div>
          <button type="button" className="grid h-9 w-9 place-items-center rounded-md border border-line 2xl:hidden" onClick={() => setOpen(!open)} aria-expanded={open} aria-label={t('เมนู', 'Menu')}>
            <Icon name={open ? 'close' : 'grip'} />
          </button>
        </div>
      </div>
      {open && (
        <nav className="border-t border-line px-4 py-3 2xl:hidden">
          <div className="grid grid-cols-2 gap-1 sm:grid-cols-3">
            {NAV.map((n) => <NavLink key={n.to} to={n.to} end={n.to === '/'} className={link}>{lang === 'th' ? n.th : n.en}</NavLink>)}
          </div>
          <label className="mt-3 flex items-center gap-2 text-[0.9375rem] text-muted sm:hidden">
            {t('ระดับ', 'Level')}
            <select value={level} onChange={(e) => setLevel(Number(e.target.value) as Level)} className="rounded-md border border-line bg-surface px-1.5 py-1 text-[0.96875rem] text-ink">
              <option value={1}>{t('พื้นฐาน', 'Beginner')}</option>
              <option value={2}>{t('ปานกลาง', 'Intermediate')}</option>
              <option value={3}>{t('ขั้นสูง', 'Advanced')}</option>
            </select>
          </label>
        </nav>
      )}
    </header>
  );
}

function Footer() {
  const t = useT();
  return (
    <footer className="border-t border-line">
      <div className="mx-auto flex max-w-[1600px] flex-col gap-6 px-4 py-10 sm:px-6 md:flex-row md:items-start md:justify-between">
        <div className="max-w-md">
          <div className="flex items-center gap-2.5"><Logo /><span className="font-display text-[1.1875rem] font-semibold">ML Visual Lab</span></div>
          <p className="mt-3 text-[1.0625rem] text-muted">{t('สื่อการสอนแบบโต้ตอบสำหรับนักศึกษามหาวิทยาลัย ฉากทดลองแสดงค่าจากเอนจินจริง แผนผังอ้างอิงใช้ประกอบการอธิบายโครงสร้าง', 'Interactive teaching material for university students. Experiment scenes show real engine values; reference diagrams explain architecture.')}</p>
        </div>
        <div className="text-[1.0625rem]">
          <nav aria-label={t('ลิงก์ท้ายหน้า', 'Footer links')} className="flex flex-col gap-2">
            <Link to="/algorithms" className="text-muted hover:text-cyan">{t('อัลกอริทึม', 'Algorithms')}</Link>
            <Link to="/learn" className="text-muted hover:text-cyan">{t('บทเรียน', 'Learn')}</Link>
            <Link to="/about" className="text-muted hover:text-cyan">{t('เกี่ยวกับ', 'About')}</Link>
          </nav>
        </div>
      </div>
    </footer>
  );
}

export function Layout() {
  const loc = useLocation();
  const lang = useSettings((settings) => settings.lang);
  useEffect(() => { document.documentElement.lang = lang; }, [lang]);
  useEffect(() => { window.scrollTo(0, 0); }, [loc.pathname]);
  return (
    <div className="flex min-h-screen flex-col">
      <Navbar />
      <main className="flex-1"><Outlet /></main>
      <Footer />
    </div>
  );
}
