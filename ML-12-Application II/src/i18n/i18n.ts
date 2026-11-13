import { create } from 'zustand';
import { persist } from 'zustand/middleware';

export type Lang = 'th' | 'en';
export type Text = { th: string; en: string };
export type Level = 1 | 2 | 3;

interface Settings {
  lang: Lang;
  /** 1 beginner, 2 intermediate, 3 advanced */
  level: Level;
  quality: 'high' | 'low';
  setLang: (l: Lang) => void;
  setLevel: (l: Level) => void;
  setQuality: (q: 'high' | 'low') => void;
}

export const useSettings = create<Settings>()(
  persist(
    (set) => ({
      lang: 'th',
      level: 2,
      quality: 'high',
      setLang: (lang) => set({ lang }),
      setLevel: (level) => set({ level }),
      setQuality: (quality) => set({ quality }),
    }),
    { name: 'mlvl-settings' },
  ),
);

/** Returns a translator: t({th, en}) or t('th text', 'en text'). */
export function useT() {
  const lang = useSettings((s) => s.lang);
  return (a: Text | string, en?: string) => (typeof a === 'string' ? (lang === 'th' ? a : en ?? a) : a[lang]);
}
