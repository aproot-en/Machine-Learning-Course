/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{ts,tsx}'],
  theme: {
    extend: {
      fontSize: {
        xs: ['0.9375rem', { lineHeight: '1.5rem' }],
        sm: ['1.0625rem', { lineHeight: '1.625rem' }],
        base: ['1.125rem', { lineHeight: '1.875rem' }],
        lg: ['1.25rem', { lineHeight: '1.875rem' }],
        xl: ['1.375rem', { lineHeight: '2rem' }],
        '2xl': ['1.625rem', { lineHeight: '2.25rem' }],
        '3xl': ['2rem', { lineHeight: '2.5rem' }],
      },
      colors: {
        bg: '#070B14',
        surface: '#101827',
        'surface-2': '#162036',
        line: '#1F2A44',
        ink: '#E5E7EB',
        muted: '#94A3B8',
        cyan: { DEFAULT: '#00D9FF' },
        blue: { DEFAULT: '#3B82F6' },
        purple: { DEFAULT: '#A855F7' },
        green: { DEFAULT: '#22C55E' },
        orange: { DEFAULT: '#F59E0B' },
        red: { DEFAULT: '#EF4444' },
      },
      fontFamily: {
        display: ['"Chakra Petch"', '"IBM Plex Sans Thai"', 'system-ui', 'sans-serif'],
        sans: ['"IBM Plex Sans Thai"', '"Leelawadee UI"', 'Tahoma', 'system-ui', 'sans-serif'],
        mono: ['"IBM Plex Mono"', 'ui-monospace', 'Consolas', 'monospace'],
      },
    },
  },
  plugins: [],
};
