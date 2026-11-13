import { useEffect, useRef } from 'react';
import Plotly from 'plotly.js-dist-min';
import type { Series } from './common';

export default function PlotImpl({ series, xTitle, yTitle, height = 260, yRange, xRange, markerX, yLog }: { series: Series[]; xTitle: string; yTitle: string; height?: number; yRange?: [number, number]; xRange?: [number, number]; markerX?: number; yLog?: boolean }) {
  const el = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!el.current) return;
    const axis = { gridcolor: '#1F2A44', zerolinecolor: '#1F2A44', linecolor: '#1F2A44', tickfont: { family: 'IBM Plex Mono, monospace', size: 14, color: '#94A3B8' } };
    Plotly.react(el.current, series.map((s) => ({ type: 'scatter', mode: s.mode ?? 'lines', name: s.name, x: s.x, y: s.y, text: s.text, line: { color: s.color, width: 2, dash: s.dash }, marker: { color: s.color, size: s.size ?? 7 } })), {
      height, margin: { l: 68, r: 20, t: 32, b: 60 }, paper_bgcolor: 'rgba(0,0,0,0)', plot_bgcolor: 'rgba(0,0,0,0)',
      font: { family: 'IBM Plex Sans Thai, sans-serif', color: '#E5E7EB', size: 15 },
      xaxis: { ...axis, title: { text: xTitle, font: { size: 15, color: '#94A3B8' } }, range: xRange },
      yaxis: { ...axis, title: { text: yTitle, font: { size: 15, color: '#94A3B8' } }, range: yRange, type: yLog ? 'log' : 'linear' },
      legend: { orientation: 'h', y: 1.12, x: 0, font: { size: 15 } },
      shapes: markerX != null ? [{ type: 'line', x0: markerX, x1: markerX, yref: 'paper', y0: 0, y1: 1, line: { color: '#00D9FF', width: 1, dash: 'dot' } }] : [],
    }, { displayModeBar: false, responsive: true });
  }, [series, xTitle, yTitle, height, yRange, xRange, markerX, yLog]);
  useEffect(() => () => { if (el.current) Plotly.purge(el.current); }, []);
  return <div ref={el} className="w-full" style={{ height }} />;
}
