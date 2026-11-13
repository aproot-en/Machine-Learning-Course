import { useRef, useState, type ReactNode } from 'react';
import { Stage, StageToolbar, useCameraState } from '../engine3d/Stage';
import { ExperimentScene } from '../engine3d/ExperimentScene';
import type { ModelExperiment } from '../engine3d/experiment';
import { flowVector } from '../engine3d/experiment';
import { useT } from '../i18n/i18n';
import { Button, FloatingPanel, Icon, NumberField, Segmented } from './ui';
import { PlaybackBar, type PlaybackProps } from './PlaybackBar';

export function ExperimentResults({ data }: { data: ModelExperiment }) {
  const t = useT();
  return <section aria-label={t('ผลการทดลอง', 'Experiment results')} className="mx-auto max-w-[1600px] px-4 py-6 sm:px-6">
    <h2 className="font-display text-xl font-semibold">{t('ผลการทดลอง', 'Experiment results')} · {data.model}</h2>
    <p className="mt-2 break-words text-base text-cyan">{data.output}</p>
    {data.note && <p className="mt-1 text-sm text-muted">{data.note}</p>}
    <dl className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-4">{data.metrics.map((metric, i) => <div key={i} className="rounded-lg border border-line bg-surface p-3">
      <dt className="text-sm text-muted">{metric.label}</dt><dd className="mt-1 break-words font-mono text-lg">{metric.value}</dd>
    </div>)}</dl>
    <div className="mt-4 grid gap-4 lg:grid-cols-2">
      {data.history?.map((series, i) => {
        const finite = series.values.map((value, index) => ({ value, index })).filter(({ value }) => Number.isFinite(value));
        const lo = finite.length ? Math.min(...finite.map((p) => p.value)) : 0, hi = finite.length ? Math.max(...finite.map((p) => p.value)) : 1;
        const points = finite.map(({ value, index }) => `${48 + index / Math.max(1, series.values.length - 1) * 490},${126 - (value - lo) / (hi - lo || 1) * 100}`).join(' ');
        return <figure key={i} className="min-w-0 rounded-lg border border-line bg-surface p-4">
          <figcaption className="text-base">{series.label} <span className="text-sm text-muted">· {finite.length} {t('จุดที่บันทึก', 'recorded points')}</span></figcaption>
          {finite.length > 0 ? <svg viewBox="0 0 560 160" className="mt-2 w-full" role="img" aria-label={`${series.label}: ${finite[0].value} → ${finite[finite.length - 1].value}`}>
            <path d="M48 20V130H540" fill="none" stroke="#425472" /><polyline points={points} fill="none" stroke={series.color ?? '#00D9FF'} strokeWidth="2.5" />
            {finite.length === 1 && <circle cx="48" cy={hi === lo ? 126 : 26} r="3" fill={series.color ?? '#00D9FF'} />}
            <text x="0" y="26" fill="#94A3B8" fontSize="13">{Number(hi.toPrecision(3))}</text><text x="0" y="130" fill="#94A3B8" fontSize="13">{Number(lo.toPrecision(3))}</text>
            <text x="48" y="152" fill="#94A3B8" fontSize="13">0</text><text x="540" y="152" textAnchor="end" fill="#94A3B8" fontSize="13">{series.values.length - 1}</text>
          </svg> : <p className="py-8 text-sm text-muted">{t('เริ่มฝึกเพื่อบันทึกผลการทดลอง', 'Start training to record results')}</p>}
        </figure>;
      })}
      {data.confusion && <div className="scroll-thin overflow-auto rounded-lg border border-line bg-surface p-4">
        <p className="mb-2 text-base">{t('คำตอบจริง × ผลทำนาย (ชุดทดสอบ)', 'Actual × predicted (test set)')}</p>
        <table className="w-full text-center font-mono text-sm"><thead><tr><th>y / ŷ</th>{data.confusion.map((_, i) => <th key={i} className="p-1">{i}</th>)}</tr></thead>
          <tbody>{data.confusion.map((row, i) => <tr key={i}><th className="p-1">{i}</th>{row.map((value, j) => <td key={j} className={`border border-line p-1 ${i === j ? 'bg-cyan/10 text-cyan' : value ? 'text-orange' : 'text-muted'}`}>{value}</td>)}</tr>)}</tbody>
        </table>
      </div>}
    </div>
  </section>;
}

export function ExperimentLab({ data, controls, playback, detail, detailLabel }: { data: ModelExperiment; controls: ReactNode; playback?: PlaybackProps; detail: ReactNode; detailLabel?: string }) {
  const t = useT();
  const [view, setView] = useState<'network' | 'detail'>('network');
  const [flowing, setFlowing] = useState(() => !(typeof window !== 'undefined' && window.matchMedia?.('(prefers-reduced-motion: reduce)').matches));
  const [speed, setSpeed] = useState('1');
  const [selected, setSelected] = useState(0);
  const [cam, setCam] = useCameraState({ preset: 'front' });
  const box = useRef<HTMLDivElement>(null);
  const stageIndex = Math.min(selected, Math.max(0, data.stages.length - 1));
  return <>
    <div className="mx-auto flex max-w-[1600px] flex-wrap items-end gap-3 px-4 pb-4 sm:px-6">
      <Segmented value={view} onChange={setView} options={[{ value: 'network', label: t('ข้อมูลผ่านโมเดล 3D', '3D model execution') }, { value: 'detail', label: detailLabel ?? t('สำรวจอัลกอริทึม', 'Algorithm detail') }]} />
      {view === 'network' && <>
        <Button size="sm" onClick={() => setFlowing((v) => !v)}><Icon name={flowing ? 'pause' : 'play'} />{flowing ? t('หยุดสัญญาณ', 'Pause signal') : t('เล่นสัญญาณ', 'Play signal')}</Button>
        <Segmented label={t('ความเร็วสัญญาณ', 'Signal speed')} value={speed} onChange={setSpeed} options={['0.5', '1', '2'].map((v) => ({ value: v, label: `${v}×` }))} />
        {data.sample && <div className="w-60"><NumberField label={t('ตัวอย่างข้อมูล', 'Data sample')} value={Math.min(data.sample.index, data.sample.count - 1)} min={0} max={Math.max(0, data.sample.count - 1)} onChange={(index) => data.sample?.onChange(Math.max(0, Math.min(data.sample.count - 1, Math.trunc(index))))} /></div>}
      </>}
    </div>
    {view === 'detail' ? detail : <>
      <div ref={box} className="relative flex h-[calc(100svh-220px)] min-h-[620px] border-y border-line bg-bg">
        <div className="relative min-w-0 flex-1">
          <Stage cam={cam} distance={Math.max(24, data.stages.length * 8)} fogEnabled={false}>
            <ExperimentScene data={data} playing={flowing} speed={Number(speed)} selected={stageIndex} onSelect={setSelected} />
          </Stage>
          <div className="pointer-events-none absolute inset-0 flex flex-col justify-between p-3 sm:p-4">
            <div className="flex flex-wrap items-start justify-between gap-2">
              <div className="pointer-events-auto max-w-sm"><StageToolbar cam={cam} setCam={setCam} container={box} allowPresets={false} />
                <p className="mt-2 rounded bg-bg/85 px-2 py-1 text-sm text-muted">{data.source}</p>
              </div>
              <span className="rounded bg-bg/85 px-2 py-1 text-sm text-muted">{t('คลิกชั้นเพื่อดูค่าจริง · ลากเพื่อหมุน · เลื่อนเพื่อซูม', 'Click a layer for values · drag to orbit · scroll to zoom')}</span>
            </div>
            <div className="flex flex-col items-center gap-2">
              <p className="max-w-full break-words rounded bg-bg/90 px-3 py-1 font-mono text-sm text-cyan">{data.stages[stageIndex]?.name}: {flowVector(data.stages[stageIndex]?.values, 10)}</p>
              <div className="pointer-events-auto flex max-w-full flex-wrap justify-center gap-1 rounded-lg border border-line bg-bg/90 p-1">
                {data.stages.map((stage, i) => <button key={i} type="button" aria-pressed={stageIndex === i} onClick={() => setSelected(i)} className={`rounded px-2 py-1 text-xs ${stageIndex === i ? 'bg-cyan/15 text-cyan' : 'text-muted hover:text-ink'}`}>{stage.name}</button>)}
              </div>
              {playback && <PlaybackBar {...playback} />}
            </div>
          </div>
        </div>
        <aside className="hidden w-[350px] shrink-0 border-l border-line p-3 xl:block">
          <FloatingPanel title={t('ตั้งค่าการทดลอง', 'Experiment controls')} className="h-full"><div className="pb-4">{controls}</div></FloatingPanel>
        </aside>
      </div>
      <div className="border-b border-line bg-surface/40 p-4 xl:hidden">{controls}</div>
      <p className="mx-auto max-w-[1600px] px-4 pt-3 text-sm text-muted sm:px-6">{t('สีฟ้า = ค่าบวก · ม่วง = ค่าลบ · ส้ม = ค่าที่เลือก · แต่ละชั้นปรับสเกลเพื่อแสดงภาพ ตัวเลขในแถบตรวจค่าเป็นค่าจริง · เส้นเชื่อมแสดงการส่งข้อมูลระหว่างขั้น · ปุ่มสัญญาณควบคุมภาพเคลื่อนไหว ปุ่มฝึกด้านล่างฉากอัปเดตโมเดลจริง', 'Cyan = positive · purple = negative · orange = selected · layers use relative visual scales; inspected numbers are actual values · links show data transfer · signal controls animate the trace; training controls update the model')}</p>
    </>}
    <ExperimentResults data={data} />
  </>;
}
