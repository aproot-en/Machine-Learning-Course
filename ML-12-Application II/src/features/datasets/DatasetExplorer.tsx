import { extent } from '../../lib/numbers';
import { useEffect, useMemo, useState } from 'react';
import { useCameraState } from '../../engine3d/Stage';
import { LabShell, Metrics } from '../../components/LabShell';
import { PointCloud, Segments } from '../../engine3d/plots';
import { BOX, FrameBox, Label } from '../../engine3d/primitives';
import { CLASS_COLORS } from '../../engine3d/palette';
import { BUILTINS, builtin, columnKinds, columnStats, handleMissing, parseCSV, pca, scaleColumns, tableFromCSV, type BuiltinId, type Table } from '../../data/explorer';
import { trainTestSplit } from '../../data/datasets';
import { IRIS_ROWS } from '../../data/iris';
import { useT } from '../../i18n/i18n';
import { Button, Icon, Section, Segmented, Select, Slider, Toggle, fmt, pct } from '../../components/ui';
import { Callout, download, Quiz } from '../../components/common';
import * as THREE from 'three';

const R = 5;
const PALETTE = [...CLASS_COLORS, '#F472B6', '#94A3B8', '#FDE047', '#2DD4BF'];
const ramp = (u: number) => new THREE.Color('#1e3a8a').lerp(new THREE.Color('#00D9FF'), Math.min(1, u * 2)).lerp(new THREE.Color('#F59E0B'), Math.max(0, u * 2 - 1));

function Histogram({ values, labels, classNames, bins = 16, w = 220, h = 90 }: { values: number[]; labels: number[] | null; classNames: string[]; bins?: number; w?: number; h?: number }) {
  const t = useT();
  if (!values.length) return <p className="py-4 text-sm text-muted">{t('ไม่มีข้อมูลหลังทำความสะอาด', 'No rows remain after cleaning')}</p>;
  const [lo, hi] = extent(values), span = hi - lo || 1;
  const K = labels ? classNames.length : 1;
  const counts = Array.from({ length: bins }, () => Array(K).fill(0));
  values.forEach((v, i) => counts[Math.min(bins - 1, Math.floor(((v - lo) / span) * bins))][labels ? labels[i] : 0]++);
  const max = Math.max(1, ...counts.map((c) => c.reduce((a, b) => a + b, 0)));
  const bw = w / bins;
  return (
    <svg viewBox={`0 0 ${w} ${h + 22}`} width={w} height={h + 22} role="img" aria-label="histogram">
      {counts.map((c, b) => {
        let y = h;
        return c.map((n, k) => { const bh = (n / max) * h; y -= bh; return <rect key={`${b}-${k}`} x={b * bw + 0.5} y={y} width={bw - 1} height={bh} fill={labels ? PALETTE[k % PALETTE.length] : '#00D9FF'} opacity={0.85} />; });
      })}
      <line x1={0} y1={h} x2={w} y2={h} stroke="#1F2A44" />
      <text x={0} y={h + 17} fontSize={13} fill="#94A3B8" fontFamily="IBM Plex Mono, monospace">{fmt(lo, 2)}</text>
      <text x={w} y={h + 17} fontSize={13} fill="#94A3B8" textAnchor="end" fontFamily="IBM Plex Mono, monospace">{fmt(hi, 2)}</text>
    </svg>
  );
}

export default function DatasetExplorer() {
  const t = useT();
  const [src, setSrc] = useState<'builtin' | 'csv'>('builtin');
  const [bid, setBid] = useState<BuiltinId>('iris');
  const [syn, setSyn] = useState({ n: 300, noise: 0.2, seed: 3, classes: 3 });
  const [csv, setCsv] = useState<{ name: string; p: { header: string[]; rows: string[][] } } | null>(null);
  const [labelCol, setLabelCol] = useState<number | null>(null);
  const [csvErr, setCsvErr] = useState<string | null>(null);
  const [missing, setMissing] = useState<'mean' | 'drop'>('mean');
  const [scaling, setScaling] = useState<'none' | 'standardize' | 'minmax'>('standardize');
  const [mode, setMode] = useState<'pca' | 'features'>('pca');
  const [axes, setAxes] = useState([0, 1, 2]);
  const [colorBy, setColorBy] = useState(-1);
  const [loadings, setLoadings] = useState(true);
  const [split, setSplit] = useState({ show: false, ratio: 0.25, seed: 42 });
  const [sel, setSel] = useState<number | null>(null);
  const [histF, setHistF] = useState(0);
  const [cam, setCam] = useCameraState();

  const table: Table = useMemo(() => (src === 'csv' && csv ? tableFromCSV(csv.name, csv.p, labelCol) : builtin(bid, syn)), [src, csv, labelCol, bid, syn]);
  const d = table.columns.length;
  const stats = useMemo(() => columnStats(table.X), [table]);
  const missingCells = stats.reduce((s, c) => s + c.missing, 0);
  const clean = useMemo(() => handleMissing(table, missing), [table, missing]);
  useEffect(() => { setSel(null); }, [clean]);
  const X = useMemo(() => scaleColumns(clean.X, scaling), [clean, scaling]);
  const y = clean.y;
  const P = useMemo(() => (X.length > 1 && d >= 2 ? pca(X, Math.min(10, d)) : null), [X, d]);
  const ax = axes.map((a) => Math.min(a, d - 1));
  const coords = useMemo(() => {
    const raw = X.map((r, i) => (mode === 'pca' && P ? [P.projected[i][0] ?? 0, P.projected[i][1] ?? 0, P.projected[i][2] ?? 0] : [r[ax[0]] ?? 0, r[ax[1]] ?? 0, d >= 3 ? r[ax[2]] ?? 0 : 0]));
    const bounds = [0, 1, 2].map((k) => raw.length ? extent(raw.map((v) => v[k])) : [0, 0]);
    const lo = bounds.map((b) => b[0]), hi = bounds.map((b) => b[1]);
    const s = (2 * R * 0.92) / Math.max(1e-9, ...hi.map((h, k) => h - lo[k]));
    const c = lo.map((l, k) => (l + hi[k]) / 2);
    return { pts: raw.map((v) => [(v[0] - c[0]) * s, (v[1] - c[1]) * s, (v[2] - c[2]) * s] as [number, number, number]), s, c };
  }, [X, P, mode, ax, d]);
  const sp = useMemo(() => (y ? trainTestSplit(y, split.ratio, split.seed) : null), [y, split]);
  const testSet = useMemo(() => new Set(sp?.test ?? []), [sp]);

  const colors = useMemo(() => {
    if (colorBy < 0 && y) return y.map((c) => PALETTE[c % PALETTE.length]);
    const j = Math.max(0, Math.min(colorBy, d - 1));
    const v = clean.X.map((r) => r[j]);
    const [lo, hi] = extent(v);
    return v.map((x) => ramp((x - lo) / ((hi - lo) || 1)));
  }, [colorBy, y, clean, d]);

  const axisNames = mode === 'pca' && P ? P.ratio.slice(0, 3).map((r, k) => `PC${k + 1} (${pct(r)})`) : ax.map((a) => table.columns[a] ?? '');
  const loadSegs = useMemo(() => {
    if (mode !== 'pca' || !P || !loadings) return [];
    const L = table.columns.map((name, j) => ({ name, v: [P.components[0][j], P.components[1]?.[j] ?? 0, P.components[2]?.[j] ?? 0] }));
    return L.map((l) => ({ ...l, m: Math.hypot(...l.v) })).sort((a, b) => b.m - a.m).slice(0, 8);
  }, [mode, P, loadings, table.columns]);
  const origin = coords.c.map((c) => -c * coords.s) as [number, number, number];

  const ptsTrain = coords.pts.filter((_, i) => !split.show || !testSet.has(i));
  const colTrain = colors.filter((_, i) => !split.show || !testSet.has(i));
  const idxTrain = coords.pts.map((_, i) => i).filter((i) => !split.show || !testSet.has(i));
  const idxTest = coords.pts.map((_, i) => i).filter((i) => split.show && testSet.has(i));
  const scene = (
    <group>
      <FrameBox size={R} labels={[axisNames[0], axisNames[1], d >= 3 || mode === 'pca' ? axisNames[2] : '']} />
      <PointCloud points={ptsTrain} colors={colTrain} sizes={X.length > 1000 ? 0.07 : 0.12} onClick={(i) => setSel(idxTrain[i])} />
      {idxTest.length > 0 && <PointCloud points={idxTest.map((i) => coords.pts[i])} colors={idxTest.map((i) => colors[i])} sizes={0.15} geometry={BOX} glow={0.15} onClick={(i) => setSel(idxTest[i])} />}
      {sel != null && coords.pts[sel] && <><PointCloud points={[coords.pts[sel]]} colors={['#ffffff']} sizes={0.3} basic wire opacity={0.9} /><Label position={[coords.pts[sel][0], coords.pts[sel][1] + 0.5, coords.pts[sel][2]]}>#{sel}{y ? ` · ${table.classNames[y[sel]]}` : ''}</Label></>}
      {loadSegs.length > 0 && <>
        <Segments segs={loadSegs.map((l) => ({ a: origin, b: [origin[0] + l.v[0] * R * 0.9, origin[1] + l.v[1] * R * 0.9, origin[2] + l.v[2] * R * 0.9] as [number, number, number], c: '#E5E7EB', i: 0.9 }))} />
        {loadSegs.map((l) => <Label key={l.name} position={[origin[0] + l.v[0] * R * 0.98, origin[1] + l.v[1] * R * 0.98, origin[2] + l.v[2] * R * 0.98]} muted>{l.name.length > 18 ? l.name.slice(0, 17) + '…' : l.name}</Label>)}
      </>}
    </group>
  );

  const onFile = async (f: File) => {
    try {
      const p = parseCSV(await f.text());
      if (!p.header.length || p.rows.length < 2) { setCsvErr(t('ไฟล์ต้องมีแถวหัวตารางและข้อมูลอย่างน้อย 2 แถว', 'The file needs a header row and at least 2 data rows')); return; }
      const kinds = columnKinds(p);
      if (kinds.filter((k) => k === 'numeric').length < 2) { setCsvErr(t('ต้องมีคอลัมน์ตัวเลขอย่างน้อย 2 คอลัมน์', 'At least 2 numeric columns are required')); return; }
      // default label: the last text column, otherwise none
      const lastText = kinds.lastIndexOf('text');
      setCsv({ name: f.name, p }); setLabelCol(lastText >= 0 ? lastText : null); setSrc('csv'); setCsvErr(null); setSel(null); setAxes([0, 1, 2]); setColorBy(-1);
    } catch { setCsvErr(t('อ่านไฟล์ไม่ได้', 'Could not read the file')); }
  };
  const classCounts = useMemo(() => { const counts = table.classNames.map(() => 0); y?.forEach((label) => counts[label]++); return counts; }, [y, table.classNames]);
  const info = BUILTINS.find((b) => b.id === bid);

  return (
    <LabShell crumb={t('ชุดข้อมูล', 'Datasets')} title="Dataset Explorer" chips={[t('ข้อมูลจริง + สังเคราะห์', 'Real + synthetic data'), 'PCA', 'CSV']}
      intro={t('เลือกชุดข้อมูลหรืออัปโหลด CSV แล้วดูข้อมูลทุกแถวเป็นจุดใน 3 มิติ ทั้งในแกนฟีเจอร์จริงหรือในแกน PCA ที่คำนวณจากข้อมูล เส้นสีขาวคือทิศของฟีเจอร์เดิม (loadings) คลิกจุดเพื่อดูค่าในแถวนั้น', 'Pick a dataset or upload a CSV and see every row as a point in 3D, on real feature axes or on PCA axes computed from the data. White lines are the original features’ directions (loadings). Click a point to see its row.')}
      cam={cam} setCam={setCam} distance={18} scene={scene} onPointerMissed={() => setSel(null)}
      readout={{ title: table.name, rows: [
        [t('แหล่งข้อมูล', 'source'), table.source.length > 34 ? table.source.slice(0, 33) + '…' : table.source],
        [t('แถว · ฟีเจอร์', 'rows · features'), `${table.X.length} · ${d}`],
        [t('คลาส', 'classes'), y ? `${table.classNames.length}` : t('ไม่มี label', 'no label')],
        [t('ค่าที่หายไป', 'missing cells'), `${missingCells}${clean.dropped ? ` (${t('ตัดทิ้ง', 'dropped')} ${clean.dropped} ${t('แถว', 'rows')})` : ''}`, missingCells ? '#F59E0B' : undefined],
        [t('มุมมอง', 'view'), mode === 'pca' ? `PCA · ${t('อธิบาย', 'explains')} ${P ? pct(P.ratio.slice(0, 3).reduce((a, b) => a + b, 0)) : '—'}` : t('ฟีเจอร์จริง', 'raw features')],
        ...(sel != null && clean.X[sel] ? [[`#${sel}`, clean.X[sel].slice(0, 4).map((v) => fmt(v, 2)).join(', ') + (d > 4 ? ' …' : '')] as [string, string]] : []),
      ] }}
      controls={
        <>
          <Section title={t('แหล่งข้อมูล', 'Source')}>
            <Segmented value={src} onChange={(s) => { if (s === 'csv' && !csv) return; setSrc(s); setSel(null); }} options={[{ value: 'builtin', label: t('ชุดในระบบ', 'Built-in') }, { value: 'csv', label: csv ? `CSV · ${csv.name.slice(0, 12)}` : t('CSV (ยังไม่อัปโหลด)', 'CSV (none yet)') }]} />
            {src === 'builtin' && <Select<BuiltinId> label={t('ชุดข้อมูล', 'Dataset')} value={bid} onChange={(v) => { setBid(v); setAxes([0, 1, 2]); setSel(null); setColorBy(-1); setHistF(0); }} options={BUILTINS.map((b) => ({ value: b.id, label: `${b.name}${b.real ? '' : t(' (สังเคราะห์)', ' (synthetic)')}` }))} />}
            {src === 'builtin' && info && !info.real && <>
              <Slider label={t('จำนวนตัวอย่าง', 'Samples')} value={syn.n} min={50} max={1000} step={50} onChange={(n) => setSyn({ ...syn, n })} />
              <Slider label="Noise" value={syn.noise} min={0} max={1} step={0.05} format={(v) => v.toFixed(2)} onChange={(noise) => setSyn({ ...syn, noise })} />
              {(bid === 'blobs3d' || bid === 'spiral') && <Slider label={t('จำนวนคลาส', 'Classes')} value={syn.classes} min={2} max={6} onChange={(classes) => setSyn({ ...syn, classes })} />}
              <Slider label="Seed" value={syn.seed} min={1} max={99} onChange={(seed) => setSyn({ ...syn, seed })} />
            </>}
            <label className="text-[0.96875rem] text-muted">{t('อัปโหลด CSV (แถวแรกเป็นหัวตาราง)', 'Upload CSV (first row = header)')}
              <input type="file" accept=".csv,text/csv,.txt" className="mt-1 block w-full text-[0.9375rem]" onChange={(e) => { const f = e.target.files?.[0]; if (f) onFile(f); e.target.value = ''; }} />
            </label>
            {csvErr && <p className="text-[0.9375rem] text-red">{csvErr}</p>}
            {src === 'csv' && csv && <Select<number> label={t('คอลัมน์ label (คลาส)', 'Label (class) column')} value={labelCol ?? -1} onChange={(v) => setLabelCol(v < 0 ? null : v)} options={[{ value: -1, label: t('— ไม่มี —', '— none —') }, ...csv.p.header.map((h, i) => ({ value: i, label: h }))]} />}
            <Button size="sm" onClick={() => download('iris-sample.csv', ['sepal_length,sepal_width,petal_length,petal_width,species', ...IRIS_ROWS.map((r) => [...r.slice(0, 4), ['setosa', 'versicolor', 'virginica'][r[4]]].join(','))].join('\n'), 'text/csv')}><Icon name="download" />{t('ตัวอย่าง CSV', 'Sample CSV')}</Button>
          </Section>
          <Section title={t('เตรียมข้อมูล', 'Preprocessing')}>
            <Segmented label={t('ค่าที่หายไป', 'Missing values')} value={missing} onChange={setMissing} options={[{ value: 'mean', label: t('แทนด้วยค่าเฉลี่ย', 'Mean impute') }, { value: 'drop', label: t('ตัดแถวทิ้ง', 'Drop rows') }]} />
            <Select<'none' | 'standardize' | 'minmax'> label={t('การปรับสเกล', 'Scaling')} value={scaling} onChange={setScaling} options={[{ value: 'standardize', label: 'Standardise (z-score)' }, { value: 'minmax', label: 'Min–max (0–1)' }, { value: 'none', label: t('ไม่ปรับ', 'None') }]} />
          </Section>
          <Section title={t('การแสดงผล 3 มิติ', '3D view')}>
            <Segmented value={mode} onChange={setMode} options={[{ value: 'pca', label: t('แกน PCA', 'PCA axes') }, { value: 'features', label: t('เลือกฟีเจอร์', 'Pick features') }]} />
            {mode === 'features' && (
              <div className="grid grid-cols-1 gap-2">
                {(['x', 'y', 'z'] as const).slice(0, d >= 3 ? 3 : 2).map((n, k) => (
                  <Select<number> key={n} label={n} value={ax[k]} onChange={(v) => { const a = [...axes]; a[k] = v; setAxes(a); }} options={table.columns.map((c, j) => ({ value: j, label: c }))} />
                ))}
              </div>
            )}
            {mode === 'pca' && <Toggle label={t('แสดง loadings (8 ฟีเจอร์ที่ยาวที่สุด)', 'Show loadings (8 longest)')} checked={loadings} onChange={setLoadings} />}
            <Select<number> label={t('ระบายสีตาม', 'Colour by')} value={colorBy < 0 && y ? -1 : Math.max(0, Math.min(colorBy, d - 1))} onChange={setColorBy} options={[...(y ? [{ value: -1, label: t('คลาส', 'class') }] : []), ...table.columns.map((c, j) => ({ value: j, label: c }))]} />
            {y && <Toggle label={t('แสดงการแบ่ง train/test (ลูกบาศก์ = test)', 'Show train/test split (cubes = test)')} checked={split.show} onChange={(show) => setSplit({ ...split, show })} />}
            {y && split.show && <>
              <Slider label={t('สัดส่วน test', 'Test ratio')} value={split.ratio} min={0.1} max={0.5} step={0.05} format={pct} onChange={(ratio) => setSplit({ ...split, ratio })} />
              <Slider label="Split seed" value={split.seed} min={1} max={99} onChange={(seed) => setSplit({ ...split, seed })} />
            </>}
          </Section>
        </>
      }
      tabs={[
        { id: 'summary', label: t('สรุปสถิติ', 'Summary'), content: (
          <div className="flex flex-col gap-6">
            <Metrics items={[[t('แถว', 'Rows'), String(table.X.length)], [t('ฟีเจอร์', 'Features'), String(d)], [t('คลาส', 'Classes'), y ? String(table.classNames.length) : '—'], [t('ค่าที่หายไป', 'Missing'), String(missingCells)]]} />
            <div className="scroll-thin max-h-[440px] overflow-auto">
              <table className="num w-full min-w-[620px] text-left font-mono text-[0.96875rem]">
                <thead className="sticky top-0 bg-bg text-muted"><tr className="border-b border-line"><th className="py-1.5">{t('ฟีเจอร์', 'feature')}</th><th>min</th><th>max</th><th>mean</th><th>std</th><th>{t('หายไป', 'missing')}</th><th>{t('การกระจาย', 'distribution')}</th></tr></thead>
                <tbody>{stats.map((s, j) => (
                  <tr key={j} className="border-b border-line/50">
                    <td className="py-1 pr-3">{table.columns[j]}</td><td>{fmt(s.min, 3)}</td><td>{fmt(s.max, 3)}</td><td>{fmt(s.mean, 3)}</td><td>{fmt(s.std, 3)}</td>
                    <td className={s.missing ? 'text-orange' : ''}>{s.missing}</td>
                    <td><Histogram values={clean.X.map((r) => r[j])} labels={null} classNames={[]} bins={12} w={110} h={24} /></td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          </div>
        ) },
        { id: 'dist', label: t('การกระจายของฟีเจอร์', 'Feature distributions'), content: (
          <div className="flex flex-col gap-5">
            <div className="max-w-xs"><Select<number> label={t('ฟีเจอร์ (ภาพใหญ่)', 'Feature (large view)')} value={Math.min(histF, d - 1)} onChange={setHistF} options={table.columns.map((c, j) => ({ value: j, label: c }))} /></div>
            <Histogram values={clean.X.map((r) => r[Math.min(histF, d - 1)])} labels={y} classNames={table.classNames} bins={30} w={640} h={180} />
            {y && <div className="flex flex-wrap gap-3 text-[1rem]">{table.classNames.map((c, k) => <span key={c}><span style={{ color: PALETTE[k % PALETTE.length] }}>■</span> {c}</span>)}</div>}
            <div className="grid grid-cols-2 gap-4 sm:grid-cols-3 lg:grid-cols-5">
              {table.columns.slice(0, 15).map((c, j) => (
                <button key={c} type="button" onClick={() => setHistF(j)} className={`flex flex-col gap-1 rounded-lg border p-2 text-left ${j === histF ? 'border-cyan' : 'border-line'}`}>
                  <span className="truncate font-mono text-[0.875rem] text-muted">{c}</span>
                  <Histogram values={clean.X.map((r) => r[j])} labels={y} classNames={table.classNames} bins={14} w={170} h={50} />
                </button>
              ))}
            </div>
            {d > 15 && <p className="text-[1rem] text-muted">{t(`แสดง 15 จาก ${d} ฟีเจอร์ เลือกฟีเจอร์อื่นจากเมนูด้านบน`, `Showing 15 of ${d} features; pick others from the menu above.`)}</p>}
          </div>
        ) },
        { id: 'classes', label: t('คลาสและการแบ่งข้อมูล', 'Classes & split'), content: y ? (
          <div className="grid gap-8 lg:grid-cols-2">
            <div className="flex flex-col gap-2">
              <h3 className="label-xs">{t('การกระจายของคลาส', 'Class distribution')}</h3>
              {table.classNames.map((c, k) => (
                <div key={c} className="flex items-center gap-3 font-mono text-[1rem]">
                  <span className="w-28 truncate" style={{ color: PALETTE[k % PALETTE.length] }}>{c}</span>
                  <span className="h-2.5 flex-1 rounded bg-line"><span className="block h-full rounded" style={{ width: pct(classCounts[k] / Math.max(...classCounts)), background: PALETTE[k % PALETTE.length] }} /></span>
                  <span className="w-24 text-right">{classCounts[k]} ({pct(classCounts[k] / y.length)})</span>
                </div>
              ))}
              {Math.max(...classCounts) / Math.max(1, Math.min(...classCounts)) > 1.5 && <Callout tone="warn">{t('คลาสไม่สมดุล: accuracy อาจทำให้เข้าใจผิด ควรดู precision/recall/F1 และใช้ stratified split', 'Imbalanced classes: accuracy can mislead; check precision/recall/F1 and use a stratified split.')}</Callout>}
            </div>
            <div className="flex flex-col gap-2">
              <h3 className="label-xs">{t(`Stratified split · test ${pct(split.ratio)} · seed ${split.seed}`, `Stratified split · test ${pct(split.ratio)} · seed ${split.seed}`)}</h3>
              <table className="num w-full text-left font-mono text-[1rem]">
                <thead className="text-muted"><tr className="border-b border-line"><th className="py-1">{t('คลาส', 'class')}</th><th>train</th><th>test</th></tr></thead>
                <tbody>{table.classNames.map((c, k) => (
                  <tr key={c} className="border-b border-line/50"><td className="py-1">{c}</td><td>{sp!.train.filter((i) => y[i] === k).length}</td><td>{sp!.test.filter((i) => y[i] === k).length}</td></tr>
                ))}</tbody>
              </table>
              <p className="text-[1rem] text-muted">{t('การแบ่งแบบ stratified รักษาสัดส่วนของแต่ละคลาสให้เท่ากันทั้งใน train และ test ทุกห้องทดลองในเว็บนี้ใช้วิธีนี้', 'A stratified split keeps each class’s proportion the same in train and test; every lab on this site uses it.')}</p>
            </div>
          </div>
        ) : <p className="text-muted">{t('ชุดข้อมูลนี้ไม่มีคอลัมน์ label เลือกคอลัมน์ label ในแผงควบคุม', 'This dataset has no label column; choose one in the controls.')}</p> },
        { id: 'pca', label: 'PCA', content: P ? (
          <div className="grid gap-8 lg:grid-cols-2">
            <div className="flex flex-col gap-2">
              <h3 className="label-xs">{t('สัดส่วน variance ที่อธิบายได้', 'Explained variance ratio')}</h3>
              {P.ratio.map((r, k) => {
                const cum = P.ratio.slice(0, k + 1).reduce((a, b) => a + b, 0);
                return (
                  <div key={k} className="flex items-center gap-3 font-mono text-[1rem]">
                    <span className="w-10">PC{k + 1}</span>
                    <span className="h-2.5 flex-1 rounded bg-line"><span className="block h-full rounded bg-cyan" style={{ width: pct(r) }} /></span>
                    <span className="w-36 text-right">{pct(r)} · {t('สะสม', 'cum.')} {pct(cum)}</span>
                  </div>
                );
              })}
              <p className="mt-2 text-[1rem] text-muted">{t('ผลขึ้นกับการปรับสเกล ลองเปลี่ยนเป็น "ไม่ปรับ" กับชุด Wine แล้วดูว่า PC1 ถูกครอบงำโดยฟีเจอร์ proline ที่มีหน่วยใหญ่', 'Results depend on scaling: switch to “None” on Wine and watch PC1 get dominated by the large-unit proline feature.')}</p>
            </div>
            <div className="scroll-thin max-h-[420px] overflow-auto">
              <table className="num w-full text-left font-mono text-[0.96875rem]">
                <thead className="sticky top-0 bg-bg text-muted"><tr className="border-b border-line"><th className="py-1">loadings</th><th>PC1</th><th>PC2</th><th>PC3</th></tr></thead>
                <tbody>{table.columns.map((c, j) => (
                  <tr key={c} className="border-b border-line/50"><td className="py-1 pr-2">{c}</td>{[0, 1, 2].map((k) => { const v = P.components[k]?.[j]; return <td key={k} style={{ color: v == null ? undefined : v >= 0 ? '#3B82F6' : '#EF4444' }}>{v == null ? '—' : fmt(v, 3)}</td>; })}</tr>
                ))}</tbody>
              </table>
            </div>
          </div>
        ) : <p className="text-muted">{t('PCA ต้องมีอย่างน้อย 2 ฟีเจอร์', 'PCA needs at least 2 features')}</p> },
        { id: 'table', label: t('ตารางข้อมูล', 'Data table'), content: (
          <div className="scroll-thin max-h-[480px] overflow-auto">
            <table className="num w-full text-left font-mono text-[0.9375rem]">
              <thead className="sticky top-0 bg-bg text-muted"><tr className="border-b border-line"><th className="py-1 pr-2">#</th>{table.columns.slice(0, 16).map((c) => <th key={c} className="whitespace-nowrap pr-3">{c}</th>)}{y && <th>label</th>}</tr></thead>
              <tbody>{table.X.slice(0, 100).map((r, i) => (
                <tr key={i} className={`border-b border-line/40 ${sel === i ? 'bg-cyan/10' : ''}`} onClick={() => setSel(i)}>
                  <td className="py-0.5 pr-2 text-muted">{i}</td>
                  {r.slice(0, 16).map((v, j) => <td key={j} className={`pr-3 ${v == null ? 'text-orange' : ''}`}>{v == null ? 'NA' : fmt(v, 3)}</td>)}
                  {y && table.y && <td style={{ color: PALETTE[table.y[i] % PALETTE.length] }}>{table.classNames[table.y[i]]}</td>}
                </tr>
              ))}</tbody>
            </table>
            <p className="mt-2 text-[0.96875rem] text-muted">{t(`แสดง 100 แถวแรก${d > 16 ? ' และ 16 คอลัมน์แรก' : ''} ค่าดิบก่อนเติมค่าหายและปรับสเกล`, `First 100 rows${d > 16 ? ' and 16 columns' : ''}; raw values before imputation and scaling.`)}</p>
          </div>
        ) },
        { id: 'ex', label: t('แบบฝึกหัด', 'Exercises'), content: (
          <Quiz id="datasets" items={[
            { q: t('ทำไมต้องปรับสเกลก่อนทำ PCA', 'Why scale before PCA?'), options: [t('PCA หาทิศที่ variance มากสุด ฟีเจอร์หน่วยใหญ่จะชนะเสมอ', 'PCA finds max-variance directions, so large-unit features always win'), t('ทำให้คำนวณเร็วขึ้น', 'It is faster')], answer: 0, why: t('ลองชุด Wine กับ "ไม่ปรับ"', 'Try Wine with “None”.') },
            { q: t('ใน Iris, PC1 อธิบาย variance ได้ราว 92% (ไม่ปรับสเกล) หมายความว่า', 'In Iris, PC1 explains ~92% of variance (unscaled). This means'), options: [t('ข้อมูลเกือบทั้งหมดเรียงตัวตามแนวเดียว', 'the data lies almost along one direction'), t('มีเพียงฟีเจอร์เดียวที่ใช้ได้', 'only one feature is useful')], answer: 0, why: t('PC เป็นการผสมของทุกฟีเจอร์ ดู loadings', 'A PC mixes all features; see the loadings.') },
            { q: t('เมื่อไรไม่ควรแทนค่าที่หายไปด้วยค่าเฉลี่ย', 'When is mean imputation a poor choice?'), options: [t('เมื่อค่าหายไปมากหรือไม่สุ่ม', 'When many values are missing or missingness is not random'), t('ไม่มีกรณีไหน', 'Never')], answer: 0, why: t('ค่าเฉลี่ยลด variance และอาจสร้างอคติ', 'Means shrink variance and can introduce bias.') },
          ]} />
        ) },
      ]}
    />
  );
}
