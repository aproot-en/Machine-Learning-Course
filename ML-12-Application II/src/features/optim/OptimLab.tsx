import { useMemo, useState } from 'react';
import { useCameraState } from '../../engine3d/Stage';
import { LabShell, Metrics } from '../../components/LabShell';
import { flowVector } from '../../engine3d/experiment';
import { HeightSurface, PointCloud, Polyline, Segments, contour, rgb } from '../../engine3d/plots';
import { FrameBox } from '../../engine3d/primitives';
import { GRAD } from '../../engine3d/palette';
import { SURFACES, customSurface, runOptimizer, type OptName, type OptParams, type Surface } from '../../ml/optim';
import { useSettings, useT } from '../../i18n/i18n';
import { useInterval } from '../../lib/useLoop';
import { Section, Select, Slider, Toggle, fmt } from '../../components/ui';
import { LinePlot, Quiz, Tex } from '../../components/common';

const R = 5, G = 64;
const COLORS = ['#00D9FF', '#F59E0B'];
const NAMES: Record<OptName, string> = { gd: 'Batch GD', sgd: 'SGD (noisy gradient)', momentum: 'Momentum', rmsprop: 'RMSprop', adam: 'Adam' };

export default function OptimLab() {
  const t = useT();
  const level = useSettings((s) => s.level);
  const [sid, setSid] = useState('rosenbrock');
  const [expr, setExpr] = useState('0.2*x^2 + sin(x)*cos(y) + 0.1*y^2');
  const [iters, setIters] = useState(300);
  const [compare, setCompare] = useState(true);
  const [logH, setLogH] = useState(true);
  const [opts, setOpts] = useState<OptParams[]>([
    { name: 'gd', lr: 0.01, momentum: 0.9, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, noise: 2, seed: 1 },
    { name: 'adam', lr: 0.05, momentum: 0.9, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, noise: 2, seed: 1 },
  ]);
  const [start, setStart] = useState<[number, number] | null>(null);
  const [k, setK] = useState(0);
  const [running, setRunning] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [cam, setCam] = useCameraState();

  const parsed = useMemo((): { s: Surface; err: string | null } => {
    if (sid !== 'custom') return { s: SURFACES.find((s) => s.id === sid)!, err: null };
    try { const s = customSurface(expr, [-5, 5, -5, 5]); s.f(0.3, 0.7); return { s, err: null }; }
    catch (e) { return { s: SURFACES[0], err: (e as Error).message }; }
  }, [sid, expr]);
  const S = parsed.s;
  const st = start ?? S.start;
  const [x0, x1, y0, y1] = S.range;
  const toX = (x: number) => ((x - x0) / (x1 - x0) - 0.5) * 2 * R;
  const toZ = (y: number) => ((y - y0) / (y1 - y0) - 0.5) * 2 * R;

  const grid = useMemo(() => {
    const raw = new Float32Array((G + 1) ** 2);
    for (let r = 0; r <= G; r++) for (let c = 0; c <= G; c++) raw[r * (G + 1) + c] = S.f(x0 + (c / G) * (x1 - x0), y0 + (r / G) * (y1 - y0));
    let lo = Infinity, hi = -Infinity;
    raw.forEach((v) => { if (Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); } });
    const H = (v: number) => { const u = logH ? Math.log1p(Math.max(0, v - lo)) / Math.log1p(hi - lo || 1) : (v - lo) / (hi - lo || 1); return Math.min(1.05, Math.max(-0.05, u)) * 4.5 - 2.5; };
    const heights = raw.map((v) => H(v));
    const lowC = rgb('#1e3a8a'), midC = rgb('#00D9FF'), hiC = rgb('#F59E0B');
    const cols = new Float32Array(raw.length * 3);
    heights.forEach((h, i) => { const u = (h + 2.5) / 4.5; const [a, b, w] = u < 0.5 ? [lowC, midC, u * 2] : [midC, hiC, (u - 0.5) * 2]; for (let j = 0; j < 3; j++) cols[i * 3 + j] = (a[j] + (b[j] - a[j]) * w) * 0.75; });
    const levels = Array.from({ length: 9 }, (_, i) => -2.5 + ((i + 0.5) / 9) * 4.5);
    const iso = levels.flatMap((lv) => contour(G, heights, lv).map(([c0, r0, c1, r1]) => ({ a: [(c0 / G - 0.5) * 2 * R, -2.99, (r0 / G - 0.5) * 2 * R] as [number, number, number], b: [(c1 / G - 0.5) * 2 * R, -2.99, (r1 / G - 0.5) * 2 * R] as [number, number, number], c: '#3b5687', i: 0.7 })));
    return { heights, cols, H, iso };
  }, [S, x0, x1, y0, y1, logH]);

  const runs = useMemo(() => (compare ? opts : opts.slice(0, 1)).map((o) => runOptimizer(S, o, st, iters)), [S, opts, compare, st, iters]);
  const maxLen = Math.max(1, ...runs.map((r) => r.path.length));
  const idx = Math.min(k, maxLen - 1);
  useInterval(running, 40, speed, () => { if (idx >= maxLen - 1) return false; setK((q) => Math.min(maxLen - 1, q + 1)); return true; }, () => setRunning(false));

  const P = (x: number, y: number, lift = 0.06): [number, number, number] => [toX(x), grid.H(S.f(x, y)) + lift, toZ(y)];
  const scene = (
    <group>
      <FrameBox size={R} labels={['x', 'f', 'y']} />
      <HeightSurface G={G} heights={grid.heights} colors={grid.cols} size={2 * R} opacity={0.82} wire={0.05} />
      <Segments segs={grid.iso} additive={false} opacity={0.8} />
      {runs.map((r, i) => {
        const pts = r.path.slice(0, idx + 1);
        const cur = pts[pts.length - 1];
        return (
          <group key={i}>
            <Polyline points={pts.map((p) => P(p.x, p.y))} color="#ffffff" colorEnd={COLORS[i]} />
            <Polyline points={pts.map((p) => [toX(p.x), -2.97, toZ(p.y)] as [number, number, number])} color={COLORS[i]} opacity={0.6} />
            {cur && (
              <>
                <PointCloud points={[P(cur.x, cur.y, 0.15)]} colors={[COLORS[i]]} sizes={0.2} glow={0.9} />
                <Segments segs={[{ a: P(cur.x, cur.y, 0.15), b: (() => { const n = Math.hypot(cur.gx, cur.gy) || 1; const l = Math.min(1.2, n) / n; return P(cur.x - cur.gx * l * (x1 - x0) / (2 * R), cur.y - cur.gy * l * (y1 - y0) / (2 * R), 0.15); })(), c: GRAD }]} />
              </>
            )}
          </group>
        );
      })}
      {S.minimum && <PointCloud points={[P(S.minimum[0], S.minimum[1], 0.05)]} colors={['#22C55E']} sizes={0.14} basic wire />}
      <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -2.98, 0]} onClick={(e) => { e.stopPropagation(); setStart([x0 + ((e.point.x / (2 * R)) + 0.5) * (x1 - x0), y0 + ((e.point.z / (2 * R)) + 0.5) * (y1 - y0)]); setK(0); }}>
        <planeGeometry args={[2 * R, 2 * R]} /><meshBasicMaterial transparent opacity={0} depthWrite={false} />
      </mesh>
    </group>
  );

  const setO = (i: number, patch: Partial<OptParams>) => { setOpts(opts.map((o, j) => (j === i ? { ...o, ...patch } : o))); setK(0); };
  const optControls = (i: number) => {
    const o = opts[i];
    return (
      <Section key={i} title={`${i === 0 ? '●' : '◆'} ${t('ตัวที่', 'Optimizer')} ${i + 1} (${i === 0 ? t('ฟ้า', 'cyan') : t('ส้ม', 'orange')})`}>
        <Select<OptName> label={t('ชนิด', 'Type')} value={o.name} onChange={(name) => setO(i, { name })} options={(Object.keys(NAMES) as OptName[]).map((n) => ({ value: n, label: NAMES[n] }))} />
        <Slider label="η" value={Math.log10(o.lr)} min={-4} max={0.5} step={0.05} format={(v) => (10 ** v).toPrecision(2)} onChange={(v) => setO(i, { lr: Number((10 ** v).toPrecision(3)) })} />
        {o.name === 'momentum' && <Slider label="μ" value={o.momentum} min={0} max={0.99} step={0.01} format={(v) => v.toFixed(2)} onChange={(momentum) => setO(i, { momentum })} />}
        {o.name === 'adam' && level >= 2 && <Slider label="β₁" value={o.beta1} min={0} max={0.999} step={0.001} format={(v) => v.toFixed(3)} onChange={(beta1) => setO(i, { beta1 })} />}
        {(o.name === 'adam' || o.name === 'rmsprop') && level >= 2 && <Slider label="β₂" value={o.beta2} min={0.8} max={0.9999} step={0.0001} format={(v) => v.toFixed(4)} onChange={(beta2) => setO(i, { beta2 })} />}
        {(o.name === 'adam' || o.name === 'rmsprop') && level >= 3 && <Slider label="ε (log₁₀)" value={Math.log10(o.epsilon)} min={-10} max={-2} step={1} format={(v) => `1e${v}`} onChange={(v) => setO(i, { epsilon: 10 ** v })} />}
        {o.name === 'sgd' && <Slider label={t('ความแรงของ noise', 'Gradient noise σ')} value={o.noise} min={0} max={10} step={0.25} format={(v) => v.toFixed(2)} onChange={(noise) => setO(i, { noise })} />}
      </Section>
    );
  };
  const cur = runs.map((r) => r.path[Math.min(idx, r.path.length - 1)]).filter(Boolean);
  const status = (i: number) => { const r = runs[i]; const c = cur[i]; if (r.diverged && idx >= r.path.length - 1) return t('ลู่ออก', 'diverged'); return c && Math.hypot(c.gx, c.gy) < 1e-3 ? t('ลู่เข้า', 'converged') : t('กำลังเดิน', 'moving'); };

  return (
    <LabShell crumb="Optimization" title={t('Gradient Descent และ Optimizer', 'Gradient Descent & Optimizers')} chips={[t('การหาค่าเหมาะสม', 'Optimization'), t('ปานกลาง', 'Intermediate')]}
      experiment={{ model: 'Gradient Descent / Optimizer', source: `${t('ฟังก์ชันเป้าหมาย', 'Objective function')}: ${S.name}`,
        stages: [
          { name: 'Initial θ₀', values: st, labels: ['x₀', 'y₀'] },
          { name: `Current θ · step ${idx}`, values: cur.flatMap((c) => [c.x, c.y]) },
          { name: 'Gradient ∇f', values: cur.flatMap((c) => [c.gx, c.gy]) },
          { name: 'Optimizer update Δθ', values: cur.flatMap((c) => [c.ux, c.uy]) },
          { name: 'Objective f(θ)', values: cur.map((c) => c.f), labels: opts.slice(0, runs.length).map((o) => NAMES[o.name]) },
        ],
        metrics: [{ label: 'Iteration', value: String(idx) }, ...cur.map((c, i) => ({ label: `${NAMES[opts[i].name]} · f`, value: fmt(c.f, 5) })), { label: '‖∇f‖', value: cur[0] ? fmt(Math.hypot(cur[0].gx, cur[0].gy), 5) : '—' }],
        history: runs.map((run, i) => ({ label: `${NAMES[opts[i].name]} · objective`, values: run.path.slice(0, idx + 1).map((point) => point.f), color: COLORS[i] })),
        output: cur.map((c, i) => `${NAMES[opts[i].name]}: ${flowVector([c.x, c.y])} · f=${fmt(c.f, 4)} · ${status(i)}`).join('\n') || t('ไม่มีขั้นที่คำนวณได้ ตรวจฟังก์ชัน จุดเริ่ม และ gradient', 'No valid optimization step; check the function, starting point and gradient'),
        note: t('แล็บนี้รับฟังก์ชันและพารามิเตอร์เริ่มต้น แทนชุดข้อมูลฝึก', 'This lab takes an objective function and starting parameters as input'),
      }}
      intro={t('กด ▶ เพื่อปล่อย optimizer สองตัวจากจุดเดียวกัน ลูกศรสีม่วงคือทิศ −∇f คลิกที่พื้นเพื่อเลือกจุดเริ่มใหม่', 'Press ▶ to release two optimizers from the same start. The purple arrow is −∇f. Click the floor to pick a new start.')}
      cam={cam} setCam={setCam} distance={17} scene={scene}
      readout={{ title: S.name, rows: [
        [t('รอบที่', 'iteration'), `${idx} / ${iters}`],
        ...cur.flatMap((c, i) => c ? [
          [`${NAMES[opts[i].name]}`, status(i), COLORS[i]] as [string, string, string],
          ['  (x, y)', `(${fmt(c.x, 3)}, ${fmt(c.y, 3)})`] as [string, string],
          ['  f · ‖∇f‖', `${fmt(c.f, 4)} · ${fmt(Math.hypot(c.gx, c.gy), 3)}`] as [string, string],
        ] : []),
      ] }}
      playback={{ playing: running, onToggle: () => { if (idx >= maxLen - 1) setK(0); setRunning(!running); }, onNext: () => setK(Math.min(maxLen - 1, idx + 1)), onPrev: () => { setRunning(false); setK(Math.max(0, idx - 1)); },
        canPrev: idx > 0, canNext: idx < maxLen - 1, onReset: () => { setRunning(false); setK(0); }, speed, onSpeed: setSpeed, status: `iter ${idx}` }}
      controls={
        <>
          <Section title={t('ฟังก์ชัน', 'Function')}>
            <Select label={t('พื้นผิว loss', 'Loss surface')} value={sid} onChange={(v) => { setSid(v); setStart(null); setK(0); }} options={[...SURFACES.map((s) => ({ value: s.id, label: s.name })), { value: 'custom', label: t('สมการเอง', 'Custom expression') }]} />
            {sid === 'custom' && (
              <label className="flex flex-col gap-1 text-[1rem]">
                <span>f(x, y) =</span>
                <input value={expr} onChange={(e) => { setExpr(e.target.value); setK(0); }} spellCheck={false} className="rounded-md border border-line bg-bg px-2 py-1.5 font-mono text-[1rem]" />
                <span className={`text-[0.9375rem] ${parsed.err ? 'text-red' : 'text-muted'}`}>{parsed.err ?? t('ใช้ได้: + − * / ^ ( ) sin cos tan exp log sqrt abs tanh pi · gradient คำนวณเชิงตัวเลข', 'Allowed: + − * / ^ ( ) sin cos tan exp log sqrt abs tanh pi · gradient computed numerically')}</span>
              </label>
            )}
            <Slider label={t('จำนวนรอบ', 'Iterations')} value={iters} min={20} max={2000} step={20} onChange={(v) => { setIters(v); setK(0); }} />
            <Slider label="start x" value={st[0]} min={x0} max={x1} step={0.05} format={(v) => v.toFixed(2)} onChange={(x) => { setStart([x, st[1]]); setK(0); }} />
            <Slider label="start y" value={st[1]} min={y0} max={y1} step={0.05} format={(v) => v.toFixed(2)} onChange={(y) => { setStart([st[0], y]); setK(0); }} />
            <Toggle label={t('ความสูงแบบ log', 'Log height scale')} checked={logH} onChange={setLogH} />
            <Toggle label={t('เปรียบเทียบ 2 ตัว', 'Compare two optimizers')} checked={compare} onChange={setCompare} />
          </Section>
          {optControls(0)}
          {compare && optControls(1)}
        </>
      }
      tabs={[
        { id: 'concept', label: t('แนวคิด', 'Concept'), content: (
          <div className="grid max-w-5xl gap-8 text-[1.125rem] leading-7 lg:grid-cols-2">
            <p>{t('Gradient ชี้ไปทางที่ f เพิ่มเร็วที่สุด การเดินสวนทาง gradient ทีละก้าวขนาด η จึงลด f ลง พื้นผิวที่เป็นหุบแคบยาว (Rosenbrock) ทำให้ GD ธรรมดาเดินซิกแซกช้า Momentum สะสมความเร็วตามทิศที่สม่ำเสมอ ส่วน RMSprop/Adam ปรับขนาดก้าวแยกตามแต่ละแกน', 'The gradient points where f rises fastest, so stepping against it by η lowers f. Long narrow valleys (Rosenbrock) make plain GD zig-zag slowly. Momentum builds speed along consistent directions; RMSprop/Adam rescale the step per axis.')}</p>
            <ul className="list-disc space-y-1.5 pl-5">
              <li>{t('Oscillation: η ใหญ่ ทำให้เด้งไปมาข้ามหุบ', 'Oscillation: large η bounces across a valley.')}</li>
              <li>{t('Overshooting / divergence: η ใหญ่มาก ค่าพุ่งออกนอกพื้นผิว', 'Overshooting / divergence: very large η shoots off the surface.')}</li>
              <li>{t('Saddle: gradient ใกล้ 0 ที่จุดอานม้า GD อาจค้าง ส่วน SGD ที่มี noise หลุดออกได้', 'Saddle: the gradient is near 0 at a saddle point; GD may stall while noisy SGD escapes.')}</li>
              <li>{t('Himmelblau มีจุดต่ำสุด 4 จุด จุดเริ่มกำหนดว่าจะไปจบที่ไหน', 'Himmelblau has 4 minima; the start decides which one you reach.')}</li>
            </ul>
          </div>
        ) },
        { id: 'math', label: t('สมการอัปเดต', 'Update equations'), content: (
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="flex flex-col gap-3">
              <Tex block>{S.tex}</Tex>
              <Tex block>{'\\text{GD: } \\boldsymbol\\theta \\leftarrow \\boldsymbol\\theta - \\eta\\nabla f \\qquad \\text{SGD: } \\nabla f + \\boldsymbol\\epsilon,\\ \\boldsymbol\\epsilon\\sim\\mathcal N(0,\\sigma^2)'}</Tex>
              <Tex block>{'\\text{Momentum: } \\mathbf v \\leftarrow \\mu\\mathbf v - \\eta\\nabla f,\\ \\boldsymbol\\theta \\leftarrow \\boldsymbol\\theta + \\mathbf v'}</Tex>
              <Tex block>{'\\text{RMSprop: } \\mathbf s \\leftarrow \\beta_2\\mathbf s + (1-\\beta_2)(\\nabla f)^2,\\ \\boldsymbol\\theta \\leftarrow \\boldsymbol\\theta - \\frac{\\eta\\nabla f}{\\sqrt{\\mathbf s}+\\epsilon}'}</Tex>
              <Tex block>{'\\text{Adam: } \\hat{\\mathbf m} = \\frac{\\mathbf m}{1-\\beta_1^t},\\ \\hat{\\mathbf v} = \\frac{\\mathbf v}{1-\\beta_2^t},\\ \\boldsymbol\\theta \\leftarrow \\boldsymbol\\theta - \\eta\\frac{\\hat{\\mathbf m}}{\\sqrt{\\hat{\\mathbf v}}+\\epsilon}'}</Tex>
            </div>
            <div className="flex flex-col gap-3">
              <h3 className="label-xs">{t('ก้าวปัจจุบัน', 'Current step')}</h3>
              {cur.map((c, i) => c && (
                <div key={i}>
                  <div className="font-mono text-[1rem]" style={{ color: COLORS[i] }}>{NAMES[opts[i].name]}</div>
                  <Tex block>{`\\nabla f(${fmt(c.x, 2)}, ${fmt(c.y, 2)}) = (${fmt(c.gx, 3)},\\ ${fmt(c.gy, 3)}),\\quad \\Delta\\boldsymbol\\theta = (${fmt(c.ux, 4)},\\ ${fmt(c.uy, 4)})`}</Tex>
                </div>
              ))}
            </div>
          </div>
        ) },
        { id: 'eval', label: t('เปรียบเทียบ', 'Comparison'), content: (
          <div className="grid gap-8 lg:grid-cols-[1.4fr_1fr]">
            <div><h3 className="label-xs mb-2">{t('f ต่อรอบ', 'f per iteration')}</h3>
              <LinePlot xTitle="iteration" yTitle="f(x, y)" markerX={idx} series={runs.map((r, i) => ({ name: NAMES[opts[i].name], x: r.path.map((_, j) => j), y: r.path.map((p) => p.f), color: COLORS[i] }))} /></div>
            <div className="flex flex-col gap-4">
              {runs.map((r, i) => {
                const firstClose = r.path.findIndex((p) => Math.hypot(p.gx, p.gy) < 1e-3);
                const last = r.path[r.path.length - 1];
                return <div key={i}><div className="mb-1 font-mono text-[1rem]" style={{ color: COLORS[i] }}>{NAMES[opts[i].name]} · η={opts[i].lr}</div>
                  <Metrics items={[[t('f สุดท้าย', 'final f'), r.diverged ? '∞' : fmt(last?.f ?? NaN, 4)], [t('รอบที่ ‖∇‖<1e-3', 'iters to ‖∇‖<1e-3'), firstClose >= 0 ? String(firstClose) : '—']]} /></div>;
              })}
            </div>
          </div>
        ) },
        { id: 'ex', label: t('แบบฝึกหัด', 'Exercises'), content: (
          <Quiz id="optim" items={[
            { q: t('บน Rosenbrock ทำไม GD ธรรมดาช้า', 'Why is plain GD slow on Rosenbrock?'), options: [t('หุบแคบโค้ง gradient ชี้ข้ามหุบมากกว่าตามหุบ', 'The narrow curved valley makes the gradient point across it, not along it'), t('ฟังก์ชันไม่มีจุดต่ำสุด', 'It has no minimum')], answer: 0, why: t('ดูเส้นทางซิกแซกบนพื้น', 'Look at the zig-zag path on the floor.') },
            { q: t('η = 1.5 บน convex quadratic ให้ผลอย่างไร', 'What does η = 1.5 do on the convex quadratic?'), options: [t('ลู่เข้าเร็วขึ้น', 'Converges faster'), t('ลู่ออก', 'Diverges')], answer: 1, why: t('ตัวคูณการอัปเดตในแกน y คือ 1 − 5η ซึ่งมีขนาดเกิน 1', 'The y-axis update factor is 1 − 5η, whose magnitude exceeds 1.') },
            { q: t('Adam ต่างจาก Momentum อย่างไร', 'How does Adam differ from Momentum?'), options: [t('ปรับขนาดก้าวต่อแกนด้วยค่าเฉลี่ย g²', 'It scales each axis’s step by a running g² average'), t('ไม่ใช้ gradient', 'It does not use the gradient')], answer: 0, why: t('จึงเดินได้สม่ำเสมอแม้ความชันของแต่ละแกนต่างกันมาก', 'So it moves evenly even when axes have very different slopes.') },
          ]} />
        ) },
      ]}
    />
  );
}
