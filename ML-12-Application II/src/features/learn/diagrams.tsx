import { useMemo, useState, type ReactNode } from 'react';
import { useT } from '../../i18n/i18n';
import { activate } from '../../ml/nn/activations';
import type { Activation } from '../../ml/nn/types';
import { SMO } from '../../ml/svm';
import { buildTree, treeNodes, type TreeNode } from '../../ml/tree';
import { regionQuery } from '../../ml/clustering';
import { CLASS_COLORS, GRAD, NEG_W, POS_W } from '../../engine3d/palette';
import { Slider, fmt } from '../../components/ui';

const INK = '#E5E7EB', MUTED = '#94A3B8', LINE = '#1F2A44', CYAN = '#00D9FF';
const mono = { fontFamily: 'IBM Plex Mono, monospace' };

/** Frame for an SVG diagram with a caption. */
export function Diagram({ title, children, caption, w, h, controls }: { title: string; children: ReactNode; caption?: ReactNode; w: number; h: number; controls?: ReactNode }) {
  return (
    <figure className="my-6 rounded-xl border border-line bg-surface/50 p-4">
      <figcaption className="label-xs mb-3">{title}</figcaption>
      <div className="scroll-thin overflow-x-auto">
        <svg viewBox={`0 0 ${w} ${h}`} width="100%" style={{ maxWidth: w, minWidth: Math.min(w, 520) }} role="img" aria-label={title}>
          <defs>
            <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" fill={MUTED} /></marker>
            <marker id="arrC" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" fill={CYAN} /></marker>
            <marker id="arrP" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse"><path d="M0 0 10 5 0 10z" fill={GRAD} /></marker>
          </defs>
          {children}
        </svg>
      </div>
      {controls && <div className="mt-3 grid max-w-xl gap-3 sm:grid-cols-2">{controls}</div>}
      {caption && <p className="mt-3 text-[1.03125rem] leading-6 text-muted">{caption}</p>}
    </figure>
  );
}

const Txt = ({ x, y, children, size = 13, color = INK, anchor = 'middle', m }: { x: number; y: number; children: ReactNode; size?: number; color?: string; anchor?: 'start' | 'middle' | 'end'; m?: boolean }) => (
  <text x={x} y={y} fontSize={Math.max(13, size + 2)} fill={color} textAnchor={anchor} dominantBaseline="middle" style={m ? mono : { fontFamily: 'IBM Plex Sans Thai, sans-serif' }}>{children}</text>
);

/* ---------- 1. a single neuron: Σ w·x + b → f ---------- */
export function NeuronDiagram() {
  const t = useT();
  const [x1, setX1] = useState(0.8), [x2, setX2] = useState(-0.4);
  const [act, setAct] = useState<Activation>('sigmoid');
  const x = [x1, x2, 1.0], w = [0.9, -1.2, 0.5], b = 0.1;
  const z = x.reduce((s, v, i) => s + v * w[i], b);
  const a = activate(act, [z])[0];
  const ys = [50, 130, 210];
  return (
    <Diagram title={t('การคำนวณภายในหนึ่งโหนด (นิวรอน)', 'Computation inside one node (neuron)')} w={760} h={260}
      controls={<>
        <Slider label="x₁" value={x1} min={-2} max={2} step={0.1} format={(v) => v.toFixed(1)} onChange={setX1} />
        <Slider label="x₂" value={x2} min={-2} max={2} step={0.1} format={(v) => v.toFixed(1)} onChange={setX2} />
        <label className="flex flex-col gap-1 text-[1rem]">Activation
          <select value={act} onChange={(e) => setAct(e.target.value as Activation)} className="rounded-md border border-line bg-bg px-2 py-1">{(['sigmoid', 'tanh', 'relu', 'linear'] as Activation[]).map((v) => <option key={v}>{v}</option>)}</select>
        </label>
      </>}
      caption={t('แต่ละอินพุตคูณกับน้ำหนักของเส้นเชื่อม รวมกันแล้วบวก bias ได้ z จากนั้นผ่าน activation function ได้ผลลัพธ์ a ที่ส่งต่อไปชั้นถัดไป ลองเลื่อนค่า x แล้วดูตัวเลขทุกจุดเปลี่ยนตาม', 'Each input is multiplied by its connection weight, summed with the bias to give z, then passed through the activation function to give the output a sent to the next layer. Move x and watch every number update.')}>
      {x.map((v, i) => (
        <g key={i}>
          <circle cx={70} cy={ys[i]} r={26} fill="#101827" stroke={CYAN} />
          <Txt x={70} y={ys[i]} m>{fmt(v, 2)}</Txt>
          <Txt x={22} y={ys[i]} color={MUTED} size={12}>{`x${'₁₂₃'[i]}`}</Txt>
          <line x1={96} y1={ys[i]} x2={330} y2={130} stroke={w[i] >= 0 ? POS_W : NEG_W} strokeWidth={1.5 + Math.abs(w[i]) * 2} opacity={0.8} markerEnd="url(#arr)" />
          <Txt x={175} y={ys[i] + (130 - ys[i]) * 0.33 - 13} m size={12} color={w[i] >= 0 ? POS_W : NEG_W}>{`w=${w[i]}  →  ${fmt(v * w[i], 2)}`}</Txt>
        </g>
      ))}
      <circle cx={370} cy={130} r={40} fill="#101827" stroke={INK} />
      <Txt x={370} y={118} size={22}>Σ</Txt>
      <Txt x={370} y={146} m size={12}>{`+b=${b}`}</Txt>
      <line x1={410} y1={130} x2={500} y2={130} stroke={MUTED} markerEnd="url(#arr)" />
      <Txt x={455} y={114} m size={12}>{`z=${fmt(z, 3)}`}</Txt>
      <rect x={500} y={100} width={110} height={60} rx={10} fill="#101827" stroke="#A855F7" />
      <Txt x={555} y={122} size={13}>f(z)</Txt>
      <Txt x={555} y={142} size={11} color={MUTED} m>{act}</Txt>
      <line x1={610} y1={130} x2={690} y2={130} stroke={MUTED} markerEnd="url(#arr)" />
      <circle cx={720} cy={130} r={28} fill="#101827" stroke="#22C55E" />
      <Txt x={720} y={130} m>{fmt(a, 3)}</Txt>
      <Txt x={720} y={175} color={MUTED} size={12}>a</Txt>
      <Txt x={380} y={238} m size={12} color={MUTED}>{`z = (${x.map((v, i) => `${fmt(v, 1)}×${w[i]}`).join(') + (')}) + ${b} = ${fmt(z, 3)}   →   a = ${act}(${fmt(z, 3)}) = ${fmt(a, 3)}`}</Txt>
    </Diagram>
  );
}

/* ---------- 2. backprop on a tiny computational graph ---------- */
export function BackpropDiagram() {
  const t = useT();
  const [y, setY] = useState(1);
  const x = 1.5, w = 0.8, b = -0.2;
  const z = w * x + b, a = 1 / (1 + Math.exp(-z)), L = -(y * Math.log(a) + (1 - y) * Math.log(1 - a));
  const dLda = -(y / a) + (1 - y) / (1 - a), dadz = a * (1 - a), dLdz = dLda * dadz, dLdw = dLdz * x, dLdb = dLdz;
  const nodes = [['x', fmt(x, 2), 90], ['z = wx+b', fmt(z, 3), 280], ['a = σ(z)', fmt(a, 3), 470], ['L (BCE)', fmt(L, 3), 660]] as const;
  return (
    <Diagram title={t('Backpropagation: กฎลูกโซ่บนกราฟการคำนวณ', 'Backpropagation: the chain rule on a computational graph')} w={760} h={230}
      controls={<label className="flex items-center gap-2 text-[1rem]">{t('ค่าจริง y', 'Target y')}<select value={y} onChange={(e) => setY(Number(e.target.value))} className="rounded-md border border-line bg-bg px-2 py-1"><option value={1}>1</option><option value={0}>0</option></select></label>}
      caption={t('ลูกศรสีฟ้าคือ forward pass (คำนวณค่า) ลูกศรสีม่วงคือ backward pass ที่คูณอนุพันธ์ย่อยต่อกันจากขวาไปซ้าย ผลลัพธ์ ∂L/∂w และ ∂L/∂b คือสิ่งที่ optimizer ใช้ปรับน้ำหนัก', 'Cyan arrows are the forward pass (computing values); purple arrows are the backward pass, multiplying local derivatives right to left. The resulting ∂L/∂w and ∂L/∂b are what the optimizer uses.')}>
      {nodes.map(([n, v, cx], i) => (
        <g key={n}>
          <rect x={cx - 70} y={60} width={140} height={56} rx={10} fill="#101827" stroke={i === 3 ? '#F59E0B' : INK} />
          <Txt x={cx} y={78} size={12} color={MUTED}>{n}</Txt>
          <Txt x={cx} y={100} m>{v}</Txt>
          {i < 3 && <line x1={cx + 72} y1={74} x2={nodes[i + 1][2] - 72} y2={74} stroke={CYAN} markerEnd="url(#arrC)" />}
          {i < 3 && <line x1={nodes[i + 1][2] - 72} y1={104} x2={cx + 72} y2={104} stroke={GRAD} markerEnd="url(#arrP)" />}
        </g>
      ))}
      <Txt x={185} y={40} m size={11} color={MUTED}>{`w=${w}, b=${b}`}</Txt>
      <Txt x={375} y={140} m size={11} color={GRAD}>{`∂a/∂z = a(1−a) = ${fmt(dadz, 3)}`}</Txt>
      <Txt x={565} y={140} m size={11} color={GRAD}>{`∂L/∂a = ${fmt(dLda, 3)}`}</Txt>
      <Txt x={380} y={180} m size={12} color={GRAD}>{`∂L/∂z = ${fmt(dLda, 3)} × ${fmt(dadz, 3)} = ${fmt(dLdz, 3)}  (= a − y)`}</Txt>
      <Txt x={380} y={206} m size={12} color={GRAD}>{`∂L/∂w = ∂L/∂z · x = ${fmt(dLdw, 3)}     ∂L/∂b = ${fmt(dLdb, 3)}`}</Txt>
    </Diagram>
  );
}

/* ---------- 3. KNN distances ---------- */
const KNN_PTS: [number, number, number][] = [[1, 1, 0], [1.6, 2.2, 0], [2.4, 1.2, 0], [0.8, 3, 0], [4.2, 3.8, 1], [3.6, 4.6, 1], [4.8, 2.8, 1], [3.1, 3.2, 1], [2.2, 3.9, 1], [5.2, 4.4, 1]];
export function KnnDiagram() {
  const t = useT();
  const [k, setK] = useState(3);
  const q = [2.7, 2.7];
  const ranked = KNN_PTS.map((p, i) => ({ i, d: Math.hypot(p[0] - q[0], p[1] - q[1]) })).sort((a, b) => a.d - b.d);
  const nb = ranked.slice(0, k), votes = [0, 1].map((c) => nb.filter((n) => KNN_PTS[n.i][2] === c).length);
  const S = 80, X = (v: number) => 40 + v * S, Y = (v: number) => 420 - v * S;
  return (
    <Diagram title={t('KNN: วัดระยะ เรียงลำดับ แล้วโหวต', 'KNN: measure, rank, vote')} w={720} h={440}
      controls={<Slider label="K" value={k} min={1} max={9} onChange={setK} />}
      caption={t(`วงกลมคือรัศมีถึงเพื่อนบ้านตัวที่ K ตัวเลขบนเส้นคือระยะ Euclidean ผลโหวต: A = ${votes[0]}, B = ${votes[1]} → ทำนาย ${votes[0] >= votes[1] ? 'A' : 'B'}`, `The circle reaches the K-th neighbour; numbers are Euclidean distances. Votes: A = ${votes[0]}, B = ${votes[1]} → predict ${votes[0] >= votes[1] ? 'A' : 'B'}`)}>
      {[0, 1, 2, 3, 4, 5].map((g) => <g key={g}><line x1={X(g)} y1={Y(0)} x2={X(g)} y2={Y(5)} stroke={LINE} /><line x1={X(0)} y1={Y(g)} x2={X(5.5)} y2={Y(g)} stroke={LINE} /></g>)}
      <circle cx={X(q[0])} cy={Y(q[1])} r={nb[nb.length - 1].d * S} fill="rgba(229,231,235,0.05)" stroke={INK} strokeDasharray="4 4" />
      {nb.map((n) => <g key={n.i}><line x1={X(q[0])} y1={Y(q[1])} x2={X(KNN_PTS[n.i][0])} y2={Y(KNN_PTS[n.i][1])} stroke={CLASS_COLORS[KNN_PTS[n.i][2]]} /><Txt x={(X(q[0]) + X(KNN_PTS[n.i][0])) / 2 + 14} y={(Y(q[1]) + Y(KNN_PTS[n.i][1])) / 2} m size={11}>{fmt(n.d, 2)}</Txt></g>)}
      {KNN_PTS.map((p, i) => <circle key={i} cx={X(p[0])} cy={Y(p[1])} r={nb.some((n) => n.i === i) ? 10 : 7} fill={CLASS_COLORS[p[2]]} opacity={nb.some((n) => n.i === i) ? 1 : 0.55} />)}
      <rect x={X(q[0]) - 9} y={Y(q[1]) - 9} width={18} height={18} fill="#fff" transform={`rotate(45 ${X(q[0])} ${Y(q[1])})`} />
      <Txt x={X(q[0]) + 22} y={Y(q[1]) - 18} size={12}>q</Txt>
      <g transform="translate(500 40)">
        <Txt x={0} y={0} anchor="start" size={12} color={MUTED}>{t('อันดับ', 'rank')} · {t('ระยะ', 'distance')}</Txt>
        {ranked.slice(0, 7).map((r, j) => <Txt key={r.i} x={0} y={24 + j * 22} anchor="start" m size={12} color={j < k ? CLASS_COLORS[KNN_PTS[r.i][2]] : MUTED}>{`#${j + 1}  ${'AB'[KNN_PTS[r.i][2]]}  ${fmt(r.d, 3)}${j < k ? '  ✓' : ''}`}</Txt>)}
      </g>
    </Diagram>
  );
}

/* ---------- 4. SVM margin ---------- */
export function SvmMarginDiagram() {
  const t = useT();
  const X = [[1, 1], [1.5, 2.2], [2, 0.8], [0.6, 2.4], [3.6, 3.4], [4.2, 2.4], [3.2, 4.4], [4.6, 4]];
  const y = [-1, -1, -1, -1, 1, 1, 1, 1];
  const m = useMemo(() => { const s = new SMO(X, y, { kernel: 'linear', C: 100, gamma: 1, degree: 3, coef0: 1, tol: 1e-5, classWeight: [1, 1] }); while (!s.done) s.pass(); return s; }, []); // eslint-disable-line react-hooks/exhaustive-deps
  const w = m.weights(), b = m.b, nw = Math.hypot(...w), sv = new Set(m.supportVectors());
  const S = 80, PX = (v: number) => 60 + v * S, PY = (v: number) => 420 - v * S;
  const lineAt = (c: number) => { const xs = [0, 5.5]; return xs.map((x0) => [x0, (c - b - w[0] * x0) / w[1]]); };
  const L = (c: number, col: string, dash?: string) => { const [[a1, b1], [a2, b2]] = lineAt(c); return <line x1={PX(a1)} y1={PY(b1)} x2={PX(a2)} y2={PY(b2)} stroke={col} strokeDasharray={dash} strokeWidth={dash ? 1.2 : 2} />; };
  return (
    <Diagram title={t('SVM: ระนาบที่ margin กว้างที่สุด (คำนวณด้วย SMO)', 'SVM: the maximum-margin boundary (solved with SMO)')} w={720} h={440}
      caption={t(`w = (${w.map((v) => fmt(v, 3)).join(', ')}), b = ${fmt(b, 3)} · ความกว้าง margin = 2/‖w‖ = ${fmt(2 / nw, 3)} · จุดที่มีวงล้อมคือ support vector (α > 0) มีเพียงจุดเหล่านี้ที่กำหนดเส้น`, `w = (${w.map((v) => fmt(v, 3)).join(', ')}), b = ${fmt(b, 3)} · margin width = 2/‖w‖ = ${fmt(2 / nw, 3)} · ringed points are support vectors (α > 0); only they define the line`)}>
      <clipPath id="svmclip"><rect x={60} y={0} width={440} height={420} /></clipPath>
      <g clipPath="url(#svmclip)">{L(0, INK)}{L(1, CLASS_COLORS[0], '6 5')}{L(-1, CLASS_COLORS[1], '6 5')}</g>
      {X.map((p, i) => <g key={i}><circle cx={PX(p[0])} cy={PY(p[1])} r={8} fill={CLASS_COLORS[y[i] > 0 ? 0 : 1]} />{sv.has(i) && <circle cx={PX(p[0])} cy={PY(p[1])} r={15} fill="none" stroke="#fff" />}</g>)}
      <g transform="translate(520 60)">
        <Txt x={0} y={0} anchor="start" size={12} color={INK}>w·x + b = 0</Txt>
        <Txt x={0} y={24} anchor="start" size={12} color={CLASS_COLORS[0]}>w·x + b = +1</Txt>
        <Txt x={0} y={48} anchor="start" size={12} color={CLASS_COLORS[1]}>w·x + b = −1</Txt>
        <Txt x={0} y={84} anchor="start" size={12} color={MUTED} m>{`margin = ${fmt(2 / nw, 3)}`}</Txt>
        {[...sv].map((i, j) => <Txt key={i} x={0} y={114 + j * 20} anchor="start" size={11} color={MUTED} m>{`α${i} = ${fmt(m.alpha[i], 3)}`}</Txt>)}
      </g>
    </Diagram>
  );
}

/* ---------- 5. decision tree nodes ---------- */
export function TreeDiagram() {
  const t = useT();
  const X = [[1, 5], [2, 4], [2.5, 6], [3, 2], [4, 1], [5, 2], [6, 6], [6.5, 5], [7, 7], [5.5, 6.5], [3.5, 7], [1.5, 1.5]];
  const y = [0, 0, 0, 1, 1, 1, 2, 2, 2, 2, 0, 1];
  const root = useMemo(() => buildTree(X, y, 3, { maxDepth: 2, minSamplesSplit: 2, minSamplesLeaf: 1, criterion: 'gini', maxFeatures: 'all', ccpAlpha: 0 }), []); // eslint-disable-line react-hooks/exhaustive-deps
  const pos = new Map<number, [number, number]>();
  const place = (n: TreeNode, x: number, dx: number) => { pos.set(n.id, [x, 50 + n.depth * 130]); if (n.left && n.right) { place(n.left, x - dx, dx / 2); place(n.right, x + dx, dx / 2); } };
  place(root, 380, 190);
  const nodes = treeNodes(root);
  return (
    <Diagram title={t('Decision tree: สิ่งที่คำนวณในแต่ละโหนด', 'Decision tree: what each node computes')} w={760} h={370}
      caption={t('ทุกโหนดเก็บจำนวนตัวอย่างของแต่ละคลาส คำนวณ Gini = 1 − Σp² แล้วลองทุกเกณฑ์ "ฟีเจอร์ ≤ ค่า" เลือกเกณฑ์ที่ลด impurity ถ่วงน้ำหนักได้มากที่สุด (Δ) ใบไม้ทำนายคลาสที่มีจำนวนมากที่สุด', 'Each node counts samples per class, computes Gini = 1 − Σp², tries every “feature ≤ value” rule and keeps the one with the largest weighted impurity decrease (Δ). Leaves predict their majority class.')}>
      {nodes.map((n) => n.left && n.right && [n.left, n.right].map((c, k) => { const a = pos.get(n.id)!, b = pos.get(c.id)!; return <g key={`${n.id}-${c.id}`}><line x1={a[0]} y1={a[1] + 38} x2={b[0]} y2={b[1] - 38} stroke={MUTED} /><Txt x={(a[0] + b[0]) / 2 + (k ? 18 : -18)} y={(a[1] + b[1]) / 2} size={11} color={MUTED}>{k ? t('ไม่ใช่', 'no') : t('ใช่', 'yes')}</Txt></g>; }))}
      {nodes.map((n) => {
        const [x, yy] = pos.get(n.id)!, leaf = !n.left;
        return (
          <g key={n.id}>
            <rect x={x - 82} y={yy - 38} width={164} height={76} rx={10} fill="#101827" stroke={leaf ? CLASS_COLORS[n.prediction] : INK} />
            <Txt x={x} y={yy - 22} size={12} m>{leaf ? `→ ${'ABC'[n.prediction]}` : `x${n.feature! + 1} ≤ ${fmt(n.threshold!, 2)}`}</Txt>
            <Txt x={x} y={yy - 3} size={11} m color={MUTED}>{`n=${n.samples.length}  [${n.counts.join(', ')}]`}</Txt>
            <Txt x={x} y={yy + 16} size={11} m color={MUTED}>{`gini=${fmt(n.impurity, 3)}${n.gain != null && !leaf ? `  Δ=${fmt(n.gain, 3)}` : ''}`}</Txt>
          </g>
        );
      })}
    </Diagram>
  );
}

/* ---------- 6. convolution on a small grid ---------- */
const IMG = [[0, 0, 1, 1, 0], [0, 1, 1, 1, 0], [0, 1, 1, 0, 0], [0, 0, 1, 0, 0], [0, 0, 1, 1, 1]];
const KER = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]];
export function ConvDiagram() {
  const t = useT();
  const [pos, setPos] = useState(4);
  const r0 = Math.floor(pos / 3), c0 = pos % 3;
  const out = [0, 1, 2].map((r) => [0, 1, 2].map((c) => KER.reduce((s, row, i) => s + row.reduce((q, k, j) => q + k * IMG[r + i][c + j], 0), 0)));
  const C = 44;
  const grid = (M: number[][], ox: number, oy: number, hl?: (r: number, c: number) => boolean, tint = CYAN) => M.map((row, r) => row.map((v, c) => (
    <g key={`${ox}-${r}-${c}`}>
      <rect x={ox + c * C} y={oy + r * C} width={C - 2} height={C - 2} rx={4} fill={hl?.(r, c) ? 'rgba(0,217,255,0.25)' : v > 0 ? `rgba(229,231,235,${0.12 + 0.18 * Math.min(1, v)})` : v < 0 ? 'rgba(239,68,68,0.2)' : '#101827'} stroke={hl?.(r, c) ? tint : LINE} />
      <Txt x={ox + c * C + C / 2 - 1} y={oy + r * C + C / 2 - 1} m size={13}>{v}</Txt>
    </g>
  )));
  const terms = KER.flatMap((row, i) => row.map((k, j) => `${k}×${IMG[r0 + i][c0 + j]}`));
  return (
    <Diagram title={t('Convolution: เลื่อน kernel แล้วคูณทีละช่องและรวม', 'Convolution: slide, multiply element-wise, sum')} w={760} h={300}
      controls={<Slider label={t('ตำแหน่ง kernel', 'Kernel position')} value={pos} min={0} max={8} onChange={setPos} />}
      caption={t(`ผลลัพธ์ที่ (${r0},${c0}) = ${terms.join(' + ')} = ${out[r0][c0]} · kernel นี้ตอบสนองต่อขอบแนวตั้ง (ซ้ายมืด ขวาสว่าง) ภาพ 5×5 กับ kernel 3×3 ไม่มี padding ได้ผลลัพธ์ 3×3`, `Output at (${r0},${c0}) = ${terms.join(' + ')} = ${out[r0][c0]} · this kernel responds to vertical edges (dark left, bright right). A 5×5 image with a 3×3 kernel and no padding gives a 3×3 output.`)}>
      <Txt x={110} y={18} size={12} color={MUTED}>{t('ภาพอินพุต 5×5', 'Input 5×5')}</Txt>
      {grid(IMG, 0, 34, (r, c) => r >= r0 && r < r0 + 3 && c >= c0 && c < c0 + 3)}
      <Txt x={300} y={150} size={22} color={MUTED}>∗</Txt>
      <Txt x={400} y={78} size={12} color={MUTED}>kernel 3×3</Txt>
      {grid(KER, 334, 94, () => false)}
      <Txt x={520} y={150} size={22} color={MUTED}>=</Txt>
      <Txt x={630} y={78} size={12} color={MUTED}>{t('ผลลัพธ์ 3×3', 'Output 3×3')}</Txt>
      {grid(out, 564, 94, (r, c) => r === r0 && c === c0, '#F59E0B')}
    </Diagram>
  );
}

/* ---------- 7. RNN unrolled ---------- */
export function RnnUnrolledDiagram() {
  const t = useT();
  const T = 4;
  return (
    <Diagram title={t('RNN ที่คลี่ตามเวลา (ใช้น้ำหนักชุดเดียวกันทุก step)', 'An RNN unrolled through time (same weights at every step)')} w={760} h={250}
      caption={t('h_t = tanh(W_x x_t + W_h h_{t−1} + b) ลูกศรแนวนอนคือ W_h ที่ส่งความจำต่อไป ลูกศรสีม่วงคือ gradient ที่ย้อนเวลา (BPTT) ซึ่งถูกคูณด้วย W_h และ tanh′ ทุก step จึงหดลงเร็ว', 'h_t = tanh(W_x x_t + W_h h_{t−1} + b). Horizontal arrows are W_h carrying memory forward; purple arrows are gradients flowing back in time (BPTT), multiplied by W_h and tanh′ at every step, so they shrink quickly.')}>
      {Array.from({ length: T }, (_, i) => {
        const x = 110 + i * 170;
        return (
          <g key={i}>
            <rect x={x - 45} y={95} width={90} height={56} rx={10} fill="#101827" stroke={CYAN} />
            <Txt x={x} y={123} m>{`h${'₁₂₃₄'[i]}`}</Txt>
            <circle cx={x} cy={210} r={18} fill="#101827" stroke={INK} /><Txt x={x} y={210} m size={12}>{`x${'₁₂₃₄'[i]}`}</Txt>
            <line x1={x} y1={192} x2={x} y2={153} stroke={MUTED} markerEnd="url(#arr)" /><Txt x={x + 22} y={174} m size={11} color={MUTED}>W_x</Txt>
            <circle cx={x} cy={36} r={18} fill="#101827" stroke="#22C55E" /><Txt x={x} y={36} m size={12}>{`ŷ${'₁₂₃₄'[i]}`}</Txt>
            <line x1={x} y1={93} x2={x} y2={56} stroke={MUTED} markerEnd="url(#arr)" /><Txt x={x + 22} y={74} m size={11} color={MUTED}>W_y</Txt>
            {i < T - 1 && <><line x1={x + 46} y1={115} x2={x + 123} y2={115} stroke={CYAN} markerEnd="url(#arrC)" /><Txt x={x + 85} y={102} m size={11} color={CYAN}>W_h</Txt>
              <line x1={x + 123} y1={135} x2={x + 46} y2={135} stroke={GRAD} markerEnd="url(#arrP)" strokeOpacity={0.4 + 0.2 * i} /></>}
          </g>
        );
      })}
      <Txt x={30} y={123} m size={11} color={MUTED}>h₀=0</Txt>
    </Diagram>
  );
}

/* ---------- 8. LSTM cell with real numbers ---------- */
export function LstmCellDiagram() {
  const t = useT();
  const [cPrev, setCPrev] = useState(0.8);
  const [zf, setZf] = useState(2);
  const sig = (v: number) => 1 / (1 + Math.exp(-v));
  const f = sig(zf), i = sig(-0.5), g = Math.tanh(0.9), o = sig(1.0);
  const c = f * cPrev + i * g, h = o * Math.tanh(c);
  const box = (x: number, y: number, label: string, v: number, col: string) => (
    <g><rect x={x - 38} y={y - 24} width={76} height={48} rx={8} fill="#101827" stroke={col} /><Txt x={x} y={y - 8} size={12} color={col}>{label}</Txt><Txt x={x} y={y + 10} m size={12}>{fmt(v, 3)}</Txt></g>
  );
  return (
    <Diagram title={t('เซลล์ LSTM หนึ่งตัว: gate ควบคุมการลืมและการเขียนความจำ', 'One LSTM cell: gates control forgetting and writing memory')} w={760} h={300}
      controls={<>
        <Slider label="c_{t−1}" value={cPrev} min={-1} max={1} step={0.05} format={(v) => v.toFixed(2)} onChange={setCPrev} />
        <Slider label={t('z ของ forget gate', 'forget-gate z')} value={zf} min={-4} max={4} step={0.1} format={(v) => v.toFixed(1)} onChange={setZf} />
      </>}
      caption={t(`c_t = f·c_{t−1} + i·g = ${fmt(f, 3)}×${fmt(cPrev, 2)} + ${fmt(i, 3)}×${fmt(g, 3)} = ${fmt(c, 3)} · h_t = o·tanh(c_t) = ${fmt(h, 3)} · เมื่อ f ≈ 1 ความจำเดิมผ่านเส้น cell state ไปได้แทบไม่ลดลง ซึ่งเป็นเหตุผลที่ LSTM จำได้นานกว่า RNN`, `c_t = f·c_{t−1} + i·g = ${fmt(f, 3)}×${fmt(cPrev, 2)} + ${fmt(i, 3)}×${fmt(g, 3)} = ${fmt(c, 3)} · h_t = o·tanh(c_t) = ${fmt(h, 3)} · With f ≈ 1 the old memory passes along the cell-state line almost undiminished, which is why an LSTM remembers longer than an RNN.`)}>
      <rect x={120} y={30} width={540} height={240} rx={16} fill="none" stroke={LINE} />
      <line x1={40} y1={70} x2={720} y2={70} stroke="#F59E0B" strokeWidth={3} markerEnd="url(#arr)" />
      <Txt x={60} y={52} m size={12} color="#F59E0B">{`c_{t−1}=${fmt(cPrev, 2)}`}</Txt>
      <Txt x={690} y={52} m size={12} color="#F59E0B">{`c_t=${fmt(c, 3)}`}</Txt>
      <circle cx={260} cy={70} r={16} fill="#101827" stroke={INK} /><Txt x={260} y={70}>×</Txt>
      <circle cx={430} cy={70} r={16} fill="#101827" stroke={INK} /><Txt x={430} y={70}>+</Txt>
      {box(260, 170, 'f = σ', f, '#F59E0B')}
      <line x1={260} y1={146} x2={260} y2={88} stroke={MUTED} markerEnd="url(#arr)" />
      {box(380, 200, 'i = σ', i, '#22C55E')}
      {box(480, 200, 'g = tanh', g, CYAN)}
      <line x1={430} y1={176} x2={430} y2={88} stroke={MUTED} markerEnd="url(#arr)" /><Txt x={445} y={130} size={11} color={MUTED}>i·g</Txt>
      {box(590, 200, 'o = σ', o, '#A855F7')}
      <line x1={590} y1={70} x2={590} y2={176} stroke={MUTED} strokeDasharray="4 4" />
      <Txt x={640} y={130} m size={11} color={MUTED}>tanh(c)</Txt>
      <line x1={630} y1={240} x2={740} y2={240} stroke="#22C55E" markerEnd="url(#arr)" /><Txt x={700} y={258} m size={12} color="#22C55E">{`h_t=${fmt(h, 3)}`}</Txt>
      <Txt x={300} y={258} size={11} color={MUTED}>{t('อินพุตของทุก gate: [x_t, h_{t−1}]', 'Every gate reads [x_t, h_{t−1}]')}</Txt>
    </Diagram>
  );
}

/* ---------- 9. attention on 3 tokens ---------- */
export function AttentionDiagram() {
  const t = useT();
  const [tau, setTau] = useState(1);
  const toks = ['cat', 'sat', 'mat'];
  const Q = [[1, 0], [0.5, 1], [0, 1]], K = [[1, 0.2], [0.3, 1], [-0.2, 0.9]], V = [[1, 0], [0, 1], [0.5, 0.5]];
  const dk = 2;
  const S = Q.map((q) => K.map((k) => (q[0] * k[0] + q[1] * k[1]) / Math.sqrt(dk)));
  const A = S.map((r) => { const e = r.map((s) => Math.exp(s / tau)); const z = e.reduce((a, b) => a + b, 0); return e.map((v) => v / z); });
  const O = A.map((r) => [0, 1].map((n) => r.reduce((s, a, j) => s + a * V[j][n], 0)));
  const cell = (x: number, y: number, v: number, alpha?: number) => <g><rect x={x} y={y} width={58} height={30} rx={4} fill={alpha != null ? `rgba(0,217,255,${0.08 + 0.8 * alpha})` : '#101827'} stroke={LINE} /><Txt x={x + 29} y={y + 15} m size={12} color={alpha != null && alpha > 0.55 ? '#070B14' : INK}>{fmt(v, 2)}</Txt></g>;
  return (
    <Diagram title={t('Self-attention บน 3 คำ (d_k = 2)', 'Self-attention on 3 tokens (d_k = 2)')} w={760} h={250}
      controls={<Slider label={t('อุณหภูมิ τ', 'Temperature τ')} value={tau} min={0.2} max={3} step={0.1} format={(v) => v.toFixed(1)} onChange={setTau} />}
      caption={t('ซ้าย: คะแนน q·k/√d_k ของทุกคู่ · กลาง: softmax ในแต่ละแถว (รวมเป็น 1) · ขวา: ผลลัพธ์ = Σ α·v ของแต่ละคำ ลด τ แล้วดูว่า attention แหลมขึ้น', 'Left: scores q·k/√d_k for every pair · middle: row-wise softmax (sums to 1) · right: output = Σ α·v per token. Lower τ to sharpen attention.')}>
      <Txt x={120} y={20} size={12} color={MUTED}>QKᵀ / √d_k</Txt>
      <Txt x={400} y={20} size={12} color={MUTED}>softmax → α</Txt>
      <Txt x={650} y={20} size={12} color={MUTED}>α V</Txt>
      {toks.map((tk, i) => (
        <g key={tk}>
          <Txt x={22} y={60 + i * 40} m size={12} anchor="start">{tk}</Txt>
          {toks.map((tj, j) => <g key={tj}>{i === 0 && <><Txt x={92 + j * 62} y={38} m size={11} color={MUTED}>{tj}</Txt><Txt x={372 + j * 62} y={38} m size={11} color={MUTED}>{tj}</Txt></>}{cell(63 + j * 62, 45 + i * 40, S[i][j])}{cell(343 + j * 62, 45 + i * 40, A[i][j], A[i][j])}</g>)}
          {O[i].map((v, n) => <g key={n}>{cell(620 + n * 62, 45 + i * 40, v)}</g>)}
        </g>
      ))}
      <Txt x={380} y={190} m size={12} color={MUTED}>{`α_cat = softmax([${S[0].map((v) => fmt(v, 2)).join(', ')}] / ${fmt(tau, 1)}) = [${A[0].map((v) => fmt(v, 2)).join(', ')}]`}</Txt>
      <Txt x={380} y={215} m size={12} color={MUTED}>{`out_cat = ${A[0].map((a, j) => `${fmt(a, 2)}·v_${toks[j]}`).join(' + ')} = (${O[0].map((v) => fmt(v, 2)).join(', ')})`}</Txt>
    </Diagram>
  );
}

/* ---------- 10. DBSCAN point types ---------- */
export function DbscanDiagram() {
  const t = useT();
  const [eps, setEps] = useState(1.1);
  const minPts = 4;
  const P = [[1, 1], [1.6, 1.3], [1.2, 1.9], [2, 2], [1.7, 0.6], [2.6, 2.4], [3.2, 2.9], [5.2, 4.4], [5.8, 4.2], [5.5, 4.9], [6.1, 4.8], [5.9, 3.6], [3.9, 0.7]];
  const nb = P.map((_, i) => regionQuery(P, i, eps, 'euclidean'));
  const core = nb.map((n) => n.length >= minPts);
  const border = P.map((_, i) => !core[i] && nb[i].some((j) => core[j]));
  const S = 75, X = (v: number) => 50 + v * S, Y = (v: number) => 400 - v * S;
  return (
    <Diagram title={t(`DBSCAN: core / border / noise (MinPts = ${minPts})`, `DBSCAN: core / border / noise (MinPts = ${minPts})`)} w={720} h={420}
      controls={<Slider label="ε" value={eps} min={0.4} max={2} step={0.05} format={(v) => v.toFixed(2)} onChange={setEps} />}
      caption={t('จุดใหญ่ = core (มีเพื่อนบ้านในรัศมี ε รวมตัวเอง ≥ MinPts) · จุดกลาง = border (อยู่ในรัศมีของ core) · กากบาท = noise · วงกลมจางคือรัศมี ε ของ core แต่ละจุด', 'Large = core (≥ MinPts neighbours within ε, itself included) · medium = border (within ε of a core) · cross = noise · faint circles are each core point’s ε radius.')}>
      {P.map((p, i) => core[i] && <circle key={`c${i}`} cx={X(p[0])} cy={Y(p[1])} r={eps * S} fill="rgba(0,217,255,0.04)" stroke="rgba(0,217,255,0.25)" />)}
      {P.map((p, i) => core[i] ? <circle key={i} cx={X(p[0])} cy={Y(p[1])} r={10} fill={CYAN} /> : border[i] ? <circle key={i} cx={X(p[0])} cy={Y(p[1])} r={7} fill="none" stroke={CYAN} strokeWidth={2} /> : <g key={i}><line x1={X(p[0]) - 7} y1={Y(p[1]) - 7} x2={X(p[0]) + 7} y2={Y(p[1]) + 7} stroke="#EF4444" strokeWidth={2} /><line x1={X(p[0]) - 7} y1={Y(p[1]) + 7} x2={X(p[0]) + 7} y2={Y(p[1]) - 7} stroke="#EF4444" strokeWidth={2} /></g>)}
      {P.map((p, i) => <Txt key={`t${i}`} x={X(p[0]) + 16} y={Y(p[1]) - 14} m size={10} color={MUTED}>{nb[i].length}</Txt>)}
      <g transform="translate(560 60)">
        <Txt x={0} y={0} anchor="start" size={12} color={CYAN}>{`core: ${core.filter(Boolean).length}`}</Txt>
        <Txt x={0} y={24} anchor="start" size={12} color={CYAN}>{`border: ${border.filter(Boolean).length}`}</Txt>
        <Txt x={0} y={48} anchor="start" size={12} color="#EF4444">{`noise: ${P.length - core.filter(Boolean).length - border.filter(Boolean).length}`}</Txt>
        <Txt x={0} y={80} anchor="start" size={11} color={MUTED}>{t('ตัวเลข = |N_ε|', 'numbers = |N_ε|')}</Txt>
      </g>
    </Diagram>
  );
}

/* ---------- 11. random forest pipeline ---------- */
export function ForestDiagram() {
  const t = useT();
  const votes = ['A', 'B', 'A', 'A', 'B'];
  return (
    <Diagram title={t('Random forest: bootstrap → หลายต้นไม้ → โหวต', 'Random forest: bootstrap → many trees → vote')} w={760} h={250}
      caption={t('แต่ละต้นสุ่มตัวอย่างแบบใส่คืน (bootstrap) และสุ่มฟีเจอร์ที่ใช้แบ่งทุกโหนด ต้นไม้จึงต่างกัน ผลรวมจากการโหวตเสียงข้างมาก (ในตัวอย่างนี้ A ชนะ 3:2)', 'Each tree samples rows with replacement (bootstrap) and random features at every split, so trees differ. The ensemble takes a majority vote (A wins 3:2 here).')}>
      <rect x={20} y={95} width={110} height={60} rx={10} fill="#101827" stroke={INK} /><Txt x={75} y={115} size={12}>{t('ข้อมูล train', 'training data')}</Txt><Txt x={75} y={135} m size={11} color={MUTED}>n rows</Txt>
      {votes.map((v, i) => {
        const y = 22 + i * 46;
        return (
          <g key={i}>
            <line x1={130} y1={125} x2={210} y2={y + 18} stroke={MUTED} markerEnd="url(#arr)" />
            <rect x={212} y={y} width={120} height={36} rx={8} fill="#101827" stroke={LINE} /><Txt x={272} y={y + 18} m size={11} color={MUTED}>{`bootstrap ${i + 1}`}</Txt>
            <line x1={332} y1={y + 18} x2={380} y2={y + 18} stroke={MUTED} markerEnd="url(#arr)" />
            <g transform={`translate(400 ${y + 4})`}><circle cx={20} cy={4} r={5} fill={INK} /><line x1={20} y1={4} x2={8} y2={22} stroke={MUTED} /><line x1={20} y1={4} x2={32} y2={22} stroke={MUTED} /><circle cx={8} cy={24} r={4} fill={CLASS_COLORS[0]} /><circle cx={32} cy={24} r={4} fill={CLASS_COLORS[1]} /></g>
            <line x1={450} y1={y + 18} x2={510} y2={y + 18} stroke={MUTED} markerEnd="url(#arr)" />
            <Txt x={530} y={y + 18} m size={13} color={CLASS_COLORS[v === 'A' ? 0 : 1]}>{v}</Txt>
            <line x1={548} y1={y + 18} x2={612} y2={125} stroke={MUTED} />
          </g>
        );
      })}
      <rect x={612} y={95} width={130} height={60} rx={10} fill="#101827" stroke={CLASS_COLORS[0]} /><Txt x={677} y={115} size={12}>{t('โหวต 3 : 2', 'vote 3 : 2')}</Txt><Txt x={677} y={135} m size={13} color={CLASS_COLORS[0]}>→ A</Txt>
    </Diagram>
  );
}

/* ---------- 12. K-means one iteration ---------- */
export function KMeansStepDiagram() {
  const t = useT();
  const [phase, setPhase] = useState<0 | 1>(0);
  const P = [[1, 1], [1.5, 2], [2, 1.2], [1.2, 1.7], [4, 4], [4.6, 3.6], [4.2, 4.8], [5, 4.4], [3, 2.6]];
  const C0 = [[2.6, 1.6], [3.4, 3.2]];
  const assign = P.map((p) => (Math.hypot(p[0] - C0[0][0], p[1] - C0[0][1]) <= Math.hypot(p[0] - C0[1][0], p[1] - C0[1][1]) ? 0 : 1));
  const C1 = [0, 1].map((k) => { const m = P.filter((_, i) => assign[i] === k); return [m.reduce((s, p) => s + p[0], 0) / m.length, m.reduce((s, p) => s + p[1], 0) / m.length]; });
  const S = 75, X = (v: number) => 50 + v * S, Y = (v: number) => 420 - v * S;
  return (
    <Diagram title={t('K-Means หนึ่งรอบ: Assign แล้ว Update', 'One K-Means iteration: assign, then update')} w={720} h={440}
      controls={<div className="flex gap-2">{[t('1. Assign', '1. Assign'), t('2. Update', '2. Update')].map((l, k) => <button key={l} type="button" onClick={() => setPhase(k as 0 | 1)} className={`rounded-md border px-3 py-1 text-[1rem] ${phase === k ? 'border-cyan text-cyan' : 'border-line'}`}>{l}</button>)}</div>}
      caption={phase === 0 ? t('Assign: แต่ละจุดเลือก centroid ที่ใกล้ที่สุด (เส้นประ)', 'Assign: each point picks its nearest centroid (dashed lines).')
        : t(`Update: centroid ย้ายไปที่ค่าเฉลี่ยของสมาชิก μ₀ = (${C1[0].map((v) => fmt(v, 2)).join(', ')}), μ₁ = (${C1[1].map((v) => fmt(v, 2)).join(', ')})`, `Update: centroids move to their members’ mean μ₀ = (${C1[0].map((v) => fmt(v, 2)).join(', ')}), μ₁ = (${C1[1].map((v) => fmt(v, 2)).join(', ')})`)}>
      {P.map((p, i) => <line key={`l${i}`} x1={X(p[0])} y1={Y(p[1])} x2={X(C0[assign[i]][0])} y2={Y(C0[assign[i]][1])} stroke={CLASS_COLORS[assign[i]]} strokeDasharray="4 4" opacity={phase === 0 ? 0.7 : 0.15} />)}
      {P.map((p, i) => <circle key={i} cx={X(p[0])} cy={Y(p[1])} r={8} fill={CLASS_COLORS[assign[i]]} />)}
      {C0.map((c, k) => <g key={k}>
        <rect x={X(c[0]) - 10} y={Y(c[1]) - 10} width={20} height={20} fill="none" stroke={CLASS_COLORS[k]} strokeWidth={2} transform={`rotate(45 ${X(c[0])} ${Y(c[1])})`} opacity={phase ? 0.4 : 1} />
        {phase === 1 && <><line x1={X(c[0])} y1={Y(c[1])} x2={X(C1[k][0])} y2={Y(C1[k][1])} stroke={INK} markerEnd="url(#arr)" /><rect x={X(C1[k][0]) - 10} y={Y(C1[k][1]) - 10} width={20} height={20} fill={CLASS_COLORS[k]} transform={`rotate(45 ${X(C1[k][0])} ${Y(C1[k][1])})`} /></>}
      </g>)}
    </Diagram>
  );
}
