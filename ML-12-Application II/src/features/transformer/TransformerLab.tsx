import { useMemo, useState } from 'react';
import * as THREE from 'three';
import { useCameraState } from '../../engine3d/Stage';
import { LabShell } from '../../components/LabShell';
import { flowVector } from '../../engine3d/experiment';
import { PointCloud, Segments } from '../../engine3d/plots';
import { BOX, Label } from '../../engine3d/primitives';
import { CLASS_COLORS, NEG_W } from '../../engine3d/palette';
import { encoder } from '../../ml/transformer';
import { useT } from '../../i18n/i18n';
import { Section, Select, Slider, Toggle, fmt } from '../../components/ui';
import { Callout, Quiz, Tex } from '../../components/common';

type V3 = [number, number, number];
const HEAD_COLORS = CLASS_COLORS;

function AttnGrid({ attn, tokens, sel, onPick, size = 26 }: { attn: number[][]; tokens: string[]; sel: [number, number] | null; onPick?: (i: number, j: number) => void; size?: number }) {
  return (
    <div className="scroll-thin overflow-x-auto">
      <table className="border-separate border-spacing-0.5 font-mono text-[0.84375rem]">
        <thead><tr><th />{tokens.map((tk, j) => <th key={j} className="max-w-[48px] truncate px-0.5 font-normal text-muted" style={{ writingMode: 'vertical-rl' }}>{tk}</th>)}</tr></thead>
        <tbody>{attn.map((row, i) => (
          <tr key={i}>
            <th className="whitespace-nowrap pr-1 text-right font-normal text-muted">{tokens[i]}</th>
            {row.map((a, j) => (
              <td key={j} onClick={() => onPick?.(i, j)} title={`${tokens[i]} → ${tokens[j]}: ${a.toFixed(3)}`}
                className={`cursor-pointer text-center ${sel && sel[0] === i && sel[1] === j ? 'outline outline-1 outline-white' : ''}`}
                style={{ width: size, height: size, background: `rgba(0,217,255,${0.06 + 0.94 * a})`, color: a > 0.5 ? '#070B14' : '#94A3B8' }}>
                {a >= 0.1 ? (a * 100).toFixed(0) : ''}
              </td>
            ))}
          </tr>
        ))}</tbody>
      </table>
    </div>
  );
}

export default function TransformerLab() {
  const t = useT();
  const [text, setText] = useState('the cat sat on the mat because it was warm');
  const [cfg, setCfg] = useState({ dModel: 16, heads: 4, ffDim: 32, blocks: 1, causal: false, seed: 4 });
  const [temp, setTemp] = useState(1);
  const [block, setBlock] = useState(0);
  const [head, setHead] = useState(0);
  const [allHeads, setAllHeads] = useState(true);
  const [pair, setPair] = useState<[number, number]>([7, 1]);
  const [cam, setCam] = useCameraState();
  const tokens = useMemo(() => text.trim().split(/\s+/).filter(Boolean).slice(0, 12), [text]);
  const tr = useMemo(() => (tokens.length ? encoder(tokens, cfg, temp) : null), [tokens, cfg, temp]);
  const b = Math.min(block, cfg.blocks - 1), h = Math.min(head, cfg.heads - 1);
  const qi = Math.min(pair[0], tokens.length - 1), kj = Math.min(pair[1], tokens.length - 1);
  const dk = cfg.dModel / cfg.heads;

  const scene = useMemo(() => {
    if (!tr) return null;
    const n = tokens.length, dx = Math.min(2.2, 20 / n), X0 = -((n - 1) * dx) / 2;
    const cellH = Math.min(0.24, 4 / cfg.dModel);
    const pts: V3[] = [], cols: THREE.Color[] = [], sizes: [number, number, number][] = [];
    const dim = new THREE.Color('#0d1424');
    const vec = (v: number[], x: number, y0: number, z: number) => {
      const mx = Math.max(1e-9, ...v.map(Math.abs));
      v.forEach((val, k) => {
        pts.push([x, y0 + k * cellH, z]);
        cols.push(new THREE.Color(val >= 0 ? '#00D9FF' : NEG_W).lerp(dim, 1 - (0.1 + 0.9 * Math.abs(val) / mx)));
        sizes.push([0.5, cellH * 0.85, 0.5]);
      });
    };
    const blk = tr.blocks[b];
    tr.x0.forEach((v, i) => vec(v, X0 + i * dx, -5, 0));
    blk.output.forEach((v, i) => vec(v, X0 + i * dx, 2.2, 0));
    const segs: { a: V3; b: V3; c: string; i?: number }[] = [];
    const heads = allHeads ? blk.heads.map((_, k) => k) : [h];
    heads.forEach((hk) => {
      const A = blk.heads[hk].attn, z = (hk - (cfg.heads - 1) / 2) * 1.1;
      for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const w = A[i][j];
        if (w < 0.03) continue;
        // arc from key j (bottom) to query i (top) through the attention layer
        const a: V3 = [X0 + j * dx, -5 + cfg.dModel * cellH + 0.2, z], c: V3 = [X0 + i * dx, 2.0, z];
        const mid: V3 = [(a[0] + c[0]) / 2, (a[1] + c[1]) / 2 + Math.abs(i - j) * 0.15, z];
        let prev = a;
        for (let s = 1; s <= 12; s++) {
          const u = s / 12, p: V3 = [(1 - u) ** 2 * a[0] + 2 * (1 - u) * u * mid[0] + u * u * c[0], (1 - u) ** 2 * a[1] + 2 * (1 - u) * u * mid[1] + u * u * c[1], z];
          const focus = i === qi ? 1 : 0.25;
          segs.push({ a: prev, b: p, c: hk === h && i === qi && j === kj ? '#ffffff' : HEAD_COLORS[hk % 6], i: w * focus });
          prev = p;
        }
      }
    });
    return (
      <group>
        <PointCloud points={pts} colors={cols} sizes={sizes} geometry={BOX} glow={0.4} />
        <Segments segs={segs} />
        {tokens.map((tk, i) => <Label key={i} position={[X0 + i * dx, -5.6, 0]} className={i === qi ? 'text-cyan' : ''}>{tk}</Label>)}
        {tokens.map((_, i) => <mesh key={`m${i}`} position={[X0 + i * dx, -1.5, 0]} onClick={(e) => { e.stopPropagation(); setPair([i, kj]); }}><boxGeometry args={[dx * 0.8, 8, 0.6]} /><meshBasicMaterial transparent opacity={0} depthWrite={false} /></mesh>)}
        <Label position={[X0 - dx * 1.1, -5 + (cfg.dModel * cellH) / 2, 0]} muted>{t('ฝัง + ตำแหน่ง', 'embed + PE')}</Label>
        <Label position={[X0 - dx * 1.1, 2.2 + (cfg.dModel * cellH) / 2, 0]} muted>{t(`ผลลัพธ์ block ${b + 1}`, `block ${b + 1} output`)}</Label>
        {heads.map((hk) => <Label key={`h${hk}`} position={[X0 + (n - 1) * dx + dx * 0.9, -1.2, (hk - (cfg.heads - 1) / 2) * 1.1]}><span style={{ color: HEAD_COLORS[hk % 6] }}>head {hk + 1}</span></Label>)}
      </group>
    );
  }, [tr, tokens, cfg, b, h, allHeads, qi, kj, t]);

  if (!tr) return <div className="mx-auto max-w-3xl space-y-3 p-6"><p className="text-muted">{t('พิมพ์ประโยคอย่างน้อย 1 คำ', 'Type at least one word')}</p><textarea aria-label={t('ข้อความเข้าโมเดล', 'Model input text')} value={text} onChange={(event) => setText(event.target.value)} className="w-full rounded-lg border border-line bg-surface p-3 text-base" rows={3} /></div>;
  const blk = tr.blocks[b], H = blk.heads[h];
  const q = H.Q[qi], k = H.K[kj], dot = q.reduce((s, v, i) => s + v * k[i], 0);
  const vecTex = (v: number[]) => `(${v.slice(0, 6).map((x) => fmt(x, 2)).join(', ')}${v.length > 6 ? ', \\dots' : ''})`;
  const entropy = H.attn[qi].reduce((s, a) => (a > 0 ? s - a * Math.log2(a) : s), 0);

  return (
    <LabShell crumb="Transformer" title={t('Transformer — Self-Attention', 'Transformer — Self-Attention')} chips={[t('ข้อมูลลำดับ', 'Sequence model'), t('ขั้นสูง', 'Advanced')]}
      experiment={{ model: 'Transformer Encoder', source: text,
        stages: [
          { name: 'Embedding + position', kind: 'tensor', shape: [tokens.length, cfg.dModel, 1], values: tr.x0.flat() },
          { name: `Q / K · head ${h + 1}`, kind: 'tensor', shape: [tokens.length, dk, 2], values: [...H.Q.flat(), ...H.K.flat()] },
          { name: 'QKᵀ / √dₖ', kind: 'tensor', shape: [tokens.length, tokens.length, 1], values: H.scores.flat() },
          { name: 'Softmax attention', kind: 'tensor', shape: [tokens.length, tokens.length, 1], values: H.attn.flat(), highlight: [qi * tokens.length + kj] },
          { name: 'Attention × V', kind: 'tensor', shape: [tokens.length, dk, 1], values: H.out.flat() },
          { name: 'FF + residual + norm', kind: 'tensor', shape: [tokens.length, cfg.dModel, 1], values: blk.output.flat() },
          { name: `Output: ${tokens[qi]}`, values: blk.output[qi] },
        ],
        metrics: [{ label: 'Tokens', value: String(tokens.length) }, { label: 'Selected block / head', value: `${b + 1} / ${h + 1}` }, { label: 'Attention entropy', value: fmt(entropy, 4) }, { label: 'Weights', value: t('สุ่ม · ยังไม่ฝึก', 'Random · untrained') }],
        output: `${t('เวกเตอร์บริบท', 'Context vectors')} ${tokens.length}×${cfg.dModel} · “${tokens[qi]}”=${flowVector(blk.output[qi])}`,
        note: t('น้ำหนักสุ่ม ยังไม่ฝึก ผลลัพธ์เป็นเวกเตอร์ ไม่ใช่คำที่โมเดลสร้าง', 'Random untrained weights; output is a vector, not generated text'),
      }}
      intro={t('แต่ละคำถูกแปลงเป็นเวกเตอร์ (แท่งด้านล่าง) เส้นโค้งคือน้ำหนัก attention ที่คำด้านบน (query) ดึงข้อมูลจากคำด้านล่าง (key) แต่ละสีคือหนึ่ง head คลิกคำหรือช่องใน heatmap เพื่อตรวจ q·k', 'Each word becomes a vector (columns at the bottom). Arcs are attention weights from a query word (top) to key words (bottom); each colour is one head. Click a word or a heat-map cell to inspect q·k.')}
      cam={cam} setCam={setCam} distance={20} target={[0, -1.5, 0]} scene={scene}
      readout={{ title: t('Encoder (น้ำหนักสุ่ม ยังไม่ฝึก)', 'Encoder (random, untrained weights)'), rows: [
        ['tokens', tokens.length], ['d_model · heads · d_k', `${cfg.dModel} · ${cfg.heads} · ${dk}`],
        ['block · head', `${b + 1} · ${h + 1}`], ['query → key', `“${tokens[qi]}” → “${tokens[kj]}”`],
        ['q·k / √d_k', fmt(dot / Math.sqrt(dk), 3)], ['attention', fmt(H.attn[qi][kj], 4)],
        [t('ความกระจายของ attention (entropy)', 'attention entropy'), `${fmt(entropy, 2)} bits`],
      ] }}
      aside={<div className="glass pointer-events-auto hidden max-w-[380px] p-3 lg:block"><div className="label-xs mb-2">{t(`ตาราง attention · head ${h + 1}`, `Attention matrix · head ${h + 1}`)}</div><AttnGrid attn={H.attn} tokens={tokens} sel={[qi, kj]} onPick={(i, j) => setPair([i, j])} size={22} /></div>}
      controls={
        <>
          <Section title={t('ประโยค', 'Sentence')}>
            <label className="flex flex-col gap-1 text-[1rem]">
              <span>{t('ข้อความ (สูงสุด 12 คำ แยกด้วยช่องว่าง)', 'Text (up to 12 space-separated tokens)')}</span>
              <textarea value={text} onChange={(e) => setText(e.target.value)} rows={2} className="rounded-md border border-line bg-bg px-2 py-1.5 text-[1rem]" />
            </label>
            <Select<number> label="Query token" value={qi} onChange={(i) => setPair([i, kj])} options={tokens.map((tk, i) => ({ value: i, label: `${i}: ${tk}` }))} />
            <Select<number> label="Key token" value={kj} onChange={(j) => setPair([qi, j])} options={tokens.map((tk, i) => ({ value: i, label: `${i}: ${tk}` }))} />
          </Section>
          <Section title={t('สถาปัตยกรรม', 'Architecture')}>
            <Select<number> label="d_model" value={cfg.dModel} onChange={(dModel) => setCfg({ ...cfg, dModel, heads: Math.min(cfg.heads, dModel / 2) })} options={[8, 16, 32].map((v) => ({ value: v, label: String(v) }))} />
            <Select<number> label={t('จำนวน head', 'Heads')} value={cfg.heads} onChange={(heads) => setCfg({ ...cfg, heads })} options={[1, 2, 4, 8].filter((v) => cfg.dModel % v === 0 && cfg.dModel / v >= 2).map((v) => ({ value: v, label: `${v} (d_k = ${cfg.dModel / v})` }))} />
            <Select<number> label="FFN dim" value={cfg.ffDim} onChange={(ffDim) => setCfg({ ...cfg, ffDim })} options={[16, 32, 64].map((v) => ({ value: v, label: String(v) }))} />
            <Slider label={t('จำนวน encoder block', 'Encoder blocks')} value={cfg.blocks} min={1} max={3} onChange={(blocks) => setCfg({ ...cfg, blocks })} />
            <Toggle label={t('Causal mask (มองไม่เห็นคำอนาคต)', 'Causal mask (no peeking ahead)')} checked={cfg.causal} onChange={(causal) => setCfg({ ...cfg, causal })} />
            <Slider label={t('อุณหภูมิ softmax', 'Softmax temperature')} value={temp} min={0.2} max={3} step={0.1} format={(v) => v.toFixed(1)} onChange={setTemp} />
            <Slider label={t('seed ของน้ำหนัก', 'Weight seed')} value={cfg.seed} min={1} max={99} onChange={(seed) => setCfg({ ...cfg, seed })} />
          </Section>
          <Section title={t('การแสดงผล', 'View')}>
            <Slider label="Block" value={b} min={0} max={cfg.blocks - 1} onChange={setBlock} />
            <Slider label="Head" value={h} min={0} max={cfg.heads - 1} onChange={setHead} />
            <Toggle label={t('แสดงทุก head ใน 3D', 'Show all heads in 3D')} checked={allHeads} onChange={setAllHeads} />
          </Section>
        </>
      }
      tabs={[
        { id: 'qkv', label: t('ตรวจ Q · K · V', 'Inspect Q · K · V'), content: (
          <div className="grid gap-8 lg:grid-cols-2">
            <div className="flex flex-col gap-3">
              <Tex block>{`\\mathbf q_{${qi}} = ${vecTex(q)}`}</Tex>
              <Tex block>{`\\mathbf k_{${kj}} = ${vecTex(k)}`}</Tex>
              <Tex block>{`\\frac{\\mathbf q\\cdot\\mathbf k}{\\sqrt{d_k}} = \\frac{${fmt(dot, 3)}}{\\sqrt{${dk}}} = ${fmt(dot / Math.sqrt(dk), 3)}${cfg.causal && kj > qi ? '\\;\\to -\\infty\\ (\\text{masked})' : ''}`}</Tex>
              <Tex block>{`\\alpha_{${qi},${kj}} = \\text{softmax}_j\\Big(\\frac{s_{${qi},j}}{\\tau}\\Big) = ${fmt(H.attn[qi][kj], 4)}`}</Tex>
              <Tex block>{`\\mathbf v_{${kj}} = ${vecTex(H.V[kj])}`}</Tex>
              <Tex block>{`\\text{out}_{${qi}} = \\sum_j \\alpha_{${qi},j}\\mathbf v_j = ${vecTex(H.out[qi])}`}</Tex>
            </div>
            <div className="flex flex-col gap-4">
              <h3 className="label-xs">{t(`คะแนนดิบและ softmax ของ "${tokens[qi]}"`, `Raw scores and softmax for “${tokens[qi]}”`)}</h3>
              <table className="num w-full text-left font-mono text-[0.96875rem]">
                <thead className="text-muted"><tr className="border-b border-line"><th className="py-1">key</th><th>q·k/√d</th><th>α</th><th /></tr></thead>
                <tbody>{tokens.map((tk, j) => (
                  <tr key={j} className={`border-b border-line/50 ${j === kj ? 'bg-cyan/10' : ''}`} onClick={() => setPair([qi, j])}>
                    <td className="py-1">{tk}</td><td>{Number.isFinite(H.scores[qi][j]) ? fmt(H.scores[qi][j], 3) : '−∞'}</td><td>{fmt(H.attn[qi][j], 4)}</td>
                    <td className="w-1/3"><span className="block h-1.5 rounded bg-cyan" style={{ width: `${H.attn[qi][j] * 100}%` }} /></td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          </div>
        ) },
        { id: 'heads', label: t('ทุก head', 'All heads'), content: (
          <div className="flex flex-wrap gap-6">
            {blk.heads.map((hd, k2) => <div key={k2}><div className="label-xs mb-1" style={{ color: HEAD_COLORS[k2 % 6] }}>head {k2 + 1}</div><AttnGrid attn={hd.attn} tokens={tokens} sel={null} onPick={(i, j) => { setHead(k2); setPair([i, j]); }} size={18} /></div>)}
          </div>
        ) },
        { id: 'pipe', label: t('ขั้นตอนใน block', 'Inside a block'), content: (
          <div className="grid gap-8 lg:grid-cols-2">
            <ol className="list-decimal space-y-2 pl-5 text-[1.125rem]">
              <li>{t('Token embedding + positional encoding (sin/cos) เพราะ attention เองไม่รู้ลำดับคำ', 'Token embedding + sinusoidal positional encoding, since attention itself is order-blind.')}</li>
              <li>{t('แต่ละ head ฉาย X เป็น Q = XW_Q, K = XW_K, V = XW_V', 'Each head projects X into Q = XW_Q, K = XW_K, V = XW_V.')}</li>
              <li>{t('คะแนน QKᵀ/√d_k → (mask) → softmax ได้น้ำหนัก attention แต่ละแถวรวมเป็น 1', 'Scores QKᵀ/√d_k → (mask) → softmax gives attention weights; each row sums to 1.')}</li>
              <li>{t('ผลรวมถ่วงน้ำหนักของ V จากทุก head นำมาต่อกันแล้วคูณ W_O', 'Weighted sums of V from all heads are concatenated and multiplied by W_O.')}</li>
              <li>{t('Residual (X + attention) แล้ว LayerNorm', 'Residual (X + attention), then LayerNorm.')}</li>
              <li>{t('Feed-forward (ReLU) ต่อทีละตำแหน่ง + residual + LayerNorm', 'Position-wise feed-forward (ReLU) + residual + LayerNorm.')}</li>
            </ol>
            <div className="flex flex-col gap-2 font-mono text-[0.96875rem]">
              {([[t('อินพุต', 'input'), blk.input], ['attention W_O', blk.attnOut], ['residual 1', blk.res1], ['LayerNorm 1', blk.norm1], ['FFN', blk.ff], ['LayerNorm 2 (output)', blk.output]] as [string, number[][]][]).map(([name, M]) => (
                <div key={name} className="flex justify-between border-b border-line/50 py-1"><span className="text-muted">{name}</span><span>‖·‖ {t('ของ', 'of')} “{tokens[qi]}” = {fmt(Math.hypot(...M[qi]), 3)}</span></div>
              ))}
            </div>
          </div>
        ) },
        { id: 'math', label: t('คณิตศาสตร์', 'Math'), content: (
          <div className="flex flex-col gap-3">
            <Tex block>{'\\text{Attention}(Q,K,V) = \\text{softmax}\\Big(\\frac{QK^\\top}{\\sqrt{d_k}} + M\\Big)V,\\quad M_{ij} = -\\infty \\text{ if causal and } j>i'}</Tex>
            <Tex block>{'\\text{MultiHead}(X) = [\\text{head}_1;\\dots;\\text{head}_h]W_O,\\qquad PE_{p,2i} = \\sin\\!\\big(p/10000^{2i/d}\\big),\\ PE_{p,2i+1} = \\cos(\\cdot)'}</Tex>
            <Tex block>{'\\text{LayerNorm}(\\mathbf x) = \\frac{\\mathbf x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}},\\qquad \\text{FFN}(\\mathbf x) = \\max(0, \\mathbf x W_1)W_2'}</Tex>
            <Callout tone="warn">{t('น้ำหนักทั้งหมดในห้องนี้สุ่มจาก seed (ยังไม่ได้ฝึก) และ embedding ของแต่ละคำมาจาก hash ของตัวคำ การคำนวณ attention ถูกต้องตามสูตร แต่รูปแบบ attention ไม่ได้สะท้อนความหมายของภาษาแบบโมเดลภาษาที่ฝึกแล้ว', 'All weights here are seeded random (untrained) and each word’s embedding comes from a hash of the word. The attention computation follows the formulas exactly, but the patterns do not reflect linguistic meaning the way a trained language model would.')}</Callout>
          </div>
        ) },
        { id: 'ex', label: t('แบบฝึกหัด', 'Exercises'), content: (
          <Quiz id="transformer" items={[
            { q: t('ทำไมต้องหารด้วย √d_k', 'Why divide by √d_k?'), options: [t('ให้คะแนนไม่ใหญ่เกินจน softmax อิ่มตัว', 'To keep scores from growing so large that softmax saturates'), t('ให้ attention รวมเป็น 1', 'So attention sums to 1')], answer: 0, why: t('ผลคูณภายในของเวกเตอร์สุ่มมี variance ∝ d_k ลองลดอุณหภูมิ softmax แล้วดูว่า attention แหลมขึ้น', 'Dot products of random vectors have variance ∝ d_k. Lower the temperature and watch attention sharpen.') },
            { q: t('causal mask ใช้เมื่อไร', 'When is a causal mask used?'), options: [t('ในโมเดลที่สร้างข้อความทีละคำ (decoder)', 'In models generating text token by token (decoders)'), t('ในทุก encoder', 'In every encoder')], answer: 0, why: t('ตำแหน่ง i ห้ามเห็นตำแหน่งอนาคต j > i ดูสามเหลี่ยมใน heatmap', 'Position i must not see future positions j > i; see the triangle in the heat-map.') },
            { q: t('ถ้าไม่มี positional encoding การสลับลำดับคำจะเปลี่ยนผลของ self-attention หรือไม่', 'Without positional encoding, does shuffling the words change self-attention’s outputs (other than their order)?'), options: [t('เปลี่ยน', 'Yes'), t('ไม่เปลี่ยน (permutation-equivariant)', 'No (permutation-equivariant)')], answer: 1, why: t('attention ดูเพียงเนื้อหาของเวกเตอร์ ไม่รู้ตำแหน่ง', 'Attention only sees vector contents, not positions.') },
          ]} />
        ) },
      ]}
    />
  );
}
