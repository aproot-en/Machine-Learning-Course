import { useEffect, useRef, useState } from 'react';
import { useT } from '../../i18n/i18n';
import type { Network } from '../../ml/nn/network';
import type { LayerSpec } from '../../ml/nn/types';
import type { Validation } from '../../ml/nn/validate';
import { CLASS_COLORS } from '../../engine3d/palette';
import { Callout, download, Quiz, Tex } from '../../components/common';
import { Button, Icon } from '../../components/ui';

/** Heat-map of the network's prediction over the 2-feature input plane, computed with Network.output. */
export function DecisionMap({ net, tick, X, y, train, test, toModel, classNames, picked, onPick }: {
  net: Network; tick: number; X: number[][]; y: number[]; train: number[]; test: number[];
  toModel: (raw: number[]) => number[]; classNames: string[]; picked: number[] | null; onPick: (raw: number[]) => void;
}) {
  const t = useT();
  const ref = useRef<HTMLCanvasElement>(null);
  const S = 360, G = 60;
  const xs = X.map((r) => r[0]), ys = X.map((r) => r[1]);
  const pad = 0.4;
  const [x0, x1, y0, y1] = [Math.min(...xs) - pad, Math.max(...xs) + pad, Math.min(...ys) - pad, Math.max(...ys) + pad];
  const toPx = (v: number[]) => [((v[0] - x0) / (x1 - x0)) * S, S - ((v[1] - y0) / (y1 - y0)) * S];
  useEffect(() => {
    const c = ref.current?.getContext('2d');
    if (!c) return;
    const rgb = CLASS_COLORS.map((h) => [parseInt(h.slice(1, 3), 16), parseInt(h.slice(3, 5), 16), parseInt(h.slice(5, 7), 16)]);
    const cell = S / G;
    for (let gy = 0; gy < G; gy++) for (let gx = 0; gx < G; gx++) {
      const raw = [x0 + ((gx + 0.5) / G) * (x1 - x0), y1 - ((gy + 0.5) / G) * (y1 - y0)];
      const o = net.output(toModel(raw));
      const pr = o.length === 1 ? [1 - o[0], o[0]] : o;
      let r = 0, g = 0, b = 0;
      pr.forEach((p, k) => { r += p * rgb[k][0]; g += p * rgb[k][1]; b += p * rgb[k][2]; });
      const conf = Math.max(...pr);
      const a = 0.18 + 0.55 * (conf - 1 / pr.length) / (1 - 1 / pr.length);
      c.fillStyle = `rgb(${7 + (r - 7) * a | 0},${11 + (g - 11) * a | 0},${20 + (b - 20) * a | 0})`;
      c.fillRect(gx * cell, gy * cell, cell + 0.5, cell + 0.5);
    }
    const dot = (i: number, isTest: boolean) => {
      const [px, py] = toPx(X[i]);
      c.beginPath(); c.arc(px, py, isTest ? 3.6 : 3, 0, Math.PI * 2);
      if (isTest) { c.strokeStyle = CLASS_COLORS[y[i]]; c.lineWidth = 1.6; c.stroke(); }
      else { c.fillStyle = CLASS_COLORS[y[i]]; c.fill(); c.strokeStyle = 'rgba(7,11,20,.8)'; c.lineWidth = 0.8; c.stroke(); }
    };
    train.forEach((i) => dot(i, false));
    test.forEach((i) => dot(i, true));
    if (picked) {
      const [px, py] = toPx(picked);
      c.strokeStyle = '#fff'; c.lineWidth = 2; c.beginPath(); c.arc(px, py, 7, 0, Math.PI * 2); c.stroke();
      c.beginPath(); c.moveTo(px - 11, py); c.lineTo(px + 11, py); c.moveTo(px, py - 11); c.lineTo(px, py + 11); c.stroke();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [net, tick, X, y, train, test, picked, toModel]);
  return (
    <div className="flex flex-col gap-2">
      <canvas ref={ref} width={S} height={S} className="aspect-square w-full max-w-[360px] cursor-crosshair rounded-lg border border-line"
        onClick={(e) => { const r = e.currentTarget.getBoundingClientRect(); const px = ((e.clientX - r.left) / r.width) * S, py = ((e.clientY - r.top) / r.height) * S; onPick([x0 + (px / S) * (x1 - x0), y1 - (py / S) * (y1 - y0)]); }} />
      <div className="flex flex-wrap gap-x-4 gap-y-1 text-[0.9375rem] text-muted">
        {classNames.map((c, i) => <span key={c}><span style={{ color: CLASS_COLORS[i] }}>●</span> {c}</span>)}
        <span>● train · ○ test</span>
      </div>
      <p className="text-[0.9375rem] text-muted">{t('คลิกบนแผนที่เพื่อส่งจุดนั้นเข้าโครงข่ายและดู forward pass ในฉาก 3D', 'Click the map to feed that point through the network and watch its forward pass in 3D.')}</p>
    </div>
  );
}

export function architectureSvg(layers: LayerSpec[], widths: number[]): string {
  const colW = 130, H = 420, shown = widths.map((w) => Math.min(w, 12));
  const W = colW * layers.length + 40;
  const ys = shown.map((n) => Array.from({ length: n }, (_, j) => 70 + ((j + 0.5) / n) * (H - 120)));
  let s = `<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" font-family="sans-serif"><rect width="100%" height="100%" fill="#070B14"/>`;
  layers.forEach((l, k) => {
    const x = 20 + colW * k + colW / 2;
    if (k > 0) {
      const px = x - colW;
      const dense = l.kind === 'dense' || l.kind === 'output';
      ys[k].forEach((yj, j) => ys[k - 1].forEach((yi, i) => { if (dense || i === j) s += `<line x1="${px}" y1="${yi}" x2="${x}" y2="${yj}" stroke="#3B82F6" stroke-opacity="0.25"/>`; }));
    }
  });
  layers.forEach((l, k) => {
    const x = 20 + colW * k + colW / 2;
    const fill = l.kind === 'input' ? '#E5E7EB' : l.kind === 'output' ? '#22C55E' : l.kind === 'dropout' ? '#64748B' : l.kind === 'activation' ? '#A855F7' : '#00D9FF';
    ys[k].forEach((y) => (s += `<circle cx="${x}" cy="${y}" r="7" fill="${fill}"/>`));
    if (widths[k] > shown[k]) s += `<text x="${x}" y="${H - 38}" fill="#94A3B8" font-size="14" text-anchor="middle">+${widths[k] - shown[k]} more</text>`;
    const desc = l.kind === 'dropout' ? `rate ${l.rate}` : l.kind === 'activation' ? l.activation : l.kind === 'input' ? `${l.units} features` : `${l.units} · ${l.activation}`;
    s += `<text x="${x}" y="30" fill="#E5E7EB" font-size="16" text-anchor="middle">${l.name.replace(/[<&>]/g, '')}</text><text x="${x}" y="48" fill="#94A3B8" font-size="14" text-anchor="middle">${desc}</text>`;
  });
  return s + '</svg>';
}

export function modelSummaryText(layers: LayerSpec[], v: Validation): string {
  const rows = layers.map((l) => {
    const i = v.info[l.id];
    return `${l.name.padEnd(18)} ${l.kind.padEnd(11)} (None, ${String(i?.outShape ?? '?')})`.padEnd(48) + `${l.activation ?? ''}`.padEnd(11) + `${i?.params ?? 0}`;
  });
  return ['Layer              Type        Output shape              Activation Params', '-'.repeat(78), ...rows, '-'.repeat(78), `Total params: ${v.totalParams}`].join('\n');
}

export function ModelSummary({ layers, validation, onExportJson, onImport, onSvg, onPng, saves, onSave, onLoad, onDelete }: {
  layers: LayerSpec[]; validation: Validation; onExportJson: () => void; onImport: (text: string) => string | null; onSvg: () => void; onPng: () => void;
  saves: string[]; onSave: (name: string) => void; onLoad: (name: string) => void; onDelete: (name: string) => void;
}) {
  const t = useT();
  const [name, setName] = useState('');
  const [err, setErr] = useState<string | null>(null);
  return (
    <div className="grid gap-10 lg:grid-cols-[1.4fr_1fr]">
      <div className="scroll-thin overflow-x-auto">
        <table className="num w-full min-w-[520px] text-left font-mono text-[1rem]">
          <thead className="text-muted"><tr className="border-b border-line"><th className="py-2">{t('ชั้น', 'Layer')}</th><th>{t('ชนิด', 'Type')}</th><th>{t('รูปร่างเอาต์พุต', 'Output shape')}</th><th>Activation</th><th className="text-right">{t('พารามิเตอร์', 'Params')}</th></tr></thead>
          <tbody>
            {layers.map((l) => {
              const i = validation.info[l.id];
              return (
                <tr key={l.id} className="border-b border-line/50">
                  <td className="py-1.5">{l.name}</td><td>{l.kind}</td><td>(None, {i?.outShape ?? '?'})</td><td>{l.activation ?? '—'}</td>
                  <td className="text-right">{(i?.params ?? 0).toLocaleString()} {i && i.params > 0 && <span className="text-muted">= {i.inShape}×{i.outShape}{l.useBias === false ? '' : ` + ${i.outShape}`}</span>}</td>
                </tr>
              );
            })}
          </tbody>
          <tfoot><tr><td colSpan={4} className="pt-2 text-muted">{t('รวม', 'Total')}</td><td className="pt-2 text-right">{validation.totalParams.toLocaleString()}</td></tr></tfoot>
        </table>
      </div>
      <div className="flex flex-col gap-5">
        <div>
          <h3 className="label-xs mb-2">{t('ส่งออก', 'Export')}</h3>
          <div className="flex flex-wrap gap-2">
            <Button size="sm" onClick={onExportJson}><Icon name="download" />JSON</Button>
            <Button size="sm" onClick={onSvg}><Icon name="download" />{t('แผนภาพ SVG', 'Diagram SVG')}</Button>
            <Button size="sm" onClick={onPng}><Icon name="camera" />{t('ภาพฉาก PNG', 'Scene PNG')}</Button>
            <Button size="sm" onClick={() => download('model-summary.txt', modelSummaryText(layers, validation), 'text/plain')}><Icon name="download" />Summary</Button>
          </div>
        </div>
        <div>
          <h3 className="label-xs mb-2">{t('นำเข้า JSON', 'Import JSON')}</h3>
          <input type="file" accept="application/json,.json" className="text-[1rem] file:mr-3 file:rounded-md file:border file:border-line file:bg-surface-2 file:px-3 file:py-1.5 file:text-ink"
            onChange={async (e) => { const f = e.target.files?.[0]; if (f) setErr(onImport(await f.text())); e.target.value = ''; }} />
          {err && <p className="mt-2 text-[1rem] text-red">{err}</p>}
        </div>
        <div>
          <h3 className="label-xs mb-2">{t('บันทึกในเบราว์เซอร์นี้', 'Save in this browser')}</h3>
          <form className="flex gap-2" onSubmit={(e) => { e.preventDefault(); if (name.trim()) { onSave(name.trim()); setName(''); } }}>
            <input value={name} onChange={(e) => setName(e.target.value)} placeholder={t('ชื่อการทดลอง', 'Experiment name')} aria-label={t('ชื่อการทดลอง', 'Experiment name')} className="flex-1 rounded-md border border-line bg-bg px-2 py-1.5 text-[1rem]" />
            <Button size="sm" type="submit" variant="solid">{t('บันทึก', 'Save')}</Button>
          </form>
          <ul className="mt-2 flex flex-col gap-1">
            {saves.map((s) => (
              <li key={s} className="flex items-center justify-between gap-2 rounded-md border border-line px-2.5 py-1 text-[1rem]">
                <span className="truncate">{s}</span>
                <span className="flex gap-1"><Button size="sm" onClick={() => onLoad(s)}>{t('โหลด', 'Load')}</Button><Button size="sm" variant="danger" onClick={() => onDelete(s)} title={t('ลบ', 'Delete')}><Icon name="trash" /></Button></span>
              </li>
            ))}
            {!saves.length && <li className="text-[0.96875rem] text-muted">{t('ยังไม่มีการทดลองที่บันทึกไว้', 'No saved experiments yet')}</li>}
          </ul>
        </div>
      </div>
    </div>
  );
}

export function MlpMath({ level }: { level: number }) {
  const t = useT();
  return (
    <div className="grid gap-8 lg:grid-cols-2">
      <div className="flex flex-col gap-3">
        <h3 className="label-xs">Forward propagation</h3>
        <Tex block>{'\\mathbf{z}^{(l)} = W^{(l)}\\mathbf{a}^{(l-1)} + \\mathbf{b}^{(l)}, \\qquad \\mathbf{a}^{(l)} = f\\big(\\mathbf{z}^{(l)}\\big)'}</Tex>
        <h3 className="label-xs">Softmax · Loss</h3>
        <Tex block>{'p_k = \\frac{e^{z_k}}{\\sum_j e^{z_j}}, \\quad L_{CE} = -\\sum_k y_k \\log p_k, \\quad L_{BCE} = -[y\\log p + (1-y)\\log(1-p)]'}</Tex>
        <h3 className="label-xs">Backpropagation</h3>
        <Tex block>{'\\boldsymbol{\\delta}^{(L)} = \\mathbf{p} - \\mathbf{y} \\;\\;(\\text{softmax+CE or sigmoid+BCE})'}</Tex>
        <Tex block>{'\\boldsymbol{\\delta}^{(l)} = \\big(W^{(l+1)}\\big)^{\\!\\top}\\boldsymbol{\\delta}^{(l+1)} \\odot f\'\\big(\\mathbf{z}^{(l)}\\big), \\qquad \\frac{\\partial L}{\\partial W^{(l)}} = \\boldsymbol{\\delta}^{(l)}\\big(\\mathbf{a}^{(l-1)}\\big)^{\\!\\top}'}</Tex>
      </div>
      <div className="flex flex-col gap-3">
        <h3 className="label-xs">{t('การอัปเดตด้วย optimizer', 'Optimizer updates')}</h3>
        <Tex block>{'\\text{SGD: } w \\leftarrow w - \\eta g'}</Tex>
        <Tex block>{'\\text{Momentum: } v \\leftarrow \\mu v - \\eta g,\\; w \\leftarrow w + v'}</Tex>
        {level >= 2 && <Tex block>{'\\text{RMSprop: } s \\leftarrow \\beta_2 s + (1-\\beta_2)g^2,\\; w \\leftarrow w - \\frac{\\eta g}{\\sqrt{s}+\\epsilon}'}</Tex>}
        {level >= 2 && <Tex block>{'\\text{Adam: } m \\leftarrow \\beta_1 m + (1-\\beta_1) g,\\; v \\leftarrow \\beta_2 v + (1-\\beta_2) g^2,\\; w \\leftarrow w - \\eta\\frac{\\hat m}{\\sqrt{\\hat v}+\\epsilon}'}</Tex>}
        {level >= 3 && <Tex block>{'\\text{L2: } g \\leftarrow g + \\lambda_2 w,\\quad \\text{L1: } g \\leftarrow g + \\lambda_1\\,\\mathrm{sign}(w),\\quad \\text{Dropout: } a \\leftarrow \\frac{m \\odot a}{1-p},\\; m_i\\sim\\text{Bernoulli}(1-p)'}</Tex>}
        <Callout>{t('ภาพเคลื่อนไหวในฉากแสดงปริมาณที่คำนวณได้จริง: ขนาดอนุภาคขาไปคือ |w·a| ขากลับคือ |δ·a| = |∂L/∂w| ไม่ใช่สัญญาณไฟฟ้าของเซลล์ประสาทจริง', 'The scene animates computed quantities: forward particle size is |w·a|, backward is |δ·a| = |∂L/∂w|. It is not a model of real neural electrical signals.')}</Callout>
      </div>
    </div>
  );
}

export function MlpExercises({ live }: { live: { dataset: string; nClasses: number; params: number; testAcc: number; epoch: number } }) {
  const t = useT();
  const challenges = [
    { title: t('XOR ด้วย perceptron', 'XOR with a perceptron'), goal: t('เลือก XOR + preset Single Perceptron ฝึก 200 epoch แล้วสังเกตว่าความแม่นยำติดอยู่ราว 50% จากนั้นเพิ่มชั้นซ่อน 1 ชั้น', 'Pick XOR + the Single Perceptron preset, train 200 epochs and see accuracy stall near 50%; then add one hidden layer.'), ok: live.dataset === 'xor' && live.testAcc >= 0.9, need: 'XOR · test ≥ 90%' },
    { title: t('เกลียวด้วยโมเดลเล็ก', 'Small model, spiral data'), goal: t('ทำให้ Spiral 3 คลาส ได้ test ≥ 85% โดยใช้พารามิเตอร์ไม่เกิน 400 ตัว', 'Reach ≥ 85% test accuracy on 3-class Spiral with at most 400 parameters.'), ok: live.dataset === 'spiral' && live.nClasses === 3 && live.params <= 400 && live.testAcc >= 0.85, need: 'Spiral(3) · ≤400 params · test ≥ 85%' },
    { title: t('จับ overfitting', 'Catch overfitting'), goal: t('ใช้ข้อมูล 60 ตัวอย่าง noise สูง กับโมเดลกว้าง 64×64 แล้วดูกราฟ: val loss จะเริ่มสูงขึ้นขณะ train loss ลดลง ลองเปิด L2 หรือ Dropout', 'Use 60 noisy samples with a 64×64 model and watch val loss rise while train loss falls. Then try L2 or Dropout.'), ok: null, need: '' },
  ];
  return (
    <div className="grid gap-10 lg:grid-cols-2">
      <div className="flex flex-col gap-4">
        <h3 className="font-display text-xl font-semibold">{t('โจทย์ท้าทาย (ตรวจจากผลการฝึกจริง)', 'Challenges (checked against your live run)')}</h3>
        {challenges.map((c) => (
          <div key={c.title} className="rounded-xl border border-line bg-surface/60 p-4">
            <div className="flex items-center justify-between gap-3">
              <b className="font-display">{c.title}</b>
              {c.ok != null && <span className={`chip ${c.ok ? 'border-green text-green' : ''}`}>{c.ok ? t('ผ่าน ✓', 'Passed ✓') : c.need}</span>}
            </div>
            <p className="mt-2 text-[1.0625rem] text-muted">{c.goal}</p>
          </div>
        ))}
        <p className="font-mono text-[0.9375rem] text-muted">{t('ตอนนี้', 'Now')}: {live.dataset} · {live.params} params · epoch {live.epoch} · test {(live.testAcc * 100).toFixed(1)}%</p>
      </div>
      <Quiz id="mlp" items={[
        { q: t('ทำไม perceptron ชั้นเดียวแก้ XOR ไม่ได้', 'Why can’t a single perceptron solve XOR?'), options: [t('learning rate ต่ำเกินไป', 'The learning rate is too low'), t('เส้นแบ่งของมันเป็นเส้นตรง แต่ XOR แยกด้วยเส้นตรงไม่ได้', 'Its boundary is a straight line, and XOR is not linearly separable'), t('ข้อมูลน้อยเกินไป', 'Too little data')], answer: 1, why: t('ต้องมีชั้นซ่อนที่มี activation ไม่เชิงเส้น เพื่อบิดพื้นที่ให้แยกได้', 'You need a hidden layer with a non-linear activation to bend the space.') },
        { q: t('ค่า δ ของนิวรอนหมายถึงอะไร', 'What does a neuron’s δ mean?'), options: ['∂L/∂a', '∂L/∂z', '∂L/∂b + ∂L/∂w'], answer: 1, why: t('δ = ∂L/∂z และ ∂L/∂w = δ·a ของนิวรอนต้นทาง ลองกดที่นิวรอนแล้วตามด้วยน้ำหนักขาเข้า', 'δ = ∂L/∂z, and ∂L/∂w = δ times the source activation. Click a neuron, then one of its incoming weights.') },
        { q: t('ถ้า val loss เพิ่มขึ้นขณะที่ train loss ยังลดลง แปลว่าอะไร', 'Val loss rises while train loss keeps falling. What is happening?'), options: [t('underfitting', 'Underfitting'), t('overfitting', 'Overfitting'), t('learning rate ต่ำไป', 'Learning rate too low')], answer: 1, why: t('โมเดลจำข้อมูล train แทนที่จะเรียนรู้รูปแบบทั่วไป early stopping, L2 หรือ dropout ช่วยได้', 'The model memorises training data instead of general patterns; early stopping, L2 or dropout help.') },
        { q: t('Dropout ทำงานตอนไหน', 'When is dropout active?'), options: [t('ทั้งตอนฝึกและตอนทำนาย', 'Training and inference'), t('เฉพาะตอนฝึก', 'Training only'), t('เฉพาะตอนทำนาย', 'Inference only')], answer: 1, why: t('ตอนทำนายใช้ทุกนิวรอน ส่วนตอนฝึกค่าที่เหลือถูกคูณ 1/(1−p) เพื่อให้ค่าคาดหวังเท่าเดิม', 'Inference uses every unit; in training the kept units are scaled by 1/(1−p) to keep the expectation.') },
      ]} />
    </div>
  );
}
