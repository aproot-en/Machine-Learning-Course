import type { ReactNode } from 'react';
import { useT } from '../../i18n/i18n';
import type { Network } from '../../ml/nn/network';
import type { Trainer } from '../../ml/nn/trainer';
import type { LayerSpec } from '../../ml/nn/types';
import { ACTIVATION_TEX } from '../../ml/nn/activations';
import { Tex } from '../../components/common';
import { fmt } from '../../components/ui';
import { GRAD, NEG_W, POS_W } from '../../engine3d/palette';
import type { SampleTrace, Selection } from './NetworkScene';

const Row = ({ k, v, c }: { k: string; v: ReactNode; c?: string }) => (
  <div className="flex justify-between gap-4 border-b border-line/50 py-1 font-mono text-[0.96875rem]"><span className="text-muted">{k}</span><b className="num text-right font-medium" style={c ? { color: c } : undefined}>{v}</b></div>
);
const vec = (v: ArrayLike<number>, n = 6) => `[${Array.from(v).slice(0, n).map((x) => fmt(x, 3)).join(', ')}${v.length > n ? `, … +${v.length - n}` : ''}]`;

export function Inspector({ sel, net, layers, sample, trainer, viewingPast, onSelect, featureNames }: {
  sel: Selection; net: Network; layers: LayerSpec[]; sample: SampleTrace | null; trainer: Trainer | null; viewingPast: boolean;
  onSelect: (s: Selection) => void; featureNames: string[];
}) {
  const t = useT();
  const l = sel.l;
  const spec = layers[l];
  if (!spec || (sel.type === 'neuron' && (sel.j < 0 || sel.j >= net.widths[l]))) return null;
  const A = sample?.trace.a, Z = sample?.trace.z, D = sample?.grads.delta;
  const p = net.params[l];
  if (sel.type === 'edge' && (!p || sel.i < 0 || sel.i >= p.nin || sel.j < 0 || sel.j >= p.nout)) return null;
  const lr = trainer?.cfg.optimizer.lr ?? 0;
  const opt = trainer?.cfg.optimizer.name ?? 'sgd';

  if (sel.type === 'edge' && p) {
    const k = sel.j * p.nin + sel.i;
    const w = p.W[k];
    const ai = A?.[l - 1][sel.i];
    const dj = D?.[l]?.[sel.j];
    const g = trainer?.lastGradW[l]?.[k];
    const prev = trainer?.lastPrevW[l]?.[k];
    const dw = trainer?.lastDeltaW[l]?.[k];
    return (
      <div className="flex flex-col gap-3">
        <div className="font-display text-[1.125rem] font-semibold">{t('น้ำหนัก', 'Weight')} w<sub>{sel.j},{sel.i}</sub><sup>({l})</sup></div>
        <div>
          <Row k={t('ต้นทาง', 'source')} v={`${layers[l - 1].name} · #${sel.i}`} />
          <Row k={t('ปลายทาง', 'target')} v={`${spec.name} · #${sel.j}`} />
          <Row k={t('ค่าน้ำหนัก', 'value')} v={fmt(w, 5)} c={w >= 0 ? POS_W : NEG_W} />
          <Row k={t('เครื่องหมาย', 'sign')} v={w >= 0 ? t('บวก (กระตุ้น)', 'positive (excitatory)') : t('ลบ (ยับยั้ง)', 'negative (inhibitory)')} />
        </div>
        {ai != null && dj != null && (
          <div>
            <div className="label-xs mb-1">{t('gradient จากตัวอย่างที่เลือก', 'Gradient for the traced sample')}</div>
            <Tex block>{`\\frac{\\partial L}{\\partial w} = \\delta_j\\, a_i = (${fmt(dj, 4)})(${fmt(ai, 4)}) = ${fmt(dj * ai, 5)}`}</Tex>
          </div>
        )}
        {!viewingPast && g != null && prev != null && dw != null ? (
          <div>
            <div className="label-xs mb-1">{t('การอัปเดตล่าสุด (mini-batch)', 'Latest update (mini-batch)')}</div>
            <Row k={t('gradient เฉลี่ยของ batch', 'batch-mean gradient g')} v={fmt(g, 5)} c={GRAD} />
            <Row k={t('น้ำหนักก่อนหน้า', 'previous w')} v={fmt(prev, 5)} />
            <Row k={t('น้ำหนักหลังอัปเดต', 'updated w')} v={fmt(prev + dw, 5)} />
            <Row k="Δw" v={fmt(dw, 6)} />
            <Row k={`−η·g  (η=${lr})`} v={fmt(-lr * g, 6)} />
            <p className="mt-2 text-[0.9375rem] text-muted">
              {opt === 'sgd' ? t('SGD: Δw = −η·g ตรงกับค่าด้านบน', 'SGD: Δw = −η·g, matching the line above')
                : t(`${opt} ปรับขนาดก้าวด้วย momentum/ค่าเฉลี่ยของ g² จึงทำให้ Δw ต่างจาก −η·g`, `${opt} rescales the step with momentum / running g² averages, so Δw differs from −η·g`)}
            </p>
          </div>
        ) : <p className="text-[0.9375rem] text-muted">{viewingPast ? t('กำลังดูสถานะย้อนหลัง ข้อมูลการอัปเดตล่าสุดแสดงเฉพาะสถานะปัจจุบัน', 'Viewing a past state; latest-update data applies only to the current state.') : t('ยังไม่มีการอัปเดต กด ▶ หรือ → เพื่อฝึก', 'No update yet; press ▶ or → to train.')}</p>}
      </div>
    );
  }

  if (sel.type !== 'neuron') return null;
  const j = sel.j;
  const a = A?.[l]?.[j];
  const z = Z?.[l]?.[j];
  const d = D?.[l]?.[j];
  const incoming = p ? Array.from({ length: p.nin }, (_, i) => ({ i, w: p.W[j * p.nin + i] })).sort((x, y) => Math.abs(y.w) - Math.abs(x.w)) : [];
  const dws = trainer?.lastDeltaW[l];
  const meanDw = p && dws && !viewingPast ? Array.from({ length: p.nin }, (_, i) => Math.abs(dws[j * p.nin + i])).reduce((s, v) => s + v, 0) / p.nin : null;
  const terms = p && A ? Array.from({ length: Math.min(4, p.nin) }, (_, i) => `(${fmt(p.W[j * p.nin + i], 2)})(${fmt(A[l - 1][i], 2)})`).join(' + ') + (p.nin > 4 ? ' + \\dots' : '') : '';
  return (
    <div className="flex flex-col gap-3">
      <div className="font-display text-[1.125rem] font-semibold">{spec.name} · {t('นิวรอน', 'neuron')} #{j} <span className="font-mono text-[0.9375rem] text-muted">({t('ชั้น', 'layer')} {l})</span></div>
      {spec.kind === 'input' && A && <Row k={featureNames[j] ?? `x${j}`} v={fmt(A[0][j], 4)} />}
      {p && A && (
        <>
          <Row k={t('อินพุต a⁽ˡ⁻¹⁾', 'input a⁽ˡ⁻¹⁾')} v={vec(A[l - 1])} />
          <Row k={t('น้ำหนัก w', 'weights w')} v={vec(p.W.subarray(j * p.nin, (j + 1) * p.nin))} />
          <Row k="bias b" v={spec.useBias === false ? t('ไม่ใช้', 'off') : fmt(p.b[j], 4)} />
          <Tex block>{`z = \\textstyle\\sum_i w_i a_i + b = ${terms}${spec.useBias === false ? '' : ` + (${fmt(p.b[j], 2)})`} = ${fmt(z ?? 0, 4)}`}</Tex>
        </>
      )}
      {(spec.kind === 'dense' || spec.kind === 'output' || spec.kind === 'activation') && z != null && a != null && (
        <Tex block>{`a = f(z) = ${ACTIVATION_TEX[spec.activation ?? 'linear'].replace(/z/g, spec.activation === 'softmax' ? 'z' : `(${fmt(z, 3)})`)} = ${fmt(a, 4)}`}</Tex>
      )}
      {spec.kind === 'dropout' && A && (
        <Row k={t('mask (ตอนฝึก)', 'mask (training)')} v={sample?.trace.mask[l] ? fmt(sample.trace.mask[l]![j], 3) : t('ปิดตอน inference', 'off at inference')} />
      )}
      {d != null && <Row k={t('gradient เฉพาะที่ δ = ∂L/∂z', 'local gradient δ = ∂L/∂z')} v={fmt(d, 5)} c={GRAD} />}
      {meanDw != null && <Row k={t('เฉลี่ย |Δw| ขาเข้า (ก้าวล่าสุด)', 'mean |Δw| incoming (last step)')} v={fmt(meanDw, 6)} />}
      {incoming.length > 0 && (
        <div>
          <div className="label-xs mb-1">{t('น้ำหนักขาเข้า (เรียงตาม |w|) คลิกเพื่อดูรายละเอียด', 'Incoming weights (by |w|), click to inspect')}</div>
          <div className="scroll-thin max-h-48 overflow-y-auto">
            {incoming.slice(0, 24).map(({ i, w }) => (
              <button key={i} type="button" onClick={() => onSelect({ type: 'edge', l, j, i })} className="flex w-full justify-between gap-3 rounded px-1 py-0.5 font-mono text-[0.9375rem] hover:bg-white/5">
                <span className="text-muted">#{i}</span>
                <span style={{ color: w >= 0 ? POS_W : NEG_W }}>{fmt(w, 4)}</span>
                {A && <span className="text-muted">w·a = {fmt(w * A[l - 1][i], 3)}</span>}
              </button>
            ))}
          </div>
        </div>
      )}
      {!sample && <p className="text-[0.9375rem] text-muted">{t('เลือกตัวอย่างเพื่อดูค่าที่คำนวณจริง', 'Pick a sample to see computed values')}</p>}
    </div>
  );
}
