import { useState } from 'react';
import { useT, type Level, type Text } from '../../i18n/i18n';
import type { KnnOptions, KnnResult } from '../../ml/knn';
import type { ClassificationReport } from '../../ml/metrics';
import { Callout, ConfusionMatrix, LinePlot, Quiz, Tex } from '../../components/common';
import { Button, Prose, Tabs, fmt, pct } from '../../components/ui';
import { CLASS_COLORS } from '../../engine3d/palette';

export const STAGES: { name: Text; text: Text }[] = [
  { name: { th: 'จุดคำถาม', en: 'Query' }, text: { th: 'จุดสีขาวคือข้อมูลใหม่ที่ยังไม่รู้คลาส ลากเพื่อย้ายตำแหน่งได้', en: 'The white point is a new sample with unknown class. Drag it to move it.' } },
  { name: { th: 'คำนวณระยะ', en: 'Distances' }, text: { th: 'วัดระยะจากจุดคำถามไปยังจุด train ทุกจุด เส้นยิ่งสว่างยิ่งใกล้', en: 'Measure the distance to every training point. Brighter lines are closer.' } },
  { name: { th: 'เรียงลำดับ', en: 'Sort' }, text: { th: 'เรียงระยะจากน้อยไปมาก ป้ายบนจุดคือ #อันดับ · ระยะ', en: 'Sort distances ascending. Tags show #rank · distance.' } },
  { name: { th: 'เลือก K จุด', en: 'Select K' }, text: { th: 'เก็บเฉพาะ K อันดับแรก ขอบเขตโปร่งแสงมีรัศมีเท่ากับระยะถึงเพื่อนบ้านตัวที่ K', en: 'Keep the first K. The translucent boundary’s radius equals the distance to the K-th neighbour.' } },
  { name: { th: 'โหวต', en: 'Vote' }, text: { th: 'เพื่อนบ้านแต่ละจุดโหวตให้คลาสของตัวเอง ดูคะแนนในกล่องด้านซ้ายบน', en: 'Each neighbour votes for its own class. See the tally in the top-left readout.' } },
  { name: { th: 'ทำนาย', en: 'Predict' }, text: { th: 'คลาสที่ได้คะแนนมากที่สุดคือคำตอบ ถ้าเสมอใช้คลาสของจุดที่ใกล้ที่สุด', en: 'The class with most votes wins; ties go to the class of the nearest point.' } },
];

export function StageCaption({ stage, k }: { stage: number; k: number }) {
  const t = useT();
  return (
    <div className="glass max-w-[640px] px-4 py-2 text-center text-[1.03125rem]">
      <b className="font-display text-cyan">{stage}. {t(STAGES[stage].name)}</b>
      <span className="ml-2 text-ink/85">{t(STAGES[stage].text).replace('K ', `K = ${k} `)}</span>
    </div>
  );
}

interface LessonProps {
  model: KnnOptions; result: KnnResult; Xtr: number[][]; ytr: number[]; q: number[]; classNames: string[];
  report: ClassificationReport; curve: { k: number; train: number; test: number }[]; featureNames: string[];
  onPlaceQuery: (v: number[]) => void; onReveal: () => void; lim: number; dims: number; nClasses: number; level: Level; selected: number | null;
}

function distanceWork(a: number[], q: number[], m: KnnOptions) {
  const d = a.map((v, i) => v - q[i]);
  if (m.metric === 'euclidean') return `√(${d.map((x) => `${fmt(x, 2)}²`).join(' + ')}) = √(${d.map((x) => fmt(x * x, 3)).join(' + ')})`;
  if (m.metric === 'manhattan') return d.map((x) => `|${fmt(x, 2)}|`).join(' + ');
  return `(${d.map((x) => `|${fmt(x, 2)}|^${m.p}`).join(' + ')})^(1/${m.p})`;
}

export function KnnLessons(p: LessonProps) {
  const t = useT();
  const shown = Math.min(p.result.ranked.length, p.result.neighbors.length + 5);
  const metricTex = p.model.metric === 'euclidean' ? 'd(\\mathbf{x},\\mathbf{q}) = \\sqrt{\\textstyle\\sum_{i=1}^{n}(x_i-q_i)^2}'
    : p.model.metric === 'manhattan' ? 'd(\\mathbf{x},\\mathbf{q}) = \\textstyle\\sum_{i=1}^{n}|x_i-q_i|'
      : `d(\\mathbf{x},\\mathbf{q}) = \\Big(\\textstyle\\sum_{i=1}^{n}|x_i-q_i|^{p}\\Big)^{1/p},\\; p=${p.model.p}`;
  const bestK = p.curve.reduce((b, c) => (c.test > b.test ? c : b), p.curve[0] ?? { k: 1, test: 0, train: 0 });

  const tabs = [
    {
      id: 'concept', label: t('แนวคิด', 'Concept'), content: (
        <div className="grid gap-10 lg:grid-cols-2">
          <Prose>
            <p>{t('KNN เป็นอัลกอริทึมจำแนกประเภทที่ "ไม่มีการฝึก" ในความหมายปกติ โมเดลคือชุดข้อมูล train ทั้งชุด เมื่อมีจุดใหม่เข้ามา จะหาจุด train ที่ใกล้ที่สุด K จุด แล้วให้จุดเหล่านั้นโหวต',
              'KNN is a classifier with no training step in the usual sense: the model is the whole training set. For a new point it finds the K closest training points and lets them vote.')}</p>
            <p>{t('แนวคิดเบื้องหลังคือ "ของที่คล้ายกันมักอยู่ใกล้กัน" ในพื้นที่ฟีเจอร์ ดังนั้นการวัดระยะและการปรับสเกลฟีเจอร์จึงสำคัญมาก', 'The idea is that similar things sit close together in feature space, which is why the distance measure and feature scaling matter so much.')}</p>
            <h3 className="pt-2 font-display text-lg font-semibold">{t('ขั้นตอนอัลกอริทึม', 'Algorithm workflow')}</h3>
            <ol>{STAGES.slice(1).map((s, i) => <li key={i}><b>{t(s.name)}</b> — {t(s.text)}</li>)}</ol>
          </Prose>
          <Prose>
            <h3 className="font-display text-lg font-semibold">{t('ค่า K กับ overfitting / underfitting', 'K, overfitting and underfitting')}</h3>
            <ul>
              <li>{t('K เล็ก (เช่น 1): เส้นแบ่งขรุขระ ตามจุด noise ทุกจุด ความแม่นยำบน train สูงมากแต่บน test อาจต่ำ = overfitting', 'Small K (e.g. 1): a jagged boundary that follows every noisy point. Train accuracy is very high but test accuracy can drop: overfitting.')}</li>
              <li>{t('K ใหญ่มาก: เส้นแบ่งเรียบเกินไป คลาสเล็กถูกกลืน = underfitting', 'Very large K: an over-smoothed boundary; small classes get swallowed: underfitting.')}</li>
              <li>{t('ลองเปิด "พื้นที่ตัดสินใจ" แล้วเลื่อน K จาก 1 ถึง 25 และดูแท็บ "การประเมินผล"', 'Turn on the decision region, slide K from 1 to 25, and check the Evaluation tab.')}</li>
            </ul>
            <h3 className="pt-2 font-display text-lg font-semibold">{t('การใช้งานจริง', 'Real-world uses')}</h3>
            <ul>
              <li>{t('ระบบแนะนำสินค้าจากผู้ใช้ที่คล้ายกัน', 'Recommendations from similar users')}</li>
              <li>{t('ค้นหาภาพหรือเอกสารที่คล้ายกันด้วย embedding', 'Similar-image or document search with embeddings')}</li>
              <li>{t('เติมค่าที่หายไป (KNN imputation)', 'Filling missing values (KNN imputation)')}</li>
            </ul>
            <h3 className="pt-2 font-display text-lg font-semibold">{t('ข้อผิดพลาดที่พบบ่อย', 'Common mistakes')}</h3>
            <ul>
              <li>{t('ไม่ปรับสเกลฟีเจอร์ ฟีเจอร์ที่มีหน่วยใหญ่จะครอบงำระยะทาง (ลองปิด "ปรับมาตรฐาน" กับชุด Iris)', 'Not scaling features, so large-unit features dominate distance (try turning off standardisation on Iris).')}</li>
              <li>{t('ใช้ข้อมูล test ในการ fit scaler หรือเลือก K ด้วย test set', 'Fitting the scaler on test data, or tuning K on the test set.')}</li>
              <li>{t('ใช้กับข้อมูลมิติสูงมาก ซึ่งระยะทางทุกคู่เริ่มใกล้เคียงกัน (curse of dimensionality)', 'Using it in very high dimensions, where all distances look alike (curse of dimensionality).')}</li>
            </ul>
          </Prose>
        </div>
      ),
    },
    {
      id: 'math', label: t('คณิตศาสตร์', 'Math'), content: (
        <div className="grid gap-8 lg:grid-cols-2">
          <div className="flex flex-col gap-4">
            <h3 className="label-xs">{t('ระยะทางที่ใช้อยู่', 'Distance in use')}</h3>
            <Tex block>{metricTex}</Tex>
            <h3 className="label-xs">{t('เพื่อนบ้าน K ตัว', 'The K neighbours')}</h3>
            <Tex block>{'N_K(\\mathbf{q}) = \\{\\text{the } K \\text{ training points with smallest } d(\\mathbf{x}_j, \\mathbf{q})\\}'}</Tex>
            <h3 className="label-xs">{t('การโหวต', 'Voting')}</h3>
            <Tex block>{p.model.weighting === 'uniform'
              ? '\\hat{y} = \\arg\\max_{c} \\sum_{\\mathbf{x}_j \\in N_K} \\mathbb{1}[y_j = c]'
              : '\\hat{y} = \\arg\\max_{c} \\sum_{\\mathbf{x}_j \\in N_K} \\frac{1}{d(\\mathbf{x}_j,\\mathbf{q})}\\,\\mathbb{1}[y_j = c]'}</Tex>
          </div>
          <div className="flex flex-col gap-4">
            <h3 className="label-xs">{t('การปรับมาตรฐาน (fit บน train เท่านั้น)', 'Standardisation (fit on train only)')}</h3>
            <Tex block>{'z_i = \\frac{x_i - \\mu_i^{\\text{train}}}{\\sigma_i^{\\text{train}}}'}</Tex>
            {p.level >= 3 && (
              <>
                <h3 className="label-xs">{t('ความซับซ้อน', 'Complexity')}</h3>
                <Tex block>{'\\text{predict: } O(N\\,n + N\\log N)\\quad\\text{(N training points, n features)}'}</Tex>
                <p className="text-[1.0625rem] text-muted">{t('ทุกการทำนายต้องวัดระยะกับจุด train ทั้งหมด ระบบจริงใช้ KD-tree, Ball tree หรือ approximate nearest neighbour เพื่อเร่งความเร็ว', 'Every prediction measures distance to all training points; real systems use KD-trees, ball trees or approximate nearest-neighbour search.')}</p>
              </>
            )}
            <Callout>{t('ระยะ Manhattan ทำให้ "ลูกบอล" ของเพื่อนบ้านเป็นทรงแปดหน้า เพราะจุดที่ระยะเท่ากันอยู่บนผิว |x|+|y|+|z| = r ลองเปลี่ยนวิธีวัดระยะในขั้น "เลือก K จุด"', 'With Manhattan distance the neighbour “ball” becomes an octahedron, because equidistant points satisfy |x|+|y|+|z| = r. Switch metrics at the “Select K” step.')}</Callout>
          </div>
        </div>
      ),
    },
    {
      id: 'inspect', label: t('ตรวจการคำนวณ', 'Calculation inspector'), content: (
        <div className="flex flex-col gap-6">
          <p className="text-[1.0625rem] text-muted">{t('ตัวเลขด้านล่างคือค่าเดียวกับที่โมเดลใช้ทำนายจุดคำถามตอนนี้', 'These are the exact numbers the model uses for the current query.')} q = ({p.q.map((v) => fmt(v, 3)).join(', ')})</p>
          <div className="scroll-thin overflow-x-auto">
            <table className="num w-full min-w-[720px] text-left font-mono text-[0.96875rem]">
              <thead className="text-muted"><tr className="border-b border-line">
                <th className="py-2 pr-3">#</th><th className="pr-3">{t('จุด', 'point')}</th><th className="pr-3">{t('คลาส', 'class')}</th>
                <th className="pr-3">{p.featureNames.join(', ')}</th><th className="pr-3">{t('การคำนวณ', 'working')}</th><th className="pr-3 text-right">d</th>
                {p.model.weighting === 'distance' && <th className="text-right">1/d</th>}
              </tr></thead>
              <tbody>
                {p.result.ranked.slice(0, shown).map((r, k) => {
                  const inK = k < p.result.neighbors.length;
                  return (
                    <tr key={r.index} className={`border-b border-line/60 ${inK ? '' : 'opacity-45'} ${p.selected === r.index ? 'bg-cyan/10' : ''}`}>
                      <td className="py-1.5 pr-3">{k + 1}</td><td className="pr-3">x{r.index}</td>
                      <td className="pr-3" style={{ color: CLASS_COLORS[p.ytr[r.index]] }}>{p.classNames[p.ytr[r.index]]}</td>
                      <td className="pr-3">({p.Xtr[r.index].map((v) => fmt(v, 2)).join(', ')})</td>
                      <td className="pr-3 text-muted">{distanceWork(p.Xtr[r.index], p.q, p.model)}</td>
                      <td className="pr-3 text-right">{fmt(r.distance, 3)}</td>
                      {p.model.weighting === 'distance' && <td className="text-right">{inK ? fmt(1 / (r.distance + 1e-9), 3) : ''}</td>}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          <p className="text-[1rem] text-muted">{t(`แถวสว่าง = K = ${p.model.k} เพื่อนบ้านที่ถูกเลือก แถวจาง = อันดับถัดไปที่ไม่ถูกนับ`, `Bright rows are the K = ${p.model.k} selected neighbours; faded rows are the next ranks, not counted.`)}</p>
          <div>
            <h3 className="label-xs mb-3">{t('รวมคะแนนโหวต', 'Vote tally')}</h3>
            <div className="flex flex-col gap-2 font-mono text-[1rem]">
              {p.result.votes.map((v, c) => {
                const terms = p.result.neighbors.filter((n) => n.label === c);
                return (
                  <div key={c} className="flex flex-wrap items-center gap-3">
                    <span className="w-24" style={{ color: CLASS_COLORS[c] }}>{p.classNames[c]}</span>
                    <span className="text-muted">{terms.length ? terms.map((n) => (p.model.weighting === 'distance' ? fmt(n.weight, 2) : '1')).join(' + ') : '0'} =</span>
                    <b>{p.model.weighting === 'distance' ? fmt(v, 3) : v}</b>
                    {c === p.result.predicted && <span className="chip border-cyan text-cyan">{t('ชนะ', 'winner')}</span>}
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      ),
    },
    {
      id: 'eval', label: t('การประเมินผล', 'Evaluation'), content: (
        <div className="grid gap-10 lg:grid-cols-[1.4fr_1fr]">
          <div>
            <h3 className="label-xs mb-2">{t('ความแม่นยำเทียบกับ K (คำนวณจริงทุกค่า K)', 'Accuracy vs K (computed for every K)')}</h3>
            <LinePlot xTitle="K" yTitle={t('ความแม่นยำ', 'accuracy')} yRange={[0, 1.02]} markerX={p.model.k} series={[
              { name: 'train', x: p.curve.map((c) => c.k), y: p.curve.map((c) => c.train), color: '#94A3B8', dash: 'dot' },
              { name: 'test', x: p.curve.map((c) => c.k), y: p.curve.map((c) => c.test), color: '#00D9FF' },
            ]} />
            <p className="mt-2 text-[1.0625rem] text-muted">{t(`K ที่ test ดีที่สุดในกราฟคือ ${bestK.k} (${pct(bestK.test)}) แต่การเลือก K จาก test set ทำให้ประเมินเกินจริง ควรใช้ validation หรือ cross-validation`, `The best test K on this chart is ${bestK.k} (${pct(bestK.test)}), but choosing K from the test set inflates the estimate; use validation or cross-validation.`)}</p>
          </div>
          <div className="flex flex-col gap-5">
            <div className="grid grid-cols-2 gap-3 font-mono">
              {([[t('ความแม่นยำ', 'Accuracy'), p.report.accuracy], ['Precision (macro)', p.report.precision], ['Recall (macro)', p.report.recall], ['F1 (macro)', p.report.f1]] as [string, number][]).map(([k, v]) => (
                <div key={k} className="rounded-lg border border-line px-3 py-2"><div className="text-[0.875rem] text-muted">{k}</div><div className="num text-lg">{pct(v)}</div></div>
              ))}
            </div>
            <div>
              <h3 className="label-xs mb-2">Confusion matrix (test, n = {p.report.support})</h3>
              <ConfusionMatrix m={p.report.confusion} classNames={p.classNames} />
            </div>
          </div>
        </div>
      ),
    },
    { id: 'ex', label: t('แบบฝึกหัด', 'Exercises'), content: <Exercises {...p} /> },
  ];
  return <Tabs tabs={tabs} />;
}

function Exercises(p: LessonProps) {
  const t = useT();
  const [asked, setAsked] = useState(false);
  const [guess, setGuess] = useState<number | null>(null);
  const [score, setScore] = useState({ ok: 0, n: 0 });
  const newQuestion = () => {
    const v = Array.from({ length: p.dims }, () => (Math.random() * 2 - 1) * p.lim * 0.8);
    p.onPlaceQuery(v);
    setAsked(true); setGuess(null);
  };
  const answer = (c: number) => {
    setGuess(c); p.onReveal();
    setScore((s) => ({ ok: s.ok + (c === p.result.predicted ? 1 : 0), n: s.n + 1 }));
  };
  return (
    <div className="grid gap-10 lg:grid-cols-2">
      <div className="flex flex-col gap-4">
        <h3 className="font-display text-xl font-semibold">{t('ทายผล KNN', 'Predict the KNN class')}</h3>
        <p className="text-[1.09375rem] text-muted">{t('กดปุ่มเพื่อวางจุดคำถามแบบสุ่ม ฉากจะหยุดที่ขั้น "เลือก K จุด" ดูสีของเพื่อนบ้านในฉาก 3D แล้วทายว่าโมเดลจะตอบคลาสไหน', 'Press the button to place a random query. The scene stops at “Select K”. Look at the neighbours’ colours in 3D and guess the class.')}</p>
        <div className="flex items-center gap-3">
          <Button variant="primary" onClick={newQuestion}>{asked ? t('ข้อถัดไป', 'Next question') : t('เริ่ม', 'Start')}</Button>
          <span className="font-mono text-[1rem] text-muted">{t('ถูก', 'correct')} {score.ok}/{score.n}</span>
        </div>
        {asked && (
          <div className="flex flex-wrap gap-2">
            {Array.from({ length: p.nClasses }, (_, c) => (
              <button key={c} type="button" disabled={guess != null} onClick={() => answer(c)}
                className={`rounded-lg border px-4 py-2 text-[1.0625rem] ${guess == null ? 'border-line hover:border-cyan/60' : c === p.result.predicted ? 'border-green bg-green/10' : guess === c ? 'border-red bg-red/10' : 'border-line opacity-50'}`}>
                <span style={{ color: CLASS_COLORS[c] }}>●</span> {p.classNames[c]}
              </button>
            ))}
          </div>
        )}
        {guess != null && (
          <p className={`text-[1.09375rem] ${guess === p.result.predicted ? 'text-green' : 'text-orange'}`}>
            {guess === p.result.predicted ? t('ถูกต้อง ', 'Correct. ') : t('ยังไม่ใช่ ', 'Not quite. ')}
            <span className="text-ink/85">{t('คะแนนโหวต: ', 'Votes: ')}{p.result.votes.map((v, c) => `${p.classNames[c]} ${p.model.weighting === 'distance' ? v.toFixed(2) : v}`).join(' · ')}{p.result.tie ? t(' (เสมอ จึงใช้คลาสของจุดที่ใกล้ที่สุด)', ' (tie, so the nearest point’s class wins)') : ''}</span>
          </p>
        )}
        <Callout tone="warn">{t('โจทย์ท้าทาย: หาตำแหน่งที่ K = 1 กับ K = 15 ให้คำตอบต่างกัน แล้วอธิบายว่าทำไม', 'Challenge: find a spot where K = 1 and K = 15 disagree, and explain why.')}</Callout>
      </div>
      <Quiz id="knn" items={[
        { q: t('ถ้า K = 1 ความแม่นยำบนข้อมูล train จะเป็นเท่าไร (เมื่อไม่มีจุดซ้ำ)', 'With K = 1, what is the training accuracy (no duplicate points)?'), options: ['50%', t('ขึ้นกับ noise', 'Depends on noise'), '100%'], answer: 2, why: t('แต่ละจุดเป็นเพื่อนบ้านที่ใกล้ที่สุดของตัวเอง (ระยะ 0) จึงทำนายถูกเสมอ ดูเส้น train ในกราฟ', 'Each point is its own nearest neighbour (distance 0), so it is always right. See the train curve in the plot.') },
        { q: t('ทำไมต้องปรับมาตรฐานฟีเจอร์ก่อนใช้ KNN', 'Why standardise features before KNN?'), options: [t('เพื่อให้คำนวณเร็วขึ้น', 'To compute faster'), t('เพื่อไม่ให้ฟีเจอร์ที่มีหน่วยใหญ่ครอบงำระยะทาง', 'So large-unit features don’t dominate distance'), t('เพราะ KNN ต้องการข้อมูลแจกแจงปกติ', 'Because KNN needs normal data')], answer: 1, why: t('ระยะทางรวมความต่างทุกฟีเจอร์ ฟีเจอร์ที่ช่วงค่ากว้างกว่าจะมีน้ำหนักมากกว่าโดยไม่ตั้งใจ', 'Distance sums differences across features, so wide-range features silently get more weight.') },
        { q: t('เมื่อเพิ่ม K มาก ๆ เส้นแบ่งการตัดสินใจจะเป็นอย่างไร', 'As K grows large, the decision boundary becomes…'), options: [t('ขรุขระขึ้น', 'More jagged'), t('เรียบขึ้น', 'Smoother'), t('ไม่เปลี่ยน', 'Unchanged')], answer: 1, why: t('การโหวตเฉลี่ยจากจุดจำนวนมาก ทำให้รายละเอียดเล็ก ๆ หายไป อาจนำไปสู่ underfitting', 'Voting averages over many points, washing out small details; this can underfit.') },
        { q: t('ระยะ Manhattan ทำให้ขอบเขตเพื่อนบ้านใน 3 มิติเป็นรูปทรงใด', 'In 3D, Manhattan distance makes the neighbour boundary a…'), options: [t('ทรงกลม', 'Sphere'), t('ทรงแปดหน้า', 'Octahedron'), t('ลูกบาศก์', 'Cube')], answer: 1, why: t('เซตของจุดที่ |x|+|y|+|z| = r คือทรงแปดหน้า ส่วนลูกบาศก์เกิดจากระยะ Chebyshev (p → ∞)', 'Points with |x|+|y|+|z| = r form an octahedron; a cube comes from Chebyshev distance (p → ∞).') },
      ]} />
    </div>
  );
}
