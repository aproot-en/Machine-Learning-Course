import { useMemo, useState, type ReactNode } from 'react';
import { VISION_IDS, VISION_MODELS } from '../../content/visionModels';
import { VisionTheory } from '../vision/VisionTheory';
import { Link } from 'react-router-dom';
import { useT, type Text } from '../../i18n/i18n';
import { LinePlot, Tex, type Series } from '../../components/common';
import { Slider, fmt } from '../../components/ui';
import { makeDataset, trainTestSplit } from '../../data/datasets';
import { accuracyByK, knnPredict } from '../../ml/knn';
import { SMO } from '../../ml/svm';
import { GDRegression, makeRegressionData } from '../../ml/regression';
import { buildForest, buildTree, predictForest, predictTree } from '../../ml/tree';
import { initCentroids, kmeansNext, silhouette, type KMeansState } from '../../ml/clustering';
import { runOptimizer, SURFACES } from '../../ml/optim';
import { activate, activationBackward } from '../../ml/nn/activations';
import { Trainer } from '../../ml/nn/trainer';
import type { Activation, LayerSpec } from '../../ml/nn/types';
import { SeqModel, makeSeqBatch } from '../../ml/sequence';
import { positionalEncoding } from '../../ml/transformer';
import { Rng } from '../../lib/random';
import { CLASS_COLORS } from '../../engine3d/palette';
import { AttentionDiagram, BackpropDiagram, ConvDiagram, DbscanDiagram, ForestDiagram, KMeansStepDiagram, KnnDiagram, LstmCellDiagram, NeuronDiagram, RnnUnrolledDiagram, SvmMarginDiagram, TreeDiagram } from './diagrams';

/* ---------- building blocks ---------- */
const linspace = (a: number, b: number, n: number) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));
const C = { cyan: '#00D9FF', orange: '#F59E0B', purple: '#A855F7', green: '#22C55E', red: '#EF4444', blue: '#3B82F6', muted: '#94A3B8' };

export const H = ({ children }: { children: ReactNode }) => <h2 className="mb-3 mt-10 font-display text-2xl font-semibold first:mt-0">{children}</h2>;
export const P = ({ children }: { children: ReactNode }) => <p className="my-3 max-w-[72ch] text-[1.15625rem] leading-[1.85] text-ink/90">{children}</p>;
const Eq = ({ children }: { children: string }) => <div className="my-3 rounded-lg border border-line bg-surface/50 px-4 py-2"><Tex block>{children}</Tex></div>;
const Steps = ({ items }: { items: string[] }) => <ol className="my-3 max-w-[72ch] list-decimal space-y-1.5 pl-6 text-[1.125rem] leading-7 marker:font-mono marker:text-cyan">{items.map((s, i) => <li key={i}>{s}</li>)}</ol>;
function ProsCons({ pros, cons }: { pros: string[]; cons: string[] }) {
  const t = useT();
  return (
    <div className="my-4 grid max-w-4xl gap-4 sm:grid-cols-2">
      <div className="rounded-lg border border-green/30 p-4"><div className="label-xs mb-2 text-green">{t('ข้อดี', 'Strengths')}</div><ul className="list-disc space-y-1 pl-5 text-[1.09375rem]">{pros.map((p) => <li key={p}>{p}</li>)}</ul></div>
      <div className="rounded-lg border border-red/30 p-4"><div className="label-xs mb-2 text-red">{t('ข้อจำกัด', 'Limitations')}</div><ul className="list-disc space-y-1 pl-5 text-[1.09375rem]">{cons.map((p) => <li key={p}>{p}</li>)}</ul></div>
    </div>
  );
}
function Figure({ title, caption, children, controls }: { title: string; caption: ReactNode; children: ReactNode; controls?: ReactNode }) {
  return (
    <figure className="my-6 rounded-xl border border-line bg-surface/50 p-4">
      <figcaption className="label-xs mb-2">{title}</figcaption>
      {children}
      {controls && <div className="mt-3 grid max-w-xl gap-3 sm:grid-cols-2">{controls}</div>}
      <p className="mt-2 text-[1.03125rem] leading-6 text-muted">{caption}</p>
    </figure>
  );
}
function LabLink({ to, label }: { to: string; label: string }) {
  const t = useT();
  return <div className="mt-10 border-t border-line pt-5 text-[1.0625rem] text-muted">{t('ทดลองในฉาก 3 มิติ:', 'Try it in 3D:')} <Link to={to} className="text-cyan hover:underline">{label} →</Link></div>;
}

/* ================= lessons ================= */

function Basics() {
  const t = useT();
  const [lr, setLr] = useState(0.3);
  const gd = useMemo(() => { const pts = [-2.5]; for (let i = 0; i < 12; i++) { const w = pts[pts.length - 1]; pts.push(w - lr * 2 * (w - 1.5)); } return pts; }, [lr]);
  const J = (w: number) => (w - 1.5) ** 2 + 0.5;
  const ws = linspace(-3, 6, 120);
  const curve = useMemo(() => {
    const ds = makeDataset('moons', { n: 800, classes: 2, noise: 0.5, seed: 7 });
    const sp = trainTestSplit(ds.y, 0.3, 1);
    const Xte = sp.test.map((i) => ds.X[i]), yte = sp.test.map((i) => ds.y[i]);
    const order = new Rng(3).shuffle([...sp.train]);
    const sizes = [6, 10, 15, 22, 32, 45, 65, 90, 130, 180, 250, 330, 450, 560];
    return sizes.map((n) => {
      const idx = order.slice(0, n), Xtr = idx.map((i) => ds.X[i]), ytr = idx.map((i) => ds.y[i]);
      const o = { k: 5, metric: 'euclidean' as const, p: 2, weighting: 'uniform' as const };
      const acc = (X: number[][], y: number[]) => X.filter((x, i) => knnPredict(Xtr, ytr, 2, x, o).predicted === y[i]).length / X.length;
      return { n, train: acc(Xtr, ytr), test: acc(Xte, yte) };
    });
  }, []);
  return (
    <>
      <H>{t('การเรียนรู้คือการลด loss', 'Learning means reducing a loss')}</H>
      <P>{t('โมเดล Machine Learning มีพารามิเตอร์ (เช่นน้ำหนัก w) ที่ต้องปรับให้เหมาะกับข้อมูล เราวัดความผิดพลาดด้วยฟังก์ชัน loss J(w) แล้วหาค่า w ที่ทำให้ J ต่ำที่สุด สำหรับปัญหาง่าย ๆ เช่น linear regression ค่า loss เป็นรูปพาราโบลา (convex) จึงมีจุดต่ำสุดเพียงจุดเดียว', 'A machine-learning model has parameters (e.g. weights w) that must be fitted to data. We measure the error with a loss function J(w) and look for the w that makes J smallest. For simple problems such as linear regression the loss is a parabola (convex), so it has a single minimum.')}</P>
      <Eq>{'J(w) = \\frac1n\\sum_{i}(\\hat y_i - y_i)^2,\\qquad w \\leftarrow w - \\eta\\,\\frac{dJ}{dw}'}</Eq>
      <Figure title={t('Gradient descent บนพาราโบลา J(w) = (w − 1.5)² + 0.5', 'Gradient descent on the parabola J(w) = (w − 1.5)² + 0.5')}
        controls={<Slider label="learning rate η" value={lr} min={0.05} max={1.05} step={0.05} format={(v) => v.toFixed(2)} onChange={setLr} />}
        caption={t(`แต่ละจุดคือหนึ่งก้าว: w ใหม่ = w − η·2(w − 1.5) η เล็กเดินช้าแต่มั่นคง η ใกล้ 1 จะเด้งข้ามไปมา และ η > 1 ลู่ออก (ตอนนี้ w สุดท้าย = ${fmt(gd[gd.length - 1], 3)})`, `Each dot is one step: new w = w − η·2(w − 1.5). Small η is slow but steady; η near 1 bounces across; η > 1 diverges (final w now = ${fmt(gd[gd.length - 1], 3)}).`)}>
        <LinePlot xTitle="w" yTitle="J(w)" height={300} xRange={[-3.2, 6.2]} yRange={[0, 22]} series={[
          { name: 'J(w)', x: ws, y: ws.map(J), color: C.muted },
          { name: t('ก้าวของ GD', 'GD steps'), x: gd, y: gd.map(J), color: C.cyan, mode: 'lines+markers', size: 9 },
        ]} />
      </Figure>
      <H>{t('Learning curve: ความแม่นยำเข้าใกล้ 1 เมื่อข้อมูลมากขึ้น', 'Learning curve: accuracy approaches 1 with more data')}</H>
      <P>{t('Learning curve แสดงความแม่นยำเทียบกับจำนวนข้อมูลฝึก (หรือจำนวนรอบการฝึก) เส้น train เริ่มสูงเพราะจำข้อมูลน้อยได้ง่าย ส่วนเส้น test เริ่มต่ำแล้วโค้งขึ้นเข้าหาเพดาน เมื่อสองเส้นเข้าใกล้กันแปลว่าโมเดลเริ่ม generalize ได้ดี ช่องว่างที่เหลือคือ variance และระยะห่างจาก 1 คือ bias + noise ของข้อมูล', 'A learning curve plots accuracy against training-set size (or training epochs). The train curve starts high because few points are easy to memorise; the test curve starts low and bends up toward a ceiling. When the two meet, the model generalises; the remaining gap is variance, and the distance from 1 is bias plus data noise.')}</P>
      <Figure title={t('Learning curve ของ KNN (K = 5) บน Two Moons — คำนวณจริง', 'Learning curve of KNN (K = 5) on Two Moons — computed live')}
        caption={t('แกน x คือจำนวนตัวอย่างฝึก ชุด test คงที่ 240 จุด ความแม่นยำ test ไต่ขึ้นตามเส้นโค้งเข้าใกล้ 1 แต่ไม่ถึงเพราะข้อมูลมี noise', 'x is the number of training samples; the test set is fixed at 240 points. Test accuracy climbs along a curve toward 1 but never reaches it because the data is noisy.')}>
        <LinePlot xTitle={t('จำนวนตัวอย่างฝึก', 'training samples')} yTitle="accuracy" yRange={[0.4, 1.02]} series={[
          { name: 'train', x: curve.map((c) => c.n), y: curve.map((c) => c.train), color: C.muted, mode: 'lines+markers', dash: 'dot' },
          { name: 'test', x: curve.map((c) => c.n), y: curve.map((c) => c.test), color: C.cyan, mode: 'lines+markers' },
          { name: '1.0', x: [0, 570], y: [1, 1], color: C.green, dash: 'dash' },
        ]} />
      </Figure>
      <H>{t('Overfitting และ Underfitting', 'Overfitting and underfitting')}</H>
      <P>{t('Underfitting: โมเดลง่ายเกินไป ทั้ง train และ test ต่ำ Overfitting: โมเดลซับซ้อนเกินไป train สูงมากแต่ test ต่ำ เราแบ่งข้อมูลเป็น train / validation / test เพื่อวัดสิ่งนี้อย่างซื่อตรง และใช้ regularization, early stopping หรือข้อมูลเพิ่มเพื่อแก้ overfitting', 'Underfitting: the model is too simple, so both train and test are low. Overfitting: too complex, with very high train but low test accuracy. We split data into train / validation / test to measure this honestly, and use regularisation, early stopping or more data to fight overfitting.')}</P>
    </>
  );
}

function KnnLesson() {
  const t = useT();
  const curve = useMemo(() => {
    const ds = makeDataset('moons', { n: 300, classes: 2, noise: 0.35, seed: 3 });
    const sp = trainTestSplit(ds.y, 0.3, 2);
    const g = (ix: number[]) => [ix.map((i) => ds.X[i]), ix.map((i) => ds.y[i])] as const;
    const [Xtr, ytr] = g(sp.train), [Xte, yte] = g(sp.test);
    return accuracyByK(Xtr, ytr, Xte, yte, 2, { k: 1, metric: 'euclidean', p: 2, weighting: 'uniform' }, 40);
  }, []);
  return (
    <>
      <H>{t('แนวคิด', 'Idea')}</H>
      <P>{t('K-Nearest Neighbors ไม่มีขั้นตอนฝึก มันเก็บข้อมูล train ทั้งหมด เมื่อต้องทำนายจุดใหม่ จะหาจุด train ที่ใกล้ที่สุด K จุด แล้วให้โหวต สมมติฐานคือ "สิ่งที่คล้ายกันอยู่ใกล้กัน" ในพื้นที่ฟีเจอร์', 'K-Nearest Neighbors has no training step: it stores the whole training set. To predict a new point it finds the K closest training points and lets them vote. The assumption is that similar things sit close together in feature space.')}</P>
      <Eq>{'d(\\mathbf x,\\mathbf q) = \\Big(\\sum_i |x_i-q_i|^p\\Big)^{1/p}\\quad (p=2:\\ \\text{Euclidean},\\ p=1:\\ \\text{Manhattan})'}</Eq>
      <Eq>{'\\hat y = \\arg\\max_c \\sum_{\\mathbf x_j\\in N_K(\\mathbf q)} \\mathbb 1[y_j=c]'}</Eq>
      <KnnDiagram />
      <H>{t('ขั้นตอน', 'Algorithm')}</H>
      <Steps items={[t('คำนวณระยะจากจุดคำถามไปยังทุกจุด train', 'Compute the distance from the query to every training point'), t('เรียงระยะจากน้อยไปมาก', 'Sort distances ascending'), t('เลือก K จุดแรก', 'Take the first K'), t('นับคะแนนโหวตของแต่ละคลาส คลาสที่มากที่สุดคือคำตอบ', 'Count votes per class; the largest wins')]} />
      <Figure title={t('ความแม่นยำเทียบกับ K (คำนวณจริงบน Two Moons ที่มี noise)', 'Accuracy vs K (computed on noisy Two Moons)')}
        caption={t('K = 1 ได้ train = 1.0 เสมอ (จำทุกจุด) แต่ test ต่ำกว่า = overfitting เมื่อ K โตขึ้น test ดีขึ้นก่อน แล้วลดลงเมื่อเส้นแบ่งเรียบเกินไป = underfitting', 'K = 1 always scores 1.0 on train (it memorises) but lower on test: overfitting. As K grows, test improves first, then drops when the boundary becomes too smooth: underfitting.')}>
        <LinePlot xTitle="K" yTitle="accuracy" yRange={[0.6, 1.02]} series={[
          { name: 'train', x: curve.map((c) => c.k), y: curve.map((c) => c.train), color: C.muted, dash: 'dot' },
          { name: 'test', x: curve.map((c) => c.k), y: curve.map((c) => c.test), color: C.cyan, mode: 'lines+markers', size: 5 },
        ]} />
      </Figure>
      <ProsCons pros={[t('เข้าใจง่าย ไม่มีขั้นตอนฝึก', 'Simple, no training'), t('รองรับหลายคลาสโดยธรรมชาติ', 'Naturally multi-class'), t('เส้นแบ่งยืดหยุ่นตามข้อมูล', 'Flexible boundaries')]}
        cons={[t('ทำนายช้า O(n·d) ต่อจุด', 'Slow prediction, O(n·d) per point'), t('ต้องปรับสเกลฟีเจอร์', 'Needs feature scaling'), t('แย่ลงในมิติสูง (curse of dimensionality)', 'Degrades in high dimensions')]} />
      <LabLink to="/algorithms/knn" label="KNN Lab" />
    </>
  );
}

function LinearLesson() {
  const t = useT();
  const [lr, setLr] = useState(0.1);
  const data = useMemo(() => makeRegressionData(40, 1, 0.4, 5), []);
  const run = useMemo(() => {
    const m = new GDRegression(data.X, data.y, { kind: 'linear', optimizer: { name: 'sgd', lr, momentum: 0.9, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }, batchSize: 0, l2: 0, l1: 0, seed: 1, init: 'zeros' }, { w: [-1.5], b: 0.5 });
    for (let i = 0; i < 40; i++) { m.step(); if (!Number.isFinite(m.history[m.history.length - 1].loss) || m.history[m.history.length - 1].loss > 1e4) break; }
    return m;
  }, [data, lr]);
  const H0 = run.history, b = H0[H0.length - 1].b;
  const ws = linspace(-2, 4, 120);
  const xs = linspace(-2, 2, 2);
  return (
    <>
      <H>{t('แนวคิด', 'Idea')}</H>
      <P>{t('Linear regression ทำนายค่าตัวเลขด้วยเส้นตรง ŷ = w·x + b โดยเลือก w และ b ที่ทำให้ผลรวมกำลังสองของ residual (ŷ − y) น้อยที่สุด เพราะ loss เป็นฟังก์ชันกำลังสองของ w จึงเป็นพาราโบลาที่มีจุดต่ำสุดเดียว', 'Linear regression predicts a number with a line ŷ = w·x + b, choosing w and b to minimise the sum of squared residuals (ŷ − y). The loss is quadratic in w, so it is a parabola with one minimum.')}</P>
      <Eq>{'J(w,b) = \\frac1n\\sum_i (w x_i + b - y_i)^2,\\quad \\frac{\\partial J}{\\partial w} = \\frac2n\\sum_i (\\hat y_i - y_i)x_i,\\quad \\frac{\\partial J}{\\partial b} = \\frac2n\\sum_i(\\hat y_i - y_i)'}</Eq>
      <Eq>{'\\text{Normal equation: } \\boldsymbol\\theta^* = (X^\\top X)^{-1}X^\\top \\mathbf y'}</Eq>
      <Figure title={t('พาราโบลา J(w) (ตรึง b ไว้) และก้าวของ gradient descent', 'The parabola J(w) (b held) and gradient-descent steps')}
        controls={<Slider label="η" value={lr} min={0.02} max={0.6} step={0.02} format={(v) => v.toFixed(2)} onChange={setLr} />}
        caption={t('จุดสีฟ้าคือ w ในแต่ละรอบ เริ่มจาก w = −1.5 แล้วไหลลงพาราโบลาสู่จุดต่ำสุด ลองเพิ่ม η เพื่อดูการแกว่ง', 'Cyan dots are w at each iteration, starting at −1.5 and sliding down the parabola to the minimum. Raise η to see oscillation.')}>
        <LinePlot xTitle="w" yTitle="J(w)" height={280} series={[
          { name: 'J(w)', x: ws, y: ws.map((w) => run.loss([w], b)), color: C.muted },
          { name: t('ก้าวของ GD', 'GD steps'), x: H0.map((h) => h.w[0]), y: H0.map((h) => run.loss([h.w[0]], b)), color: C.cyan, mode: 'lines+markers', size: 7 },
        ]} />
      </Figure>
      <Figure title={t('Loss ลดลงตามรอบ (เส้นโค้งการเรียนรู้)', 'Loss over iterations (learning curve)')}
        caption={t(`ผลสุดท้าย: ŷ = ${fmt(H0[H0.length - 1].w[0], 3)}x + ${fmt(b, 3)} (ค่าจริงที่สร้างข้อมูล: 1.6x + 0.5)`, `Final fit: ŷ = ${fmt(H0[H0.length - 1].w[0], 3)}x + ${fmt(b, 3)} (true generator: 1.6x + 0.5)`)}>
        <div className="grid gap-4 lg:grid-cols-2">
          <LinePlot xTitle="iteration" yTitle="MSE" height={240} series={[{ name: 'MSE', x: H0.map((h) => h.iter), y: H0.map((h) => h.loss), color: C.orange, mode: 'lines+markers', size: 5 }]} />
          <LinePlot xTitle="x" yTitle="y" height={240} series={[
            { name: 'data', x: data.X.map((x) => x[0]), y: data.y, color: C.cyan, mode: 'markers', size: 6 },
            { name: 'ŷ', x: xs, y: xs.map((x) => H0[H0.length - 1].w[0] * x + b), color: C.orange },
          ]} />
        </div>
      </Figure>
      <ProsCons pros={[t('ตีความง่าย ค่า w บอกผลของแต่ละฟีเจอร์', 'Interpretable: w is each feature’s effect'), t('มีคำตอบแบบปิด', 'Closed-form solution')]} cons={[t('จับความสัมพันธ์ไม่เป็นเส้นตรงไม่ได้ (เว้นแต่สร้างฟีเจอร์เพิ่ม)', 'Misses non-linear relations unless features are engineered'), t('ไวต่อ outlier เพราะยกกำลังสอง', 'Sensitive to outliers due to squaring')]} />
      <LabLink to="/algorithms/linear-regression" label="Regression Lab" />
    </>
  );
}

function LogisticLesson() {
  const t = useT();
  const [w, setW] = useState(1.5);
  const zs = linspace(-6, 6, 160), ps = linspace(0.005, 0.995, 120);
  return (
    <>
      <H>{t('แนวคิด', 'Idea')}</H>
      <P>{t('Logistic regression ใช้สำหรับการจำแนก 2 คลาส คำนวณคะแนนเชิงเส้น z = w·x + b แล้วแปลงเป็นความน่าจะเป็นด้วย sigmoid ซึ่งเป็นเส้นโค้งรูปตัว S ที่เข้าใกล้ 1 เมื่อ z มาก และเข้าใกล้ 0 เมื่อ z ติดลบมาก แต่ไม่เคยแตะ', 'Logistic regression classifies two classes. It computes a linear score z = w·x + b and turns it into a probability with the sigmoid, an S-shaped curve that approaches 1 for large z and 0 for very negative z without ever touching them.')}</P>
      <Eq>{'p = \\sigma(z) = \\frac{1}{1+e^{-z}},\\qquad \\lim_{z\\to\\infty}\\sigma(z) = 1'}</Eq>
      <Figure title={t('Sigmoid: เส้นโค้งที่เข้าใกล้ 1', 'Sigmoid: a curve that approaches 1')}
        controls={<Slider label={t('ความชัน w', 'Slope w')} value={w} min={0.2} max={5} step={0.1} format={(v) => v.toFixed(1)} onChange={setW} />}
        caption={t('w ใหญ่ทำให้ช่วงเปลี่ยนจาก 0 ไป 1 แคบลง (โมเดลมั่นใจขึ้น) เส้นประคือเส้นกำกับ p = 1 และ p = 0.5 ซึ่งเป็นเส้นแบ่งการตัดสินใจ', 'A larger w narrows the 0→1 transition (a more confident model). Dashed lines are the asymptote p = 1 and the decision threshold p = 0.5.')}>
        <LinePlot xTitle="x" yTitle="p(y=1)" yRange={[-0.05, 1.08]} series={[
          { name: `σ(${w.toFixed(1)}x)`, x: zs, y: zs.map((z) => 1 / (1 + Math.exp(-w * z))), color: C.cyan },
          { name: 'p = 1', x: [-6, 6], y: [1, 1], color: C.green, dash: 'dash' },
          { name: 'p = 0.5', x: [-6, 6], y: [0.5, 0.5], color: C.muted, dash: 'dot' },
        ]} />
      </Figure>
      <H>{t('Cross-entropy loss', 'Cross-entropy loss')}</H>
      <Eq>{'L = -\\big[y\\log p + (1-y)\\log(1-p)\\big],\\qquad \\frac{\\partial L}{\\partial z} = p - y'}</Eq>
      <Figure title={t('โทษของการทายผิด', 'The penalty for being wrong')} caption={t('เมื่อ y = 1 แต่โมเดลให้ p ใกล้ 0 ค่า loss พุ่งไปอนันต์ cross-entropy จึงลงโทษความมั่นใจผิด ๆ หนักมาก และให้ gradient p − y ที่เรียบง่าย', 'When y = 1 but the model gives p near 0, the loss shoots to infinity; cross-entropy punishes confident mistakes heavily and yields the simple gradient p − y.')}>
        <LinePlot xTitle="p" yTitle="loss" yRange={[0, 5]} series={[
          { name: 'y = 1: −log p', x: ps, y: ps.map((p) => -Math.log(p)), color: CLASS_COLORS[1] },
          { name: 'y = 0: −log(1−p)', x: ps, y: ps.map((p) => -Math.log(1 - p)), color: CLASS_COLORS[0] },
        ]} />
      </Figure>
      <ProsCons pros={[t('ให้ความน่าจะเป็น', 'Outputs probabilities'), t('เร็ว ตีความได้', 'Fast and interpretable')]} cons={[t('เส้นแบ่งเป็นเส้นตรงเท่านั้น', 'Only linear boundaries'), t('หลายคลาสต้องใช้ softmax หรือ one-vs-rest', 'Multi-class needs softmax or one-vs-rest')]} />
      <LabLink to="/algorithms/logistic-regression" label="Regression Lab" />
    </>
  );
}

function SvmLesson() {
  const t = useT();
  const m = linspace(-2, 3, 200);
  const svByC = useMemo(() => {
    const ds = makeDataset('moons', { n: 120, classes: 2, noise: 0.25, seed: 2 });
    const y = ds.y.map((c) => (c ? 1 : -1));
    return [0.03, 0.1, 0.3, 1, 3, 10, 30, 100].map((Cv) => {
      const s = new SMO(ds.X, y, { kernel: 'rbf', C: Cv, gamma: 1, degree: 3, coef0: 1, tol: 1e-3, classWeight: [1, 1] }, 1);
      while (!s.done) s.pass();
      const acc = ds.X.filter((x, i) => Math.sign(s.decision(x)) === y[i]).length / ds.X.length;
      return { C: Cv, sv: s.supportVectors().length, acc };
    });
  }, []);
  return (
    <>
      <H>{t('แนวคิด', 'Idea')}</H>
      <P>{t('SVM หาเส้นแบ่งที่ห่างจากจุดที่ใกล้ที่สุดของทั้งสองคลาสมากที่สุด (maximum margin) แนวคิดคือเส้นที่ห่างจากข้อมูลมากจะทนต่อข้อมูลใหม่ได้ดีกว่า จุดที่อยู่บนขอบ margin เรียกว่า support vector', 'An SVM finds the boundary that is as far as possible from the nearest points of both classes (maximum margin), on the idea that a boundary far from the data generalises better. Points on the margin are support vectors.')}</P>
      <SvmMarginDiagram />
      <Eq>{'\\min_{\\mathbf w,b}\\ \\tfrac12\\|\\mathbf w\\|^2 + C\\sum_i \\max\\big(0,\\, 1 - y_i(\\mathbf w\\cdot\\mathbf x_i + b)\\big)'}</Eq>
      <Figure title={t('Hinge loss เทียบกับ 0-1 loss และ logistic loss', 'Hinge loss vs 0-1 loss and logistic loss')} caption={t('แกน x คือ margin m = y·f(x) hinge loss เป็น 0 ทันทีที่ m ≥ 1 (จุดอยู่นอก margin) จึงไม่สนใจจุดที่จำแนกถูกอย่างมั่นใจ ต่างจาก logistic loss ที่ลดลงแต่ไม่ถึงศูนย์', 'x is the margin m = y·f(x). Hinge loss is exactly 0 once m ≥ 1 (outside the margin), so confidently correct points are ignored, unlike the logistic loss which shrinks but never hits zero.')}>
        <LinePlot xTitle="m = y·f(x)" yTitle="loss" yRange={[0, 3.2]} series={[
          { name: '0-1', x: m, y: m.map((v) => (v < 0 ? 1 : 0)), color: C.muted },
          { name: 'hinge', x: m, y: m.map((v) => Math.max(0, 1 - v)), color: C.cyan },
          { name: 'logistic', x: m, y: m.map((v) => Math.log2(1 + Math.exp(-v))), color: C.orange, dash: 'dot' },
        ]} />
      </Figure>
      <H>Kernel trick</H>
      <P>{t('เมื่อข้อมูลแยกด้วยเส้นตรงไม่ได้ SVM แทนผลคูณภายในด้วย kernel K(x, x′) ซึ่งเท่ากับผลคูณภายในในพื้นที่ฟีเจอร์มิติสูงโดยไม่ต้องคำนวณพื้นที่นั้นจริง', 'When data is not linearly separable, SVM replaces dot products with a kernel K(x, x′), equal to a dot product in a high-dimensional feature space that is never computed explicitly.')}</P>
      <Eq>{'f(\\mathbf x) = \\sum_{i\\in SV}\\alpha_i y_i K(\\mathbf x_i,\\mathbf x) + b,\\qquad K_{rbf}(\\mathbf x,\\mathbf x\') = e^{-\\gamma\\|\\mathbf x-\\mathbf x\'\\|^2}'}</Eq>
      <Figure title={t('ผลของ C (RBF, γ = 1, Two Moons) — ฝึกจริงด้วย SMO', 'Effect of C (RBF, γ = 1, Two Moons) — trained with SMO')} caption={t('C เล็ก: ยอมให้ล้ำ margin มาก จึงมี support vector มาก C ใหญ่: เข้มงวด จำนวน support vector ลดลง และความแม่นยำบน train เข้าใกล้ 1', 'Small C tolerates many margin violations, so many support vectors; large C is strict, support vectors drop and train accuracy approaches 1.')}>
        <div className="grid gap-4 lg:grid-cols-2">
          <LinePlot xTitle="C (log)" yTitle="support vectors" height={230} series={[{ name: '#SV', x: svByC.map((s) => Math.log10(s.C)), y: svByC.map((s) => s.sv), color: C.purple, mode: 'lines+markers' }]} />
          <LinePlot xTitle="C (log)" yTitle="train accuracy" height={230} yRange={[0.7, 1.02]} series={[{ name: 'acc', x: svByC.map((s) => Math.log10(s.C)), y: svByC.map((s) => s.acc), color: C.cyan, mode: 'lines+markers' }, { name: '1.0', x: [-1.6, 2.1], y: [1, 1], color: C.green, dash: 'dash' }]} />
        </div>
      </Figure>
      <ProsCons pros={[t('ทำงานดีในมิติสูง', 'Works well in high dimensions'), t('คำตอบขึ้นกับ support vector เท่านั้น', 'Depends only on support vectors')]} cons={[t('ฝึกช้าเมื่อข้อมูลมาก (O(n²)–O(n³))', 'Slow on large datasets (O(n²)–O(n³))'), t('ต้องเลือก kernel, C, γ', 'Needs kernel, C and γ tuning'), t('ไม่ได้ให้ความน่าจะเป็นโดยตรง', 'No direct probabilities')]} />
      <LabLink to="/algorithms/svm" label="SVM Lab" />
    </>
  );
}

function TreeLesson() {
  const t = useT();
  const p = linspace(0, 1, 101);
  const depth = useMemo(() => {
    const ds = makeDataset('moons', { n: 400, classes: 2, noise: 0.35, seed: 9 });
    const sp = trainTestSplit(ds.y, 0.3, 3);
    const Xtr = sp.train.map((i) => ds.X[i]), ytr = sp.train.map((i) => ds.y[i]), Xte = sp.test.map((i) => ds.X[i]), yte = sp.test.map((i) => ds.y[i]);
    return Array.from({ length: 14 }, (_, k) => {
      const tr = buildTree(Xtr, ytr, 2, { maxDepth: k + 1, minSamplesSplit: 2, minSamplesLeaf: 1, criterion: 'gini', maxFeatures: 'all', ccpAlpha: 0 });
      const acc = (X: number[][], y: number[]) => X.filter((x, i) => predictTree(tr, x).cls === y[i]).length / X.length;
      return { d: k + 1, train: acc(Xtr, ytr), test: acc(Xte, yte) };
    });
  }, []);
  return (
    <>
      <H>{t('แนวคิด', 'Idea')}</H>
      <P>{t('Decision tree ถามคำถามแบบใช่/ไม่ใช่ทีละข้อ เช่น "x₁ ≤ 2.5 หรือไม่" แต่ละโหนดเลือกคำถามที่ทำให้กลุ่มลูกบริสุทธิ์ขึ้นมากที่สุด วัดด้วย Gini หรือ entropy', 'A decision tree asks yes/no questions one at a time, e.g. “is x₁ ≤ 2.5?”. Each node picks the question that makes its children purest, measured by Gini or entropy.')}</P>
      <TreeDiagram />
      <Eq>{'\\text{Gini} = 1 - \\sum_k p_k^2,\\quad H = -\\sum_k p_k\\log_2 p_k,\\quad \\Delta = I(S) - \\sum_{c\\in\\{L,R\\}}\\frac{|S_c|}{|S|}I(S_c)'}</Eq>
      <Figure title={t('Impurity ของ 2 คลาสเทียบกับสัดส่วน p (Gini เป็นพาราโบลา)', 'Two-class impurity vs proportion p (Gini is a parabola)')} caption={t('Gini = 2p(1 − p) เป็นพาราโบลาคว่ำ สูงสุดที่ p = 0.5 (ปนกันที่สุด) และเป็น 0 เมื่อ p = 0 หรือ 1 (บริสุทธิ์) entropy มีรูปร่างคล้ายกันแต่สูงสุดที่ 1 บิต', 'Gini = 2p(1 − p) is an upside-down parabola peaking at p = 0.5 (most mixed) and zero at p = 0 or 1 (pure). Entropy is similar but peaks at 1 bit.')}>
        <LinePlot xTitle="p" yTitle="impurity" series={[
          { name: 'Gini 2p(1−p)', x: p, y: p.map((v) => 2 * v * (1 - v)), color: C.cyan },
          { name: 'entropy', x: p, y: p.map((v) => (v <= 0 || v >= 1 ? 0 : -(v * Math.log2(v) + (1 - v) * Math.log2(1 - v)))), color: C.orange },
          { name: 'misclassification', x: p, y: p.map((v) => Math.min(v, 1 - v)), color: C.muted, dash: 'dot' },
        ]} />
      </Figure>
      <Figure title={t('ความลึกกับ overfitting (ฝึกจริงบน Two Moons ที่มี noise)', 'Depth and overfitting (trained on noisy Two Moons)')} caption={t('ความแม่นยำ train ไต่เข้าใกล้ 1 เมื่อต้นไม้ลึกขึ้น เพราะต้นไม้แบ่งจนทุกใบบริสุทธิ์ ส่วน test สูงสุดที่ความลึกปานกลางแล้วลดลง นี่คือ overfitting', 'Train accuracy climbs toward 1 as the tree deepens because it keeps splitting until every leaf is pure, while test peaks at moderate depth then falls: overfitting.')}>
        <LinePlot xTitle="max depth" yTitle="accuracy" yRange={[0.6, 1.02]} series={[
          { name: 'train', x: depth.map((d) => d.d), y: depth.map((d) => d.train), color: C.muted, mode: 'lines+markers' },
          { name: 'test', x: depth.map((d) => d.d), y: depth.map((d) => d.test), color: C.cyan, mode: 'lines+markers' },
          { name: '1.0', x: [1, 14], y: [1, 1], color: C.green, dash: 'dash' },
        ]} />
      </Figure>
      <ProsCons pros={[t('อ่านกฎได้ อธิบายได้', 'Readable, explainable rules'), t('ไม่ต้องปรับสเกล', 'No scaling needed'), t('ใช้ได้กับฟีเจอร์หลายชนิด', 'Handles mixed features')]} cons={[t('overfit ง่าย', 'Overfits easily'), t('ไม่เสถียร ข้อมูลเปลี่ยนนิดเดียวต้นไม้เปลี่ยนมาก', 'Unstable to small data changes'), t('เส้นแบ่งตั้งฉากกับแกนเท่านั้น', 'Axis-aligned boundaries only')]} />
      <LabLink to="/algorithms/decision-tree" label="Tree Lab" />
    </>
  );
}

function ForestLesson() {
  const t = useT();
  const curve = useMemo(() => {
    const ds = makeDataset('moons', { n: 400, classes: 2, noise: 0.35, seed: 9 });
    const sp = trainTestSplit(ds.y, 0.3, 3);
    const Xtr = sp.train.map((i) => ds.X[i]), ytr = sp.train.map((i) => ds.y[i]), Xte = sp.test.map((i) => ds.X[i]), yte = sp.test.map((i) => ds.y[i]);
    const f = buildForest(Xtr, ytr, 2, { maxDepth: 10, minSamplesSplit: 2, minSamplesLeaf: 1, criterion: 'gini', maxFeatures: 'sqrt', ccpAlpha: 0 }, 40, true, 5);
    const single = Xte.filter((x, i) => predictTree(f.trees[0], x).cls === yte[i]).length / Xte.length;
    return { pts: [1, 2, 3, 5, 8, 12, 16, 20, 25, 30, 35, 40].map((n) => ({ n, acc: Xte.filter((x, i) => predictForest(f, x, 2, n).cls === yte[i]).length / Xte.length })), single, oob: f.oob };
  }, []);
  return (
    <>
      <H>{t('แนวคิด', 'Idea')}</H>
      <P>{t('ต้นไม้ต้นเดียวมี variance สูง Random forest ฝึกต้นไม้จำนวนมาก แต่ละต้นเห็นข้อมูลชุด bootstrap ที่ต่างกันและสุ่มฟีเจอร์ในแต่ละโหนด เมื่อเอาผลมาโหวตรวมกัน ความผิดพลาดที่ไม่สัมพันธ์กันจะหักล้างกัน', 'A single tree has high variance. A random forest trains many trees, each on a different bootstrap sample with random features at every node. Voting averages out their uncorrelated errors.')}</P>
      <ForestDiagram />
      <Eq>{'\\hat y(\\mathbf x) = \\text{mode}\\{T_1(\\mathbf x),\\dots,T_B(\\mathbf x)\\},\\qquad \\mathrm{Var}(\\bar T) = \\rho\\sigma^2 + \\frac{1-\\rho}{B}\\sigma^2'}</Eq>
      <Figure title={t('ความแม่นยำ test เมื่อเพิ่มจำนวนต้นไม้ (ฝึกจริง)', 'Test accuracy as trees are added (trained live)')} caption={t(`เส้นโค้งไต่ขึ้นแล้วเข้าสู่ระดับคงที่ เพราะ variance ลดลงตาม 1/B แต่ไม่ต่ำกว่า ρσ² · ต้นไม้ต้นเดียว = ${fmt(curve.single * 100, 1)}% · OOB = ${curve.oob != null ? fmt(curve.oob * 100, 1) : '—'}%`, `The curve rises then levels off: variance falls like 1/B but not below ρσ² · one tree = ${fmt(curve.single * 100, 1)}% · OOB = ${curve.oob != null ? fmt(curve.oob * 100, 1) : '—'}%`)}>
        <LinePlot xTitle={t('จำนวนต้นไม้', 'number of trees')} yTitle="test accuracy" yRange={[0.7, 1]} series={[
          { name: 'forest', x: curve.pts.map((p) => p.n), y: curve.pts.map((p) => p.acc), color: C.cyan, mode: 'lines+markers' },
          { name: t('ต้นเดียว', 'single tree'), x: [1, 40], y: [curve.single, curve.single], color: C.muted, dash: 'dot' },
        ]} />
      </Figure>
      <ProsCons pros={[t('แม่นยำและทนทาน', 'Accurate and robust'), t('มี OOB error ให้ใช้แทน validation', 'OOB error as a free validation estimate'), t('มีค่า feature importance', 'Feature importance')]} cons={[t('ตีความยากกว่าต้นเดียว', 'Harder to interpret'), t('ใช้หน่วยความจำและเวลามากขึ้น', 'More memory and time')]} />
      <LabLink to="/algorithms/random-forest" label="Tree Lab" />
    </>
  );
}

function KMeansLesson() {
  const t = useT();
  const res = useMemo(() => {
    const ds = makeDataset('blobs2d', { n: 300, classes: 4, noise: 0.15, seed: 4 });
    const run = (k: number) => {
      let s: KMeansState = { iter: 0, phase: 'assign', centroids: initCentroids(ds.X, k, 'kmeans++', 2), assign: ds.X.map(() => -1), inertia: null, changed: 0, displacement: 0, converged: false };
      const hist: number[] = [];
      for (let g = 0; g < 200 && !s.converged; g++) { s = kmeansNext(ds.X, s, 1e-6, 100); if (s.phase === 'update' && s.inertia != null) hist.push(s.inertia); }
      return { s, hist };
    };
    const ks = [1, 2, 3, 4, 5, 6, 7, 8].map((k) => { const r = run(k); return { k, wcss: r.s.inertia ?? 0, sil: k > 1 ? silhouette(ds.X, r.s.assign) ?? 0 : 0 }; });
    return { ks, hist: run(4).hist };
  }, []);
  return (
    <>
      <H>{t('แนวคิด', 'Idea')}</H>
      <P>{t('K-Means แบ่งข้อมูลที่ไม่มีป้ายกำกับเป็น K กลุ่ม โดยให้ทุกจุดอยู่ใกล้จุดศูนย์กลาง (centroid) ของกลุ่มตัวเองให้มากที่สุด ทำซ้ำสองขั้นจนลู่เข้า', 'K-Means partitions unlabelled data into K groups so every point is as close as possible to its group’s centre (centroid), repeating two steps until convergence.')}</P>
      <KMeansStepDiagram />
      <Eq>{'\\min_{C,\\boldsymbol\\mu}\\ \\text{WCSS} = \\sum_{k=1}^K\\sum_{\\mathbf x\\in C_k}\\|\\mathbf x - \\boldsymbol\\mu_k\\|^2'}</Eq>
      <Steps items={[t('วาง centroid เริ่มต้น K จุด (k-means++ เลือกให้ห่างกัน)', 'Place K initial centroids (k-means++ spreads them out)'), t('Assign: ทุกจุดเข้ากลุ่มของ centroid ที่ใกล้สุด', 'Assign each point to its nearest centroid'), t('Update: ย้าย centroid ไปที่ค่าเฉลี่ยของกลุ่ม', 'Move each centroid to its group mean'), t('ทำซ้ำจนไม่มีจุดย้ายกลุ่ม', 'Repeat until nothing changes')]} />
      <Figure title={t('Elbow และ silhouette (ข้อมูลจริงมี 4 กลุ่ม)', 'Elbow and silhouette (data truly has 4 groups)')} caption={t('WCSS ลดลงเสมอเมื่อ K เพิ่ม แต่หักศอกชัดที่ K = 4 ส่วน silhouette สูงสุดที่ K เดียวกัน ทั้งสองช่วยเลือก K', 'WCSS always falls as K grows but bends sharply at K = 4, where silhouette also peaks; both help choose K.')}>
        <div className="grid gap-4 lg:grid-cols-2">
          <LinePlot xTitle="K" yTitle="WCSS" height={230} series={[{ name: 'WCSS', x: res.ks.map((k) => k.k), y: res.ks.map((k) => k.wcss), color: C.cyan, mode: 'lines+markers' }]} />
          <LinePlot xTitle="K" yTitle="silhouette" height={230} series={[{ name: 'silhouette', x: res.ks.slice(1).map((k) => k.k), y: res.ks.slice(1).map((k) => k.sil), color: C.orange, mode: 'lines+markers' }]} />
        </div>
      </Figure>
      <Figure title={t('WCSS ในแต่ละรอบ (K = 4)', 'WCSS per iteration (K = 4)')} caption={t('ลดลงทุกรอบจนคงที่ ยืนยันว่าอัลกอริทึมลู่เข้า', 'It falls every iteration until flat, confirming convergence.')}>
        <LinePlot xTitle="iteration" yTitle="WCSS" height={220} series={[{ name: 'WCSS', x: res.hist.map((_, i) => i + 1), y: res.hist, color: C.purple, mode: 'lines+markers' }]} />
      </Figure>
      <ProsCons pros={[t('เร็ว ใช้กับข้อมูลใหญ่ได้', 'Fast, scales to large data'), t('เข้าใจง่าย', 'Easy to understand')]} cons={[t('ต้องกำหนด K', 'K must be chosen'), t('สมมติกลุ่มกลม ขนาดใกล้กัน', 'Assumes round, similar-size clusters'), t('ผลขึ้นกับจุดเริ่ม', 'Depends on initialisation')]} />
      <LabLink to="/algorithms/kmeans" label="Clustering Lab" />
    </>
  );
}

function DbscanLesson() {
  const t = useT();
  const kd = useMemo(() => {
    const ds = makeDataset('moons', { n: 300, classes: 2, noise: 0.08, seed: 4 });
    return ds.X.map((x) => ds.X.map((y) => Math.hypot(x[0] - y[0], x[1] - y[1])).sort((a, b) => a - b)[4]).sort((a, b) => b - a);
  }, []);
  return (
    <>
      <H>{t('แนวคิด', 'Idea')}</H>
      <P>{t('DBSCAN มองคลัสเตอร์เป็นบริเวณที่จุดอยู่หนาแน่น ไม่ต้องกำหนดจำนวนคลัสเตอร์ล่วงหน้า หาคลัสเตอร์รูปร่างใดก็ได้ และแยกจุด noise ออกมาเอง', 'DBSCAN sees clusters as dense regions. It needs no cluster count, finds clusters of any shape and labels outliers as noise.')}</P>
      <DbscanDiagram />
      <Eq>{'N_\\varepsilon(\\mathbf p) = \\{\\mathbf q: d(\\mathbf p,\\mathbf q)\\le\\varepsilon\\},\\qquad \\mathbf p \\text{ core} \\iff |N_\\varepsilon(\\mathbf p)|\\ge \\text{MinPts}'}</Eq>
      <Steps items={[t('เยี่ยมจุดที่ยังไม่ถูกเยี่ยม หาเพื่อนบ้านในรัศมี ε', 'Visit an unvisited point and find its ε-neighbours'), t('ถ้าเพื่อนบ้านน้อยกว่า MinPts ทำเครื่องหมายเป็น noise (ชั่วคราว)', 'If fewer than MinPts, mark it noise (for now)'), t('ถ้าเป็น core ให้เริ่มคลัสเตอร์ใหม่ แล้วขยายผ่านเพื่อนบ้านที่เป็น core ต่อไปเรื่อย ๆ', 'If core, start a cluster and expand through neighbouring core points'), t('จุดที่ไปถึงได้แต่ไม่ใช่ core เป็น border', 'Reachable non-core points become border points')]} />
      <Figure title={t('กราฟ k-distance (k = 5) สำหรับเลือก ε', 'k-distance plot (k = 5) for choosing ε')} caption={t('เรียงระยะถึงเพื่อนบ้านลำดับที่ 5 ของทุกจุดจากมากไปน้อย จุดที่เส้นหักศอกคือค่า ε ที่เหมาะ จุดทางซ้ายที่สูงคือ noise', 'Each point’s distance to its 5th neighbour, sorted descending. The knee suggests ε; the tall left part is noise.')}>
        <LinePlot xTitle={t('จุด (เรียงแล้ว)', 'points (sorted)')} yTitle="5-NN distance" series={[{ name: 'k-distance', x: kd.map((_, i) => i), y: kd, color: C.cyan }]} />
      </Figure>
      <ProsCons pros={[t('หารูปร่างอิสระได้', 'Arbitrary shapes'), t('จัดการ noise ได้', 'Handles noise'), t('ไม่ต้องกำหนด K', 'No K required')]} cons={[t('ไวต่อ ε และ MinPts', 'Sensitive to ε and MinPts'), t('ลำบากเมื่อความหนาแน่นต่างกันมาก', 'Struggles with varying density')]} />
      <LabLink to="/algorithms/dbscan" label="Clustering Lab" />
    </>
  );
}

function GdLesson() {
  const t = useT();
  const xs = linspace(-3, 3, 120);
  const paths = [0.1, 0.45, 0.95, 1.05].map((eta) => { const p = [2.5]; for (let i = 0; i < 25; i++) p.push(p[p.length - 1] * (1 - 2 * eta)); return { eta, p }; });
  const cmp = useMemo(() => {
    const s = SURFACES.find((x) => x.id === 'rosenbrock')!;
    const base = { lr: 0.002, momentum: 0.9, beta1: 0.9, beta2: 0.999, epsilon: 1e-8, noise: 2, seed: 1 };
    return (['gd', 'momentum', 'rmsprop', 'adam'] as const).map((name, k) => ({ name, color: [C.muted, C.purple, C.orange, C.cyan][k], path: runOptimizer(s, { ...base, name, lr: name === 'gd' || name === 'momentum' ? 0.002 : 0.02 }, s.start, 600).path }));
  }, []);
  return (
    <>
      <H>{t('แนวคิด', 'Idea')}</H>
      <P>{t('Gradient ∇f ชี้ไปทางที่ f เพิ่มเร็วที่สุด การก้าวสวนทาง gradient ทีละนิดจึงพาไปสู่จุดต่ำ ขนาดก้าวคือ learning rate η ซึ่งสำคัญที่สุด', 'The gradient ∇f points where f grows fastest, so small steps against it lead downhill. The step size, the learning rate η, matters most.')}</P>
      <Eq>{'\\boldsymbol\\theta_{t+1} = \\boldsymbol\\theta_t - \\eta\\nabla f(\\boldsymbol\\theta_t)'}</Eq>
      <Figure title={t('GD บนพาราโบลา f(x) = x² ด้วย η ต่างกัน', 'GD on the parabola f(x) = x² with different η')} caption={t('สำหรับ f = x² ทุกก้าวคูณ x ด้วย (1 − 2η): η = 0.1 ลู่เข้าช้า, η = 0.45 เร็ว, η = 0.95 แกว่งสลับฝั่งแต่ยังลู่เข้า, η = 1.05 ลู่ออก (|1 − 2η| > 1)', 'For f = x² every step multiplies x by (1 − 2η): η = 0.1 converges slowly, 0.45 fast, 0.95 oscillates but converges, 1.05 diverges (|1 − 2η| > 1).')}>
        <div className="grid gap-4 lg:grid-cols-2">
          <LinePlot xTitle="x" yTitle="f(x) = x²" height={260} xRange={[-3.2, 3.2]} yRange={[0, 9]} series={[
            { name: 'x²', x: xs, y: xs.map((x) => x * x), color: '#475569' },
            ...paths.map((p, i) => ({ name: `η=${p.eta}`, x: p.p.slice(0, 8), y: p.p.slice(0, 8).map((v) => v * v), color: [C.green, C.cyan, C.orange, C.red][i], mode: 'lines+markers' as const, size: 6 })),
          ]} />
          <LinePlot xTitle="iteration" yTitle="|x| (log)" height={260} yLog series={paths.map((p, i) => ({ name: `η=${p.eta}`, x: p.p.map((_, k) => k), y: p.p.map((v) => Math.max(1e-12, Math.abs(v))), color: [C.green, C.cyan, C.orange, C.red][i] }))} />
        </div>
      </Figure>
      <H>{t('Optimizer ที่ฉลาดขึ้น', 'Smarter optimizers')}</H>
      <Eq>{'\\text{Momentum: } \\mathbf v \\leftarrow \\mu\\mathbf v - \\eta\\nabla f;\\quad \\text{Adam: } \\boldsymbol\\theta \\leftarrow \\boldsymbol\\theta - \\eta\\,\\hat{\\mathbf m}/(\\sqrt{\\hat{\\mathbf v}}+\\epsilon)'}</Eq>
      <Figure title={t('เปรียบเทียบบน Rosenbrock (หุบโค้งแคบ) — คำนวณจริง', 'Comparison on Rosenbrock (narrow curved valley) — computed live')} caption={t('แกน y เป็น log ของ f GD ธรรมดาช้าเพราะซิกแซกข้ามหุบ Momentum สะสมความเร็ว Adam/RMSprop ปรับขนาดก้าวต่อแกน', 'y is log f. Plain GD is slow because it zig-zags across the valley; momentum builds speed; Adam/RMSprop rescale each axis.')}>
        <LinePlot xTitle="iteration" yTitle="f (log)" yLog series={cmp.map((c) => ({ name: c.name, x: c.path.map((_, i) => i), y: c.path.map((p) => Math.max(1e-10, p.f)), color: c.color }))} />
      </Figure>
      <LabLink to="/optimization" label="Optimization Lab" />
    </>
  );
}

function MlpLesson() {
  const t = useT();
  const z = linspace(-4, 4, 161);
  const acts: Activation[] = ['sigmoid', 'tanh', 'relu', 'gelu'];
  const cols = [C.cyan, C.orange, C.green, C.purple];
  const xor = useMemo(() => {
    const ds = makeDataset('xor', { n: 240, classes: 2, noise: 0.1, seed: 2 });
    const sp = trainTestSplit(ds.y, 0.25, 1);
    const g = (ix: number[]) => [ix.map((i) => ds.X[i]), ix.map((i) => ds.y[i])] as const;
    const [Xtr, ytr] = g(sp.train), [Xte, yte] = g(sp.test);
    const run = (hidden: boolean) => {
      const layers: LayerSpec[] = [{ id: 'i', kind: 'input', name: 'in', units: 2 }, ...(hidden ? [{ id: 'h', kind: 'dense' as const, name: 'h', units: 8, activation: 'tanh' as const }] : []), { id: 'o', kind: 'output', name: 'o', units: 1, activation: 'sigmoid' }];
      const tr = new Trainer(layers, { Xtr, ytr, Xval: [], yval: [], Xte, yte, nClasses: 2 }, { optimizer: { name: 'adam', lr: 0.02, momentum: 0.9, beta1: 0.9, beta2: 0.999, epsilon: 1e-8 }, batchSize: 16, loss: 'bce', shuffle: true, seed: 3, earlyStopping: false, patience: 10 });
      for (let e = 0; e < 120; e++) tr.trainEpoch();
      return tr.history;
    };
    return { mlp: run(true), perc: run(false) };
  }, []);
  return (
    <>
      <H>{t('แนวคิด', 'Idea')}</H>
      <P>{t('โครงข่ายประสาทเทียมแบบหลายชั้น (MLP) ประกอบด้วยโหนดจำนวนมากต่อเป็นชั้น ทุกโหนดทำงานเหมือนกัน: รวมอินพุตแบบถ่วงน้ำหนัก บวก bias แล้วผ่าน activation function การมีชั้นซ่อนที่ไม่เป็นเชิงเส้นทำให้เรียนรู้เส้นแบ่งที่โค้งได้', 'A multilayer perceptron stacks many nodes in layers. Every node does the same thing: a weighted sum of inputs plus a bias, passed through an activation function. Non-linear hidden layers let it learn curved boundaries.')}</P>
      <NeuronDiagram />
      <Eq>{'\\mathbf a^{(l)} = f\\big(W^{(l)}\\mathbf a^{(l-1)} + \\mathbf b^{(l)}\\big)'}</Eq>
      <Figure title={t('Activation functions และอนุพันธ์', 'Activation functions and their derivatives')} caption={t('sigmoid และ tanh อิ่มตัว (เข้าใกล้ 1 หรือ −1) เมื่อ |z| ใหญ่ ทำให้อนุพันธ์ใกล้ 0 = vanishing gradient ReLU มีอนุพันธ์ 1 เมื่อ z > 0 จึงนิยมในโครงข่ายลึก', 'Sigmoid and tanh saturate (approach 1 or −1) for large |z|, so their derivatives approach 0: vanishing gradients. ReLU has derivative 1 for z > 0, which is why deep networks prefer it.')}>
        <div className="grid gap-4 lg:grid-cols-2">
          <LinePlot xTitle="z" yTitle="f(z)" height={240} yRange={[-1.3, 3]} series={acts.map((a, i) => ({ name: a, x: z, y: activate(a, z), color: cols[i] }))} />
          <LinePlot xTitle="z" yTitle="f′(z)" height={240} yRange={[-0.1, 1.2]} series={acts.map((a, i) => ({ name: `${a}′`, x: z, y: activationBackward(a, z, activate(a, z), z.map(() => 1)), color: cols[i] }))} />
        </div>
      </Figure>
      <H>Backpropagation</H>
      <P>{t('การฝึกใช้กฎลูกโซ่คำนวณว่าน้ำหนักแต่ละตัวส่งผลต่อ loss เท่าไร โดยเริ่มจาก output แล้วส่ง gradient ย้อนกลับทีละชั้น', 'Training uses the chain rule to compute how much each weight affects the loss, starting at the output and passing gradients back layer by layer.')}</P>
      <BackpropDiagram />
      <Eq>{'\\boldsymbol\\delta^{(l)} = \\big(W^{(l+1)}\\big)^{\\top}\\boldsymbol\\delta^{(l+1)}\\odot f\'(\\mathbf z^{(l)}),\\qquad \\frac{\\partial L}{\\partial W^{(l)}} = \\boldsymbol\\delta^{(l)}\\,\\mathbf a^{(l-1)\\top}'}</Eq>
      <Figure title={t('Learning curve บน XOR: MLP เข้าใกล้ 1 ส่วน perceptron ติดที่ ~0.5', 'Learning curve on XOR: the MLP approaches 1 while a perceptron stalls near 0.5')} caption={t('ฝึกจริงในหน้านี้ 120 epoch โมเดลที่มีชั้นซ่อน 8 โหนดเรียนรู้ XOR ได้ ความแม่นยำโค้งขึ้นเข้าหา 1 ส่วนโมเดลที่ไม่มีชั้นซ่อนวาดได้แค่เส้นตรงจึงติดอยู่ราวครึ่งหนึ่ง', 'Trained live on this page for 120 epochs. With 8 hidden nodes the model learns XOR and accuracy curves up toward 1; without a hidden layer it can only draw a straight line and stays around one half.')}>
        <LinePlot xTitle="epoch" yTitle="test accuracy" yRange={[0.3, 1.03]} series={[
          { name: t('MLP 2-8-1', 'MLP 2-8-1'), x: xor.mlp.map((h) => h.epoch), y: xor.mlp.map((h) => h.testAcc), color: C.cyan },
          { name: 'perceptron 2-1', x: xor.perc.map((h) => h.epoch), y: xor.perc.map((h) => h.testAcc), color: C.orange },
          { name: '1.0', x: [1, 120], y: [1, 1], color: C.green, dash: 'dash' },
        ]} />
      </Figure>
      <LabLink to="/network-builder" label="Network Builder" />
    </>
  );
}

function CnnLesson() {
  const t = useT();
  const sig = useMemo(() => { const r = new Rng(4); return linspace(0, 59, 60).map((i) => (i > 18 && i < 40 ? 1 : 0) + r.normal() * 0.08); }, []);
  const conv = (k: number[]) => sig.slice(0, sig.length - k.length + 1).map((_, i) => k.reduce((s, w, j) => s + w * sig[i + j], 0));
  const edge = conv([-1, 0, 1]), blur = conv([1 / 5, 1 / 5, 1 / 5, 1 / 5, 1 / 5]);
  return (
    <>
      <H>{t('แนวคิด', 'Idea')}</H>
      <P>{t('CNN ออกแบบมาสำหรับภาพ แทนที่จะเชื่อมทุกพิกเซลกับทุกนิวรอน มันใช้ตัวกรอง (kernel) ขนาดเล็กเลื่อนไปทั่วภาพ น้ำหนักชุดเดียวกันถูกใช้ทุกตำแหน่ง (weight sharing) จึงมีพารามิเตอร์น้อยและตรวจจับลวดลายได้ไม่ว่าจะอยู่ตรงไหน', 'CNNs are built for images. Instead of connecting every pixel to every neuron, they slide small kernels across the image. The same weights are used everywhere (weight sharing), so there are few parameters and a pattern is detected wherever it appears.')}</P>
      <ConvDiagram />
      <Eq>{'(I * K)_{i,j} = \\sum_{u,v} K_{u,v}\\, I_{i s+u,\\, j s+v} + b,\\qquad H_{out} = \\Big\\lfloor\\frac{H + 2p - k}{s}\\Big\\rfloor + 1'}</Eq>
      <Figure title={t('Convolution 1 มิติบนสัญญาณจริง', '1D convolution on a real signal')} caption={t('kernel [−1, 0, 1] ตอบสนองต่อการเปลี่ยนแปลง (ขอบ) ได้ค่าบวกที่ขอบขาขึ้นและลบที่ขาลง kernel เฉลี่ย [1/5 …] ทำให้สัญญาณเรียบ CNN เรียนรู้ค่า kernel เหล่านี้เองจากข้อมูล', 'The kernel [−1, 0, 1] responds to changes (edges): positive at the rising edge, negative at the falling one. The averaging kernel smooths. A CNN learns such kernels from data.')}>
        <LinePlot xTitle="position" yTitle="value" series={[
          { name: t('สัญญาณ', 'signal'), x: sig.map((_, i) => i), y: sig, color: C.muted },
          { name: 'edge [−1,0,1]', x: edge.map((_, i) => i + 1), y: edge, color: C.cyan },
          { name: 'blur', x: blur.map((_, i) => i + 2), y: blur, color: C.orange },
        ]} />
      </Figure>
      <H>{t('ชั้นต่าง ๆ ใน CNN', 'The layers of a CNN')}</H>
      <Steps items={[t('Convolution: kernel หลายตัว ได้ feature map หลายช่อง', 'Convolution: several kernels give several feature maps'), t('ReLU: ตัดค่าลบ', 'ReLU: removes negatives'), t('Pooling: ย่อขนาด เช่น max ของทุก 2×2 ช่อง ทนต่อการเลื่อนเล็กน้อย', 'Pooling: downsample, e.g. max of each 2×2, for shift tolerance'), t('ชั้นลึกขึ้นเห็นพื้นที่กว้างขึ้น (receptive field โตขึ้น) และจับลวดลายซับซ้อนขึ้น', 'Deeper layers see wider areas (growing receptive field) and complex patterns'), t('Flatten + Dense + softmax ให้ความน่าจะเป็นของแต่ละคลาส', 'Flatten + dense + softmax give class probabilities')]} />
      <Eq>{'\\text{params}_{conv} = F\\cdot C_{in}\\cdot k^2 + F \\quad\\text{vs}\\quad \\text{params}_{dense} = n_{in}\\cdot n_{out} + n_{out}'}</Eq>
      <LabLink to="/cnn-lab" label="CNN Lab" />
    </>
  );
}

function useGradDecay() {
  return useMemo(() => {
    const T = 30, out: Record<string, number[]> = {};
    (['rnn', 'lstm'] as const).forEach((cell) => {
      const m = new SeqModel({ cell, hidden: 8, layers: 1, inDim: 1, outDim: 1, seed: 3 });
      const { X, Y } = makeSeqBatch('memory', 20, T, new Rng(5));
      const acc = Array(T).fill(0);
      X.forEach((xs, i) => { const g = m.backward(m.forward(xs, 'memory'), Y[i], 'memory'); g.gradNorms.forEach((v, t2) => (acc[t2] += v)); });
      out[cell] = acc.map((v) => v / acc[T - 1]);
    });
    return out;
  }, []);
}

function RnnLesson() {
  const t = useT();
  const g = useGradDecay();
  return (
    <>
      <H>{t('แนวคิด', 'Idea')}</H>
      <P>{t('RNN ใช้กับข้อมูลลำดับ เช่นข้อความ เสียง อนุกรมเวลา โดยมี hidden state h_t ที่ส่งต่อจากขั้นหนึ่งไปอีกขั้น เป็น "ความจำ" ของสิ่งที่เห็นมาแล้ว น้ำหนักชุดเดียวกันถูกใช้ทุก time step', 'RNNs handle sequences such as text, audio and time series. A hidden state h_t passes from step to step as a memory of what was seen; the same weights are used at every time step.')}</P>
      <RnnUnrolledDiagram />
      <Eq>{'\\mathbf h_t = \\tanh(W_x\\mathbf x_t + W_h\\mathbf h_{t-1} + \\mathbf b),\\qquad \\hat y_t = W_y \\mathbf h_t + b_y'}</Eq>
      <H>Vanishing gradient</H>
      <Eq>{'\\frac{\\partial \\mathbf h_T}{\\partial \\mathbf h_k} = \\prod_{t=k+1}^{T}\\mathrm{diag}\\big(1-\\mathbf h_t^2\\big)W_h'}</Eq>
      <Figure title={t('ขนาด gradient ที่ย้อนกลับไปถึงแต่ละ time step (คำนวณด้วย BPTT จริง, T = 30)', 'Gradient size reaching each time step (real BPTT, T = 30)')} caption={t('แกน y เป็น log ปรับให้ step สุดท้าย = 1 ใน RNN gradient หดลงแบบเอกซ์โพเนนเชียลเมื่อย้อนเวลา ข้อมูลจากช่วงต้นจึงแทบไม่ถูกเรียนรู้ LSTM ลดลงช้ากว่ามาก', 'Log y-axis, normalised so the last step = 1. In the RNN the gradient shrinks exponentially going back in time, so early information is barely learned; the LSTM decays far more slowly.')}>
        <LinePlot xTitle="time step t" yTitle="‖∂L/∂h_t‖ (normalised)" yLog series={[
          { name: 'SimpleRNN', x: g.rnn.map((_, i) => i), y: g.rnn, color: C.orange, mode: 'lines+markers', size: 5 },
          { name: 'LSTM', x: g.lstm.map((_, i) => i), y: g.lstm, color: C.cyan, mode: 'lines+markers', size: 5 },
        ]} />
      </Figure>
      <ProsCons pros={[t('รับลำดับความยาวเท่าใดก็ได้', 'Any sequence length'), t('พารามิเตอร์น้อย (ใช้ร่วมทุก step)', 'Few parameters (shared)')]} cons={[t('จำระยะยาวไม่ได้ (vanishing gradient)', 'Poor long-range memory'), t('คำนวณขนานไม่ได้ ต้องทำทีละ step', 'Sequential, hard to parallelise')]} />
      <LabLink to="/algorithms/rnn" label="Sequence Lab" />
    </>
  );
}

function LstmLesson() {
  const t = useT();
  const g = useGradDecay();
  const k = linspace(0, 40, 41);
  return (
    <>
      <H>{t('แนวคิด', 'Idea')}</H>
      <P>{t('LSTM เพิ่ม cell state c_t ซึ่งเป็นเส้นทางที่ข้อมูลไหลผ่านเวลาได้ด้วยการบวก ไม่ใช่การคูณเมทริกซ์ซ้ำ ๆ และมี gate สามตัวเป็น sigmoid (ค่า 0–1) ทำหน้าที่เหมือนวาล์ว: forget gate ตัดสินว่าจะลืมแค่ไหน input gate ว่าจะเขียนใหม่แค่ไหน output gate ว่าจะแสดงออกแค่ไหน', 'An LSTM adds a cell state c_t, a path along which information flows through time by addition rather than repeated matrix products, plus three sigmoid gates (0–1) acting as valves: forget decides how much to erase, input how much to write, output how much to expose.')}</P>
      <LstmCellDiagram />
      <Eq>{'\\mathbf f,\\mathbf i,\\mathbf o = \\sigma(\\cdot),\\ \\mathbf g = \\tanh(\\cdot),\\qquad \\mathbf c_t = \\mathbf f\\odot\\mathbf c_{t-1} + \\mathbf i\\odot\\mathbf g,\\qquad \\mathbf h_t = \\mathbf o\\odot\\tanh(\\mathbf c_t)'}</Eq>
      <Figure title={t('ความจำที่เหลือหลัง k step: fᵏ', 'Memory remaining after k steps: fᵏ')} caption={t('ถ้า forget gate f = 0.99 ความจำยังเหลือ 67% หลัง 40 step แต่ถ้า f = 0.5 จะหายเกือบหมดใน 5 step LSTM เรียนรู้ที่จะตั้ง f ใกล้ 1 สำหรับข้อมูลที่ต้องจำนาน', 'With f = 0.99, 67% of the memory survives 40 steps; with f = 0.5 it is nearly gone in 5. An LSTM learns to keep f near 1 for information it must hold.')}>
        <LinePlot xTitle="k" yTitle="fᵏ" yRange={[0, 1.05]} series={[0.5, 0.9, 0.97, 0.99].map((f, i) => ({ name: `f = ${f}`, x: k, y: k.map((v) => f ** v), color: [C.red, C.orange, C.purple, C.cyan][i] }))} />
      </Figure>
      <Figure title={t('Gradient ย้อนเวลา: LSTM เทียบกับ RNN', 'Gradients through time: LSTM vs RNN')} caption={t('ค่าเดียวกับบท RNN (BPTT จริง) เส้นทาง cell state ทำให้ gradient ของ LSTM ไม่หายเร็ว', 'Same computation as the RNN lesson (real BPTT): the cell-state path keeps LSTM gradients from vanishing quickly.')}>
        <LinePlot xTitle="time step t" yTitle="normalised gradient" yLog series={[{ name: 'LSTM', x: g.lstm.map((_, i) => i), y: g.lstm, color: C.cyan }, { name: 'SimpleRNN', x: g.rnn.map((_, i) => i), y: g.rnn, color: C.orange, dash: 'dot' }]} />
      </Figure>
      <ProsCons pros={[t('จำระยะยาวได้ดีกว่า RNN', 'Much better long-range memory'), t('ใช้ได้ดีกับอนุกรมเวลาและข้อความ', 'Strong on time series and text')]} cons={[t('พารามิเตอร์มากกว่า RNN 4 เท่า', '4× the parameters of an RNN'), t('ยังคำนวณทีละ step', 'Still sequential')]} />
      <LabLink to="/algorithms/lstm" label="Sequence Lab" />
    </>
  );
}

function TransformerLesson() {
  const t = useT();
  const [tau, setTau] = useState(1);
  const scores = [2.0, 1.0, 0.5, 0.1, -0.5, -1.2];
  const soft = (tt: number) => { const e = scores.map((s) => Math.exp(s / tt)); const z = e.reduce((a, b) => a + b, 0); return e.map((v) => v / z); };
  const pe = useMemo(() => positionalEncoding(60, 16), []);
  return (
    <>
      <H>{t('แนวคิด', 'Idea')}</H>
      <P>{t('Transformer ไม่ใช้การวนซ้ำแบบ RNN แต่ให้ทุกคำ "มอง" ทุกคำในประโยคพร้อมกันผ่าน self-attention แต่ละคำสร้างเวกเตอร์ Query (สิ่งที่มองหา), Key (สิ่งที่มี) และ Value (ข้อมูลที่ส่งต่อ) น้ำหนัก attention มาจากความคล้ายระหว่าง Query กับ Key', 'A Transformer drops recurrence: every token “looks at” every other token at once through self-attention. Each token produces a Query (what it looks for), a Key (what it offers) and a Value (what it passes on); attention weights come from Query–Key similarity.')}</P>
      <AttentionDiagram />
      <Eq>{'\\text{Attention}(Q,K,V) = \\text{softmax}\\Big(\\frac{QK^\\top}{\\sqrt{d_k}}\\Big)V'}</Eq>
      <Figure title={t('Softmax กับอุณหภูมิ: คะแนนเดียวกัน ความแหลมต่างกัน', 'Softmax and temperature: same scores, different sharpness')}
        controls={<Slider label="τ" value={tau} min={0.2} max={4} step={0.1} format={(v) => v.toFixed(1)} onChange={setTau} />}
        caption={t('การหารด้วย √d_k ทำหน้าที่คล้ายอุณหภูมิ ป้องกันไม่ให้ softmax แหลมเกินไปจน gradient หาย ผลรวมของน้ำหนักเป็น 1 เสมอ', 'Dividing by √d_k acts like a temperature, keeping softmax from becoming so sharp that gradients vanish. Weights always sum to 1.')}>
        <LinePlot xTitle="key index" yTitle="attention weight" yRange={[0, 1]} series={[
          { name: `τ = ${tau.toFixed(1)}`, x: scores.map((_, i) => i), y: soft(tau), color: C.cyan, mode: 'lines+markers', size: 9 },
          { name: 'τ = 1', x: scores.map((_, i) => i), y: soft(1), color: C.muted, dash: 'dot' },
        ]} />
      </Figure>
      <H>Positional encoding</H>
      <P>{t('attention ไม่รู้ลำดับคำ จึงต้องบวกเวกเตอร์ตำแหน่งที่สร้างจาก sin/cos หลายความถี่ มิติต่ำแกว่งเร็ว มิติสูงแกว่งช้า ทำให้ทุกตำแหน่งมีลายนิ้วมือไม่ซ้ำกัน', 'Attention is order-blind, so positional vectors built from sines and cosines of many frequencies are added. Low dimensions oscillate fast, high ones slowly, giving every position a unique fingerprint.')}</P>
      <Eq>{'PE_{p,2i} = \\sin\\!\\big(p/10000^{2i/d}\\big),\\qquad PE_{p,2i+1} = \\cos\\!\\big(p/10000^{2i/d}\\big)'}</Eq>
      <Figure title={t('Positional encoding (d = 16) ของมิติต่าง ๆ', 'Positional encoding (d = 16) for several dimensions')} caption={t('แต่ละเส้นคือหนึ่งมิติของเวกเตอร์ตำแหน่ง ตามตำแหน่ง 0–59', 'Each line is one dimension of the position vector across positions 0–59.')}>
        <LinePlot xTitle="position p" yTitle="PE" series={[0, 2, 6, 10].map((d, i) => ({ name: `dim ${d}`, x: pe.map((_, p) => p), y: pe.map((r) => r[d]), color: [C.cyan, C.orange, C.purple, C.green][i] } as Series))} />
      </Figure>
      <Steps items={[t('Embedding + positional encoding', 'Embedding + positional encoding'), t('Multi-head attention (หลาย head เรียนความสัมพันธ์ต่างกัน)', 'Multi-head attention (heads learn different relations)'), t('Residual + LayerNorm', 'Residual + LayerNorm'), t('Feed-forward ทีละตำแหน่ง + Residual + LayerNorm', 'Position-wise feed-forward + residual + LayerNorm'), t('ซ้อนหลาย block', 'Stack many blocks')]} />
      <ProsCons pros={[t('คำนวณขนานได้ทั้งลำดับ', 'Fully parallel over the sequence'), t('เชื่อมคำที่อยู่ไกลกันได้ในขั้นเดียว', 'Links distant tokens in one step')]} cons={[t('ค่าใช้จ่าย O(n²) ตามความยาวลำดับ', 'O(n²) cost in sequence length'), t('ต้องการข้อมูลมากในการฝึก', 'Data-hungry')]} />
      <LabLink to="/algorithms/transformer" label="Transformer Lab" />
    </>
  );
}

export interface LessonDef { slug: string; title: Text; group: Text; minutes: number; render: () => ReactNode }

const G_BASIC = { th: 'พื้นฐาน', en: 'Foundations' }, G_SUP = { th: 'การเรียนรู้แบบมีผู้สอน', en: 'Supervised learning' }, G_UNS = { th: 'การเรียนรู้แบบไม่มีผู้สอน', en: 'Unsupervised learning' }, G_DEEP = { th: 'Deep learning', en: 'Deep learning' }, G_SEQ = { th: 'ข้อมูลลำดับ', en: 'Sequence models' };

export const LESSONS: LessonDef[] = [
  { slug: 'basics', title: { th: 'Loss, Gradient และ Learning curve', en: 'Loss, gradients and learning curves' }, group: G_BASIC, minutes: 15, render: () => <Basics /> },
  { slug: 'gradient-descent', title: { th: 'Gradient Descent และ Optimizer', en: 'Gradient descent & optimizers' }, group: G_BASIC, minutes: 20, render: () => <GdLesson /> },
  { slug: 'knn', title: { th: 'K-Nearest Neighbors', en: 'K-Nearest Neighbors' }, group: G_SUP, minutes: 15, render: () => <KnnLesson /> },
  { slug: 'linear-regression', title: { th: 'Linear Regression', en: 'Linear Regression' }, group: G_SUP, minutes: 15, render: () => <LinearLesson /> },
  { slug: 'logistic-regression', title: { th: 'Logistic Regression', en: 'Logistic Regression' }, group: G_SUP, minutes: 15, render: () => <LogisticLesson /> },
  { slug: 'svm', title: { th: 'Support Vector Machine', en: 'Support Vector Machine' }, group: G_SUP, minutes: 25, render: () => <SvmLesson /> },
  { slug: 'decision-tree', title: { th: 'Decision Tree', en: 'Decision Tree' }, group: G_SUP, minutes: 20, render: () => <TreeLesson /> },
  { slug: 'random-forest', title: { th: 'Random Forest', en: 'Random Forest' }, group: G_SUP, minutes: 15, render: () => <ForestLesson /> },
  { slug: 'kmeans', title: { th: 'K-Means', en: 'K-Means' }, group: G_UNS, minutes: 15, render: () => <KMeansLesson /> },
  { slug: 'dbscan', title: { th: 'DBSCAN', en: 'DBSCAN' }, group: G_UNS, minutes: 15, render: () => <DbscanLesson /> },
  { slug: 'mlp', title: { th: 'Neural Network (MLP) และ Backpropagation', en: 'Neural networks (MLP) & backpropagation' }, group: G_DEEP, minutes: 30, render: () => <MlpLesson /> },
  { slug: 'cnn', title: { th: 'Convolutional Neural Network', en: 'Convolutional Neural Network' }, group: G_DEEP, minutes: 25, render: () => <CnnLesson /> },
  ...VISION_IDS.map((id): LessonDef => ({ slug: id, title: { th: VISION_MODELS[id].name, en: VISION_MODELS[id].name }, group: G_DEEP, minutes: 20, render: () => <VisionTheory modelId={id} /> })),
  { slug: 'rnn', title: { th: 'Recurrent Neural Network', en: 'Recurrent Neural Network' }, group: G_SEQ, minutes: 20, render: () => <RnnLesson /> },
  { slug: 'lstm', title: { th: 'LSTM', en: 'LSTM' }, group: G_SEQ, minutes: 20, render: () => <LstmLesson /> },
  { slug: 'transformer', title: { th: 'Transformer และ Self-Attention', en: 'Transformer & self-attention' }, group: G_SEQ, minutes: 25, render: () => <TransformerLesson /> },
];

