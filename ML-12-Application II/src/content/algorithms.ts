import type { Text } from '../i18n/i18n';
import { VISION_IDS, VISION_MODELS } from './visionModels';

export type Category = 'supervised' | 'unsupervised' | 'deep' | 'optimization' | 'sequence';

export interface AlgoEntry {
  slug: string;
  name: string;
  category: Category;
  difficulty: 1 | 2 | 3;
  /** available = runnable now; otherwise the roadmap phase that delivers it */
  status: 'available' | 'planned';
  phase: number;
  route?: string;
  summary: Text;
}

/** An original publication that introduced (or defined the standard form of) an algorithm. */
export interface Paper { authors: string; year: number; title: string; venue: string; url: string }

export const CATEGORY_NAME: Record<Category, Text> = {
  supervised: { th: 'การเรียนรู้แบบมีผู้สอน', en: 'Supervised learning' },
  unsupervised: { th: 'การเรียนรู้แบบไม่มีผู้สอน', en: 'Unsupervised learning' },
  deep: { th: 'Deep learning', en: 'Deep learning' },
  optimization: { th: 'การหาค่าเหมาะสม', en: 'Optimization' },
  sequence: { th: 'ข้อมูลลำดับ', en: 'Sequence models' },
};

export const DIFFICULTY: Record<1 | 2 | 3, Text> = {
  1: { th: 'พื้นฐาน', en: 'Beginner' },
  2: { th: 'ปานกลาง', en: 'Intermediate' },
  3: { th: 'ขั้นสูง', en: 'Advanced' },
};

export const ALGORITHMS: AlgoEntry[] = [
  ...VISION_IDS.map((id): AlgoEntry => ({ slug: id, name: VISION_MODELS[id].name, category: 'deep', difficulty: 3, status: 'available', phase: 7, route: `/algorithms/${id}`, summary: { th: `${VISION_MODELS[id].summary.th} · โมเดลสาธิตขนาดย่อและโครงสร้างอ้างอิง`, en: `${VISION_MODELS[id].summary.en} Compact teaching model plus full architecture reference.` } })),
  { slug: 'knn', name: 'K-Nearest Neighbors', category: 'supervised', difficulty: 1, status: 'available', phase: 2, route: '/algorithms/knn',
    summary: { th: 'จำแนกจุดใหม่จากการโหวตของเพื่อนบ้านที่ใกล้ที่สุด K จุด เห็นระยะทาง อันดับ และผลโหวตทุกขั้น', en: 'Classify a new point by a vote of its K nearest neighbours; see every distance, rank and vote.' } },
  { slug: 'mlp', name: 'Neural Network Builder (MLP)', category: 'deep', difficulty: 2, status: 'available', phase: 3, route: '/network-builder',
    summary: { th: 'ออกแบบโครงข่ายเอง ฝึกจริงในเบราว์เซอร์ ดู forward/backward propagation และตรวจน้ำหนักทุกเส้น', en: 'Design your own network, train it in the browser, watch forward/backward propagation and inspect every weight.' } },
  { slug: 'svm', name: 'Support Vector Machine', category: 'supervised', difficulty: 2, status: 'available', phase: 2, route: '/algorithms/svm',
    summary: { th: 'ระนาบแบ่งที่ margin กว้างที่สุด support vector และ kernel trick', en: 'Maximum-margin hyperplanes, support vectors and the kernel trick.' } },
  { slug: 'linear-regression', name: 'Linear Regression', category: 'supervised', difficulty: 1, status: 'available', phase: 2, route: '/algorithms/linear-regression',
    summary: { th: 'เส้น/ระนาบที่ fit ข้อมูล residual และเส้นทาง gradient descent', en: 'Fitted line/plane, residuals and the gradient-descent path.' } },
  { slug: 'logistic-regression', name: 'Logistic Regression', category: 'supervised', difficulty: 1, status: 'available', phase: 2, route: '/algorithms/logistic-regression',
    summary: { th: 'ฟังก์ชัน sigmoid ความน่าจะเป็นของคลาส และ cross-entropy', en: 'Sigmoid probabilities, decision boundary and cross-entropy.' } },
  { slug: 'kmeans', name: 'K-Means', category: 'unsupervised', difficulty: 1, status: 'available', phase: 2, route: '/algorithms/kmeans',
    summary: { th: 'จัดกลุ่มด้วยการสลับ assign / update จุดศูนย์กลางจนลู่เข้า', en: 'Alternate assignment and centroid updates until convergence.' } },
  { slug: 'gradient-descent', name: 'Gradient Descent & Optimizers', category: 'optimization', difficulty: 2, status: 'available', phase: 2, route: '/optimization',
    summary: { th: 'พื้นผิว loss 3 มิติ เปรียบเทียบ SGD, Momentum, RMSprop, Adam', en: '3D loss landscapes comparing SGD, Momentum, RMSprop and Adam.' } },
  { slug: 'cnn', name: 'Convolutional Neural Network', category: 'deep', difficulty: 3, status: 'available', phase: 4, route: '/cnn-lab',
    summary: { th: 'kernel เลื่อนบนภาพ feature map pooling และ inference บน UCI handwritten digits', en: 'Sliding kernels, feature maps, pooling and UCI handwritten digits inference.' } },
  { slug: 'decision-tree', name: 'Decision Tree', category: 'supervised', difficulty: 2, status: 'available', phase: 5, route: '/algorithms/decision-tree',
    summary: { th: 'การแบ่งตามเกณฑ์ impurity และเส้นทางการตัดสินใจ', en: 'Impurity-based splits and decision paths.' } },
  { slug: 'random-forest', name: 'Random Forest', category: 'supervised', difficulty: 2, status: 'available', phase: 5, route: '/algorithms/random-forest',
    summary: { th: 'ต้นไม้หลายต้นจาก bootstrap และการโหวตรวม', en: 'Many bootstrapped trees and ensemble voting.' } },
  { slug: 'dbscan', name: 'DBSCAN', category: 'unsupervised', difficulty: 2, status: 'available', phase: 5, route: '/algorithms/dbscan',
    summary: { th: 'core / border / noise และการขยายคลัสเตอร์ตามความหนาแน่น', en: 'Core, border and noise points; density-based expansion.' } },
  { slug: 'rnn', name: 'RNN', category: 'sequence', difficulty: 3, status: 'available', phase: 5, route: '/algorithms/rnn',
    summary: { th: 'เครือข่ายที่คลี่ตามเวลาและ hidden state', en: 'Time-unrolled networks and hidden state.' } },
  { slug: 'lstm', name: 'LSTM', category: 'sequence', difficulty: 3, status: 'available', phase: 5, route: '/algorithms/lstm',
    summary: { th: 'forget / input / output gate และ cell state', en: 'Forget, input and output gates with the cell state.' } },
  { slug: 'transformer', name: 'Transformer', category: 'sequence', difficulty: 3, status: 'available', phase: 5, route: '/algorithms/transformer',
    summary: { th: 'Query / Key / Value และ attention matrix ที่คำนวณจริง', en: 'Query/Key/Value and real attention matrices.' } },
];

const doi = (id: string) => `https://doi.org/${id}`;

/** Original papers per algorithm slug; links verified to resolve (DOI / arXiv / publisher PDF). */
export const PAPERS: Record<string, Paper[]> = {
  ...Object.fromEntries(VISION_IDS.map((id) => [id, [VISION_MODELS[id].paper]])),
  knn: [{ authors: 'T. Cover, P. Hart', year: 1967, title: 'Nearest neighbor pattern classification', venue: 'IEEE Trans. Information Theory 13(1)', url: doi('10.1109/TIT.1967.1053964') }],
  mlp: [{ authors: 'D. Rumelhart, G. Hinton, R. Williams', year: 1986, title: 'Learning representations by back-propagating errors', venue: 'Nature 323', url: doi('10.1038/323533a0') }],
  svm: [{ authors: 'C. Cortes, V. Vapnik', year: 1995, title: 'Support-vector networks', venue: 'Machine Learning 20', url: doi('10.1007/BF00994018') }],
  'linear-regression': [{ authors: 'F. Galton', year: 1886, title: 'Regression towards mediocrity in hereditary stature', venue: 'J. Anthropological Institute 15', url: doi('10.2307/2841583') }],
  'logistic-regression': [{ authors: 'D. R. Cox', year: 1958, title: 'The regression analysis of binary sequences', venue: 'J. Royal Statistical Society B 20(2)', url: doi('10.1111/j.2517-6161.1958.tb00292.x') }],
  kmeans: [{ authors: 'S. Lloyd', year: 1982, title: 'Least squares quantization in PCM', venue: 'IEEE Trans. Information Theory 28(2)', url: doi('10.1109/TIT.1982.1056489') }],
  'gradient-descent': [
    { authors: 'H. Robbins, S. Monro', year: 1951, title: 'A stochastic approximation method', venue: 'Annals of Mathematical Statistics 22(3)', url: doi('10.1214/aoms/1177729586') },
    { authors: 'B. Polyak', year: 1964, title: 'Some methods of speeding up the convergence of iteration methods', venue: 'USSR Comput. Math. and Math. Physics 4(5)', url: doi('10.1016/0041-5553(64)90137-5') },
    { authors: 'D. Kingma, J. Ba', year: 2014, title: 'Adam: A method for stochastic optimization', venue: 'arXiv:1412.6980 (ICLR 2015)', url: 'https://arxiv.org/abs/1412.6980' },
  ],
  cnn: [
    { authors: 'Y. LeCun et al.', year: 1989, title: 'Backpropagation applied to handwritten zip code recognition', venue: 'Neural Computation 1(4)', url: doi('10.1162/neco.1989.1.4.541') },
    { authors: 'Y. LeCun, L. Bottou, Y. Bengio, P. Haffner', year: 1998, title: 'Gradient-based learning applied to document recognition', venue: 'Proceedings of the IEEE 86(11)', url: doi('10.1109/5.726791') },
  ],
  'decision-tree': [
    { authors: 'L. Breiman, J. Friedman, R. Olshen, C. Stone', year: 1984, title: 'Classification and Regression Trees (CART)', venue: 'Wadsworth (book)', url: doi('10.1201/9781315139470') },
    { authors: 'J. R. Quinlan', year: 1986, title: 'Induction of decision trees', venue: 'Machine Learning 1', url: doi('10.1007/BF00116251') },
  ],
  'random-forest': [{ authors: 'L. Breiman', year: 2001, title: 'Random forests', venue: 'Machine Learning 45', url: doi('10.1023/A:1010933404324') }],
  dbscan: [{ authors: 'M. Ester, H.-P. Kriegel, J. Sander, X. Xu', year: 1996, title: 'A density-based algorithm for discovering clusters in large spatial databases with noise', venue: 'KDD-96', url: 'https://cdn.aaai.org/KDD/1996/KDD96-037.pdf' }],
  rnn: [{ authors: 'J. Elman', year: 1990, title: 'Finding structure in time', venue: 'Cognitive Science 14(2)', url: doi('10.1207/s15516709cog1402_1') }],
  lstm: [
    { authors: 'S. Hochreiter, J. Schmidhuber', year: 1997, title: 'Long short-term memory', venue: 'Neural Computation 9(8)', url: doi('10.1162/neco.1997.9.8.1735') },
    { authors: 'F. Gers, J. Schmidhuber, F. Cummins', year: 2000, title: 'Learning to forget: continual prediction with LSTM', venue: 'Neural Computation 12(10)', url: doi('10.1162/089976600300015015') },
  ],
  transformer: [{ authors: 'A. Vaswani et al.', year: 2017, title: 'Attention is all you need', venue: 'arXiv:1706.03762 (NeurIPS 2017)', url: 'https://arxiv.org/abs/1706.03762' }],
};

export const PHASES: { n: number; title: Text; items: Text; done: boolean }[] = [
  { n: 1, title: { th: 'โครงเว็บไซต์และเอนจิน 3D', en: 'Website foundation & 3D engine' }, items: { th: 'การนำทาง ระบบดีไซน์ เอนจินเรนเดอร์ ชุดข้อมูล ตัวควบคุม', en: 'Navigation, design system, rendering engine, datasets, controls' }, done: true },
  { n: 2, title: { th: 'อัลกอริทึมคลาสสิก', en: 'Classic algorithms' }, items: { th: 'KNN · SVM · Linear/Logistic Regression · K-Means · Gradient Descent', en: 'KNN · SVM · Linear/Logistic Regression · K-Means · Gradient Descent' }, done: true },
  { n: 3, title: { th: 'ตัวสร้างโครงข่าย MLP', en: 'MLP Builder' }, items: { th: 'แก้ไขชั้น forward/backward การแสดงน้ำหนัก', en: 'Layer editing, forward/backward, weight visualisation' }, done: true },
  { n: 4, title: { th: 'CNN Lab', en: 'CNN Lab' }, items: { th: 'ตัวสร้าง CNN แอนิเมชัน convolution feature map UCI handwritten digits', en: 'CNN builder, convolution animation, feature maps, UCI handwritten digits' }, done: true },
  { n: 5, title: { th: 'โมเดลขั้นสูง', en: 'Advanced models' }, items: { th: 'Decision Tree · Random Forest · DBSCAN · RNN · LSTM · Transformer', en: 'Decision Tree · Random Forest · DBSCAN · RNN · LSTM · Transformer' }, done: true },
  { n: 6, title: { th: 'บทเรียนและการเปรียบเทียบ', en: 'Lessons & comparison' }, items: { th: 'บทเรียน แบบทดสอบ เปรียบเทียบโมเดล ส่งออกผลทดลอง', en: 'Lessons, quizzes, model comparison, experiment export' }, done: true },
  { n: 7, title: { th: 'โมเดลภาพขั้นสูง', en: 'Vision architecture labs' }, items: { th: 'VGG16 · ResNet50 · DenseNet121 · InceptionV3 · MobileNetV2 · ViT', en: 'VGG16 · ResNet50 · DenseNet121 · InceptionV3 · MobileNetV2 · ViT' }, done: true },
  { n: 8, title: { th: 'Playground', en: 'Playground' }, items: { th: 'ห้องทดลองอิสระรวมหลายเทคนิค (ยังไม่เปิดใช้งาน)', en: 'Unified free-form experiment workspace (not available yet)' }, done: false },
];
