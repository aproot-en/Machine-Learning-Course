import type { Text } from '../i18n/i18n';
import type { Paper } from './algorithms';

export const VISION_IDS = ['vgg16', 'resnet50', 'densenet121', 'inceptionv3', 'mobilenetv2', 'vit'] as const;
export type VisionId = typeof VISION_IDS[number];
export interface ReferenceStage { name: string; shape: string; operations: string; repeats?: number }
export interface VisionDefinition {
  id: VisionId; name: string; year: number; input: number; parameters: string;
  summary: Text; idea: Text; formula: string; explanation: Text;
  strengths: Text; limitations: Text; exercise: Text;
  question: Text; answers: Text[]; correct: number;
  reference: ReferenceStage[]; paper: Paper; implementation: string;
}
const tx = (th: string, en: string): Text => ({ th, en });
export const VISION_MODELS: Record<VisionId, VisionDefinition> = {
  vgg16: {
    id: 'vgg16', name: 'VGG16', year: 2014, input: 224, parameters: '~138M',
    summary: tx('ซ้อน Conv 3×3 ใน 5 บล็อก ตามด้วยชั้น Dense ดู feature map และผลจำแนกภาพ', 'Stack 3×3 convolutions in five blocks, then dense layers; inspect feature maps and image predictions.'),
    idea: tx('ความลึกจากตัวกรองขนาดเล็ก: 13 convolution และ 3 fully connected นับเป็น 16 ชั้นที่มีน้ำหนัก', 'Depth from small filters: 13 convolutions and 3 fully connected layers form 16 weight layers.'),
    formula: 'h_{l+1}=\\operatorname{ReLU}(W_l*h_l+b_l)',
    explanation: tx('Conv 3×3 สองชั้นมี receptive field 5×5 แต่เพิ่ม activation ระหว่างชั้น Pool ลดความละเอียดทีละครึ่งก่อนรวมเป็นตัวจำแนก', 'Two 3×3 convolutions cover a 5×5 receptive field while adding an intermediate activation. Pooling halves spatial resolution before the dense classifier.'),
    strengths: tx('โครงสร้างตรงไปตรงมา เหมาะศึกษาว่าความลึกสร้างลักษณะภาพอย่างไร', 'Simple structure for studying hierarchical visual features.'),
    limitations: tx('หัว Dense ขนาดใหญ่ใช้หน่วยความจำมาก และไม่มี shortcut ช่วยส่ง gradient', 'The large dense head is memory intensive and has no gradient shortcut.'),
    exercise: tx('เปรียบเทียบภาพใน block1 กับ block5 แล้วเปลี่ยน seed และฝึกหัวจำแนกใหม่ ความแม่นยำเปลี่ยนอย่างไร', 'Compare block1 and block5 activations, change the seed, and retrain the head. How does accuracy change?'),
    question: tx('เหตุใดชื่อ VGG16 จึงมีเลข 16', 'What does 16 count in VGG16?'), answers: [tx('13 Conv + 3 Dense', '13 convolutions + 3 dense layers'), tx('16 pooling layers', '16 pooling layers'), tx('16 classes', '16 classes')], correct: 0,
    reference: [
      { name: 'Input', shape: '224×224×3', operations: 'RGB → BGR; subtract ImageNet channel means' },
      ...[2, 2, 3, 3, 3].map((repeats, i) => ({ name: `Block ${i + 1}`, shape: `${224 / 2 ** (i + 1)}×${224 / 2 ** (i + 1)}×${[64, 128, 256, 512, 512][i]}`, operations: `3×3 Conv + ReLU ×${repeats}; 2×2 max pool`, repeats })),
      { name: 'FC1 / FC2', shape: '4096 → 4096', operations: 'Flatten 7×7×512; Dense + ReLU; Dense + ReLU' },
      { name: 'Classifier', shape: '1000', operations: 'Dense → softmax' },
    ],
    paper: { authors: 'K. Simonyan, A. Zisserman', year: 2014, title: 'Very Deep Convolutional Networks for Large-Scale Image Recognition', venue: 'arXiv:1409.1556 / ICLR 2015', url: 'https://arxiv.org/abs/1409.1556' },
    implementation: 'https://github.com/keras-team/keras/blob/master/keras/src/applications/vgg16.py',
  },
  resnet50: {
    id: 'resnet50', name: 'ResNet50', year: 2015, input: 224, parameters: '~25.6M',
    summary: tx('16 bottleneck blocks กับ residual shortcut เห็นทางหลักและทางลัดรวมกันจริง', '16 bottleneck blocks with residual shortcuts; inspect main and skip paths before addition.'),
    idea: tx('เรียนรู้ส่วนแก้ไข F(x) แล้วบวกกับ x ผ่าน shortcut ใช้ bottleneck 1×1 → 3×3 → 1×1', 'Learn a residual F(x) and add the shortcut x, using 1×1 → 3×3 → 1×1 bottlenecks.'),
    formula: 'y=\\operatorname{ReLU}(F(x)+S(x))',
    explanation: tx('S(x) เป็น identity เมื่อรูปร่างตรงกัน และเป็น projection 1×1 เมื่อเปลี่ยนจำนวนช่องหรือ stride ต้องบวก tensor ที่มีรูปร่างเท่ากัน', 'S(x) is an identity for matching shapes and a 1×1 projection when channels or stride change. Addition requires identical tensor shapes.'),
    strengths: tx('ทางลัดช่วยให้การฝึกโครงข่ายลึกส่งสัญญาณและ gradient ได้', 'Shortcuts support signal and gradient propagation in deep networks.'),
    limitations: tx('ยังมีต้นทุน convolution สูง การบวก shortcut ไม่เท่ากับการ concat ช่อง', 'Convolutions remain costly; residual addition is not channel concatenation.'),
    exercise: tx('เลือกบล็อกแรกของแต่ละ stage เทียบรูปร่าง main กับ projection แล้วเลือกบล็อกที่ใช้ identity shortcut', 'Inspect the first block of each stage and compare main/projection shapes, then find an identity shortcut.'),
    question: tx('shortcut ทำอะไรเมื่อจำนวนช่องเปลี่ยน', 'What happens when shortcut channels change?'), answers: [tx('ใช้ projection 1×1 ให้รูปร่างตรงกัน', 'Use a 1×1 projection to match shapes'), tx('บวก tensor คนละรูปร่างทันที', 'Add incompatible tensors'), tx('ตัดทางลัดทั้งหมด', 'Remove every shortcut')], correct: 0,
    reference: [{ name: 'Stem', shape: '56×56×64', operations: '224 RGB; 7×7 Conv s2 + BN + ReLU; max pool s2' }, ...[3, 4, 6, 3].map((repeats, i) => ({ name: `conv${i + 2}_x`, shape: `${56 / 2 ** i}×${56 / 2 ** i}×${[256, 512, 1024, 2048][i]}`, operations: '1×1 reduce → 3×3 → 1×1 expand; BN; shortcut add; ReLU', repeats })), { name: 'Head', shape: '2048 → 1000', operations: 'Global average pool → Dense → softmax' }],
    paper: { authors: 'K. He, X. Zhang, S. Ren, J. Sun', year: 2015, title: 'Deep Residual Learning for Image Recognition', venue: 'arXiv:1512.03385 / CVPR 2016', url: 'https://arxiv.org/abs/1512.03385' },
    implementation: 'https://github.com/keras-team/keras/blob/master/keras/src/applications/resnet.py',
  },
  densenet121: {
    id: 'densenet121', name: 'DenseNet121', year: 2016, input: 224, parameters: '~8M',
    summary: tx('Dense blocks [6,12,24,16] ส่ง feature เดิมให้ชั้นถัดไปผ่าน concatenation', 'Dense blocks [6,12,24,16] reuse earlier feature maps through concatenation.'),
    idea: tx('แต่ละชั้นรับช่องทั้งหมดที่สร้างก่อนหน้าในบล็อก แล้วเพิ่มช่องใหม่ตาม growth rate', 'Each layer receives all earlier feature channels in its block and appends growth-rate channels.'),
    formula: 'x_l=H_l([x_0,x_1,\\ldots,x_{l-1}]),\\quad C_l=C_0+kl',
    explanation: tx('H ประกอบด้วย BN–ReLU–1×1 และ BN–ReLU–3×3 Transition ใช้ 1×1 ลดช่องครึ่งหนึ่งและ average pool ลดขนาดภาพ', 'H uses BN–ReLU–1×1 followed by BN–ReLU–3×3. Transitions halve channels with 1×1 convolution and downsample with average pooling.'),
    strengths: tx('ใช้ feature ร่วมกัน ทำให้จำนวนพารามิเตอร์น้อยกว่าการเพิ่มช่องทุกชั้นแบบทั่วไป', 'Feature reuse can reduce parameter counts compared with independently wide layers.'),
    limitations: tx('ต้องเก็บ activation ที่จะ concat ทำให้ใช้หน่วยความจำระหว่างฝึกมาก', 'Retaining activations for concatenation can consume substantial training memory.'),
    exercise: tx('เลือก dense unit แล้วตรวจจำนวนช่องก่อนและหลัง concat เทียบกับ growth rate จากนั้นดู transition ลดช่อง', 'Inspect channels before/after a dense unit, compare with the growth rate, then inspect transition compression.'),
    question: tx('DenseNet รวม feature ด้วยวิธีใด', 'How does DenseNet combine features?'), answers: [tx('Concat ตามแกนช่อง', 'Concatenate along channels'), tx('บวกแบบ ResNet เท่านั้น', 'Only residual addition'), tx('โหวตคลาสเหมือน Random Forest', 'Class voting like a forest')], correct: 0,
    reference: [{ name: 'Stem', shape: '56×56×64', operations: '224 RGB; 7×7 Conv s2; BN–ReLU; max pool' }, ...[6, 12, 24, 16].flatMap((repeats, i) => [{ name: `Dense block ${i + 1}`, shape: `${[56, 28, 14, 7][i]}×${[56, 28, 14, 7][i]}×${[256, 512, 1024, 1024][i]}`, operations: 'BN–ReLU–Conv1×1(128) → BN–ReLU–Conv3×3(32) → concat; growth k=32', repeats }, ...(i < 3 ? [{ name: `Transition ${i + 1}`, shape: `${[28, 14, 7][i]}×${[28, 14, 7][i]}×${[128, 256, 512][i]}`, operations: 'BN–ReLU–Conv1×1; compression 0.5; average pool s2' }] : [])]), { name: 'Head', shape: '1024 → 1000', operations: 'BN–ReLU → global average pool → classifier' }],
    paper: { authors: 'G. Huang, Z. Liu, L. van der Maaten, K. Weinberger', year: 2016, title: 'Densely Connected Convolutional Networks', venue: 'arXiv:1608.06993 / CVPR 2017', url: 'https://arxiv.org/abs/1608.06993' },
    implementation: 'https://github.com/keras-team/keras/blob/master/keras/src/applications/densenet.py',
  },
  inceptionv3: {
    id: 'inceptionv3', name: 'InceptionV3', year: 2015, input: 299, parameters: '~23.9M',
    summary: tx('ประมวลผลหลายสาขาขนานกัน ใช้ convolution แบบแยกตัวประกอบ แล้ว concat feature', 'Parallel branches use factorized convolutions and concatenate their features.'),
    idea: tx('Inception-A/B/C ผสมหลาย receptive field โดยลดต้นทุนด้วย 1×1 และ factorization 1×7/7×1', 'Inception-A/B/C mix receptive fields while reducing cost with 1×1 projections and 1×7/7×1 factorization.'),
    formula: 'y=[B_1(x),B_2(x),B_3(x),B_4(x)]',
    explanation: tx('แต่ละสาขารับข้อมูลเดียวกัน ไม่ได้ต่อเรียงกัน สาขาที่จะ concat ต้องมีขนาดเชิงพื้นที่เท่ากัน Reduction-A/B ลดความละเอียดระหว่างช่วง', 'Branches receive the same input rather than forming a sequence. Spatial sizes must agree at concatenation; Reduction-A/B connect resolution stages.'),
    strengths: tx('รวมข้อมูลหลายขนาดและใช้ convolution แบบแยกตัวประกอบ', 'Combines multiple scales and efficient factorized convolutions.'),
    limitations: tx('มีหลายสาขาและกฎรูปร่างซับซ้อนกว่าลำดับ Conv ปกติ', 'Branching and shape constraints are more complex than sequential convolutions.'),
    exercise: tx('เลือก Mixed block และตามเส้นจาก input ไปแต่ละสาขา ตรวจว่าช่อง output เท่ากับผลรวมของทุกสาขา', 'Select a Mixed block, follow its parallel input paths, and check that output channels equal the sum of branch channels.'),
    question: tx('ก่อน concat สาขา Inception ต้องตรงกันที่ใด', 'What must match before Inception branch concatenation?'), answers: [tx('ความสูงและความกว้าง', 'Spatial height and width'), tx('จำนวนช่องทุกสาขาต้องเท่ากัน', 'Every branch channel count'), tx('ค่าพิกเซลทุกค่าต้องเท่ากัน', 'Every pixel value')], correct: 0,
    reference: [{ name: 'Stem', shape: '35×35×192', operations: '299 RGB; Conv3×3 → Conv3×3 → Conv3×3; pool; Conv1×1; Conv3×3; pool' }, { name: 'Inception-A ×3', shape: '35×35×256/288', operations: '1×1; 1×1→5×5; 1×1→3×3→3×3; avg pool→1×1; concat', repeats: 3 }, { name: 'Reduction-A', shape: '17×17×768', operations: 'Stride-2 Conv branch + stacked Conv branch + max-pool branch' }, { name: 'Inception-B ×4', shape: '17×17×768', operations: 'Factorized 1×7 / 7×1 branches; concat', repeats: 4 }, { name: 'Reduction-B', shape: '8×8×1280', operations: 'Two stride-2 convolution branches + pooling branch' }, { name: 'Inception-C ×2', shape: '8×8×2048', operations: 'Split 1×3 / 3×1 branches; concat', repeats: 2 }, { name: 'Head', shape: '2048 → 1000', operations: 'Global average pool → classifier (main inference path)' }],
    paper: { authors: 'C. Szegedy et al.', year: 2015, title: 'Rethinking the Inception Architecture for Computer Vision', venue: 'arXiv:1512.00567 / CVPR 2016', url: 'https://arxiv.org/abs/1512.00567' },
    implementation: 'https://github.com/keras-team/keras/blob/master/keras/src/applications/inception_v3.py',
  },
  mobilenetv2: {
    id: 'mobilenetv2', name: 'MobileNetV2', year: 2018, input: 224, parameters: '~3.5M',
    summary: tx('Expansion → depthwise → linear projection พร้อม inverted residual สำหรับอุปกรณ์พกพา', 'Expansion → depthwise → linear projection with inverted residuals for mobile computation.'),
    idea: tx('ขยายช่องก่อนคำนวณ depthwise แล้วบีบกลับด้วย linear bottleneck; shortcut ใช้เมื่อ stride=1 และช่องเท่ากัน', 'Expand channels before depthwise convolution, then project back with a linear bottleneck; shortcuts require stride 1 and matching channels.'),
    formula: 'y=\\operatorname{Project}_{linear}(\\operatorname{DWConv}(\\operatorname{Expand}(x)))',
    explanation: tx('Depthwise ใช้ kernel แยกต่อช่อง ไม่ผสมช่องจนถึง pointwise 1×1 ReLU6 ใช้ในทางขยายและ depthwise แต่ projection สุดท้ายเป็น linear', 'Depthwise uses one kernel per channel; pointwise 1×1 convolutions mix channels. Expansion and depthwise use ReLU6, while the last projection is linear.'),
    strengths: tx('ลดพารามิเตอร์และการคำนวณ เหมาะศึกษาประสิทธิภาพต่อขนาดโมเดล', 'Reduces parameters and computation for studying efficient models.'),
    limitations: tx('จำนวน FLOPs ต่ำไม่รับประกัน latency ต่ำบนฮาร์ดแวร์ทุกชนิด', 'Low FLOPs do not guarantee low latency on every device.'),
    exercise: tx('ตรวจว่าชั้น depthwise มี kernel ต่อช่อง และเปรียบเทียบพารามิเตอร์กับ Conv ปกติในขนาดเดียวกัน', 'Inspect one kernel per depthwise channel and compare parameter counts with a regular convolution of the same size.'),
    question: tx('ทำไม projection สุดท้ายเป็น linear', 'Why is the final projection linear?'), answers: [tx('หลีกเลี่ยงการตัดข้อมูลใน bottleneck แคบด้วย ReLU', 'Avoid discarding information with ReLU in a narrow bottleneck'), tx('เพราะไม่มีน้ำหนัก', 'Because it has no weights'), tx('เพราะเป็น pooling', 'Because it is pooling')], correct: 0,
    reference: [{ name: 'Stem', shape: '112×112×32', operations: '3×3 Conv s2; BN; ReLU6' }, ...[[1, 16, 1, 1], [6, 24, 2, 2], [6, 32, 3, 2], [6, 64, 4, 2], [6, 96, 3, 1], [6, 160, 3, 2], [6, 320, 1, 1]].map(([t, c, n, s], i) => ({ name: `Bottleneck group ${i + 1}`, shape: `${[112, 56, 28, 14, 14, 7, 7][i]}×${[112, 56, 28, 14, 14, 7, 7][i]}×${c}`, operations: `Expansion t=${t}; DW3×3; linear 1×1; first stride=${s}; conditional residual`, repeats: n })), { name: 'Head', shape: '7×7×1280 → 1000', operations: '1×1 Conv; BN–ReLU6; global average pool; classifier' }],
    paper: { authors: 'M. Sandler et al.', year: 2018, title: 'MobileNetV2: Inverted Residuals and Linear Bottlenecks', venue: 'arXiv:1801.04381 / CVPR 2018', url: 'https://arxiv.org/abs/1801.04381' },
    implementation: 'https://github.com/keras-team/keras/blob/master/keras/src/applications/mobilenet_v2.py',
  },
  vit: {
    id: 'vit', name: 'ViT', year: 2020, input: 224, parameters: '~86M (ViT-B/16)',
    summary: tx('แบ่งภาพเป็น patch tokens เติม CLS และตำแหน่ง แล้วผ่าน Transformer encoder', 'Split images into patch tokens, add CLS and position embeddings, then apply Transformer encoders.'),
    idea: tx('ViT-B/16 ใช้ patch 16×16, embedding 768 มิติ, 12 encoder blocks และ 12 attention heads', 'ViT-B/16 uses 16×16 patches, 768-dimensional embeddings, 12 encoder blocks and 12 attention heads.'),
    formula: 'Z_0=[x_{CLS};x_pE]+E_{pos},\\quad A=\\operatorname{softmax}(QK^T/\\sqrt{d_k})V',
    explanation: tx('patch มีลำดับตามตำแหน่งในภาพ CLS รวบรวมข้อมูลด้วย self-attention แต่ละ block ใช้ pre-LayerNorm, attention residual และ MLP residual ก่อนอ่าน CLS เพื่อจำแนก', 'Patches follow image positions. CLS aggregates information through self-attention. Each block uses pre-LayerNorm, attention residual and MLP residual before the classifier reads CLS.'),
    strengths: tx('เชื่อม patch ที่อยู่ห่างกันได้โดยตรง และตรวจ attention ระหว่างส่วนของภาพได้', 'Directly relates distant patches and exposes spatial attention.'),
    limitations: tx('attention ใช้หน่วยความจำ O(N²) ตามจำนวน patch และมักต้องการข้อมูลฝึกมาก', 'Attention uses O(N²) memory in patch count and typically needs extensive training data.'),
    exercise: tx('เปลี่ยน patch จาก 4 เป็น 8 ในภาพสาธิต 16×16 เปรียบเทียบจำนวน token และขนาด attention matrix', 'Change patches from 4 to 8 on the 16×16 teaching image; compare token counts and attention matrix sizes.'),
    question: tx('ภาพ 16×16 ใช้ patch 4×4 และ CLS หนึ่งตัว มี token เท่าใด', 'How many tokens come from a 16×16 image, 4×4 patches and one CLS token?'), answers: [tx('17', '17'), tx('16', '16'), tx('5', '5')], correct: 0,
    reference: [{ name: 'Patch projection', shape: '196×768', operations: '224×224 RGB; non-overlapping 16×16 patches; linear projection' }, { name: 'CLS + position', shape: '197×768', operations: 'Prepend learned CLS; add learned position embeddings' }, { name: 'Encoder ×12', shape: '197×768', operations: 'Pre-LN → 12-head attention → add; Pre-LN → MLP3072 + GELU → MLP768 → add', repeats: 12 }, { name: 'Head', shape: 'CLS768 → 1000', operations: 'Final LayerNorm; CLS readout; classification head' }],
    paper: { authors: 'A. Dosovitskiy et al.', year: 2020, title: 'An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale', venue: 'arXiv:2010.11929 / ICLR 2021', url: 'https://arxiv.org/abs/2010.11929' },
    implementation: 'https://github.com/google-research/vision_transformer',
  },
};
