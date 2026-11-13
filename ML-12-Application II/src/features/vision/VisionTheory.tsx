import { Link } from 'react-router-dom';
import { VISION_MODELS, type VisionId } from '../../content/visionModels';
import { useT } from '../../i18n/i18n';
import { Callout, Quiz, Tex } from '../../components/common';

export function VisionReferenceTable({ modelId }: { modelId: VisionId }) {
  const model = VISION_MODELS[modelId], t = useT();
  return <div className="scroll-thin overflow-x-auto"><table className="w-full min-w-[620px] text-left text-base">
    <thead className="text-muted"><tr><th className="p-3">{t('ส่วนของโมเดลเต็ม', 'Full-model stage')}</th><th className="p-3">{t('รูปร่างผลลัพธ์ H×W×C / tokens×D', 'Output shape H×W×C / tokens×D')}</th><th className="p-3">{t('การคำนวณ', 'Operations')}</th></tr></thead>
    <tbody>{model.reference.map((stage, i) => <tr key={i} className="border-t border-line"><th className="p-3 font-medium text-cyan">{stage.name}{stage.repeats ? ` ×${stage.repeats}` : ''}</th><td className="p-3 font-mono text-sm">{stage.shape}</td><td className="p-3 text-sm leading-relaxed">{stage.operations}</td></tr>)}</tbody>
  </table></div>;
}

export function VisionTheory({ modelId }: { modelId: VisionId }) {
  const m = VISION_MODELS[modelId], t = useT();
  return <div className="space-y-6">
    <div><h2 className="font-display text-2xl">{m.name} · {m.year}</h2><p className="mt-2 max-w-[80ch] text-base leading-relaxed">{t(m.idea)}</p></div>
    <Callout>{t('หน้า Algorithm detail แสดงการคำนวณจริงของโมเดลสาธิตบนภาพตัวเลข 16×16 ช่องสัญญาณเล็กกว่าและมี 10 คลาส ตารางอ้างอิงด้านล่างอธิบายสถาปัตยกรรม RGB ต้นฉบับ Backbone สาธิตใช้น้ำหนักสุ่มที่ตรึงไว้ ฝึกเฉพาะหัว softmax ไม่มีน้ำหนัก ImageNet และไม่ใช่ผลทดสอบโมเดลเต็ม', 'Algorithm detail shows actual computations in the 16×16 digit teaching model with fewer channels and 10 classes. The reference table below documents the original RGB architecture. The teaching backbone uses frozen random weights; only the softmax head trains. No ImageNet weights are loaded and these are not full-model benchmarks.')}</Callout>
    <section><h3 className="font-display text-xl">{t('สมการและหลักการ', 'Equations and mechanism')}</h3><Tex block>{m.formula}</Tex><p className="max-w-[80ch] text-base leading-relaxed">{t(m.explanation)}</p></section>
    <section><h3 className="mb-3 font-display text-xl">{t('โครงสร้างอ้างอิง', 'Reference architecture')} · {m.input}×{m.input} RGB · {m.parameters}</h3><VisionReferenceTable modelId={modelId} /></section>
    <div className="grid gap-4 sm:grid-cols-2"><div className="rounded-xl border border-green/30 p-4"><h3 className="font-semibold text-green">{t('จุดเด่น', 'Strengths')}</h3><p className="mt-2 text-base">{t(m.strengths)}</p></div><div className="rounded-xl border border-orange/30 p-4"><h3 className="font-semibold text-orange">{t('ข้อจำกัด', 'Limitations')}</h3><p className="mt-2 text-base">{t(m.limitations)}</p></div></div>
    <section><h3 className="font-display text-xl">{t('การทดลองที่รันจริงในเว็บ', 'What the browser experiment computes')}</h3>
      <ol className="mt-3 list-decimal space-y-2 pl-6 text-base leading-relaxed">
        <li>{t('แบ่งชุดตัวเลขจริงเป็น train/test ก่อนเลือกตัวอย่างแบบสมดุล ปรับพิกเซลเป็น 0–1 ไม่มีภาพ test เข้าไปฝึก', 'Split real digits into train/test before balanced sampling. Scale pixels to 0–1; test images do not train the classifier.')}</li>
        <li>{t('คงจำนวนบล็อกหลักและการเชื่อมต่อของแต่ละตระกูล แต่ลดช่องและขนาดภาพ การลด spatial dimension หยุดที่ 1×1 และ BatchNorm ใช้สถิติเริ่มต้นคงที่ ไม่อัปเดตระหว่างฝึก', 'Keep each family’s block counts and connections while reducing image/channel sizes. Spatial downsampling stops at 1×1; BatchNorm uses fixed initial statistics rather than updating during head training.')}</li>
        <li>{t('คำนวณ features ด้วย backbone สุ่ม แล้ว fit ค่าเฉลี่ยและส่วนเบี่ยงเบนมาตรฐานของ features จาก train เท่านั้น', 'Extract frozen random features and fit feature means/standard deviations on the training set only.')}</li>
        <li>{t('ฝึกหัว Dense 10 คลาสด้วย Adam หรือ SGD และ cross-entropy ตรวจ loss, accuracy, precision, recall, F1 และ confusion matrix บน test', 'Train the 10-class dense head with Adam or SGD and cross-entropy. Evaluate loss, accuracy, precision, recall, F1 and the test confusion matrix.')}</li>
      </ol>
      <Tex block>{'p=\\operatorname{softmax}(Wz+b),\\quad L=-\\log p_y+\\frac{\\lambda}{2}\\lVert W\\rVert^2,\\quad \\frac{\\partial L}{\\partial W}=(p-y_{onehot})z^T+\\lambda W'}</Tex>
      <p className="text-base text-muted">{t('ความแม่นยำที่เห็นเป็นของหัวจำแนกบน features สุ่ม ไม่ใช่การ fine-tune หรือฝึก backbone ทั้งตัว ผลที่ต่ำเป็นผลทดลองจริง ไม่แทนด้วยคะแนนจากงานวิจัย', 'Displayed accuracy belongs to a head on random features. This is not backbone fine-tuning. Low scores are real experiment outcomes, not replaced with published benchmark numbers.')}</p>
    </section>
    <section className="rounded-xl border border-line p-4"><h3 className="font-display text-xl">{t('ทดลองด้วยตนเอง', 'Try it')}</h3><p className="mt-2 text-base">{t(m.exercise)}</p><p className="mt-2 text-base">{t('จด seed, จำนวนตัวอย่าง, learning rate และ epoch ก่อนเปรียบเทียบผล ส่งออก JSON เพื่อเก็บค่าและน้ำหนักหัวจำแนก', 'Record seed, sample count, learning rate and epochs before comparing runs. Export JSON to retain configuration and classifier weights.')}</p></section>
    <Quiz id={`vision-${modelId}`} items={[
      { q: t(m.question), options: m.answers.map((a) => t(a)), answer: m.correct, why: t(m.explanation) },
      { q: t('คะแนน accuracy ในแล็บนี้วัดอะไร', 'What does this lab’s accuracy measure?'), options: [t('หัวจำแนก 10 คลาสบน features จาก backbone สุ่มที่ตรึงไว้', 'A 10-class head on frozen random backbone features'), t('คะแนน ImageNet ของโมเดลเต็ม', 'Full-model ImageNet accuracy'), t('การ fine-tune ทุกชั้น', 'Fine-tuning every layer')], answer: 0, why: t('ปรับน้ำหนักเฉพาะหัวจำแนก และเก็บ test แยกจากการฝึก', 'Only head weights are optimized, with a held-out test set.') },
    ]} />
    <section className="space-y-2 border-t border-line pt-4"><h3 className="font-display text-xl">{t('เอกสารต้นฉบับ', 'Primary references')}</h3><p className="text-base"><a href={m.paper.url} target="_blank" rel="noreferrer" className="text-cyan underline">{m.paper.title}</a> · {m.paper.authors} · {m.paper.venue}</p><p><a href={m.implementation} target="_blank" rel="noreferrer" className="text-cyan underline">{t('โค้ดอ้างอิงอย่างเป็นทางการ', 'Official reference implementation')}</a></p><Link to={`/algorithms/${modelId}`} className="inline-block text-cyan underline">{t('เปิดห้องทดลอง', 'Open the lab')} → {m.name}</Link></section>
  </div>;
}
