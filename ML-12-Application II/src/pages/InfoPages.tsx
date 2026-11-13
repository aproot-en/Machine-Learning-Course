import { Link } from 'react-router-dom';
import { useT, type Text } from '../i18n/i18n';
import { ALGORITHMS, CATEGORY_NAME, PAPERS, type Category } from '../content/algorithms';
import { Icon } from '../components/ui';

const PLANNED: Record<string, { title: string; phase: number; text: Text }> = {
  playground: { title: '3D ML Playground', phase: 8, text: { th: 'ผืนผ้าใบขนาดใหญ่สำหรับสลับอัลกอริทึมบนชุดข้อมูลเดียวกัน พร้อมตัวควบคุมแบบลอยและการรันทีละขั้น', en: 'A large canvas to switch algorithms on the same dataset, with floating controls and step-by-step execution.' } },
  '404': { title: 'Not found', phase: 0, text: { th: 'ไม่พบหน้านี้', en: 'This page does not exist.' } },
};

export function Planned({ page }: { page: string }) {
  const t = useT();
  const algo = ALGORITHMS.find((a) => a.slug === page);
  const info = PLANNED[page] ?? (algo ? { title: algo.name, phase: algo.phase, text: algo.summary } : PLANNED['404']);
  return (
    <div className="mx-auto max-w-3xl px-4 py-24 sm:px-6">
      {info.phase > 0 && <span className="chip">{t(`กำหนดใน Phase ${info.phase}`, `Scheduled for Phase ${info.phase}`)}</span>}
      <h1 className="mt-4 font-display text-4xl font-semibold">{info.title}</h1>
      <p className="mt-4 text-[1.1875rem] leading-7 text-muted">{t(info.text)}</p>
      {info.phase > 0 && <p className="mt-4 text-[1.125rem] text-muted">{t('หน้านี้ยังไม่เปิดใช้งาน ตามแผนการพัฒนา ระหว่างนี้ลองห้องทดลองที่พร้อมใช้:', 'This page is not open yet, per the development plan. Meanwhile, try the labs that are ready:')}</p>}
      <div className="mt-8 flex flex-wrap gap-3">
        <Link to="/algorithms" className="inline-flex h-10 items-center gap-2 rounded-lg border border-line px-4 hover:border-cyan/60">{t('อัลกอริทึมทั้งหมด', 'All algorithms')} <Icon name="arrow" /></Link>
        <Link to="/" className="inline-flex h-10 items-center gap-2 rounded-lg border border-line px-4 hover:border-cyan/60">{t('หน้าแรก', 'Home')} <Icon name="arrow" /></Link>
      </div>
    </div>
  );
}

const SECTIONS: { to: string; name: Text; text: Text }[] = [
  { to: '/', name: { th: 'หน้าแรก', en: 'Home' }, text: { th: 'ภาพรวมของห้องทดลอง พร้อมทางลัดไปยังอัลกอริทึมแต่ละหมวด', en: 'Overview of the lab with shortcuts to each algorithm category.' } },
  { to: '/algorithms', name: { th: 'อัลกอริทึม', en: 'Algorithms' }, text: { th: 'ห้องทดลอง 3 มิติ 20 อัลกอริทึม แต่ละห้องมีตัวควบคุมพารามิเตอร์ การเล่นทีละขั้น คำอธิบาย สมการ และแบบทดสอบ', en: '20 interactive 3D labs, each with parameter controls, step-by-step playback, explanations, equations and a quiz.' } },
  { to: '/network-builder', name: { th: 'สร้างโครงข่าย', en: 'Network Builder' }, text: { th: 'ออกแบบ MLP เอง (จำนวนชั้น นิวรอน activation optimizer) ฝึกจริงในเบราว์เซอร์ ดู forward/backward propagation ตรวจน้ำหนักทุกเส้น และส่งออกเป็น JSON, SVG, PNG', en: 'Design an MLP (layers, units, activations, optimizer), train it in the browser, watch forward/backward passes, inspect every weight and export JSON, SVG or PNG.' } },
  { to: '/optimization', name: { th: 'การเพิ่มประสิทธิภาพ', en: 'Optimization' }, text: { th: 'พื้นผิว loss 3 มิติ (หรือพิมพ์สมการเอง) เปรียบเทียบเส้นทางของ SGD, Momentum, RMSprop และ Adam จากจุดเริ่มเดียวกัน', en: '3D loss surfaces (or type your own formula) comparing SGD, Momentum, RMSprop and Adam paths from the same start.' } },
  { to: '/datasets', name: { th: 'ชุดข้อมูล', en: 'Datasets' }, text: { th: 'สำรวจชุดข้อมูลจริงหรืออัปโหลด CSV ของตนเอง ดูสถิติ ฮิสโทแกรม การจัดการค่าว่าง การ scale และ PCA ลงใน 3 มิติ', en: 'Explore real datasets or upload your own CSV: statistics, histograms, missing values, scaling and a 3D PCA projection.' } },
  { to: '/learn', name: { th: 'บทเรียน', en: 'Lessons' }, text: { th: 'ทฤษฎี 21 บท ตั้งแต่ข้อมูลและการวัดผล การคำนวณในนิวรอน backpropagation ไปจนถึง Transformer พร้อมกราฟและแผนภาพที่ปรับค่าได้', en: '21 theory lessons, from data and evaluation through neuron computation and backpropagation to the Transformer, with adjustable plots and diagrams.' } },
  { to: '/compare', name: { th: 'เปรียบเทียบ', en: 'Compare' }, text: { th: 'เลือกโมเดล A กับโมเดล B ฝึกบนข้อมูลชุดเดียวกัน แล้วดูขอบเขตการตัดสินใจ 3 มิติเคียงกัน จุดที่ทำนายต่างกัน และตัวชี้วัด', en: 'Pick model A and model B, train them on the same data, and compare their 3D decision boundaries, disagreements and metrics side by side.' } },
];

const DATASETS: { name: Text; text: Text }[] = [
  { name: { th: 'Iris', en: 'Iris' }, text: { th: 'ดอกไอริส 150 ตัวอย่าง 4 ลักษณะ 3 สายพันธุ์', en: '150 flowers, 4 features, 3 species' } },
  { name: { th: 'Wine', en: 'Wine' }, text: { th: 'ไวน์ 178 ตัวอย่าง 13 ลักษณะทางเคมี 3 แหล่งผลิต', en: '178 wines, 13 chemical features, 3 cultivars' } },
  { name: { th: 'Breast Cancer', en: 'Breast Cancer' }, text: { th: 'มะเร็งเต้านม 569 ตัวอย่าง 30 ลักษณะ 2 คลาส', en: '569 samples, 30 features, 2 classes' } },
  { name: { th: 'Digits', en: 'Digits' }, text: { th: 'ภาพตัวเลขเขียนมือ 0–9 จาก scikit-learn ใช้ใน CNN Lab', en: 'Handwritten digits 0–9 from scikit-learn, used by the CNN Lab' } },
  { name: { th: 'ข้อมูลสังเคราะห์', en: 'Synthetic' }, text: { th: 'moons, circles, spiral, XOR, blobs ฯลฯ สร้างใหม่ได้ด้วย seed และปรับ noise ได้', en: 'moons, circles, spiral, XOR, blobs… regenerated by seed with adjustable noise' } },
  { name: { th: 'CSV', en: 'CSV' }, text: { th: 'อัปโหลดไฟล์ของผู้ใช้เองในหน้าชุดข้อมูล ประมวลผลในเครื่อง ไม่ส่งออกไปที่ใด', en: 'upload your own file on the Datasets page; it is processed locally and never sent anywhere' } },
];

const KEYS: [string, Text][] = [
  ['Space', { th: 'เล่น / หยุด', en: 'Play / pause' }],
  ['→ / ←', { th: 'ขั้นถัดไป / ย้อนกลับ', en: 'Next / previous step' }],
  ['R', { th: 'เริ่มใหม่', en: 'Reset' }],
  ['1 2 3 4', { th: 'มุมกล้อง ไอโซ / หน้า / บน / ข้าง', en: 'Camera: iso / front / top / side' }],
  ['F', { th: 'เต็มจอ', en: 'Fullscreen' }],
];

export function About() {
  const t = useT();
  const cats = Object.keys(CATEGORY_NAME) as Category[];
  return (
    <div className="mx-auto max-w-[1100px] px-4 py-12 sm:px-6">
      <p className="label-xs">{t('เกี่ยวกับ', 'About')}</p>
      <h1 className="mt-3 font-display text-4xl font-semibold">ML Visual Lab</h1>
      <p className="mt-4 max-w-[70ch] text-[1.1875rem] leading-7 text-ink/90">
        {t('สื่อการสอนสำหรับนักศึกษามหาวิทยาลัย เพื่อสำรวจการคำนวณของ Machine Learning และ Deep Learning ผ่านฉาก 3D ที่หมุน ซูม และเดินทีละขั้นได้ แล็บคำนวณในเบราว์เซอร์ แต่ละหน้าระบุว่าส่วนใดฝึกได้ ส่วนใดใช้น้ำหนักสุ่มหรือเป็นแผนผังอ้างอิง โมเดลภาพขั้นสูงใช้รุ่นสาธิตขนาดย่อและฝึกเฉพาะหัวจำแนก',
          'A teaching tool for university students to explore machine-learning and deep-learning computations in 3D. Labs compute locally in the browser; each page identifies trainable components, random weights and reference-only diagrams. Advanced vision labs use compact teaching backbones and train only their classifier heads.')}
      </p>

      <h2 className="mt-12 font-display text-2xl font-semibold">{t('มีอะไรบ้างในเว็บไซต์', 'What is in the site')}</h2>
      <div className="mt-5 grid gap-3 sm:grid-cols-2">
        {SECTIONS.map((s) => (
          <Link key={s.to} to={s.to} className="rounded-xl border border-line bg-surface/50 p-4 hover:border-cyan/60">
            <div className="flex items-center justify-between font-medium">{t(s.name)}<Icon name="arrow" /></div>
            <p className="mt-1.5 text-[1.0625rem] leading-6 text-muted">{t(s.text)}</p>
          </Link>
        ))}
      </div>

      <h2 className="mt-12 font-display text-2xl font-semibold">{t('อัลกอริทึมและงานวิจัยต้นฉบับ', 'Algorithms and their original papers')}</h2>
      <p className="mt-2 text-[1.0625rem] text-muted">{t('คลิกชื่อบทความเพื่อเปิดงานวิจัยต้นฉบับที่เสนออัลกอริทึมนั้น (เปิดในแท็บใหม่ ผ่าน DOI หรือ arXiv)', 'Click a paper title to open the original publication that introduced the algorithm (new tab, via DOI or arXiv).')}</p>
      <div className="mt-5 grid gap-4 md:grid-cols-2">
        {cats.map((c) => (
          <div key={c} className="rounded-xl border border-line bg-surface/50 p-4">
            <div className="label-xs">{t(CATEGORY_NAME[c])}</div>
            <ul className="mt-3 flex flex-col gap-3.5">
              {ALGORITHMS.filter((a) => a.category === c).map((a) => (
                <li key={a.slug} className="text-[1.0625rem] leading-6">
                  <div><span className="font-medium">{a.name}</span><span className="text-muted"> — {t(a.summary)}</span></div>
                  {(PAPERS[a.slug] ?? []).map((p) => (
                    <a key={p.url} href={p.url} target="_blank" rel="noopener noreferrer" className="group mt-1 flex gap-2 text-[1rem] leading-5">
                      <span className="shrink-0 font-mono text-cyan/80">↗</span>
                      <span>
                        <span className="text-cyan group-hover:underline">{p.title}</span>
                        <span className="text-muted"> · {p.authors} ({p.year}) · <i>{p.venue}</i></span>
                      </span>
                    </a>
                  ))}
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div className="mt-12 grid gap-10 md:grid-cols-2">
        <section>
          <h2 className="font-display text-2xl font-semibold">{t('ชุดข้อมูล', 'Datasets')}</h2>
          <ul className="mt-4 flex flex-col gap-2 text-[1.09375rem] leading-6">
            {DATASETS.map((d) => (
              <li key={d.name.en}><b>{t(d.name)}</b> <span className="text-muted">— {t(d.text)}</span></li>
            ))}
          </ul>
        </section>
        <section>
          <h2 className="font-display text-2xl font-semibold">{t('วิธีใช้', 'How to use it')}</h2>
          <ul className="mt-4 flex flex-col gap-2 text-[1.09375rem] leading-6">
            <li>{t('เลือกภาษา ไทย / English และระดับ พื้นฐาน / ปานกลาง / ขั้นสูง ได้ที่แถบด้านบน ระดับที่สูงขึ้นจะแสดงสมการและตัวควบคุมเพิ่ม', 'Switch Thai / English and Beginner / Intermediate / Advanced in the top bar; higher levels show more equations and controls.')}</li>
            <li>{t('ลากเพื่อหมุนฉาก เลื่อนล้อเมาส์เพื่อซูม คลิกที่จุดหรือนิวรอนเพื่อดูค่าที่คำนวณได้', 'Drag to rotate, scroll to zoom, click a point or neuron to see its computed values.')}</li>
            <li>{t('ปุ่มลัดในห้องทดลอง:', 'Lab shortcuts:')}
              <div className="mt-2 grid grid-cols-[auto_1fr] items-center gap-x-3 gap-y-1">
                {KEYS.map(([k, d]) => [
                  <kbd key={k} className="rounded border border-line px-1.5 text-center font-mono text-[0.9375rem]">{k}</kbd>,
                  <span key={`${k}-d`} className="text-muted">{t(d)}</span>,
                ])}
              </div>
            </li>
          </ul>
        </section>
      </div>

      <div className="mt-12 grid gap-10 md:grid-cols-2">
        <section className="space-y-3 text-[1.125rem] leading-7 text-ink/90">
          <h2 className="font-display text-2xl font-semibold">{t('หลักการออกแบบ', 'Design principles')}</h2>
          <p>{t('ทุกภาพต้องตรงกับการคำนวณจริง พารามิเตอร์ทุกตัวมีผลกับโมเดล ไม่มีการแต่งค่าความแม่นยำ แอนิเมชันแสดงการคำนวณที่มีความหมาย เช่น ค่า w·a ที่ไหลผ่านเส้นเชื่อม ไม่ใช่การจำลองกระแสไฟฟ้าในสมอง',
            'Every visual matches a real computation, every parameter drives the model and no accuracy is invented. Animations show meaningful computation, such as w·a flowing along a connection, not electrical signals in a brain.')}</p>
          <p>{t('ผลลัพธ์ทำซ้ำได้ ข้อมูลสุ่มใช้ seed แบ่ง train/test แบบ stratified และ fit ตัว scale เฉพาะบนชุด train', 'Results are reproducible: random data uses a seed, train/test splits are stratified and scalers are fitted on the training set only.')}</p>
        </section>
        <section className="space-y-3 text-[1.125rem] leading-7 text-ink/90">
          <h2 className="font-display text-2xl font-semibold">{t('การตรวจสอบความถูกต้อง', 'Verification')}</h2>
          <p>{t('เอนจินคำนวณแยกจากส่วนแสดงผล และมี unit test 36 รายการ รวมถึงการเทียบ gradient จาก backpropagation (MLP, CNN, RNN/LSTM) กับ gradient เชิงตัวเลข',
            'The compute engine is separate from rendering and covered by 36 unit tests, including checking backpropagation gradients (MLP, CNN, RNN/LSTM) against numerical gradients.')}</p>
          <p>{t('ตัวอย่างผลที่ได้: CNN ที่ฝึกไว้ล่วงหน้าแม่นยำ 97.5% บนชุดทดสอบ Digits, PCA ของ Iris อธิบายความแปรปรวนได้ 92.5% ใน 2 แกนแรก, LSTM จำข้อมูลลำดับยาวได้ขณะที่ RNN ธรรมดาทำไม่ได้',
            'Sample results: the pretrained CNN reaches 97.5% test accuracy on Digits, Iris PCA explains 92.5% of variance in two axes, and an LSTM remembers long sequences where a plain RNN fails.')}</p>
        </section>
      </div>

      <section className="mt-12 rounded-xl border border-line bg-surface/50 p-5">
        <h2 className="label-xs">{t('เทคโนโลยี', 'Technology')}</h2>
        <p className="mt-2 text-[1.09375rem]">React · TypeScript · Vite · Tailwind CSS · Three.js · React Three Fiber · Drei · Zustand · Motion · Plotly.js · KaTeX</p>
        <p className="mt-2 text-[1.03125rem] text-muted">{t('อัลกอริทึมทั้งหมดเขียนขึ้นเองด้วย TypeScript ไม่ได้เรียกไลบรารี ML ภายนอก เพื่อให้แสดงทุกขั้นของการคำนวณได้ ชุดข้อมูลจริงส่งออกมาจาก scikit-learn',
          'All algorithms are written from scratch in TypeScript rather than calling an ML library, so every step of the computation can be shown. Real datasets were exported from scikit-learn.')}</p>
      </section>
    </div>
  );
}
