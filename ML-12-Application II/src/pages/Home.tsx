import { useState } from 'react';
import { Link } from 'react-router-dom';
import { motion } from 'framer-motion';
import { Stage, useCameraState } from '../engine3d/Stage';
import { HeroNetwork } from './HeroNetwork';
import { useT } from '../i18n/i18n';
import { ALGORITHMS, CATEGORY_NAME } from '../content/algorithms';
import { Icon } from '../components/ui';
import { CLASS_COLORS } from '../engine3d/palette';

export default function Home() {
  const t = useT();
  const [cam] = useCameraState({ preset: 'iso', autoRotate: true });
  const [out, setOut] = useState<number[]>([1 / 3, 1 / 3, 1 / 3]);
  const available = ALGORITHMS.filter((a) => a.status === 'available');
  return (
    <>
      <section className="relative h-[calc(100svh-56px)] min-h-[560px] overflow-hidden">
        <div className="absolute inset-0">
          <Stage cam={cam} distance={17}>
            <group position={[4.5, 0, 0]}><HeroNetwork onCycle={(_, o) => setOut(o)} /></group>
          </Stage>
        </div>
        <div className="pointer-events-none absolute inset-0 bg-gradient-to-r from-bg via-bg/70 to-transparent" />
        <div className="pointer-events-none relative mx-auto flex h-full max-w-[1600px] flex-col justify-center px-4 sm:px-6">
          <motion.div initial={{ opacity: 0.001, y: 12 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }} className="pointer-events-auto max-w-xl">
            <p className="label-xs text-cyan">ML Visual Lab</p>
            <h1 className="mt-3 font-display text-[clamp(2.3rem,5vw,4.2rem)] font-semibold leading-[1.04] tracking-tight">How AI Learns</h1>
            <p className="mt-5 max-w-[52ch] text-[1.21875rem] leading-7 text-ink/85">
              {t('ห้องทดลอง 3 มิติสำหรับนักศึกษา ปรับพารามิเตอร์ สร้างโครงข่ายเอง แล้วดูว่าโมเดลคำนวณอะไรในแต่ละขั้น ทุกเส้น ทุกจุด และทุกตัวเลขมาจากการคำนวณจริง',
                'A 3D laboratory for students. Tune parameters, build your own networks and watch what the model computes at each step. Every line, point and number comes from real computation.')}
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link to="/algorithms" className="inline-flex h-11 items-center gap-2 rounded-lg bg-cyan px-5 font-semibold text-bg hover:bg-cyan/90">
                {t('อัลกอริทึม', 'Algorithms')}<Icon name="arrow" />
              </Link>
            </div>
          </motion.div>
        </div>
        <div className="glass absolute bottom-5 right-4 max-w-[300px] px-4 py-3 font-mono text-[0.9375rem] sm:right-6">
          <div className="text-muted">{t('สด: forward pass ของโครงข่าย 4-16-16-9-3 (น้ำหนักสุ่ม ยังไม่ฝึก)', 'Live: forward pass of a 4-16-16-9-3 network (random, untrained weights)')}</div>
          <div className="mt-2 flex flex-col gap-1">
            {out.map((p, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="w-14 text-muted">softmax{i}</span>
                <span className="h-1.5 flex-1 rounded bg-line"><span className="block h-full rounded" style={{ width: `${p * 100}%`, background: CLASS_COLORS[i] }} /></span>
                <span className="num w-12 text-right">{p.toFixed(3)}</span>
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-[1600px] px-4 py-20 sm:px-6">
        <div className="grid gap-10 lg:grid-cols-[1fr_3fr]">
          <div>
            <h2 className="font-display text-3xl font-semibold">{t('การเรียนรู้ของเครื่องแบบ 3 มิติ', 'Machine learning in 3D')}</h2>
            <p className="mt-4 max-w-[46ch] text-muted">{t('ศึกษาและทดลองการทำงานของการเรียนรู้ของเครื่องผ่านการจำลองแบบ 3 มิติ พร้อมปรับแต่งพารามิเตอร์ ดูกระบวนการคำนวณทีละขั้นตอน และวิเคราะห์ผลลัพธ์ของแบบจำลอง',
              'Study and experiment with how machine learning works through 3D simulation: tune parameters, follow the computation step by step and analyse the model’s results.')}</p>
          </div>
          <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
            {available.map((a) => (
              <Link key={a.slug} to={a.route!} className="group rounded-2xl border border-line bg-surface/60 p-5 transition-colors hover:border-cyan/60">
                <span className="chip">{t(CATEGORY_NAME[a.category])}</span>
                <h3 className="mt-3 font-display text-xl font-semibold">{a.name}</h3>
                <p className="mt-3 text-[1.125rem] text-muted">{t(a.summary)}</p>
                <span className="mt-4 inline-flex items-center gap-2 text-cyan">{t('เปิดห้องทดลอง', 'Open lab')}<Icon name="arrow" className="h-4 w-4 transition-transform group-hover:translate-x-1" /></span>
              </Link>
            ))}
          </div>
        </div>
      </section>

      <section className="mx-auto max-w-[1600px] px-4 py-20 sm:px-6">
        <p className="label-xs">{t('วัตถุประสงค์การเรียนรู้', 'Learning objectives')}</p>
        <div className="mt-6 grid gap-8 md:grid-cols-3">
          {[
            [t('เรียนรู้กระบวนการคำนวณ', 'Learn the computation'), t('ศึกษาการทำงานของอัลกอริทึมผ่านภาพ 3 มิติ พร้อมแสดงค่าพารามิเตอร์ สมการ และผลการคำนวณในแต่ละขั้นตอน', 'Study how algorithms work through 3D visuals, with the parameters, equations and computed results shown at every step.')],
            [t('ทดลองปรับพารามิเตอร์', 'Experiment with parameters'), t('ปรับแต่งพารามิเตอร์และโครงสร้างของแบบจำลอง เพื่อศึกษาผลกระทบต่อกระบวนการเรียนรู้และประสิทธิภาพของอัลกอริทึม', 'Tune the parameters and structure of a model to study their effect on the learning process and the algorithm’s performance.')],
            [t('ศึกษาการทำงานทีละขั้นตอน', 'Study it step by step'), t('ควบคุมการจำลองด้วยการเริ่ม หยุด และย้อนดูขั้นตอนที่ผ่านมา เพื่อทำความเข้าใจลำดับการประมวลผลของอัลกอริทึม', 'Control the simulation by starting, pausing and stepping back through earlier stages to understand the order in which the algorithm processes data.')],
          ].map(([h, p]) => (
            <div key={h} className="border-t border-line pt-5">
              <h3 className="font-display text-xl font-semibold">{h}</h3>
              <p className="mt-2 text-[1.125rem] leading-7 text-muted">{p}</p>
            </div>
          ))}
        </div>
      </section>
    </>
  );
}
