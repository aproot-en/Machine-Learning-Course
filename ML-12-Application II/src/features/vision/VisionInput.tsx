import { useEffect, useRef, useState } from 'react';
import { useT } from '../../i18n/i18n';
import { Button } from '../../components/ui';
import { digitInput, grayscaleImage } from '../../data/imageInput';

export function TensorImage({ values, h, w, size = 144 }: { values: ArrayLike<number>; h: number; w: number; size?: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const context = ref.current?.getContext('2d'); if (!context) return;
    const image = context.createImageData(w, h); let max = 1e-12;
    for (let i = 0; i < h * w; i++) max = Math.max(max, Math.abs(values[i] ?? 0));
    for (let i = 0; i < h * w; i++) {
      const value = (values[i] ?? 0) / max, magnitude = Math.round(Math.abs(value) * 255);
      image.data.set(value < 0 ? [magnitude * 0.65, magnitude * 0.3, magnitude, 255] : [0, magnitude * 0.85, magnitude, 255], i * 4);
    }
    context.putImageData(image, 0, 0);
  }, [values, h, w]);
  return <canvas ref={ref} width={w} height={h} style={{ width: size, height: size, imageRendering: 'pixelated' }} className="max-w-full rounded border border-line bg-bg" aria-label={`Tensor ${h}×${w}`} />;
}

export function VisionInput({ image, onChange }: { image: Float32Array; onChange: (image: Float32Array) => void }) {
  const t = useT();
  const ref = useRef<HTMLCanvasElement>(null);
  const drawing = useRef(false);
  const [error, setError] = useState('');
  useEffect(() => {
    if (drawing.current) return;
    const ctx = ref.current?.getContext('2d'); if (!ctx) return;
    const small = document.createElement('canvas'); small.width = small.height = 16;
    const context = small.getContext('2d')!, data = context.createImageData(16, 16);
    image.forEach((v, i) => data.data.set([v * 255, v * 255, v * 255, 255], i * 4)); context.putImageData(data, 0, 0);
    ctx.imageSmoothingEnabled = false; ctx.drawImage(small, 0, 0, 160, 160);
  }, [image]);
  const read = (source: HTMLCanvasElement) => {
    const data = source.getContext('2d')!.getImageData(0, 0, source.width, source.height).data;
    onChange(digitInput(grayscaleImage(data, source.width, source.height, 'light'), source.width, source.height));
  };
  return <div className="space-y-3">
    <p className="text-sm text-muted">{t('วาดตัวเลขสีขาวบนพื้นดำ หรืออัปโหลดภาพ ระบบย่อเป็น grayscale 16×16 ในเครื่อง', 'Draw a white digit on black, or upload an image. Images become 16×16 grayscale locally.')}</p>
    <canvas ref={ref} width={160} height={160} className="touch-none rounded-lg border border-line bg-black" aria-label={t('กระดานวาดตัวเลข', 'Digit drawing pad')}
      onPointerDown={(event) => { event.currentTarget.setPointerCapture(event.pointerId); drawing.current = true; const rect = event.currentTarget.getBoundingClientRect(), ctx = event.currentTarget.getContext('2d')!, x = (event.clientX - rect.left) * 160 / rect.width, y = (event.clientY - rect.top) * 160 / rect.height; ctx.fillStyle = '#fff'; ctx.beginPath(); ctx.arc(x, y, 6.5, 0, Math.PI * 2); ctx.fill(); ctx.beginPath(); ctx.moveTo(x, y); }}
      onPointerMove={(event) => { if (!drawing.current) return; const rect = event.currentTarget.getBoundingClientRect(), ctx = event.currentTarget.getContext('2d')!; ctx.strokeStyle = '#fff'; ctx.lineWidth = 13; ctx.lineCap = 'round'; ctx.lineJoin = 'round'; ctx.lineTo((event.clientX - rect.left) * 160 / rect.width, (event.clientY - rect.top) * 160 / rect.height); ctx.stroke(); }}
      onPointerUp={(event) => { if (!drawing.current) return; drawing.current = false; read(event.currentTarget); }} onPointerCancel={(event) => { if (!drawing.current) return; drawing.current = false; read(event.currentTarget); }} />
    <div className="flex flex-wrap gap-2"><Button size="sm" onClick={() => onChange(new Float32Array(256))}>{t('ล้างภาพ', 'Clear')}</Button>
      <label className="cursor-pointer rounded-lg border border-line px-3 py-1.5 text-base hover:border-cyan">{t('อัปโหลดภาพ', 'Upload image')}
        <input type="file" accept="image/*" className="sr-only" onChange={(event) => {
          const file = event.target.files?.[0]; if (!file) return; setError('');
          const url = URL.createObjectURL(file), input = new Image();
          input.onload = () => {
            try {
              const scale = Math.min(1, 320 / Math.max(input.width, input.height));
              const canvas = document.createElement('canvas'); canvas.width = Math.max(1, Math.round(input.width * scale)); canvas.height = Math.max(1, Math.round(input.height * scale)); const ctx = canvas.getContext('2d')!;
              ctx.drawImage(input, 0, 0, canvas.width, canvas.height);
              const pixels = ctx.getImageData(0, 0, canvas.width, canvas.height);
              onChange(digitInput(grayscaleImage(pixels.data, canvas.width, canvas.height), canvas.width, canvas.height));
            } catch { setError(t('อ่านภาพไม่ได้', 'Unable to read the image')); }
            finally { URL.revokeObjectURL(url); }
          };
          input.onerror = () => { URL.revokeObjectURL(url); setError(t('ไฟล์ภาพไม่ถูกต้อง', 'Invalid image file')); }; input.src = url; event.target.value = '';
        }} />
      </label></div>
    {error && <p role="alert" className="text-sm text-red">{error}</p>}
  </div>;
}
