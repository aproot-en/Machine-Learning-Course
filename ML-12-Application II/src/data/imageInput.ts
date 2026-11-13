/** Composite transparency first, then infer polarity from the source border.
 * Inspecting a letterbox border would always see the padding color instead. */
export function grayscaleImage(rgba: ArrayLike<number>, width: number, height: number, polarity: 'auto' | 'light' = 'auto'): Float32Array {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width < 1 || height < 1 || rgba.length !== width * height * 4) throw new Error('Invalid image dimensions');
  let opaqueLuma = 0, opaqueCount = 0, transparent = false;
  for (let i = 0; i < width * height; i++) {
    if (rgba[i * 4 + 3] < 255) transparent = true;
    if (rgba[i * 4 + 3] > 127) { opaqueLuma += (rgba[i * 4] + rgba[i * 4 + 1] + rgba[i * 4 + 2]) / 765; opaqueCount++; }
  }
  const background = transparent && opaqueCount && opaqueLuma / opaqueCount > 0.5 ? 0 : 1;
  const values = Float32Array.from({ length: width * height }, (_, i) => {
    const alpha = rgba[i * 4 + 3] / 255;
    return ((rgba[i * 4] + rgba[i * 4 + 1] + rgba[i * 4 + 2]) / 765) * alpha + background * (1 - alpha);
  });
  let border = 0, count = 0;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) if (x === 0 || y === 0 || x === width - 1 || y === height - 1) { border += values[y * width + x]; count++; }
  return polarity === 'auto' && border / count > 0.5 ? values.map((v) => 1 - v) : values;
}

/** Fit the actual ink into a centered 14×14 region inside a 16×16 image. */
export function digitInput(values: Float32Array, width: number, height: number): Float32Array {
  const output = new Float32Array(256);
  let left = width, right = -1, top = height, bottom = -1;
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) if (values[y * width + x] > 0.15) { left = Math.min(left, x); right = Math.max(right, x); top = Math.min(top, y); bottom = Math.max(bottom, y); }
  if (right < left) return output;
  const scale = 14 / Math.max(right - left + 1, bottom - top + 1), dw = (right - left + 1) * scale, dh = (bottom - top + 1) * scale;
  // Area samples preserve thin strokes while downsampling, unlike nearest-neighbor reads.
  for (let y = 0; y < 16; y++) for (let x = 0; x < 16; x++) {
    let sum = 0;
    for (let sy = 0; sy < 4; sy++) for (let sx = 0; sx < 4; sx++) {
      const ix = (x + (sx + 0.5) / 4 - (16 - dw) / 2) / scale + left;
      const iy = (y + (sy + 0.5) / 4 - (16 - dh) / 2) / scale + top;
      if (ix >= left && ix < right + 1 && iy >= top && iy < bottom + 1) sum += values[Math.floor(iy) * width + Math.floor(ix)];
    }
    output[y * 16 + x] = sum / 16;
  }
  const max = Math.max(...output);
  return max ? output.map((v) => v / max) : output;
}
