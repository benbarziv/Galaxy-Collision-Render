/**
 * Minimal PNG reader for the test scripts.
 *
 * Only what the checks need: 8-bit non-interlaced greyscale/RGB/RGBA, which is
 * what Chrome's screenshot encoder emits. Pulling in a dependency for this
 * would be heavier than the twenty lines of inflate-and-unfilter.
 */
import { inflateSync } from 'node:zlib';

export function decodePng(buf) {
  if (buf.length < 8 || buf.readUInt32BE(0) !== 0x89504e47) {
    throw new Error('not a PNG');
  }
  let pos = 8;
  let width = 0;
  let height = 0;
  let colorType = 0;
  const idat = [];
  while (pos + 8 <= buf.length) {
    const length = buf.readUInt32BE(pos);
    const type = buf.toString('ascii', pos + 4, pos + 8);
    const data = buf.subarray(pos + 8, pos + 8 + length);
    if (type === 'IHDR') {
      width = data.readUInt32BE(0);
      height = data.readUInt32BE(4);
      colorType = data[9];
      if (data[8] !== 8) throw new Error(`unsupported bit depth ${data[8]}`);
      if (data[12] !== 0) throw new Error('interlaced PNG not supported');
    } else if (type === 'IDAT') {
      idat.push(data);
    } else if (type === 'IEND') {
      break;
    }
    pos += 12 + length;
  }

  const channels = { 0: 1, 2: 3, 4: 2, 6: 4 }[colorType];
  if (!channels) throw new Error(`unsupported colour type ${colorType}`);

  const raw = inflateSync(Buffer.concat(idat));
  const stride = width * channels;
  const out = Buffer.alloc(height * stride);
  let prev = Buffer.alloc(stride);
  let i = 0;
  for (let y = 0; y < height; y++) {
    const filter = raw[i++];
    const line = Buffer.from(raw.subarray(i, i + stride));
    i += stride;
    if (filter === 0) {
      line.copy(out, y * stride);
      prev = line;
      continue;
    }
    for (let x = 0; x < stride; x++) {
      const a = x >= channels ? line[x - channels] : 0;
      const b = prev[x];
      const c = x >= channels ? prev[x - channels] : 0;
      if (filter === 1) line[x] = (line[x] + a) & 255;
      else if (filter === 2) line[x] = (line[x] + b) & 255;
      else if (filter === 3) line[x] = (line[x] + ((a + b) >> 1)) & 255;
      else {
        const p = a + b - c;
        const pa = Math.abs(p - a);
        const pb = Math.abs(p - b);
        const pc = Math.abs(p - c);
        const pred = pa <= pb && pa <= pc ? a : pb <= pc ? b : c;
        line[x] = (line[x] + pred) & 255;
      }
    }
    line.copy(out, y * stride);
    prev = line;
  }
  return { width, height, channels, data: out };
}

/** Luminance of the pixel at (x, y). */
export function luminance(img, x, y) {
  const o = (y * img.width + x) * img.channels;
  return (
    0.2126 * img.data[o] + 0.7152 * img.data[o + 1] + 0.0722 * img.data[o + 2]
  );
}

/**
 * Mean/max luminance and the fraction of pixels above a "lit" threshold.
 *
 * The lit fraction is reported over the full frame but the caller can mask out
 * the UI, whose translucent panel is bright enough to be mistaken for content.
 */
export function summarise(img, { litThreshold = 12 } = {}) {
  const n = img.width * img.height;
  let sum = 0;
  let max = 0;
  let lit = 0;
  for (let y = 0; y < img.height; y++) {
    for (let x = 0; x < img.width; x++) {
      const l = luminance(img, x, y);
      sum += l;
      if (l > max) max = l;
      if (l > litThreshold) lit++;
    }
  }
  return { mean: sum / n, max, litFraction: lit / n, width: img.width, height: img.height };
}
