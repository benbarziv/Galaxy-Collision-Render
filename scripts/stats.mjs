/**
 * Reports a luminance histogram for a PNG.
 *
 * Used to tell "blown out" (mass piled at the top of the range) apart from
 * "correctly exposed with a bright core" (a broad distribution with a tail),
 * which a single lit-pixel percentage cannot distinguish. Reviewing a render
 * properly means looking at its histogram, not just its average.
 *
 * Usage: node scripts/stats.mjs <file.png> [more.png ...]
 */
import { readFileSync } from 'node:fs';
import { decodePng } from './png.mjs';

const BUCKETS = [0, 8, 24, 48, 96, 160, 220, 250, 256];

for (const file of process.argv.slice(2)) {
  const { width, height, channels, data } = decodePng(readFileSync(file));
  const hist = new Array(BUCKETS.length - 1).fill(0);
  let sum = 0;
  let max = 0;
  const n = width * height;

  for (let p = 0; p < n; p++) {
    const o = p * channels;
    const l = 0.2126 * data[o] + 0.7152 * data[o + 1] + 0.0722 * data[o + 2];
    sum += l;
    if (l > max) max = l;
    for (let b = 0; b < hist.length; b++) {
      if (l >= BUCKETS[b] && l < BUCKETS[b + 1]) {
        hist[b]++;
        break;
      }
    }
  }

  console.log(
    `\n${file.split('/').pop()}  ${width}x${height}  mean ${(sum / n).toFixed(2)}  max ${max.toFixed(0)}`,
  );
  for (let b = 0; b < hist.length; b++) {
    const pct = (100 * hist[b]) / n;
    console.log(
      `  ${String(BUCKETS[b]).padStart(3)}-${String(BUCKETS[b + 1]).padStart(3)} ` +
        `${pct.toFixed(2).padStart(6)}% ${'#'.repeat(Math.round(pct / 1.5))}`,
    );
  }
}
