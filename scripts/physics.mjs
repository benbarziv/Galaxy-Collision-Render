/**
 * Physics regression check.
 *
 * The renderer can look plausible while the force field is wrong, because a
 * mis-scaled potential still produces smooth motion -- just the wrong motion.
 * The numbers that discriminate:
 *
 *   separation  starts near the scenario value and evolves smoothly. A few
 *               hundred kpc means stars are being ejected, not orbiting.
 *   escaped     fraction of stars outside the compute box. A healthy run stays
 *               well under 10%.
 *   relVel      relative bulk speed; decays if bound, asymptotes if a flyby.
 *
 * A correct PM solver holds a disk together over a Gyr. An incorrect one
 * scatters it within tens of Myr, and that is invisible in a screenshot --
 * "all the stars left" and "the stars are still here" can both look plausible
 * when the core glow and background are drawn regardless.
 *
 * Usage: node scripts/physics.mjs [url] [qualityTier]
 */
import { launch } from './browser.mjs';

const url = process.argv[2] ?? 'http://127.0.0.1:5174/';
const TIER = process.argv[3] ?? '0';

const browser = await launch();
const page = await browser.newPage();
await page.setViewport({ width: 800, height: 500 });

const errors = [];
page.on('pageerror', (e) => errors.push(e.message));

await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });
await page.waitForSelector('#stage', { timeout: 30000 });
await page.evaluate((t) => {
  const q = document.getElementById('quality');
  q.value = t;
  q.dispatchEvent(new Event('input'));
}, TIER);

// Diagnostics drive the engine's readback, so the HUD only reflects the
// simulation while this is on.
await page.evaluate(() => {
  const t = document.getElementById('toggle-stats');
  t.checked = true;
  t.dispatchEvent(new Event('change'));
  const s = document.getElementById('speed');
  s.value = '100';
  s.dispatchEvent(new Event('input'));
});
await page.waitForFunction(
  () => document.getElementById('boot')?.classList.contains('gone') ?? false,
  { timeout: 120000 },
).catch(() => errors.push('boot never cleared'));

const read = () =>
  page.evaluate(() => {
    const q = (id) => document.getElementById(id)?.textContent;
    return {
      time: q('s-time'),
      separation: q('s-separation'),
      relVel: q('s-relvel'),
      escaped: q('s-escaped'),
    };
  });

console.log('t(s)     simulated    separation      relVel        escaped');
const samples = [];
for (let i = 0; i < 10; i++) {
  await new Promise((r) => setTimeout(r, 6000));
  const s = await read();
  samples.push(s);
  console.log(
    `${String((i + 1) * 6).padStart(4)}s  ${String(s.time).padStart(10)}  ` +
      `${String(s.separation).padStart(12)}  ${String(s.relVel).padStart(12)}  ` +
      `${String(s.escaped).padStart(8)}`,
  );
}

const num = (s) => parseFloat(String(s).replace(/[^0-9.\-]/g, ''));
const last = samples[samples.length - 1];
const escaped = num(last.escaped);
const separation = num(last.separation);

const problems = [];
if (escaped > 25) problems.push(`${escaped.toFixed(0)}% of stars escaped the box`);
if (separation > 500) problems.push(`separation reached ${separation.toFixed(0)} kpc`);
if (errors.length) problems.push(`page errors: ${errors.join('; ')}`);

console.log(
  problems.length
    ? `\nFAIL\n  - ${problems.join('\n  - ')}`
    : '\nPASS: the disk stayed bound over the sampled interval.',
);

await browser.close();
process.exit(problems.length ? 2 : 0);
