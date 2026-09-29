/**
 * Check that d(separation)/dt agrees with the reported relative bulk speed.
 *
 * These are two independent measurements of the same thing -- how fast the two
 * centres are approaching -- so they must agree. If they do not, one of them is
 * miscomputed, and the HUD is showing a pair of numbers that cannot both be
 * right. A merger either happens or it does not, so this is worth a direct
 * check rather than trusting either readout.
 */
import { launch } from './browser.mjs';

const url = process.argv[2] ?? 'http://127.0.0.1:5174/';
const browser = await launch();
const page = await browser.newPage();
await page.goto(url, { waitUntil: 'networkidle0' });
// Turn the stats overlay on so updateDiagnostics actually runs.
await page.evaluate(() => {
  const cb = document.getElementById('toggle-stats');
  if (cb && !cb.checked) cb.click();
});
// Reset so we start from t=0 rather than inheriting whatever the page was
// already running. Without this the first sample lands well into the
// encounter and there is no baseline to compare against.
await page.evaluate(() => document.getElementById('reset').click());
await new Promise((r) => setTimeout(r, 300));
const t0 = await page.evaluate(() => {
  const st = window.galaxySim.engine.stats;
  window.galaxySim.engine.updateDiagnostics();
  return { t: window.galaxySim.engine.currentTime, sep: window.galaxySim.engine.stats.separation,
           c0: window.galaxySim.engine.stats.centers[0], c1: window.galaxySim.engine.stats.centers[1] };
});
console.log('at t=%s: sep=%s  c0x=%s  c1x=%s', t0.t.toFixed(3), t0.sep.toFixed(2), t0.c0.x.toFixed(2), t0.c1.x.toFixed(2));

let prev = null;
console.log('   t(Myr)  sep(kpc)  dsep/dt   relVel(kpc/Myr)  ratio');
for (let i = 0; i < 10; i++) {
  await new Promise((r) => setTimeout(r, 3000));
  const d = await page.evaluate(() => {
    const s = window.galaxySim.engine.stats;
    return { t: window.galaxySim.engine.currentTime, sep: s.separation, rel: s.relativeSpeed / 977.79 };
  });
  let rate = NaN, ratio = NaN;
  if (prev && d.t > prev.t) {
    rate = (prev.sep - d.sep) / (d.t - prev.t);   // positive = approaching
    ratio = rate / d.rel;
  }
  console.log(
    `${d.t.toFixed(2).padStart(8)} ${d.sep.toFixed(2).padStart(9)} ` +
      `${rate.toFixed(3).padStart(9)} ${d.rel.toFixed(3).padStart(15)} ${ratio.toFixed(2).padStart(7)}`,
  );
  prev = d;
}
await browser.close();
