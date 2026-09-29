/**
 * Load every preset, run each briefly, and report whether it survives.
 *
 * A preset that throws mid-run is worse than one that looks wrong: the frame
 * loop dies and the canvas freezes on the last good image, so the failure is
 * invisible until you notice the sim clock has stopped. This checks the clock
 * actually advances for each preset, which catches that.
 */
import { launch } from './browser.mjs';

const url = process.argv[2] ?? 'http://127.0.0.1:5174/';
const browser = await launch();
const page = await browser.newPage();
await page.setViewport({ width: 1280, height: 720 });
const errs = [];
page.on('pageerror', (e) => errs.push(String(e)));
page.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
await page.goto(url, { waitUntil: 'networkidle0' });
await new Promise((r) => setTimeout(r, 1500));

const ids = await page.evaluate(() =>
  window.galaxySim.constructor.name && [...document.querySelectorAll('select option')].map((o) => o.value),
);
console.log('presets found:', JSON.stringify(ids));

let bad = 0;
for (const id of ids ?? []) {
  await page.evaluate((v) => {
    const sel = document.querySelector('select');
    sel.value = v;
    sel.dispatchEvent(new Event('change', { bubbles: true }));
  }, id);
  // Sample twice *within* this preset. Comparing against the previous preset's
  // accumulated time is meaningless, because switching presets resets the clock
  // to zero -- which reads as "frozen" for every preset after the first.
  await new Promise((r) => setTimeout(r, 2000));
  const tA = await page.evaluate(() => window.galaxySim.engine.currentTime);
  await new Promise((r) => setTimeout(r, 8000));
  const s = await page.evaluate(() => {
    const st = window.galaxySim.engine.stats;
    const p = window.galaxyProbe();
    return {
      t: window.galaxySim.engine.currentTime, sep: st.separation,
      esc: st.escapedFraction, n: st.particleCount,
      mean: p.mean, max: p.max, lit: p.litFraction,
    };
  });
  const advanced = s.t > tA;
  const ok = advanced && s.esc < 0.3 && s.mean > 1 && s.lit > 0.01;
  if (!ok) bad++;
  console.log(
    `${ok ? 'ok  ' : 'FAIL'} ${id.padEnd(9)} t=${s.t.toFixed(1).padStart(5)}Myr ` +
      `sep=${s.sep.toFixed(1).padStart(6)}kpc esc=${(s.esc * 100).toFixed(1).padStart(5)}% ` +
      `n=${s.n} mean=${s.mean.toFixed(1).padStart(5)} max=${s.max.toFixed(0).padStart(3)} ` +
      `lit=${(s.lit * 100).toFixed(0)}%${advanced ? '' : '  [CLOCK FROZEN]'}`,
  );
}
if (errs.length) console.log('errors:\n  ' + [...new Set(errs)].slice(0, 8).join('\n  '));
console.log(bad === 0 ? '\nPASS: all presets run.' : `\nFAIL: ${bad} preset(s) bad.`);
await browser.close();
process.exit(bad === 0 ? 0 : 1);
