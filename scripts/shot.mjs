/**
 * Capture a screenshot after letting the simulation run.
 *
 * Usage: node scripts/shot.mjs [outfile] [waitMs] [scenarioId]
 */
import { launch } from './browser.mjs';
import { mkdirSync } from 'node:fs';

const out = process.argv[2] ?? 'shots/frame.png';
const wait = Number(process.argv[3] ?? 12000);
const scenario = process.argv[4];
const url = 'http://127.0.0.1:5174/';

const browser = await launch();
const page = await browser.newPage();
await page.setViewport({ width: 1600, height: 900, deviceScaleFactor: 1 });
const errs = [];
page.on('console', (m) => { if (m.type() === 'error') errs.push(m.text()); });
page.on('pageerror', (e) => errs.push(String(e)));
await page.goto(url, { waitUntil: 'networkidle0' });
if (scenario) {
  await page.evaluate((id) => {
    const app = window.galaxySim;
    const sel = document.getElementById('scenario-select')
      ?? document.querySelector('select');
    if (sel) {
      sel.value = id;
      sel.dispatchEvent(new Event('change', { bubbles: true }));
    }
    void app;
  }, scenario);
}
await new Promise((r) => setTimeout(r, wait));
mkdirSync('shots', { recursive: true });
await page.screenshot({ path: out });
const info = await page.evaluate(() => {
  const p = window.galaxyProbe();
  const st = window.galaxySim.engine.stats;
  return { mean: p.mean, max: p.max, lit: p.litFraction, t: window.galaxySim.engine.currentTime,
           sep: st.separation, esc: st.escapedFraction };
});
console.log(`${out}  t=${info.t.toFixed(1)}Myr sep=${info.sep.toFixed(1)}kpc ` +
  `mean=${info.mean.toFixed(1)} max=${info.max} lit=${(info.lit * 100).toFixed(1)}% escaped=${(info.esc * 100).toFixed(1)}%`);
if (errs.length) console.log('console errors:\n  ' + errs.slice(0, 6).join('\n  '));
await browser.close();
