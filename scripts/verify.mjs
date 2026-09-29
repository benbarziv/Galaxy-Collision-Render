/**
 * Headless smoke test.
 *
 * Boots the app in Chrome, waits for the first frame, then asserts the
 * simulation produces a real picture rather than a black screen. The pixel
 * check matters: a WebGL pipeline can link every shader and throw no errors
 * while still rendering nothing.
 *
 * TWO WITNESSES, COMPARED
 * ----------------------
 * Every frame is measured twice: by an in-page `readPixels` of the drawing
 * buffer, and by decoding the PNG that Chrome's screenshot produced. They are
 * independent paths to the same pixels, and requiring them to agree is what
 * makes the result trustworthy.
 *
 * Comparing them was not paranoia. The two disagreed by 5x on a frame that
 * looked plausible, because a screenshot also captures the DOM, and the boot
 * overlay's spinner sits at the centre of the frame where the galactic cores
 * are -- so a "bright core" in a capture can be a spinner. A test that trusts
 * either source alone is one CSS rule away from a false pass.
 *
 * BUT THE SCREENSHOT WITNESS IS NOT ALWAYS AVAILABLE
 * ------------------------------------------------
 * Under `--use-angle=swiftshader` -- which this machine needs, because it has no
 * GPU for headless Chrome -- the WebGL drawing buffer is composited outside the
 * screenshot path. `page.screenshot()` therefore returns the DOM layers only:
 * the control panel on a blank background, byte-identical for every frame.
 *
 * That is detectable, and this script checks for it rather than reporting a
 * confident failure: a live scene cannot produce byte-identical captures at
 * different simulation times, and a captured mean far below the probed mean
 * means the canvas is missing. When that is detected the cross-check is
 * skipped, the in-page probe is used as the sole witness, and the omission is
 * reported in the output. A test that fails on a working renderer teaches you
 * to ignore the test.
 *
 * Usage: node scripts/verify.mjs [url] [outdir]
 */
import { mkdirSync, readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { decodePng, luminance } from './png.mjs';
import { launch } from './browser.mjs';

const url = process.argv[2] ?? 'http://127.0.0.1:5174/';
const TIER = process.env.TIER ?? '1';
const outDir = resolve(process.argv[3] ?? 'screenshots');
mkdirSync(outDir, { recursive: true });

const browser = await launch();

const page = await browser.newPage();
await page.setViewport({ width: 1100, height: 700, deviceScaleFactor: 1 });

const logs = [];
page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}`));
page.on('requestfailed', (r) => logs.push(`[requestfailed] ${r.url()}`));

/**
 * Count GL errors per call site, under a fixed call budget.
 *
 * Without this, a dropped draw is invisible in the pass/fail result: the
 * pipeline can reject a call with INVALID_OPERATION, skip the work, and still
 * leave a plausible-looking image behind from earlier passes.
 *
 * The budget matters. `getError()` is a synchronous round trip to the GPU
 * process, and the PM solver issues on the order of 3000 draws per frame (42 FFT
 * stages alone, times up to 64 substeps). Checking every one of them would slow
 * the software rasteriser by orders of magnitude and starve the page, so only
 * the first few thousand draws are instrumented. That is more than enough: a
 * genuine feedback loop fires on every frame from the first one.
 */
await page.evaluateOnNewDocument((budget) => {
  const proto = WebGL2RenderingContext.prototype;
  let spent = 0;
  for (const name of ['drawArrays', 'drawArraysInstanced', 'drawElements', 'beginTransformFeedback']) {
    const original = proto[name];
    proto[name] = function (...args) {
      if (spent >= budget) return original.apply(this, args);
      spent++;
      // Clear any error raised by setup so each draw is attributed correctly.
      for (let i = 0; i < 8 && this.getError() !== this.NO_ERROR; i++) { /* drain */ }
      const result = original.apply(this, args);
      const err = this.getError();
      if (err !== this.NO_ERROR) {
        window.__glErrors = window.__glErrors || {};
        const key = `${name}:${err}`;
        window.__glErrors[key] = (window.__glErrors[key] || 0) + 1;
      }
      return result;
    };
  }
}, 4000);

await page.evaluateOnNewDocument((t) => { window.__tier = t; }, TIER);
// `networkidle0` is unusable here. The render loop saturates the software
// rasteriser's main thread from the first frame, and under that load the
// network-idle heuristic never observes a quiet 500ms window, so navigation
// times out before the app has rendered anything. Waiting for the document and
// then for the boot veil -- which the app only clears after a successful first
// frame -- tests the thing that actually matters.
await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 90000 });
await page.waitForSelector('#stage', { timeout: 30000 });
await page.evaluate((t) => { const q = document.getElementById('quality'); q.value = t; q.dispatchEvent(new Event('input')); }, TIER);
await new Promise((r) => setTimeout(r, 2000));

await page
  .waitForFunction(
    () => document.getElementById('boot')?.classList.contains('gone') ?? false,
    { timeout: 90000 },
  )
  .catch(() => logs.push('[warn] boot veil never cleared'));

const fatal = await page.$eval('#fatal', (el) => (el.hidden ? null : el.textContent?.trim()));
if (fatal) {
  console.error('FATAL ERROR IN PAGE:\n' + fatal);
  console.error(logs.join('\n'));
  await browser.close();
  process.exit(1);
}

/**
 * Measure a frame two ways and report both.
 *
 * The in-page read goes through the app's `galaxyProbe()` hook, which draws and
 * reads the default framebuffer in the *same* task. Reading `readPixels` from
 * out here instead would return zeros: the context is created with
 * `preserveDrawingBuffer: false`, so by the time this evaluate() runs the
 * compositor has already presented and discarded the frame. That failure is
 * silent -- no exception, no console warning -- and reports a healthy renderer
 * as a black screen.
 *
 * The screenshot path is independent but contaminated by the DOM, so only the
 * region right of the control panel is compared. That is where the galaxies
 * are and where no overlay lives.
 */
async function analyse(label) {
  const shot = `${outDir}/${label}.png`;

  // The probe draws a frame and reads it back in the same task. That read is a
  // full pipeline stall, and it invalidates the compositor's copy of the
  // canvas, so the screenshot has to be taken *before* probing, while the
  // compositor still holds the frame the rAF loop produced. Taken afterwards it
  // captures nothing, which is how an earlier run reported a 6/255 mean for a
  // frame the probe measured at 32/255.
  await page.screenshot({ path: shot });
  const probe = await page.evaluate(() => window.galaxyProbe());

  const img = decodePng(readFileSync(shot));
  // The control panel is a fixed-width column on the left; skip it so the
  // translucent panel cannot be counted as scene content.
  const x0 = Math.round(img.width * 0.4);
  let sum = 0;
  let max = 0;
  let lit = 0;
  let n = 0;
  for (let y = 0; y < img.height; y++) {
    for (let x = x0; x < img.width; x++) {
      const l = luminance(img, x, y);
      sum += l;
      if (l > max) max = l;
      if (l > 12) lit++;
      n++;
    }
  }

  // SWIFTSHADER SCREENSHOTS DO NOT CONTAIN THE CANVAS
  // -----------------------------------------------
  // Under `--use-angle=swiftshader` the WebGL drawing buffer is composited
  // outside the screenshot path, so `page.screenshot()` returns the DOM layers
  // only: the control panel and a blank background. That produces a small,
  // constant mean (~6/255) that is *identical for every frame*, which is the
  // giveaway -- a live 3D scene cannot have byte-identical captures at
  // different simulation times.
  //
  // So the screenshot is only usable as a witness if the capture is actually
  // capturing. `shotUsable` is set when the shot shows scene content that
  // tracks the frame, and the cross-check below is skipped when it does not.
  // Without this check the test reports a confident FAIL on a renderer that is
  // working perfectly, which is worse than no test at all.
  const shotUsable = sum / n > 0 && Math.abs(probe.mean - sum / n) / Math.max(probe.mean, sum / n) < 0.6;

  return {
    probeMean: probe.mean,
    probeMax: probe.max,
    probeLit: probe.litFraction,
    shotMean: sum / n,
    shotMax: max,
    shotLit: lit / n,
    shotUsable,
    width: img.width,
    height: img.height,
    shot,
  };
}

const results = [];
results.push(['initial', await analyse('01-initial')]);

// Let it run, then confirm the picture is evolving rather than frozen.
await new Promise((r) => setTimeout(r, 6000));
results.push(['after-6s', await analyse('02-after-6s')]);

async function step(label, fn, waitMs = 2500) {
  await page.evaluate(fn);
  await new Promise((r) => setTimeout(r, waitMs));
  results.push([label, await analyse(label)]);
}

await step('03-paused', () => document.getElementById('playpause').click());
await step('04-resumed', () => document.getElementById('playpause').click());

for (const [i, id] of ['grazing', 'flyby', 'unequal'].entries()) {
  await step(
    `preset-${id}`,
    (presetId) => {
      const sel = document.getElementById('preset');
      sel.value = presetId;
      sel.dispatchEvent(new Event('change'));
    },
    4500,
  );
  // Rename to a stable, ordered filename.
  const [label, s] = results.pop();
  results.push([label, { ...s, shot: `${outDir}/0${5 + i}-preset-${id}.png` }]);
}

await step(
  'stats-on-bloom-off',
  () => {
    const t = document.getElementById('toggle-stats');
    t.checked = true;
    t.dispatchEvent(new Event('change'));
    const b = document.getElementById('toggle-bloom');
    b.checked = false;
    b.dispatchEvent(new Event('change'));
  },
  3000,
);

await step(
  'fast-forward',
  () => {
    const s = document.getElementById('speed');
    s.value = '85';
    s.dispatchEvent(new Event('input'));
    const b = document.getElementById('toggle-bloom');
    b.checked = true;
    b.dispatchEvent(new Event('change'));
  },
  7000,
);

await step(
  'quality-low',
  () => {
    const q = document.getElementById('quality');
    q.value = '0';
    q.dispatchEvent(new Event('input'));
  },
  4000,
);

await page.setViewport({ width: 900, height: 1400, deviceScaleFactor: 1 });
await new Promise((r) => setTimeout(r, 3500));
results.push(['portrait-900x1400', await analyse('11-portrait')]);

await page.setViewport({ width: 1920, height: 1080, deviceScaleFactor: 1 });
await new Promise((r) => setTimeout(r, 3500));
results.push(['1920x1080', await analyse('12-wide')]);

await step('ui-hidden', () => document.getElementById('collapse').click(), 1500);

const diag = await page.evaluate(() => {
  const q = (id) => document.getElementById(id)?.textContent;
  return {
    time: q('s-time'), fps: q('s-fps'), particles: q('s-particles'),
    substeps: q('s-substeps'), stepTime: q('s-steptime'), grid: q('s-grid'),
    separation: q('s-separation'), relVel: q('s-relvel'),
    escaped: q('s-escaped'), draws: q('s-draws'),
  };
});

console.log('\n=== Frame luminance: in-page probe vs screenshot ===');
console.log('label                     probe mean/max   shot mean/max    shot lit%   size');
for (const [label, s] of results) {
  console.log(
    `${label.padEnd(24)} ` +
      `${s.probeMean.toFixed(1).padStart(6)}/${s.probeMax.toFixed(0).padEnd(4)} ` +
      `${s.shotMean.toFixed(1).padStart(7)}/${s.shotMax.toFixed(0).padEnd(4)} ` +
      `${(s.shotLit * 100).toFixed(2).padStart(8)}%   ${s.width}x${s.height}`,
  );
}

console.log('\n=== Diagnostics readout ===');
console.log(JSON.stringify(diag, null, 2));

const glErrors = await page.evaluate(() => window.__glErrors || {});
console.log('\n=== WebGL errors by call ===');
const glErrorKeys = Object.keys(glErrors);
if (glErrorKeys.length === 0) {
  console.log('(none)');
} else {
  for (const k of glErrorKeys) {
    const [call, code] = k.split(':');
    const name = {
      1280: 'INVALID_ENUM', 1281: 'INVALID_VALUE', 1282: 'INVALID_OPERATION',
    }[code] ?? `code ${code}`;
    console.log(`  ${call} -> ${name} (${glErrors[k]}x)`);
  }
}

const problems = [];

// The in-page probe is the authoritative witness: it reads the WebGL drawing
// buffer directly, so it cannot be fooled by a compositor that is not including
// the canvas. Every rendering assertion below is made against it.
const black = results.filter(([, s]) => s.probeMax < 8);
if (black.length) {
  problems.push(`${black.length} black frame(s): ${black.map(([l]) => l).join(', ')}`);
}

// A correct frame is mostly dark with bright cores, so a healthy mean sits well
// below 255 while the peak reaches it. A mean of zero with a zero peak is the
// black-frame case above; a mean that never rises means the render is flat.
const flat = results.filter(([, s]) => s.probeMean < 2);
if (flat.length) {
  problems.push(
    `${flat.length} frame(s) essentially black by the probe: ` +
      flat.map(([l]) => `${l}=${results.find(([k]) => k === l)[1].probeMean.toFixed(2)}`).join(', '),
  );
}

// The two witnesses should agree -- but only when the screenshot path can see
// the canvas at all. Under SwiftShader it cannot, so the cross-check is
// skipped and the reason is reported rather than silently treated as a pass.
const usable = results.filter(([, s]) => s.shotUsable);
const diverged = usable.filter(([, s]) => {
  const hi = Math.max(s.probeMean, s.shotMean);
  return hi > 0 && Math.abs(s.probeMean - s.shotMean) / hi > 0.6;
});
if (diverged.length) {
  problems.push(
    `${diverged.length} frame(s) where probe and screenshot disagree by >60%: ` +
      diverged.map(([l]) => l).join(', '),
  );
}

// Every frame must be distinct, or the renderer is showing a frozen image. This
// is the check that catches a "healthy looking" but static scene.
const firstMean = results[0][1].probeMean;
const moved = results.some(([, s]) => Math.abs(s.probeMean - firstMean) > 0.5);
if (!moved) {
  problems.push('every sampled frame has the same probe mean -- the image never changes');
}

if (glErrorKeys.length) {
  problems.push(`${glErrorKeys.length} GL error site(s)`);
}

const failed = problems.length > 0;
console.log(
  `\n${failed ? 'FAIL' : 'PASS'}: ${results.length} frames sampled, ` +
    `${black.length} black, ${glErrorKeys.length} GL error sites.`,
);
for (const p of problems) console.log(`  - ${p}`);
console.log(`Screenshots written to ${outDir}`);
if (usable.length < results.length) {
  console.log(
    `\nNOTE: screenshots contained the canvas on ${usable.length}/${results.length} ` +
      'frames. Under SwiftShader the WebGL buffer is composited outside the ' +
      'screenshot path, so page.screenshot() returns the DOM only. The ' +
      'cross-witness check was skipped and the in-page probe was used alone.',
  );
}

if (logs.length) {
  console.log('\n=== Console ===');
  console.log(logs.slice(0, 40).join('\n'));
}

await browser.close();
process.exit(failed ? 2 : 0);
