/**
 * Reports exactly why the app fails to start.
 *
 * The page shows a friendly "Unable to start" panel instead of a stack trace,
 * which is right for a user and useless for a developer. This prints the raw
 * message, plus any console output and failed requests, so a startup
 * regression is diagnosable without opening devtools.
 *
 * Usage: node scripts/startup.mjs [url]
 */
import { launch } from './browser.mjs';

const url = process.argv[2] ?? 'http://127.0.0.1:5174/';

const browser = await launch();
const page = await browser.newPage();
await page.setViewport({ width: 900, height: 600 });

const logs = [];
page.on('console', (m) => logs.push(`[${m.type()}] ${m.text()}`));
page.on('pageerror', (e) => logs.push(`[pageerror] ${e.message}\n${e.stack ?? ''}`));
page.on('requestfailed', (r) =>
  logs.push(`[requestfailed] ${r.url()} ${r.failure()?.errorText ?? ''}`),
);
page.on('response', (r) => {
  if (r.status() >= 400) logs.push(`[http ${r.status()}] ${r.url()}`);
});

await page.goto(url, { waitUntil: 'domcontentloaded', timeout: 60000 });

// Give the app time to either boot or fail.
await new Promise((r) => setTimeout(r, 8000));

const state = await page.evaluate(() => {
  const fatal = document.getElementById('fatal');
  const boot = document.getElementById('boot');
  // Ask the *computed style*, not the `hidden` attribute.
  //
  // `hidden` is only a user-agent default of `display: none`, so any author
  // `display` rule for the same element outranks it. The error panel sets
  // `display: grid`, which made `fatal.hidden === true` while the panel was
  // still painted over the whole viewport -- so this test reported a clean
  // startup for a page whose entire body read "Unable to start". A test that
  // reads the attribute is testing the DOM's intent rather than what the user
  // can see.
  const shown = (el) => {
    if (!el) return false;
    const cs = getComputedStyle(el);
    return cs.display !== 'none' && cs.visibility !== 'hidden' && Number(cs.opacity) > 0;
  };
  return {
    fatalShown: shown(fatal),
    fatalHiddenAttr: fatal?.hidden ?? null,
    fatalMessage: document.getElementById('fatal-message')?.textContent ?? null,
    fatalText: fatal ? fatal.innerText.slice(0, 120) : null,
    bootShown: shown(boot),
    bootGone: boot?.classList.contains('gone') ?? null,
    hasProbe: typeof window.galaxyProbe === 'function',
    hasSim: typeof window.galaxySim === 'object',
    // True when any overlay is actually covering the canvas, whatever its
    // markup says.
    blockedByOverlay: shown(fatal) || shown(boot),
  };
});

console.log('=== Startup state ===');
console.log(JSON.stringify(state, null, 2));
console.log('\n=== Console / network ===');
console.log(logs.length ? logs.join('\n') : '(clean)');

await browser.close();

const problems = [];
if (state.fatalShown) {
  problems.push(`the fatal error panel is VISIBLE: "${state.fatalText}"`);
}
if (state.bootShown) {
  problems.push('the boot overlay is still covering the page after 8s');
}
if (!state.hasSim) {
  problems.push('window.galaxySim was never set -- the app did not finish starting');
}
if (!state.hasProbe) {
  problems.push('window.galaxyProbe was never set -- the test hook is missing');
}

if (problems.length) {
  console.log('\nFAIL');
  for (const p of problems) console.log(`  - ${p}`);
  process.exit(1);
}
console.log('\nPASS: the app started and nothing is covering the canvas.');
process.exit(0);
