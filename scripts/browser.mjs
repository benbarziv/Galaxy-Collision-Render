/**
 * Shared browser launch for the test scripts.
 *
 * Two things here are load-bearing.
 *
 * The `--use-angle=swiftshader` flags force software rasterisation. This
 * machine has no GPU available to headless Chrome, and without them Chrome
 * either falls back to a stub that silently no-ops every draw call or fails to
 * create a context at all. The simulation genuinely runs on SwiftShader --
 * float render targets, additive blending and transform feedback all work -- it
 * is just slow, which is why the verify script uses the low quality tier.
 *
 * The executable path is resolved explicitly. `PUPPETEER_CACHE_DIR` is set by
 * the surrounding tooling to a per-session temporary directory, so the browser
 * that was downloaded during an earlier run is not where the next run looks.
 * Discovering the installed build and passing it through keeps the scripts
 * reproducible instead of silently re-downloading or failing to launch.
 */
import { existsSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import puppeteer from 'puppeteer';

/** Chrome flags needed to get a real WebGL2 context in headless Chrome. */
export const CHROME_ARGS = [
  '--use-gl=angle',
  '--use-angle=swiftshader',
  '--enable-unsafe-swiftshader',
  '--ignore-gpu-blocklist',
  '--no-sandbox',
];

/**
 * Find an installed chrome-headless-shell.
 *
 * `PUPPETEER_CACHE_DIR` points at a session-scoped temporary directory, and a
 * fresh session gets a fresh path -- so a browser installed in an earlier
 * session is invisible, and every run would otherwise re-download it. Search
 * the configured directory first, then the sibling session directories under
 * the same stable root, and fall back to puppeteer's own resolution only if
 * none of them holds a build.
 */
function findHeadlessShell() {
  const candidates = [];
  const configured = process.env.PUPPETEER_CACHE_DIR;
  if (configured) {
    candidates.push(configured);
    // .../<shared>/<session>/puppeteer
    const session = dirname(configured);
    const shared = dirname(session);
    try {
      for (const entry of readdirSync(shared, { withFileTypes: true })) {
        if (entry.isDirectory()) candidates.push(join(shared, entry.name, 'puppeteer'));
      }
    } catch {
      /* shared root unreadable; the configured directory is the only candidate */
    }
  }

  for (const root of candidates) {
    const dir = join(root, 'chrome-headless-shell');
    if (!existsSync(dir)) continue;
    // Layout: <root>/chrome-headless-shell/<platform>/<build>/chrome-headless-shell
    for (const platform of readdirSync(dir)) {
      const pdir = join(dir, platform);
      let builds;
      try {
        builds = readdirSync(pdir);
      } catch {
        continue;
      }
      // Sorted so the newest build is preferred.
      for (const build of builds.sort()) {
        const exe = join(pdir, build, 'chrome-headless-shell');
        if (existsSync(exe)) return exe;
      }
    }
  }
  return undefined;
}

export async function launch() {
  const executablePath = findHeadlessShell();
  return puppeteer.launch({
    headless: 'shell',
    args: CHROME_ARGS,
    // Passing an explicit path keeps the run reproducible. If none is found we
    // fall through to puppeteer's own resolution, which is correct on a normal
    // machine and fails with a clear message if the browser is genuinely absent.
    ...(executablePath ? { executablePath } : {}),
  });
}
