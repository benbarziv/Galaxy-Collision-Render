import './ui/styles.css';
import { createContext } from './render/gl/utils.ts';
import { Renderer } from './render/renderer.ts';
import { SimulationEngine } from './sim/engine.ts';
import { CameraController } from './ui/camera.ts';
import { QUALITY_PROFILES, QUALITY_TIERS } from './sim/constants.ts';
import { SCENARIOS, DEFAULT_SCENARIO_ID } from './core/presets.ts';
import type { QualityProfile, QualityTier, Scenario } from './core/types.ts';

/**
 * Application entry point.
 *
 * Owns the fixed-timestep render loop and wires the DOM controls to the engine
 * and renderer. All physics lives in sim/, all drawing in render/, and this
 * file only coordinates them.
 *
 * FRAME LOOP
 * ----------
 * requestAnimationFrame drives the loop; the simulation inside it is on its own
 * fixed timestep (see sim/engine.ts), so the physics is decoupled from the
 * display rate. Rendering happens exactly once per animation frame regardless
 * of how many substeps ran, so raising the speed control costs simulation time
 * rather than frame rate until the substep cap is hit.
 */

interface Elements {
  canvas: HTMLCanvasElement;
  ui: HTMLElement;
  preset: HTMLSelectElement;
  presetBlurb: HTMLElement;
  presetDescription: HTMLElement;
  speed: HTMLInputElement;
  speedValue: HTMLOutputElement;
  quality: HTMLInputElement;
  qualityValue: HTMLOutputElement;
  starSize: HTMLInputElement;
  starSizeValue: HTMLOutputElement;
  exposure: HTMLInputElement;
  exposureValue: HTMLOutputElement;
  trailLength: HTMLInputElement;
  trailValue: HTMLOutputElement;
  toggleTrails: HTMLInputElement;
  toggleBloom: HTMLInputElement;
  toggleStarfield: HTMLInputElement;
  toggleFollow: HTMLInputElement;
  toggleStats: HTMLInputElement;
  playpause: HTMLButtonElement;
  reset: HTMLButtonElement;
  collapse: HTMLButtonElement;
  stats: HTMLElement;
  hint: HTMLElement;
  boot: HTMLElement;
  fatal: HTMLElement;
  fatalMessage: HTMLElement;
  sTime: HTMLElement;
  sFps: HTMLElement;
  sParticles: HTMLElement;
  sSubsteps: HTMLElement;
  sStepTime: HTMLElement;
  sGrid: HTMLElement;
  sSeparation: HTMLElement;
  sRelVel: HTMLElement;
  sEscaped: HTMLElement;
  sDraws: HTMLElement;
}

/** Throws with a clear message rather than letting a null propagate. */
function need<T extends Element>(id: string): T {
  const el = document.getElementById(id);
  if (!el) throw new Error(`Missing required element #${id}`);
  return el as unknown as T;
}

function collectElements(): Elements {
  return {
    canvas: need<HTMLCanvasElement>('stage'),
    ui: need('ui'),
    preset: need<HTMLSelectElement>('preset'),
    presetBlurb: need('preset-blurb'),
    presetDescription: need('preset-description'),
    speed: need<HTMLInputElement>('speed'),
    speedValue: need<HTMLOutputElement>('speed-value'),
    quality: need<HTMLInputElement>('quality'),
    qualityValue: need<HTMLOutputElement>('quality-value'),
    starSize: need<HTMLInputElement>('star-size'),
    starSizeValue: need<HTMLOutputElement>('star-size-value'),
    exposure: need<HTMLInputElement>('exposure'),
    exposureValue: need<HTMLOutputElement>('exposure-value'),
    trailLength: need<HTMLInputElement>('trail-length'),
    trailValue: need<HTMLOutputElement>('trail-value'),
    toggleTrails: need<HTMLInputElement>('toggle-trails'),
    toggleBloom: need<HTMLInputElement>('toggle-bloom'),
    toggleStarfield: need<HTMLInputElement>('toggle-starfield'),
    toggleFollow: need<HTMLInputElement>('toggle-follow'),
    toggleStats: need<HTMLInputElement>('toggle-stats'),
    playpause: need<HTMLButtonElement>('playpause'),
    reset: need<HTMLButtonElement>('reset'),
    collapse: need<HTMLButtonElement>('collapse'),
    stats: need('stats'),
    hint: need('hint'),
    boot: need('boot'),
    fatal: need('fatal'),
    fatalMessage: need('fatal-message'),
    sTime: need('s-time'),
    sFps: need('s-fps'),
    sParticles: need('s-particles'),
    sSubsteps: need('s-substeps'),
    sStepTime: need('s-steptime'),
    sGrid: need('s-grid'),
    sSeparation: need('s-separation'),
    sRelVel: need('s-relvel'),
    sEscaped: need('s-escaped'),
    sDraws: need('s-draws'),
  };
}

/** Maps the speed slider's 0-100 range onto a logarithmic time multiplier. */
function sliderToSpeed(value: number): number {
  // Exponential so the slider spends most of its travel in the slow range,
  // where the interesting dynamics are, while still reaching fast merger
  // timescales. Range: 0.05x to 8x.
  const t = value / 100;
  return 0.05 * Math.pow(8 / 0.05, t);
}

function formatSpeed(v: number): string {
  return `${v < 0.1 ? v.toFixed(3) : v.toFixed(2)}x`;
}

function formatTime(myr: number): string {
  if (myr < 1000) return `${myr.toFixed(0)} Myr`;
  return `${(myr / 1000).toFixed(2)} Gyr`;
}

class App {
  private readonly el: Elements;
  private readonly gl: WebGL2RenderingContext;
  private readonly engine: SimulationEngine;
  private readonly renderer: Renderer;
  private readonly camera = new CameraController();

  private scenario: Scenario;
  private tier: QualityTier;
  private quality: QualityProfile;

  private paused = false;
  private speed = 1;
  private showStats = false;
  private uiHidden = false;
  private running = true;

  private lastTime = 0;
  private fpsAccumulator = 0;
  private fpsFrames = 0;
  private fps = 0;
  private uiAccumulator = 0;
  private resizePending = true;
  private booted = false;

  constructor(el: Elements) {
    this.el = el;
    this.gl = createContext(el.canvas);
    this.scenario = SCENARIOS.find((s) => s.id === DEFAULT_SCENARIO_ID) ?? SCENARIOS[0];
    this.tier = 'high';
    this.quality = QUALITY_PROFILES[this.tier];

    this.engine = new SimulationEngine(this.gl);
    this.renderer = new Renderer(this.gl);

    this.camera.attach(el.canvas);
    this.populatePresets();
    this.bindControls();
    this.bindKeyboard();
    this.bindResize();
    this.applyScenario(this.scenario, true);
  }

  /** Fill the scenario dropdown from the preset table. */
  private populatePresets(): void {
    for (const s of SCENARIOS) {
      const opt = document.createElement('option');
      opt.value = s.id;
      opt.textContent = s.name;
      this.el.preset.appendChild(opt);
    }
    this.el.preset.value = this.scenario.id;
  }

  /**
   * Rebuild the simulation for a scenario.
   *
   * `refit` controls whether the camera is re-framed. Reset re-frames so the
   * user is returned to a known view; a live preset switch also re-frames,
   * because the new scenario has a different separation.
   */
  private applyScenario(scenario: Scenario, refit: boolean): void {
    this.scenario = scenario;
    this.engine.reset(scenario, this.quality);
    this.renderer.setScenario(scenario);
    this.renderer.bindStarBuffer(this.engine.stateBuffer, this.engine.particleCount);

    this.el.presetBlurb.textContent = scenario.blurb;
    this.el.presetDescription.textContent = scenario.description;

    if (refit) {
      const maxRadius = Math.max(...scenario.galaxies.map((g) => g.radius));
      this.camera.frame(scenario.separation, maxRadius);
    }
    // Seed the core positions from the initial conditions so the first frame
    // already has bright cores rather than a frame of bare stars.
    const stats = this.engine.stats;
    this.renderer.setCoreCenters(
      [stats.centers[0].x, stats.centers[0].y, stats.centers[0].z],
      [stats.centers[1].x, stats.centers[1].y, stats.centers[1].z],
    );
    this.renderer.applySettings({
      trailLength: this.quality.trailLength,
      bloomEnabled: this.quality.bloom,
      resolutionScale: this.quality.resolutionScale,
    });
    this.syncTrailLabel();
  }

  /**
   * Change the quality tier.
   *
   * This rebuilds the particle buffers and the PM grid, so it is a full reset
   * by necessity. The camera is preserved so the user does not lose their view
   * when they change quality mid-flight.
   */
  private applyQuality(tier: QualityTier): void {
    this.tier = tier;
    this.quality = QUALITY_PROFILES[tier];
    const view = this.camera.snapshot();
    this.applyScenario(this.scenario, false);
    this.camera.restore(view);
    this.el.qualityValue.textContent = this.quality.label;
  }

  private syncTrailLabel(): void {
    const v = this.el.trailLength.valueAsNumber;
    this.el.trailValue.textContent = v === 0 ? 'off' : v < 8 ? 'short' : v < 20 ? 'medium' : 'long';
  }

  private bindControls(): void {
    const el = this.el;

    el.preset.addEventListener('change', () => {
      const s = SCENARIOS.find((x) => x.id === el.preset.value);
      if (s) this.applyScenario(s, true);
    });

    el.speed.addEventListener('input', () => {
      this.speed = sliderToSpeed(el.speed.valueAsNumber);
      el.speedValue.textContent = formatSpeed(this.speed);
    });

    el.quality.addEventListener('input', () => {
      const idx = Math.round(el.quality.valueAsNumber);
      const tier = QUALITY_TIERS[Math.max(0, Math.min(QUALITY_TIERS.length - 1, idx))];
      this.applyQuality(tier);
    });

    el.starSize.addEventListener('input', () => {
      const v = el.starSize.valueAsNumber / 100;
      el.starSizeValue.textContent = `${v.toFixed(2)}x`;
      this.renderer.applySettings({ starSize: v });
    });

    el.exposure.addEventListener('input', () => {
      const v = el.exposure.valueAsNumber / 100;
      el.exposureValue.textContent = `${v.toFixed(2)}x`;
      this.renderer.applySettings({ exposure: v });
    });

    el.trailLength.addEventListener('input', () => {
      const v = el.trailLength.valueAsNumber;
      this.renderer.applySettings({ trailLength: v, trailsEnabled: v > 0 });
      el.toggleTrails.checked = v > 0;
      this.syncTrailLabel();
    });

    el.toggleTrails.addEventListener('change', () => {
      this.renderer.applySettings({ trailsEnabled: el.toggleTrails.checked });
    });

    el.toggleBloom.addEventListener('change', () => {
      this.renderer.applySettings({ bloomEnabled: el.toggleBloom.checked });
    });

    el.toggleStarfield.addEventListener('change', () => {
      this.renderer.applySettings({ starfield: el.toggleStarfield.checked });
    });

    el.toggleFollow.addEventListener('change', () => {
      this.camera.setFollow(el.toggleFollow.checked);
    });

    el.toggleStats.addEventListener('change', () => {
      this.showStats = el.toggleStats.checked;
      el.stats.hidden = !this.showStats;
    });

    el.playpause.addEventListener('click', () => this.togglePause());
    el.reset.addEventListener('click', () => this.applyScenario(this.scenario, true));
    el.collapse.addEventListener('click', () => this.toggleUi());

    // Reflect the initial control values into the simulation.
    this.speed = sliderToSpeed(el.speed.valueAsNumber);
    el.speedValue.textContent = formatSpeed(this.speed);
    el.qualityValue.textContent = this.quality.label;
    el.starSizeValue.textContent = `${(el.starSize.valueAsNumber / 100).toFixed(2)}x`;
    el.exposureValue.textContent = `${(el.exposure.valueAsNumber / 100).toFixed(2)}x`;
    this.renderer.applySettings({
      starSize: el.starSize.valueAsNumber / 100,
      exposure: el.exposure.valueAsNumber / 100,
    });
  }

  private togglePause(): void {
    this.paused = !this.paused;
    this.el.playpause.textContent = this.paused ? 'Resume' : 'Pause';
  }

  private toggleUi(): void {
    this.uiHidden = !this.uiHidden;
    this.el.ui.classList.toggle('hidden', this.uiHidden);
  }

  private bindKeyboard(): void {
    window.addEventListener('keydown', (e) => {
      // Ignore key handling while a form control has focus, so typing in the
      // dropdown does not trigger shortcuts.
      const tag = (e.target as HTMLElement | null)?.tagName;
      const typing = tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA';
      if (typing && e.key !== 'Escape') return;

      switch (e.key.toLowerCase()) {
        case ' ':
          this.togglePause();
          e.preventDefault();
          break;
        case 'h':
          this.toggleUi();
          break;
        case 'r':
          this.applyScenario(this.scenario, true);
          break;
        case 'd':
          this.el.toggleStats.checked = !this.el.toggleStats.checked;
          this.el.toggleStats.dispatchEvent(new Event('change'));
          break;
        default:
          break;
      }
    });
  }

  private bindResize(): void {
    const onResize = (): void => {
      this.resizePending = true;
    };
    window.addEventListener('resize', onResize);
    // devicePixelRatio changes when a window moves between displays; there is
    // no event for it, so poll cheaply and only act on change.
    let lastDpr = window.devicePixelRatio;
    const watchDpr = (): void => {
      if (this.running && window.devicePixelRatio !== lastDpr) {
        lastDpr = window.devicePixelRatio;
        this.resizePending = true;
      }
      if (this.running) requestAnimationFrame(watchDpr);
    };
    requestAnimationFrame(watchDpr);
  }

  private applyResize(): void {
    this.resizePending = false;
    // Cap the DPR at 2. Beyond that the extra pixels cost far more fill rate
    // than they add in visible sharpness, since the content is soft glows.
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    this.renderer.resize(
      this.el.canvas.clientWidth || window.innerWidth,
      this.el.canvas.clientHeight || window.innerHeight,
      dpr,
    );
  }

  /**
   * One animation frame.
   *
   * Order matters: resize first (it may reallocate targets the frame writes to),
   * then simulate, then update the camera from the new state, then draw.
   */
  private frame = (now: number): void => {
    if (!this.running) return;
    requestAnimationFrame(this.frame);

    if (this.lastTime === 0) this.lastTime = now;
    // Clamp the delta: a tab switch produces a multi-second gap, and feeding
    // that to the integrator would either stall or dump the substep cap.
    const realDt = Math.min((now - this.lastTime) / 1000, 0.1);
    this.lastTime = now;

    if (this.resizePending) this.applyResize();

    // ---- Simulate ---------------------------------------------------------
    this.engine.step(realDt, this.speed, this.paused);
    this.engine.tickDiagnostics(realDt, this.showStats);

    // ---- Camera -----------------------------------------------------------
    const stats = this.engine.stats;
    const mid = {
      x: (stats.centers[0].x + stats.centers[1].x) * 0.5,
      y: (stats.centers[0].y + stats.centers[1].y) * 0.5,
      z: (stats.centers[0].z + stats.centers[1].z) * 0.5,
    };
    this.camera.setFocus(mid.x, mid.y, mid.z);
    this.camera.update(realDt);

    // ---- Draw -------------------------------------------------------------
    this.renderer.render(
      this.engine.stateBuffer,
      this.engine.particleCount,
      this.camera.viewMatrix,
      this.camera.fov,
      this.engine.currentTime,
    );

    this.renderer.setCoreCenters(
      [stats.centers[0].x, stats.centers[0].y, stats.centers[0].z],
      [stats.centers[1].x, stats.centers[1].y, stats.centers[1].z],
    );

    // ---- Bookkeeping ------------------------------------------------------
    this.fpsAccumulator += realDt;
    this.fpsFrames++;
    if (this.fpsAccumulator >= 0.5) {
      this.fps = this.fpsFrames / this.fpsAccumulator;
      this.fpsAccumulator = 0;
      this.fpsFrames = 0;
    }

    // The readout is rate-limited to ~12Hz: updating text at 60Hz causes
    // layout thrash and makes the numbers unreadable.
    this.uiAccumulator += realDt;
    if (this.uiAccumulator >= 0.083) {
      this.uiAccumulator = 0;
      this.updateStats();
    }

    if (!this.booted) {
      this.booted = true;
      this.el.boot.classList.add('gone');
      // Fade the keyboard hint once the user has had time to read it.
      window.setTimeout(() => this.el.hint.classList.add('faded'), 9000);
    }
  };

  private updateStats(): void {
    const el = this.el;
    const s = this.engine.stats;
    el.sTime.textContent = formatTime(this.engine.currentTime);
    el.sFps.textContent = this.fps.toFixed(0);
    el.sParticles.textContent = this.engine.particleCount.toLocaleString();
    el.sSubsteps.textContent = String(s.substepsLastFrame);
    el.sStepTime.textContent = `${s.stepMs.toFixed(2)} ms`;
    el.sGrid.textContent = `${this.engine.gridSize}^3`;
    el.sSeparation.textContent = `${s.separation.toFixed(1)} kpc`;
    // Relative *bulk* speed, taken from the difference of the two centres of
    // mass. Reading the separation here instead would report 60 kpc as
    // "30,000 km/s" and hide the one number that reveals whether the pair is
    // still bound.
    el.sRelVel.textContent = `${s.relativeSpeed.toFixed(0)} km/s`;
    el.sEscaped.textContent = `${(s.escapedFraction * 100).toFixed(1)}%`;
    el.sDraws.textContent = String(this.renderer.lastFrameInfo.drawCalls);
  }

  /**
   * Sample the live star state for the physics diagnostic.
   *
   * Reads the integrator's current buffer back to the CPU, which stalls the
   * pipeline, so this is for tests only and is never called from the frame
   * loop. See `SimulationEngine.sampleState` for what the numbers mean.
   */
  sampleState(): ReturnType<SimulationEngine['sampleState']> {
    return this.engine.sampleState();
  }

  start(): void {
    this.lastTime = 0;
    requestAnimationFrame(this.frame);
  }

  /**
   * Draw one frame and immediately read the pixels back, for automated tests.
   *
   * The context is created with `preserveDrawingBuffer: false`, which is the
   * right setting for a real app (it lets the compositor discard the buffer and
   * keeps the fast path fast) but means a `readPixels` issued from a later task
   * returns zeros -- the browser has already presented and cleared the frame.
   * Chrome does not warn about this, so an out-of-band probe silently reports a
   * perfectly healthy renderer as a black screen.
   *
   * Drawing and reading in the same task sidesteps that without giving up the
   * performance setting. Returns mean/max luminance and the lit-pixel fraction
   * so a test can assert the image is neither black nor blown out.
   */
  probeFrame(): {
    mean: number;
    max: number;
    litFraction: number;
    width: number;
    height: number;
    /** Row-major RGB triples, for comparing against a real screenshot. */
    rgb: number[];
  } {
    const gl = this.gl;
    this.renderer.render(
      this.engine.stateBuffer,
      this.engine.particleCount,
      this.camera.viewMatrix,
      this.camera.fov,
      this.engine.currentTime,
    );

    const w = this.renderer.canvasSize.width;
    const h = this.renderer.canvasSize.height;
    const px = new Uint8Array(w * h * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, px);

    let sum = 0;
    let max = 0;
    let lit = 0;
    const n = w * h;
    const rgb = new Array<number>(n * 3);
    for (let i = 0; i < n; i++) {
      const r = px[i * 4];
      const g = px[i * 4 + 1];
      const b = px[i * 4 + 2];
      rgb[i * 3] = r;
      rgb[i * 3 + 1] = g;
      rgb[i * 3 + 2] = b;
      const l = 0.2126 * r + 0.7152 * g + 0.0722 * b;
      sum += l;
      if (l > max) max = l;
      if (l > 12) lit++;
    }
    return { mean: sum / n, max, litFraction: lit / n, width: w, height: h, rgb };
  }

  destroy(): void {
    this.running = false;
    this.camera.dispose();
    this.engine.destroy();
    this.renderer.destroy();
  }
}

/** Surface a startup failure in the page instead of only the console. */
function fatal(message: string): void {
  const boot = document.getElementById('boot');
  const box = document.getElementById('fatal');
  const pre = document.getElementById('fatal-message');
  if (boot) boot.classList.add('gone');
  if (box) box.hidden = false;
  if (pre) pre.textContent = message;
  console.error(message);
}

try {
  const elements = collectElements();
  const app = new App(elements);
  app.start();

  // Expose for debugging from the console; harmless in production and useful
  // when tuning a preset by hand. `probeFrame` is the test hook described on
  // the method: it exists because reading the drawing buffer from outside a
  // rAF callback returns zeros under `preserveDrawingBuffer: false`.
  (window as unknown as { galaxySim: App }).galaxySim = app;
  (window as unknown as { galaxyProbe: () => unknown }).galaxyProbe = () =>
    app.probeFrame();
  (window as unknown as { galaxySample: () => unknown }).galaxySample = () =>
    app.sampleState();} catch (err) {
  fatal(err instanceof Error ? err.message : String(err));
}
