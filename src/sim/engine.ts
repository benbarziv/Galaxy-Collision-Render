/**
 * The simulation engine.
 *
 * Owns the particle state, the PM solver and the transform-feedback integrator,
 * and exposes a deliberately small surface to the rest of the app:
 *
 *     engine.reset(scenario, quality)
 *     engine.step(realDtSeconds)     // fixed-timestep, substepped
 *     engine.readState()             // for readback / diagnostics
 *
 * Everything physics-related lives here; the renderer only ever reads buffers.
 * That separation is what lets the same simulation run headless for tests.
 *
 * DOUBLE BUFFERING
 * ----------------
 * The integrator is a transform-feedback program, which cannot read and write
 * the same buffer in one draw. We ping-pong between two state buffers and swap
 * after each substep. `read` is always the buffer holding current state.
 *
 * The readback for diagnostics is deliberately *not* done every frame. Calling
 * `getBufferSubData` stalls the pipeline hard, so it is sampled at a low rate
 * and only when the debug overlay is actually open.
 */

import { createBuffer, createProgram, glCaps, type GL } from '../render/gl/utils.ts';
import { INTEGRATE_FS, INTEGRATE_VS, INTEGRATE_VARYINGS } from '../render/gl/shaders/integrator.ts';
import { PmSolver } from '../render/pm-solver.ts';
import { buildInitialConditions, STRIDE } from './initial-conditions.ts';
import { clampGridSize, defaultTuning, KPC_PER_MYR_TO_KM_PER_S } from './constants.ts';
import type { QualityProfile, Scenario, SimTuning } from '../core/types.ts';
import { vec3, type Vec3 } from '../core/math.ts';

export interface EngineStats {
  /** Elapsed simulated time in Myr. */
  timeMyr: number;
  /** Number of integration substeps executed in the last `step`. */
  substepsLastFrame: number;
  /** Wall-clock ms spent inside the last `step`. */
  stepMs: number;
  particleCount: number;
  /** Fraction of stars that have left the simulation box. */
  escapedFraction: number;
  /** Current centres of mass of the two galaxies, for core placement. */
  centers: [{ x: number; y: number; z: number }, { x: number; y: number; z: number }];
  /** Separation between the two galaxy centres, kpc. */
  separation: number;
  /** Centre-of-mass speed of each galaxy, km/s. */
  speeds: [number, number];
  /** Bulk velocity of each galaxy's centre of mass, kpc/Myr. */
  bulkVelocities: [Vec3, Vec3];
  /** Magnitude of the relative bulk velocity, km/s. */
  relativeSpeed: number;
}

export class SimulationEngine {
  private readonly gl: GL;
  private readonly solver: PmSolver;
  private readonly integrate: ReturnType<typeof createProgram>;

  private read: WebGLBuffer;
  private write: WebGLBuffer;

  private stateVao!: WebGLVertexArrayObject;
  private depositVao!: WebGLVertexArrayObject;
  private transformFeedback: WebGLTransformFeedback | null = null;

  private count = 0;
  private tuning: SimTuning = defaultTuning();
  private scenario: Scenario | null = null;

  private timeMyr = 0;
  /** Accumulated simulated time not yet consumed by a whole dt substep. */
  private accumulator = 0;

  private statsAccumulator = 0;
  private latestStats: EngineStats = {
    timeMyr: 0,
    substepsLastFrame: 0,
    stepMs: 0,
    particleCount: 0,
    escapedFraction: 0,
    centers: [{ x: 0, y: 0, z: 0 }, { x: 0, y: 0, z: 0 }],
    separation: 0,
    speeds: [0, 0],
    bulkVelocities: [vec3(), vec3()],
    relativeSpeed: 0,
  };

  /** Readback scratch, reused to avoid per-call allocation. */
  private readback: Float32Array | null = null;

  constructor(gl: GL) {
    this.gl = gl;
    this.solver = new PmSolver(gl);
    this.integrate = createProgram(
      gl,
      INTEGRATE_VS,
      INTEGRATE_FS,
      'integrate',
      INTEGRATE_VARYINGS,
    );
    this.read = createBuffer(gl, gl.ARRAY_BUFFER, null, gl.DYNAMIC_COPY);
    this.write = createBuffer(gl, gl.ARRAY_BUFFER, null, gl.DYNAMIC_COPY);
  }
  /** The buffer currently holding valid particle state. */
  get stateBuffer(): WebGLBuffer {
    return this.read;
  }

  get particleCount(): number {
    return this.count;
  }

  /**
   * Side length of the periodic PM compute box, in kpc.
   *
   * This must contain the *whole encounter*, not just a galaxy: the box has to
   * hold both disks at their initial separation, plus the halo radius of each,
   * plus room for the tidal tails that get thrown out during the merger. Sizing
   * it from the galaxy radius alone leaves the galaxies sitting outside the box
   * at t=0, where the deposit pass clips them, and the simulation starts with
   * no mass in the grid at all.
   *
   * The margin factor of 1.35 is a compromise: a larger box means fewer PM cells
   * per galaxy and a softer, less detailed force field, while a smaller box
   * truncates the tails. 1.35 covers the widest preset with room to spare.
   */
  get boxSize(): number {
    if (!this.scenario) return 200;
    const maxRadius = Math.max(...this.scenario.galaxies.map((g) => g.radius));
    const halo = this.tuning.haloRadius;
    const needed = this.scenario.separation + 2 * (maxRadius + halo);
    const base = this.tuning.pmBoxScale * maxRadius;
    return Math.max(base, needed * 1.35);
  }

  get boxMin(): [number, number, number] {
    const s = this.boxSize * 0.5;
    return [-s, -s, -s];
  }

  get gridSize(): number {
    return this.solver.cellsPerAxis;
  }

  get currentTime(): number {
    return this.timeMyr;
  }

  get stats(): EngineStats {
    return this.latestStats;
  }

  /**
   * Build fresh initial conditions and reset the clock.
   *
   * Called on preset change, quality change, and the Reset button. Rebuilding
   * from the deterministic generator means Reset is exactly reproducible.
   */
  reset(scenario: Scenario, quality: QualityProfile, tuning?: Partial<SimTuning>): void {
    const gl = this.gl;
    this.scenario = scenario;
    this.tuning = defaultTuning(tuning);

    // The atlas is (N*N) x N, so the texture's long edge is N*N. Clamp to the
    // largest power of two that fits the device's MAX_TEXTURE_SIZE; without
    // this, a 128^3 grid needs a 16384-px texture and the allocation fails on
    // any device reporting 8192 (which includes most mobile and some software
    // rasterisers).
    this.solver.resize(clampGridSize(quality.gridSize, glCaps().maxTextureSize));

    const data = buildInitialConditions(scenario, quality.particles, this.tuning.haloFactor);
    this.count = data.count;

    // (Re)allocate the state buffers. gl.bufferData must be given an explicit
    // size: passing a null initial data with no size argument is an
    // INVALID_VALUE and leaves the store unallocated.
    //
    // The two buffers are allocated together and then only ever written through
    // transform feedback, so re-allocating on reset is cheap relative to the
    // initial condition build and avoids tracking capacity separately.
    const bytes = data.buffer.byteLength;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.read);
    gl.bufferData(gl.ARRAY_BUFFER, bytes, gl.DYNAMIC_COPY);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, data.buffer);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.write);
    gl.bufferData(gl.ARRAY_BUFFER, bytes, gl.DYNAMIC_COPY);

    this.buildVaos();
    this.timeMyr = 0;
    this.accumulator = 0;

    this.latestStats = {
      timeMyr: 0,
      substepsLastFrame: 0,
      stepMs: 0,
      particleCount: this.count,
      escapedFraction: 0,
      centers: [
        { x: data.galaxyCenters[0].x, y: data.galaxyCenters[0].y, z: data.galaxyCenters[0].z },
        { x: data.galaxyCenters[1].x, y: data.galaxyCenters[1].y, z: data.galaxyCenters[1].z },
      ],
      separation: scenario.separation,
      speeds: [0, 0],
      bulkVelocities: [
        vec3(data.galaxyCenters[0].vx, data.galaxyCenters[0].vy, data.galaxyCenters[0].vz),
        vec3(data.galaxyCenters[1].vx, data.galaxyCenters[1].vy, data.galaxyCenters[1].vz),
      ],
      relativeSpeed: Math.hypot(
        scenario.relativeVelocity.x,
        scenario.relativeVelocity.y,
        scenario.relativeVelocity.z,
      ) * KPC_PER_MYR_TO_KM_PER_S,
    };
  }

  /**
   * Bind the state buffer into VAOs for the two programs that consume it.
   *
   * The integrator and the deposit pass both read the same interleaved buffer
   * but declare different attribute sets, so they need separate VAOs. Rebuilt
   * on reset because the attribute offsets depend on the buffer size, which in
   * turn depends on the particle count.
   */
  private buildVaos(): void {
    const gl = this.gl;
    if (this.stateVao) gl.deleteVertexArray(this.stateVao);
    if (this.depositVao) gl.deleteVertexArray(this.depositVao);

    const stride = STRIDE * 4;

    // Integrator VAO: position, velocity and the per-star invariants.
    this.stateVao = gl.createVertexArray()!;
    gl.bindVertexArray(this.stateVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.read);
    const p = this.integrate.attribs;
    gl.enableVertexAttribArray(p.aPosition);
    gl.vertexAttribPointer(p.aPosition, 3, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(p.aVelocity);
    gl.vertexAttribPointer(p.aVelocity, 3, gl.FLOAT, false, stride, 12);
    gl.enableVertexAttribArray(p.aGalaxy);
    gl.vertexAttribPointer(p.aGalaxy, 1, gl.FLOAT, false, stride, 24);
    gl.enableVertexAttribArray(p.aSeed);
    gl.vertexAttribPointer(p.aSeed, 1, gl.FLOAT, false, stride, 28);
    gl.enableVertexAttribArray(p.aR0);
    gl.vertexAttribPointer(p.aR0, 1, gl.FLOAT, false, stride, 32);
    gl.enableVertexAttribArray(p.aRNorm);
    gl.vertexAttribPointer(p.aRNorm, 1, gl.FLOAT, false, stride, 36);
    gl.enableVertexAttribArray(p.aBulkVel);
    gl.vertexAttribPointer(p.aBulkVel, 3, gl.FLOAT, false, stride, 40);
    gl.enableVertexAttribArray(p.aMass);
    gl.vertexAttribPointer(p.aMass, 1, gl.FLOAT, false, stride, 52);
    gl.enableVertexAttribArray(p.aSpin);
    gl.vertexAttribPointer(p.aSpin, 1, gl.FLOAT, false, stride, 56);
    gl.enableVertexAttribArray(p.aV0);
    gl.vertexAttribPointer(p.aV0, 1, gl.FLOAT, false, stride, 60);

    // Deposit VAO: only position and mass are read.
    this.depositVao = gl.createVertexArray()!;
    gl.bindVertexArray(this.depositVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.read);
    const d = this.solver.depositAttribs;
    gl.enableVertexAttribArray(d.aPosition);
    gl.vertexAttribPointer(d.aPosition, 3, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(d.aMass);
    gl.vertexAttribPointer(d.aMass, 1, gl.FLOAT, false, stride, 52);

    gl.bindVertexArray(null);
  }

  /**
   * Repoint the deposit VAO at whichever buffer now holds current state.
   *
   * A VAO stores the buffer object its attribute pointers reference, and the
   * ping-pong swaps that object every substep. The integrator VAO gets rebound
   * explicitly in `integrateOnce`, but the deposit VAO used to be left pointing
   * at whatever buffer happened to be current at *build* time.
   *
   * The consequence is subtle and was very expensive to find: on alternating
   * substeps the mass deposit reads a one-step-stale buffer, so the force field
   * the stars respond to is built from positions that are one substep out of
   * date, and a star that has just been integrated out to a new position is
   * absent from the field that is supposed to pull it back. The field is still
   * smooth and plausible, so nothing errors and no shader is obviously wrong --
   * but self-gravity is effectively evaluated at the wrong place. The disks
   * expand instead of orbiting, and within tens of Myr every star is ejected.
   *
   * Rebinding is two `vertexAttribPointer` calls, which is far cheaper than
   * rebuilding the VAO.
   */
  private bindDepositAttributes(): void {
    const gl = this.gl;
    const stride = STRIDE * 4;
    const d = this.solver.depositAttribs;
    gl.bindVertexArray(this.depositVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.read);
    gl.enableVertexAttribArray(d.aPosition);
    gl.vertexAttribPointer(d.aPosition, 3, gl.FLOAT, false, stride, 0);
    gl.enableVertexAttribArray(d.aMass);
    gl.vertexAttribPointer(d.aMass, 1, gl.FLOAT, false, stride, 52);
  }

  /**
   * Advance the simulation by a real-time delta.
   *
   * The accumulated wall-clock time is consumed in whole `dt` substeps. The
   * substep count is capped so that a stall (a tab switch, a long GC pause, a
   * breakpoint) cannot trigger a spiral of death where catching up takes longer
   * than the frame budget. The cap means the simulation runs in slow motion
   * after a stall rather than freezing, which is the right failure mode for a
   * visualisation.
   */
  step(realDt: number, speedMultiplier: number, paused: boolean): void {
    if (paused || this.count === 0) {
      this.latestStats.substepsLastFrame = 0;
      this.latestStats.stepMs = 0;
      return;
    }

    const t0 = performance.now();
    const maxSubsteps = 64;

    // How much simulated time this frame is allowed to consume.
    this.accumulator += realDt * speedMultiplier;

    let steps = Math.floor(this.accumulator / this.tuning.dt);
    if (steps > maxSubsteps) {
      // Drop the backlog rather than trying to catch up. Time is not lost from
      // the simulation's perspective; it simply advances more slowly.
      this.accumulator = 0;
      steps = maxSubsteps;
    } else {
      this.accumulator -= steps * this.tuning.dt;
    }

    if (steps === 0) {
      this.latestStats.stepMs = performance.now() - t0;
      return;
    }

    for (let i = 0; i < steps; i++) {
      // The deposit VAO must point at the buffer holding the state the solver is
      // about to read. It ping-pongs every substep, so this has to happen per
      // substep, not once per frame. See `bindDepositAttributes`.
      this.bindDepositAttributes();
      this.solver.compute(this.depositVao, this.count, {
        G: this.tuning.G,
        boxSize: this.boxSize,
        pointSize: 1,
      });
      this.integrateOnce();
      this.timeMyr += this.tuning.dt;
    }

    this.latestStats.substepsLastFrame = steps;
    this.latestStats.stepMs = performance.now() - t0;
    this.latestStats.timeMyr = this.timeMyr;
    this.latestStats.particleCount = this.count;
  }

  /** One leapfrog substep via transform feedback, then swap the buffers. */
  private integrateOnce(): void {
    const gl = this.gl;
    const uniforms = this.integrate.uniforms;

    // Detach any framebuffer the renderer left bound.
    //
    // The solver's final pass renders into its `force` target, and that binding
    // survives into this draw. ANGLE's framebuffer-feedback validation rejects
    // a transform-feedback draw when a bound framebuffer has a texture
    // attachment that is also bound to a texture unit -- even though the
    // integrator samples only the force field, and even though rasterisation is
    // discarded. The substep is then silently skipped: time advances, the
    // diagnostics keep updating, and the stars never move.
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);

    gl.useProgram(this.integrate.program);
    gl.bindVertexArray(this.stateVao);
    gl.disable(gl.BLEND);
    gl.disable(gl.DEPTH_TEST);

    // Bind the current state buffer's attributes (the VAO was built against
    // `read`, which alternates, so rebind each substep).
    this.bindIntegrateAttributes();

    // Clear the other units. The renderer's composite leaves the scene, bloom
    // and trail textures bound, and the same validation applies to them.
    for (let unit = 1; unit < 3; unit++) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, null);
    }
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, this.solver.forceTexture);
    if (uniforms.uForce) gl.uniform1i(uniforms.uForce, 0);

    const [bx, by, bz] = this.boxMin;
    const box = this.boxSize;
    const N = this.gridSize;
    if (uniforms.uG) gl.uniform1f(uniforms.uG, this.tuning.G);
    if (uniforms.uDt) gl.uniform1f(uniforms.uDt, this.tuning.dt);
    if (uniforms.uGridSize) gl.uniform1f(uniforms.uGridSize, N);
    if (uniforms.uBoxSize) gl.uniform1f(uniforms.uBoxSize, box);
    if (uniforms.uBoxMin) gl.uniform3f(uniforms.uBoxMin, bx, by, bz);
    if (uniforms.uHaloSoftening) gl.uniform1f(uniforms.uHaloSoftening, this.tuning.softening);

    // Halo centres track the two galaxies. They are refreshed from the periodic
    // readback, so between readbacks they hold the last known values.
    //
    // THE ANALYTIC HALO IS DELIBERATELY SWITCHED OFF
    // -----------------------------------------------
    // Every star is deposited with a full share of its galaxy's combined
    // stellar+dark mass (`galaxyTotalMass / starCount` in
    // initial-conditions.ts), and the circular velocities those stars start
    // with are derived from that same total. The grid-deposited force therefore
    // already reproduces the flat rotation curve on its own.
    //
    // Adding an analytic cored-isothermal halo on top -- as this once did, with
    // 0.9x the total -- counts the same mass twice. The effective gravitational
    // strength roughly doubles, every orbit is over-bound, the disks expand,
    // the pair becomes unbound, and the whole system disperses within tens of
    // Myr. It looks like a working simulation right up until the moment the
    // tidal tails are supposed to form.
    //
    // The uniform plumbing stays in place: the mass is zero, and the shader
    // still evaluates the halo term, which simply contributes nothing. That
    // keeps a single code path for "grid force plus optional analytic
    // potential", and re-enabling a real halo becomes a one-line change.
    const stats = this.latestStats;
    for (let g = 0; g < 2; g++) {
      const spec = this.scenario!.galaxies[g];
      const haloMass = 0;
      const coreRadius = this.tuning.haloRadius * (spec.radius / 14);
      const center = stats.centers[g];
      if (uniforms[`uHaloMass${g}`]) gl.uniform1f(uniforms[`uHaloMass${g}`], haloMass);
      if (uniforms[`uHaloRadius${g}`]) gl.uniform1f(uniforms[`uHaloRadius${g}`], coreRadius);
      if (uniforms[`uHaloCenter${g}`]) gl.uniform3f(uniforms[`uHaloCenter${g}`], center.x, center.y, center.z);
    }

    if (!this.transformFeedback) this.transformFeedback = gl.createTransformFeedback();
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, this.transformFeedback);
    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, this.write);

    // Rasterisation is disabled during transform feedback; without this some
    // drivers still run the fragment stage and stall.
    gl.enable(gl.RASTERIZER_DISCARD);
    gl.beginTransformFeedback(gl.POINTS);
    gl.drawArrays(gl.POINTS, 0, this.count);
    gl.endTransformFeedback();
    gl.disable(gl.RASTERIZER_DISCARD);

    gl.bindBufferBase(gl.TRANSFORM_FEEDBACK_BUFFER, 0, null);
    gl.bindTransformFeedback(gl.TRANSFORM_FEEDBACK, null);

    // Swap.
    const tmp = this.read;
    this.read = this.write;
    this.write = tmp;
  }

  /**
   * Repoint the integrator VAO at whichever buffer now holds current state.
   *
   * A VAO records the buffer object its attribute pointers reference, and the
   * ping-pong swaps those objects every substep. Rebinding is a handful of
   * cheap `vertexAttribPointer` calls and is far cheaper than rebuilding the
   * VAO, which would mean a delete and a create per substep.
   */
  private bindIntegrateAttributes(): void {
    const gl = this.gl;
    const stride = STRIDE * 4;
    const p = this.integrate.attribs;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.read);
    gl.bindVertexArray(this.stateVao);

    const bind = (name: string, size: number, offset: number): void => {
      const loc = p[name];
      if (loc === undefined || loc < 0) return;
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, offset);
    };

    bind('aPosition', 3, 0);
    bind('aVelocity', 3, 12);
    bind('aGalaxy', 1, 24);
    bind('aSeed', 1, 28);
    bind('aR0', 1, 32);
    bind('aRNorm', 1, 36);
    bind('aBulkVel', 3, 40);
    bind('aMass', 1, 52);
    bind('aSpin', 1, 56);
    bind('aV0', 1, 60);
  }

  /**
   * Pull the particle state back to the CPU and derive the diagnostics the HUD
   * shows: galaxy centres, separation, bulk speeds, and the escaped fraction.
   *
   * This is a full pipeline stall, so it is rate-limited by the caller and only
   * runs while the debug overlay is open.
   */
  updateDiagnostics(): void {
    const gl = this.gl;
    const need = this.count * STRIDE;
    if (this.readback === null || this.readback.length < need) {
      this.readback = new Float32Array(need);
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this.read);
    gl.getBufferSubData(gl.ARRAY_BUFFER, 0, this.readback, 0, need);

    const b = this.readback;
    const acc = [
      { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, n: 0 },
      { x: 0, y: 0, z: 0, vx: 0, vy: 0, vz: 0, n: 0 },
    ];
    const half = this.boxSize * 0.5;
    let escaped = 0;

    for (let i = 0; i < this.count; i++) {
      const w = i * STRIDE;
      const x = b[w];
      const y = b[w + 1];
      const z = b[w + 2];
      if (Math.abs(x) > half || Math.abs(y) > half || Math.abs(z) > half) escaped++;
      const g = b[w + 6] < 0.5 ? 0 : 1;
      const a = acc[g];
      a.x += x; a.y += y; a.z += z;
      a.vx += b[w + 3]; a.vy += b[w + 4]; a.vz += b[w + 5];
      a.n++;
    }

    const toKms = KPC_PER_MYR_TO_KM_PER_S;
    for (let g = 0; g < 2; g++) {
      const a = acc[g];
      if (a.n === 0) continue;
      this.latestStats.centers[g] = { x: a.x / a.n, y: a.y / a.n, z: a.z / a.n };
      const vx = a.vx / a.n;
      const vy = a.vy / a.n;
      const vz = a.vz / a.n;
      this.latestStats.bulkVelocities[g] = vec3(vx, vy, vz);
      this.latestStats.speeds[g] = Math.hypot(vx, vy, vz) * toKms;
    }

    const c0 = this.latestStats.centers[0];
    const c1 = this.latestStats.centers[1];
    this.latestStats.separation = Math.hypot(c1.x - c0.x, c1.y - c0.y, c1.z - c0.z);

    // The relative bulk speed is what says whether the encounter is bound and
    // therefore whether a merger is even possible: it falls steadily through
    // the encounter as orbital energy is exchanged with the halos, and levels
    // off at the remnant's rotation speed.
    const v0 = this.latestStats.bulkVelocities[0];
    const v1 = this.latestStats.bulkVelocities[1];
    this.latestStats.relativeSpeed = Math.hypot(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z) * toKms;

    this.latestStats.escapedFraction = this.count > 0 ? escaped / this.count : 0;
  }

  /**
   * Read back a sample of the live star state and summarise it.
   *
   * The centre-of-mass diagnostics above average over whole galaxies, which
   * hides exactly the failure this is here to catch: a force field that is far
   * too strong ejects stars individually, so each galaxy's mean velocity stays
   * plausible while its dispersion runs away. Summarising the *distribution*
   * (median, high percentile, and the radius at which the bulk of the stars
   * still are) separates the two cases:
   *
   *   - healthy: median near the rotation speed, p99 within a few x of it
   *   - runaway: median still reasonable but p99 orders of magnitude larger
   *
   * Returns raw arrays so a test can make assertions about them directly.
   */
  sampleState(): {
    count: number;
    boxSize: number;
    radii: number[];
    speeds: number[];
  } {
    const gl = this.gl;
    const need = this.count * STRIDE;
    if (this.readback === null || this.readback.length < need) {
      this.readback = new Float32Array(need);
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this.read);
    gl.getBufferSubData(gl.ARRAY_BUFFER, 0, this.readback, 0, need);
    const b = this.readback;

    const radii: number[] = [];
    const speeds: number[] = [];
    // Sample rather than measure every star: this runs from a test hook, and
    // 8k samples characterise a 100k-star disk amply.
    const step = Math.max(1, Math.floor(this.count / 8000));
    for (let i = 0; i < this.count; i += step) {
      const w = i * STRIDE;
      // Distance from the origin. The two centres start symmetrically placed,
      // so the origin is a stable reference even mid-encounter.
      radii.push(Math.hypot(b[w], b[w + 1], b[w + 2]));
      speeds.push(Math.hypot(b[w + 3], b[w + 4], b[w + 5]));
    }
    return { count: this.count, boxSize: this.boxSize, radii, speeds };
  }

  /**
   * Advance the rate-limited diagnostics clock. Kept separate from `step` so
   * that pausing the simulation still refreshes the readouts.
   */
  tickDiagnostics(realDt: number, enabled: boolean): void {
    this.statsAccumulator += realDt;
    // 8 Hz is fast enough to look live and slow enough that the pipeline stall
    // is not visible.
    if (enabled && this.statsAccumulator >= 0.125) {
      this.statsAccumulator = 0;
      this.updateDiagnostics();
    } else if (!enabled) {
      this.statsAccumulator = 0;
    }
  }

  destroy(): void {
    const gl = this.gl;
    this.solver.destroy();
    gl.deleteProgram(this.integrate.program);
    gl.deleteBuffer(this.read);
    gl.deleteBuffer(this.write);
    if (this.stateVao) gl.deleteVertexArray(this.stateVao);
    if (this.depositVao) gl.deleteVertexArray(this.depositVao);
    if (this.transformFeedback) gl.deleteTransformFeedback(this.transformFeedback);
    this.readback = null;
  }
}

