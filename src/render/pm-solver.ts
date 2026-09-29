/**
 * The Particle-Mesh gravity solver.
 *
 * Owns the force-field pipeline and exposes one method:
 *   solver.compute(vao, count, params) -> forces in forceTexture
 *
 * The 3D grid is stored as a 2D "atlas" texture of size (N*N) x N, where slice
 * z occupies rows [z*N, (z+1)*N). Every pass is then an ordinary 2D fragment
 * pass, which is a good trade: a 3D texture would need per-layer framebuffer
 * views that are awkward in WebGL2, and the integrator interpolates z by hand
 * either way.
 *
 * Pass order per step:
 *
 *   1. deposit  -> density     (8 additive point splats per star, CIC)
 *   2. scale    -> source      (4*pi*G*rho)
 *   3. FFT      -> sourceFreq  (bit-reversal + log2 N butterflies per axis)
 *   4. Green    -> phiFreq     (multiply by -1/k^2, zero the DC mode)
 *   5. FFT      -> phi         (inverse transform)
 *   6. gradient -> force       (7-point central difference, a = -grad phi)
 */

import {
  createProgram,
  createTextureTarget,
  destroyTarget,
  type CompiledProgram,
  type GL,
  type RenderTarget,
} from './gl/utils.ts';
import { FULLSCREEN_VS } from './gl/shaders/common.ts';
import {
  CLEAR_FS,
  COPY_FS,
  DENSITY_SCALE_FS,
  DEPOSIT_FS,
  DEPOSIT_VS,
  FFT_FS,
  GREENS_FS,
  POTENTIAL_GRADIENT_FS,
} from './gl/shaders/pm.ts';

export interface PmSolveParams {
  /** Gravitational constant in simulation units. */
  G: number;
  /** Side length of the periodic compute box, kpc. */
  boxSize: number;
  /** Splat footprint in pixels. */
  pointSize: number;
}

/** Number of CIC corners splatted per star. */
const CIC_CORNERS = 8;

export class PmSolver {
  private readonly gl: GL;
  private gridSize = 0;

  private deposit!: CompiledProgram;
  private copy!: CompiledProgram;
  private densityScale!: CompiledProgram;
  private fft!: CompiledProgram;
  private greens!: CompiledProgram;
  private gradient!: CompiledProgram;
  private clear!: CompiledProgram;

  /** Mass and CIC-weight accumulation, the Poisson source, and the ping-pong. */
  private deposited!: RenderTarget;
  private source!: RenderTarget;
  private phi!: RenderTarget;
  private force!: RenderTarget;
  private scratchA!: RenderTarget;
  private scratchB!: RenderTarget;
  private scratchC!: RenderTarget;

  private emptyVao: WebGLVertexArrayObject | null = null;

  constructor(gl: GL) {
    this.gl = gl;

    this.deposit = createProgram(gl, DEPOSIT_VS, DEPOSIT_FS, 'deposit');
    this.copy = createProgram(gl, FULLSCREEN_VS, COPY_FS, 'copy');
    this.densityScale = createProgram(gl, FULLSCREEN_VS, DENSITY_SCALE_FS, 'densityScale');
    this.fft = createProgram(gl, FULLSCREEN_VS, FFT_FS, 'fft');
    this.greens = createProgram(gl, FULLSCREEN_VS, GREENS_FS, 'greens');
    this.gradient = createProgram(gl, FULLSCREEN_VS, POTENTIAL_GRADIENT_FS, 'potentialGradient');
    this.clear = createProgram(gl, FULLSCREEN_VS, CLEAR_FS, 'clear');

    // Fullscreen passes read no vertex attributes, but WebGL2 still requires a
    // bound VAO with no enabled arrays for a legal draw call.
    this.emptyVao = gl.createVertexArray();
  }

  get forceTexture(): WebGLTexture {
    return this.force.texture;
  }

  get cellsPerAxis(): number {
    return this.gridSize;
  }

  /**
   * Attribute locations of the deposit program, so the engine can build a VAO
   * with exactly the layout this shader expects instead of hardcoding offsets.
   */
  get depositAttribs(): Record<string, number> {
    return this.deposit.attribs;
  }

  /** (Re)allocate the grid. No-op if the resolution is unchanged. */
  resize(gridSize: number): void {
    if (gridSize === this.gridSize) return;
    this.disposeTargets();
    this.gridSize = gridSize;

    const gl = this.gl;
    // The atlas is N^2 cells WIDE by N tall: slice z occupies rows
    // [z*N, (z+1)*N), so the long horizontal edge carries the flattened
    // (x, y, z) index and the short vertical edge is the row within a slice.
    // Every solver shader addresses it as uv * vec2(N, N*N) / vec2(N, N*N)
    // respectively; the deposit pass converts to clip space in DEPOSIT_VS.
    //
    // The long edge is N^2, which caps N at sqrt(MAX_TEXTURE_SIZE) -- see
    // clampGridSize.
    const w = gridSize * gridSize;
    const h = gridSize;
    // RG32F carries the complex FFT pairs and the mass/weight pair; R32F and
    // RGBA32F cover the rest.
    this.deposited = createTextureTarget(gl, w, h, gl.RG32F, gl.NEAREST);
    this.source = createTextureTarget(gl, w, h, gl.RG32F, gl.NEAREST);
    this.phi = createTextureTarget(gl, w, h, gl.R32F, gl.NEAREST);
    this.force = createTextureTarget(gl, w, h, gl.RGBA32F, gl.LINEAR);
    this.scratchA = createTextureTarget(gl, w, h, gl.RG32F, gl.NEAREST);
    this.scratchB = createTextureTarget(gl, w, h, gl.RG32F, gl.NEAREST);
    this.scratchC = createTextureTarget(gl, w, h, gl.RG32F, gl.NEAREST);
  }

  private disposeTargets(): void {
    const gl = this.gl;
    for (const t of [
      this.deposited, this.source, this.phi, this.force,
      this.scratchA, this.scratchB, this.scratchC,
    ]) {
      destroyTarget(gl, t ?? null);
    }
    this.deposited = null as unknown as RenderTarget;
    this.source = null as unknown as RenderTarget;
    this.phi = null as unknown as RenderTarget;
    this.force = null as unknown as RenderTarget;
    this.scratchA = null as unknown as RenderTarget;
    this.scratchB = null as unknown as RenderTarget;
    this.scratchC = null as unknown as RenderTarget;
  }

  /** Draw a fullscreen triangle into `target` with `program` already bound. */
  private blit(target: RenderTarget, program: CompiledProgram): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, target.framebuffer);
    gl.viewport(0, 0, target.width, target.height);
    gl.useProgram(program.program);
    gl.bindVertexArray(this.emptyVao);
    gl.disable(gl.BLEND);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /**
   * Rebind `program` and republish its uniform locations.
   *
   * Every `blit` switches the current program, and a WebGLUniformLocation is
   * only valid for the program that produced it. The solver caches locations in
   * CompiledProgram and therefore has to re-bind the program *after* any blit
   * and before writing uniforms. Centralising that here means the ordering
   * requirement is stated once instead of being repeated -- and silently
   * forgotten -- at each call site.
   */
  private use(p: CompiledProgram): void {
    this.gl.useProgram(p.program);
  }

  private bindTex(unit: number, tex: WebGLTexture): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
  }

  /**
   * Unbind the units the solver uses, so a pass never inherits a stale binding.
   *
   * The FFT rotates its output through three scratch targets, so any unit left
   * bound from the previous stage can alias the target the next stage renders
   * into. ANGLE treats *any* bound texture that is also a current framebuffer
   * attachment as a feedback loop, even when the active program samples only
   * unit 0, and drops the draw with GL_INVALID_OPERATION. A dropped stage
   * corrupts every transform that follows it, so this is not a cosmetic issue.
   */
  private clearTextureUnits(): void {
    const gl = this.gl;
    for (let unit = 0; unit < 3; unit++) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, null);
    }
    gl.activeTexture(gl.TEXTURE0);
  }

  /**
   * Set a float uniform, asserting the program is current.
   *
   * A WebGLUniformLocation is only valid for the program that produced it, and
   * passing a location from another program is silently ignored by some drivers
   * and a hard INVALID_OPERATION on others. Since the solver alternates
   * programs several times per pass, the check is cheap insurance against the
   * class of bug where a uniform quietly stops taking effect.
   */
  private setU(p: CompiledProgram, name: string, v: number): void {
    const loc = p.uniforms[name];
    if (!loc) return;
    this.gl.uniform1f(loc, v);
  }

  private setUi(p: CompiledProgram, name: string, v: number): void {
    const loc = p.uniforms[name];
    if (!loc) return;
    this.gl.uniform1i(loc, v);
  }

  private setU3(p: CompiledProgram, name: string, x: number, y: number, z: number): void {
    const loc = p.uniforms[name];
    if (!loc) return;
    this.gl.uniform3f(loc, x, y, z);
  }

  /**
   * One complete 1D transform along a single axis.
   *
   * Runs log2(N) + 1 passes: the bit-reversal permutation plus log2(N) butterfly
   * stages, rotating through the three complex scratch textures.
   *
   * WHY THREE TARGETS, NOT TWO
   * --------------------------
   * The input is often one of the scratch textures themselves (the first
   * forward pass reads `source`). With only two scratch targets, a pass would
   * have to write into the texture it is reading, which is a framebuffer
   * feedback loop: the driver rejects the draw with
   * GL_INVALID_OPERATION and the pass silently produces nothing. Three targets
   * break the cycle for any entry point.
   */
  private transformAxis(input: RenderTarget, sign: -1 | 1, axis: 0 | 1 | 2): RenderTarget {
    const N = this.gridSize;
    const stages = Math.round(Math.log2(N)) + 1;
    const pool = [this.scratchA, this.scratchB, this.scratchC];

    let read = input;
    // The first write must land on a target that is not `read`.
    let write = pool.find((t) => t !== read)!;

    for (let stage = 0; stage < stages; stage++) {
      // Re-bind every pass: blit() leaves its own program current.
      this.use(this.fft);
      this.setU(this.fft, 'uGridSize', N);
      this.setUi(this.fft, 'uAxis', axis);
      this.setU(this.fft, 'uSign', sign);
      this.setUi(this.fft, 'uSrc', 0);
      this.setUi(this.fft, 'uStage', stage);
      this.bindTex(0, read.texture);
      this.blit(write, this.fft);
      read = write;
      write = pool.find((t) => t !== read && t !== input)!;
    }
    return read;
  }

  /** Forward transform along all three axes. */
  private forward3(input: RenderTarget): RenderTarget {
    let t = this.transformAxis(input, -1, 0);
    t = this.transformAxis(t, -1, 1);
    t = this.transformAxis(t, -1, 2);
    return t;
  }

  /** Inverse transform along all three axes, with the 1/N folded in. */
  private inverse3(input: RenderTarget): RenderTarget {
    let t = this.transformAxis(input, 1, 0);
    t = this.transformAxis(t, 1, 1);
    t = this.transformAxis(t, 1, 2);
    return t;
  }

  /**
   * Compute the gravitational force field for the current particle positions.
   *
   * `vao` must already have the deposit program's attribute layout bound; the
   * caller owns the VAO and the solver only draws from it.
   */
  compute(vao: WebGLVertexArrayObject, count: number, params: PmSolveParams): void {
    const gl = this.gl;
    const N = this.gridSize;
    if (N === 0 || count === 0) return;

    const { G, boxSize, pointSize } = params;
    const half = boxSize * 0.5;
    const cell = boxSize / N;
    const cellVolume = cell * cell * cell;

    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    this.clearTextureUnits();

    // ---- 1. Deposit --------------------------------------------------------
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.deposited.framebuffer);
    gl.viewport(0, 0, this.deposited.width, this.deposited.height);
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    gl.useProgram(this.deposit.program);
    gl.bindVertexArray(vao);
    gl.enable(gl.BLEND);
    // Every star contributes additively to the cells of its CIC stencil.
    gl.blendFunc(gl.ONE, gl.ONE);
    this.setU(this.deposit, 'uGridSize', N);
    this.setU(this.deposit, 'uBoxSize', boxSize);
    this.setU3(this.deposit, 'uBoxMin', -half, -half, -half);
    this.setU(this.deposit, 'uPointSize', pointSize);
    for (let corner = 0; corner < CIC_CORNERS; corner++) {
      this.setUi(this.deposit, 'uCorner', corner);
      gl.drawArrays(gl.POINTS, 0, count);
    }
    gl.disable(gl.BLEND);

    // ---- 2. Mass -> density -> Poisson source ------------------------------
    this.use(this.densityScale);
    this.bindTex(0, this.deposited.texture);
    this.setUi(this.densityScale, 'uSrc', 0);
    this.setU(this.densityScale, 'uG', G);
    this.setU(this.densityScale, 'uCellVolume', cellVolume);
    this.blit(this.source, this.densityScale);

    // ---- 3. Forward FFT, Green's function, inverse FFT ---------------------
    const sourceFreq = this.forward3(this.source);

    this.use(this.greens);
    this.setUi(this.greens, 'uSrc', 0);
    this.setU(this.greens, 'uGridSize', N);
    this.setU(this.greens, 'uBoxSize', boxSize);
    this.bindTex(0, sourceFreq.texture);
    // The Green's pass must not sample and write the same texture, so it goes
    // into one of the other scratch targets.
    const greensOut = [this.scratchA, this.scratchB, this.scratchC].find(
      (t) => t !== sourceFreq,
    )!;
    this.blit(greensOut, this.greens);

    const phiReal = this.inverse3(greensOut);
    // phiReal is one of the RG32F scratch targets; copy its real channel into
    // the dedicated single-channel target the gradient pass samples.
    this.use(this.copy);
    this.bindTex(0, phiReal.texture);
    this.setUi(this.copy, 'uSrc', 0);
    this.blit(this.phi, this.copy);

    // ---- 4. Potential gradient ---------------------------------------------
    this.use(this.gradient);
    this.bindTex(0, this.phi.texture);
    this.setUi(this.gradient, 'uPhi', 0);
    this.setU(this.gradient, 'uGridSize', N);
    this.setU(this.gradient, 'uCellSize', cell);
    this.blit(this.force, this.gradient);
  }

  destroy(): void {
    this.disposeTargets();
    const gl = this.gl;
    for (const p of [
      this.deposit, this.copy, this.densityScale,
      this.fft, this.greens, this.gradient, this.clear,
    ]) {
      if (p) gl.deleteProgram(p.program);
    }
    if (this.emptyVao) gl.deleteVertexArray(this.emptyVao);
  }
}
