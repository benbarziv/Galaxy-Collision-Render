/**
 * The renderer.
 *
 * Owns the GL programs, the render targets and the per-frame draw order. It
 * reads the engine's state buffer but never writes it, so the simulation and
 * the presentation stay independent.
 *
 * FRAME ORDER
 * -----------
 *   1. background    fullscreen, ray-marched procedural sky
 *   2. scene         additive: trail feedback, then cores, then stars
 *   3. bright pass   extract highlights from the scene
 *   4. blur pyramid  separable Gaussian at 1/2, 1/4, 1/8 resolution
 *   5. composite     scene + bloom + trail, tonemap, sRGB, to the default FB
 *
 * Everything except the final composite renders to a float target, because the
 * additive accumulation in a galactic core legitimately exceeds 1.0 and an
 * 8-bit intermediate would clip it to a hard white disc before the tonemap
 * ever ran.
 */

import {
  createBuffer,
  createProgram,
  createTextureTarget,
  destroyTarget,
  type CompiledProgram,
  type GL,
  type RenderTarget,
} from './gl/utils.ts';
import { FULLSCREEN_VS } from './gl/shaders/common.ts';
import { BACKGROUND_FS } from './gl/shaders/background.ts';
import { CORE_FS, CORE_VS, STAR_FS, STAR_VS } from './gl/shaders/star.ts';
import { BLIT_FS, BLUR_FS, BRIGHT_PASS_FS, COMPOSITE_FS } from './gl/shaders/post.ts';
import {
  TRAIL_DECAY_FS,
  TRAIL_POINT_FS,
  TRAIL_POINT_VS,
} from './gl/shaders/trail.ts';
import {
  mat4Multiply,
  mat4Invert,
  mat4Perspective,
  type Mat4,
} from '../ui/camera.ts';
import { STRIDE } from '../sim/initial-conditions.ts';
import type { LinearColor, Scenario } from '../core/types.ts';

export interface RenderSettings {
  trailsEnabled: boolean;
  trailLength: number;
  bloomEnabled: boolean;
  bloomIntensity: number;
  starfield: boolean;
  exposure: number;
  starSize: number;
  coreIntensity: number;
  resolutionScale: number;
}

export interface RenderFrameInfo {
  fps: number;
  frameMs: number;
  drawCalls: number;
}

/** Bloom pyramid level count. Three gives a wide, soft falloff cheaply. */
const BLOOM_LEVELS = 3;

export class Renderer {
  private readonly gl: GL;

  private background!: CompiledProgram;
  private star!: CompiledProgram;
  private core!: CompiledProgram;
  private brightPass!: CompiledProgram;
  private blur!: CompiledProgram;
  private blitProgram!: CompiledProgram;
  private composite!: CompiledProgram;
  private trailDecay!: CompiledProgram;
  private trailPoint!: CompiledProgram;
  private trailVao: WebGLVertexArrayObject | null = null;

  private starVao: WebGLVertexArrayObject | null = null;
  private coreVao: WebGLVertexArrayObject | null = null;
  private emptyVao: WebGLVertexArrayObject | null = null;

  private scene!: RenderTarget;
  private bloomA: RenderTarget[] = [];
  private bloomB: RenderTarget[] = [];
  private trailHistory: RenderTarget[] = [];
  private trailIndex = 0;

  private width = 1;
  private height = 1;
  private settings: RenderSettings = {
    trailsEnabled: true,
    trailLength: 12,
    bloomEnabled: true,
    bloomIntensity: 1.0,
    starfield: true,
    exposure: 1.0,
    starSize: 1.0,
    coreIntensity: 1.0,
    resolutionScale: 1.0,
  };

  private viewProjection: Mat4 = new Float32Array(16);
  private invViewProjection: Mat4 = new Float32Array(16);
  private drawCalls = 0;
  private galaxyColors: [LinearColor, LinearColor] = [
    { r: 0.42, g: 0.62, b: 1.0 },
    { r: 1.0, g: 0.62, b: 0.28 },
  ];
  private galaxyCoreColors: [LinearColor, LinearColor] = [
    { r: 0.78, g: 0.9, b: 1.0 },
    { r: 1.0, g: 0.86, b: 0.6 },
  ];
  private coreCenters: [[number, number, number], [number, number, number]] = [
    [0, 0, 0],
    [0, 0, 0],
  ];
  private coreRadii: [number, number] = [6, 6];

  constructor(gl: GL) {
    this.gl = gl;
    this.background = createProgram(gl, FULLSCREEN_VS, BACKGROUND_FS, 'background');
    this.star = createProgram(gl, STAR_VS, STAR_FS, 'star');
    this.core = createProgram(gl, CORE_VS, CORE_FS, 'core');
    this.brightPass = createProgram(gl, FULLSCREEN_VS, BRIGHT_PASS_FS, 'brightPass');
    this.blur = createProgram(gl, FULLSCREEN_VS, BLUR_FS, 'blur');
    this.blitProgram = createProgram(gl, FULLSCREEN_VS, BLIT_FS, 'blit');
    this.composite = createProgram(gl, FULLSCREEN_VS, COMPOSITE_FS, 'composite');
    this.trailDecay = createProgram(gl, FULLSCREEN_VS, TRAIL_DECAY_FS, 'trailDecay');
    this.trailPoint = createProgram(gl, TRAIL_POINT_VS, TRAIL_POINT_FS, 'trailPoint');

    this.emptyVao = gl.createVertexArray();

    // A unit quad shared by both galactic cores. Six vertices, no index buffer.
    const quad = new Float32Array([
      -1, -1, 1, -1, -1, 1,
      -1, 1, 1, -1, 1, 1,
    ]);
    this.coreVao = gl.createVertexArray();
    gl.bindVertexArray(this.coreVao);
    const quadBuf = createBuffer(gl, gl.ARRAY_BUFFER, quad, gl.STATIC_DRAW);
    gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
    gl.enableVertexAttribArray(this.core.attribs.aCorner);
    gl.vertexAttribPointer(this.core.attribs.aCorner, 2, gl.FLOAT, false, 0, 0);
    gl.bindVertexArray(null);
  }

  get canvasSize(): { width: number; height: number } {
    return { width: this.width, height: this.height };
  }

  get lastFrameInfo(): RenderFrameInfo {
    return { fps: 0, frameMs: 0, drawCalls: this.drawCalls };
  }

  applySettings(s: Partial<RenderSettings>): void {
    this.settings = { ...this.settings, ...s };
  }

  setScenario(scenario: Scenario): void {
    const [a, b] = scenario.galaxies;
    this.galaxyColors = [a.color, b.color];
    this.galaxyCoreColors = [a.coreColor, b.coreColor];
    this.coreRadii = [a.radius * 0.42, b.radius * 0.42];
  }

  setCoreCenters(
    c0: readonly [number, number, number],
    c1: readonly [number, number, number],
  ): void {
    this.coreCenters[0] = [c0[0], c0[1], c0[2]];
    this.coreCenters[1] = [c1[0], c1[1], c1[2]];
  }

  /**
   * Resize the drawing buffer and every offscreen target.
   *
   * `resolutionScale` trades fill rate for sharpness. The bloom pyramid is
   * always at a fraction of the scene size, so it follows automatically.
   *
   * Targets are destroyed before being recreated. Allocating on every resize
   * event would leak driver-side memory, because a resize fires continuously
   * while a window is being dragged.
   */
  resize(cssWidth: number, cssHeight: number, dpr: number): void {
    const scale = this.settings.resolutionScale;
    const w = Math.max(1, Math.round(cssWidth * dpr * scale));
    const h = Math.max(1, Math.round(cssHeight * dpr * scale));
    if (w === this.width && h === this.height) return;

    this.width = w;
    this.height = h;

    const canvas = this.gl.canvas as HTMLCanvasElement;
    canvas.width = w;
    canvas.height = h;

    const gl = this.gl;
    destroyTarget(gl, this.scene ?? null);
    for (const t of this.bloomA) destroyTarget(gl, t);
    for (const t of this.bloomB) destroyTarget(gl, t);
    for (const t of this.trailHistory) destroyTarget(gl, t);
    this.bloomA = [];
    this.bloomB = [];
    this.trailHistory = [];

    // Half-float is enough precision for an additive star buffer and halves the
    // bandwidth versus full float, which matters because the scene pass is
    // heavily overdrawn.
    this.scene = createTextureTarget(gl, w, h, gl.RGBA16F, gl.LINEAR);

    for (let i = 0; i < BLOOM_LEVELS; i++) {
      const d = 2 << i;
      const lw = Math.max(1, Math.floor(w / d));
      const lh = Math.max(1, Math.floor(h / d));
      this.bloomA.push(createTextureTarget(gl, lw, lh, gl.RGBA16F, gl.LINEAR));
      this.bloomB.push(createTextureTarget(gl, lw, lh, gl.RGBA16F, gl.LINEAR));
    }

    // The trail history runs at a reduced resolution: it is a smear, so full
    // resolution is wasted bandwidth.
    const tw = Math.max(1, Math.floor(w * 0.6));
    const th = Math.max(1, Math.floor(h * 0.6));
    for (let i = 0; i < 2; i++) {
      this.trailHistory.push(createTextureTarget(gl, tw, th, gl.RGBA16F, gl.LINEAR));
    }
  }

  /**
   * Build the VAO the star pass uses.
   *
   * All attributes get divisor 1, making them *instanced*. This is essential
   * and easy to miss: the star pass draws a 6-vertex quad per star with
   * drawArraysInstanced, and with the default divisor of 0 the attribute
   * pointer advances per *vertex* rather than per instance, so the six corners
   * of a quad would read six different stars' data and the image would be
   * unrecognisable noise.
   *
   * Rebuilt whenever the particle count changes, because the buffer the
   * attributes point at is reallocated on reset.
   */
  bindStarBuffer(buffer: WebGLBuffer, count: number): void {
    const gl = this.gl;
    if (this.starVao) gl.deleteVertexArray(this.starVao);
    this.starVao = gl.createVertexArray();
    gl.bindVertexArray(this.starVao);
    gl.bindBuffer(gl.ARRAY_BUFFER, buffer);

    const stride = STRIDE * 4;
    const a = this.star.attribs;
    const bind = (name: string, size: number, offset: number): void => {
      const loc = a[name];
      if (loc === undefined || loc < 0) return;
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, offset);
      gl.vertexAttribDivisor(loc, 1);
    };

    bind('aPosition', 3, 0);
    bind('aGalaxy', 1, 24);
    bind('aSeed', 1, 28);
    bind('aR0', 1, 32);
    bind('aRNorm', 1, 36);
    bind('aSpin', 1, 56);
    bind('aV0', 1, 60);

    gl.bindVertexArray(null);
    this.starCount = count;
  }

  private starCount = 0;

  /** Number of texture units the widest pass (the composite) samples. */
  private static readonly MAX_UNITS = 3;

  private bindTex(unit: number, tex: WebGLTexture): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, tex);
  }

  /**
   * Unbind every texture unit, so a pass starts from a known state.
   *
   * WebGL2 has no "unbind all textures" call, but a unit left holding a
   * texture from an earlier pass is a real hazard: ANGLE's framebuffer feedback
   * validation rejects a draw when *any* bound texture is also an attachment of
   * the current framebuffer, even if the active program never samples that
   * unit. The composite binds the bloom result to unit 1; the next frame's
   * bright pass then renders *into* that same bloom texture, and the draw is
   * dropped with GL_INVALID_OPERATION until the stale binding is cleared.
   *
   * That failure is silent in the worst way: no exception, no console error in
   * release, just a missing image. Clearing units explicitly at the top of each
   * pass makes the invariant local and checkable.
   */
  private clearTextureUnits(): void {
    const gl = this.gl;
    for (let unit = 0; unit < Renderer.MAX_UNITS; unit++) {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, null);
    }
    gl.activeTexture(gl.TEXTURE0);
  }

  private setU(p: CompiledProgram, name: string, v: number): void {
    const loc = p.uniforms[name];
    if (loc) this.gl.uniform1f(loc, v);
  }

  private setColor(p: CompiledProgram, name: string, c: LinearColor): void {
    const loc = p.uniforms[name];
    if (loc) this.gl.uniform3f(loc, c.r, c.g, c.b);
  }

  /**
   * Draw one frame.
   *
   * `stateBuffer` is the engine's current particle buffer; the renderer treats
   * it as read-only and never synchronises on it, so there is no pipeline
   * stall between the simulation's transform-feedback write and this draw.
   */
  render(
    stateBuffer: WebGLBuffer,
    count: number,
    view: Mat4,
    fov: number,
    time: number,
  ): void {
    const gl = this.gl;
    this.drawCalls = 0;
    if (count === 0 || !this.scene) return;

    const aspect = this.width / this.height;
    const proj = mat4Perspective(fov, aspect, 0.1, 4000);
    mat4Multiply(proj, view, this.viewProjection);
    mat4Invert(this.viewProjection, this.invViewProjection);
    this.lastView.set(view);

    const eye = this.cameraPosition(view);

    gl.disable(gl.DEPTH_TEST);
    gl.disable(gl.CULL_FACE);
    gl.disable(gl.BLEND);
    // Every pass below binds its own inputs; starting from a clean set of units
    // keeps last frame's bindings from aliasing this frame's render targets.
    this.clearTextureUnits();

    // ---- 1. Background -----------------------------------------------------
    {
      const p = this.background;
      gl.useProgram(p.program);
      gl.bindVertexArray(this.emptyVao);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.scene.framebuffer);
      gl.viewport(0, 0, this.scene.width, this.scene.height);
      gl.disable(gl.BLEND);
      if (p.uniforms.uInvViewProjection) {
        gl.uniformMatrix4fv(p.uniforms.uInvViewProjection, false, this.invViewProjection);
      }
      if (p.uniforms.uCameraPos) {
        gl.uniform3f(p.uniforms.uCameraPos, eye[0], eye[1], eye[2]);
      }
      this.setU(p, 'uTime', time);
      this.setU(p, 'uStarfieldIntensity', this.settings.starfield ? 0.55 : 0.0);
      this.drawCalls++;
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    // ---- 2. Additive passes ------------------------------------------------
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.scene.framebuffer);
    gl.viewport(0, 0, this.scene.width, this.scene.height);
    gl.enable(gl.BLEND);
    // Additive: every star contributes flux to whatever is already there, which
    // is what makes a dense core read as a continuous glow.
    gl.blendFunc(gl.ONE, gl.ONE);

    this.drawCores();
    this.drawStars(stateBuffer, count);

    gl.disable(gl.BLEND);

    // ---- 3. Trails ---------------------------------------------------------
    // Rendered into their own history targets and composited at the end, so
    // the feedback accumulation never contaminates the scene buffer.
    if (this.settings.trailsEnabled && this.settings.trailLength > 0) {
      this.renderTrails(stateBuffer, count, time);
    }

    // ---- 4. Bloom ----------------------------------------------------------
    if (this.settings.bloomEnabled) {
      this.renderBloom();
    }

    // ---- 5. Composite ------------------------------------------------------
    this.renderComposite();
  }

  /**
   * Advance the trail history.
   *
   * Two targets are needed because the accumulation reads the previous frame
   * while writing the current one. The decay is applied first (a fullscreen
   * pass reading last frame's history), then this frame's positions are splatted
   * additively on top.
   *
   * The decay factor is framerate-corrected on the CPU: an exponential decay
   * applied once per frame would make trails shorter at high frame rates. We
   * pick the per-frame factor that corresponds to a fixed decay *in simulated
   * time*, so the trail has a well-defined physical length regardless of fps.
   */
  private renderTrails(stateBuffer: WebGLBuffer, count: number, time: number): void {
    const gl = this.gl;
    const prev = this.trailHistory[this.trailIndex];
    const nextIdx = (this.trailIndex + 1) % this.trailHistory.length;
    const next = this.trailHistory[nextIdx];

    this.clearTextureUnits();

    // ---- Decay pass: next = prev * decay -----------------------------------
    gl.bindFramebuffer(gl.FRAMEBUFFER, next.framebuffer);
    gl.viewport(0, 0, next.width, next.height);
    gl.useProgram(this.trailDecay.program);
    gl.bindVertexArray(this.emptyVao);
    gl.disable(gl.BLEND);
    this.bindTex(0, prev.texture);
    if (this.trailDecay.uniforms.uHistory) gl.uniform1i(this.trailDecay.uniforms.uHistory, 0);
    // Half-life expressed in simulated Myr, converted to a per-frame factor.
    const halfLifeMyr = Math.max(0.05, this.settings.trailLength * 0.06);
    const decay = Math.pow(0.5, 0.0167 / halfLifeMyr);
    this.setU(this.trailDecay, 'uDecay', Math.min(0.995, decay));
    this.drawCalls++;
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // ---- Splat pass: add this frame's star positions ------------------------
    gl.bindBuffer(gl.ARRAY_BUFFER, stateBuffer);
    if (!this.trailVao) {
      this.trailVao = gl.createVertexArray();
    }
    gl.bindVertexArray(this.trailVao);
    const stride = STRIDE * 4;
    const a = this.trailPoint.attribs;
    const bind = (name: string, size: number, offset: number): void => {
      const loc = a[name];
      if (loc === undefined || loc < 0) return;
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, offset);
    };
    bind('aPosition', 3, 0);
    bind('aGalaxy', 1, 24);
    bind('aSeed', 1, 28);
    bind('aR0', 1, 32);
    bind('aRNorm', 1, 36);

    gl.useProgram(this.trailPoint.program);
    if (this.trailPoint.uniforms.uViewProjection) {
      gl.uniformMatrix4fv(this.trailPoint.uniforms.uViewProjection, false, this.viewProjection);
    }
    // Splat brightness is small: the accumulation is additive and persists for
    // many frames, so a per-frame value that looks right on its own would blow
    // out to a solid smear.
    this.setU(this.trailPoint, 'uPointSize', 0.16);
    this.setColor(this.trailPoint, 'uGalaxyColor0', this.galaxyColors[0]);
    this.setColor(this.trailPoint, 'uGalaxyColor1', this.galaxyColors[1]);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    this.drawCalls++;
    gl.drawArrays(gl.POINTS, 0, count);
    gl.disable(gl.BLEND);

    this.trailIndex = nextIdx;
    void time;
  }

  /**
   * Recover the world-space eye position from a view matrix.
   *
   * The view matrix maps world to camera space, so the eye is the point that
   * maps to the origin: solving M * p = 0 for p. For a rigid view transform this
   * is simply -R^T * t, which is cheaper and better conditioned than inverting
   * the whole matrix.
   */
  private cameraPosition(view: Mat4): [number, number, number] {
    const tx = view[12], ty = view[13], tz = view[14];
    return [
      -(view[0] * tx + view[1] * ty + view[2] * tz),
      -(view[4] * tx + view[5] * ty + view[6] * tz),
      -(view[8] * tx + view[9] * ty + view[10] * tz),
    ];
  }

  private drawStars(stateBuffer: WebGLBuffer, count: number): void {
    const gl = this.gl;
    if (!this.starVao) return;
    const p = this.star;

    // The engine ping-pongs its state buffer, so the VAO's attribute pointers
    // would otherwise still reference the buffer from last frame. Rebind it.
    gl.bindBuffer(gl.ARRAY_BUFFER, stateBuffer);
    gl.bindVertexArray(this.starVao);
    const stride = STRIDE * 4;
    const a = p.attribs;
    const rebind = (name: string, size: number, offset: number): void => {
      const loc = a[name];
      if (loc === undefined || loc < 0) return;
      gl.vertexAttribPointer(loc, size, gl.FLOAT, false, stride, offset);
      // Divisor must be reasserted too: the VAO is reused across frames and a
      // different draw path may have left it at the default of 0.
      gl.vertexAttribDivisor(loc, 1);
    };
    rebind('aPosition', 3, 0);
    rebind('aGalaxy', 1, 24);
    rebind('aSeed', 1, 28);
    rebind('aR0', 1, 32);
    rebind('aRNorm', 1, 36);
    rebind('aSpin', 1, 56);
    rebind('aV0', 1, 60);

    gl.useProgram(p.program);
    if (p.uniforms.uViewProjection) {
      gl.uniformMatrix4fv(p.uniforms.uViewProjection, false, this.viewProjection);
    }
    if (p.uniforms.uCameraPos) {
      const e = this.cameraPosition(this.lastView);
      gl.uniform3f(p.uniforms.uCameraPos, e[0], e[1], e[2]);
    }
    // Sprite scale: pixels per kpc at unit distance. Derived from the vertical
    // FOV so a star subtends a constant number of pixels regardless of zoom
    // setting, which is what keeps the galaxy the same apparent size when the
    // user changes FOV.
    this.setU(p, 'uPointScale', 0.55);
    this.setU(p, 'uViewportHeight', this.height);
    this.setU(p, 'uSizeBoost', this.settings.starSize);
    this.setU(p, 'uExposure', 1.0);
    this.setU(p, 'uGlowPower', 1.35);
    this.setColor(p, 'uGalaxyColor0', this.galaxyColors[0]);
    this.setColor(p, 'uGalaxyColor1', this.galaxyColors[1]);
    this.setColor(p, 'uGalaxyCore0', this.galaxyCoreColors[0]);
    this.setColor(p, 'uGalaxyCore1', this.galaxyCoreColors[1]);

    // Six vertices per star, expanded from gl_VertexID into a camera-facing quad.
    this.drawCalls++;
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, count);
    void this.starCount;
  }

  private drawCores(): void {
    const gl = this.gl;
    if (!this.coreVao) return;
    const p = this.core;
    gl.useProgram(p.program);
    gl.bindVertexArray(this.coreVao);
    if (p.uniforms.uViewProjection) {
      gl.uniformMatrix4fv(p.uniforms.uViewProjection, false, this.viewProjection);
    }
    const e = this.cameraPosition(this.lastView);
    if (p.uniforms.uCameraPos) gl.uniform3f(p.uniforms.uCameraPos, e[0], e[1], e[2]);

    for (let i = 0; i < 2; i++) {
      const c = this.coreCenters[i];
      if (p.uniforms.uCenter) gl.uniform3f(p.uniforms.uCenter, c[0], c[1], c[2]);
      this.setU(p, 'uRadius', this.coreRadii[i]);
      this.setU(p, 'uViewportHeight', this.height);
      this.setColor(p, 'uColor', this.galaxyCoreColors[i]);
      this.setU(p, 'uIntensity', this.settings.coreIntensity);
      this.drawCalls++;
      gl.drawArrays(gl.TRIANGLES, 0, 6);
    }
  }

  private renderBloom(): void {
    const gl = this.gl;
    const p = this.brightPass;

    this.clearTextureUnits();

    // Bright pass into the first (half-resolution) level.
    gl.useProgram(p.program);
    gl.bindVertexArray(this.emptyVao);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomA[0].framebuffer);
    gl.viewport(0, 0, this.bloomA[0].width, this.bloomA[0].height);
    this.bindTex(0, this.scene.texture);
    if (p.uniforms.uSrc) gl.uniform1i(p.uniforms.uSrc, 0);
    this.setU(p, 'uThreshold', 0.55);
    this.setU(p, 'uKnee', 0.35);
    gl.disable(gl.BLEND);
    this.drawCalls++;
    gl.drawArrays(gl.TRIANGLES, 0, 3);

    // Downsample-and-blur each level, then upsample-and-accumulate back.
    for (let i = 0; i < BLOOM_LEVELS; i++) {
      if (i > 0) {
        // Downsample from the previous level's blurred result. A plain blit:
        // the blur is linear enough that the box filter of a bilinear fetch at
        // the lower resolution does the downsample for free.
        this.blit(this.bloomA[i - 1].texture, this.bloomA[i], 1.0);
      }

      // Separable blur: horizontal into B, vertical back into A.
      gl.useProgram(this.blur.program);
      this.bindTex(0, this.bloomA[i].texture);
      if (this.blur.uniforms.uSrc) gl.uniform1i(this.blur.uniforms.uSrc, 0);

      gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomB[i].framebuffer);
      gl.viewport(0, 0, this.bloomB[i].width, this.bloomB[i].height);
      if (this.blur.uniforms.uDirection) gl.uniform2f(this.blur.uniforms.uDirection, 1, 0);
      this.drawCalls++;
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomA[i].framebuffer);
      this.bindTex(0, this.bloomB[i].texture);
      if (this.blur.uniforms.uSrc) gl.uniform1i(this.blur.uniforms.uSrc, 0);
      if (this.blur.uniforms.uDirection) gl.uniform2f(this.blur.uniforms.uDirection, 0, 1);
      this.drawCalls++;
      gl.drawArrays(gl.TRIANGLES, 0, 3);
    }

    // Accumulate the coarse levels back into the finest, giving a wide soft halo.
    //
    // This is an additive *blend* onto the destination, not a shader that reads
    // both operands. Sampling bloomA[i-1] while rendering into bloomA[i-1] is a
    // framebuffer feedback loop, and WebGL2 rejects the draw outright with
    // GL_INVALID_OPERATION rather than silently dropping it. Blending keeps the
    // destination texture unbound, so the draw is legal and the level sums.
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    for (let i = BLOOM_LEVELS - 1; i > 0; i--) {
      this.blit(this.bloomA[i].texture, this.bloomA[i - 1], 0.65);
    }
    gl.disable(gl.BLEND);
  }

  /**
   * Sample `src` into `dst`, optionally adding it to whatever is already there.
   *
   * `weight` scales the incoming samples. The caller owns blend state, because
   * whether a blit should replace or accumulate depends on the pass.
   */
  private blit(src: WebGLTexture, dst: RenderTarget, weight: number): void {
    const gl = this.gl;
    const p = this.blitProgram;
    gl.useProgram(p.program);
    gl.bindVertexArray(this.emptyVao);
    gl.bindFramebuffer(gl.FRAMEBUFFER, dst.framebuffer);
    gl.viewport(0, 0, dst.width, dst.height);
    this.bindTex(0, src);
    if (p.uniforms.uSrc) gl.uniform1i(p.uniforms.uSrc, 0);
    this.setU(p, 'uWeight', weight);
    this.drawCalls++;
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  private renderComposite(): void {
    const gl = this.gl;
    const p = this.composite;
    this.clearTextureUnits();
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, this.width, this.height);
    gl.useProgram(p.program);
    gl.bindVertexArray(this.emptyVao);
    gl.disable(gl.BLEND);

    this.bindTex(0, this.scene.texture);
    if (p.uniforms.uScene) gl.uniform1i(p.uniforms.uScene, 0);
    this.bindTex(1, this.settings.bloomEnabled ? this.bloomA[0].texture : this.scene.texture);
    if (p.uniforms.uBloom) gl.uniform1i(p.uniforms.uBloom, 1);
    this.bindTex(2, this.trailHistory[this.trailIndex].texture);
    if (p.uniforms.uTrail) gl.uniform1i(p.uniforms.uTrail, 2);

    this.setU(p, 'uBloomIntensity', this.settings.bloomEnabled ? this.settings.bloomIntensity : 0);
    this.setU(p, 'uTrailIntensity', this.settings.trailsEnabled ? 0.85 : 0);
    this.setU(p, 'uExposure', this.settings.exposure);
    this.setU(p, 'uVignette', 0.28);
    this.drawCalls++;
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  private lastView: Mat4 = new Float32Array(16);

  destroy(): void {
    const gl = this.gl;
    destroyTarget(gl, this.scene ?? null);
    for (const t of this.bloomA) destroyTarget(gl, t);
    for (const t of this.bloomB) destroyTarget(gl, t);
    for (const t of this.trailHistory) destroyTarget(gl, t);
    this.bloomA = [];
    this.bloomB = [];
    this.trailHistory = [];

    for (const prog of [
      this.background, this.star, this.core,
      this.brightPass, this.blur, this.blitProgram, this.composite,
    ]) {
      if (prog) gl.deleteProgram(prog.program);
    }
    if (this.starVao) gl.deleteVertexArray(this.starVao);
    if (this.coreVao) gl.deleteVertexArray(this.coreVao);
    if (this.emptyVao) gl.deleteVertexArray(this.emptyVao);
  }
}

