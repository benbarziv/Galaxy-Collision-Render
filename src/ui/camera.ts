import { clamp, lerp } from '../core/math.ts';
import type { CameraState } from '../core/types.ts';

/**
 * A column-major 4x4 matrix.
 *
 * Pinned to `Float32Array<ArrayBuffer>` rather than the bare `Float32Array`
 * alias because TypeScript 5.7 made the latter generic over the backing buffer
 * type, and `uniformMatrix4fv` needs the concrete `ArrayBuffer` case. Without
 * the annotation the compiler cannot prove `view` is not backed by a
 * SharedArrayBuffer.
 */
export type Mat4 = Float32Array<ArrayBuffer>;

/**
 * Perspective projection mapping depth to [-1, 1], the OpenGL convention that
 * WebGL's clip space expects (DirectX's [0, 1] would need a remap).
 */
export function mat4Perspective(fovYRad: number, aspect: number, near: number, far: number): Mat4 {
  const f = 1 / Math.tan(fovYRad * 0.5);
  const m = new Float32Array(16);
  m[0] = f / aspect;
  m[5] = f;
  m[10] = (far + near) / (near - far);
  m[11] = -1;
  m[14] = (2 * far * near) / (near - far);
  return m;
}

/** Right-handed look-at view matrix. */
export function mat4LookAt(
  eye: readonly [number, number, number],
  target: readonly [number, number, number],
  up: readonly [number, number, number],
): Mat4 {
  const [ex, ey, ez] = eye;
  let zx = ex - target[0];
  let zy = ey - target[1];
  let zz = ez - target[2];
  let len = Math.hypot(zx, zy, zz) || 1;
  zx /= len; zy /= len; zz /= len;

  // x = normalize(cross(up, z))
  let xx = up[1] * zz - up[2] * zy;
  let xy = up[2] * zx - up[0] * zz;
  let xz = up[0] * zy - up[1] * zx;
  len = Math.hypot(xx, xy, xz) || 1;
  xx /= len; xy /= len; xz /= len;

  // y = cross(z, x)
  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  const m = new Float32Array(16);
  m[0] = xx; m[4] = xy; m[8] = xz;
  m[1] = yx; m[5] = yy; m[9] = yz;
  m[2] = zx; m[6] = zy; m[10] = zz;
  m[12] = -(xx * ex + xy * ey + xz * ez);
  m[13] = -(yx * ex + yy * ey + yz * ez);
  m[14] = -(zx * ex + zy * ey + zz * ez);
  m[15] = 1;
  return m;
}

/** Column-major 4x4 product, written into `out`. */
export function mat4Multiply(a: Mat4, b: Mat4, out = new Float32Array(16)): Mat4 {
  for (let c = 0; c < 4; c++) {
    const b0 = b[c * 4], b1 = b[c * 4 + 1], b2 = b[c * 4 + 2], b3 = b[c * 4 + 3];
    out[c * 4 + 0] = a[0] * b0 + a[4] * b1 + a[8] * b2 + a[12] * b3;
    out[c * 4 + 1] = a[1] * b0 + a[5] * b1 + a[9] * b2 + a[13] * b3;
    out[c * 4 + 2] = a[2] * b0 + a[6] * b1 + a[10] * b2 + a[14] * b3;
    out[c * 4 + 3] = a[3] * b0 + a[7] * b1 + a[11] * b2 + a[15] * b3;
  }
  return out;
}

/** General 4x4 inverse, by cofactor expansion. */
export function mat4Invert(m: Mat4, out = new Float32Array(16)): Mat4 {
  const a00 = m[0], a01 = m[1], a02 = m[2], a03 = m[3];
  const a10 = m[4], a11 = m[5], a12 = m[6], a13 = m[7];
  const a20 = m[8], a21 = m[9], a22 = m[10], a23 = m[11];
  const a30 = m[12], a31 = m[13], a32 = m[14], a33 = m[15];

  const b00 = a00 * a11 - a01 * a10;
  const b01 = a00 * a12 - a02 * a10;
  const b02 = a00 * a13 - a03 * a10;
  const b03 = a01 * a12 - a02 * a11;
  const b04 = a01 * a13 - a03 * a11;
  const b05 = a02 * a13 - a03 * a12;
  const b06 = a20 * a31 - a21 * a30;
  const b07 = a20 * a32 - a22 * a30;
  const b08 = a20 * a33 - a23 * a30;
  const b09 = a21 * a32 - a22 * a31;
  const b10 = a21 * a33 - a23 * a31;
  const b11 = a22 * a33 - a23 * a32;

  let det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;
  if (!det) {
    // Singular: return a harmless perspective so the background pass still
    // produces something rather than propagating NaNs into every pixel.
    return mat4Perspective(1, 1, 0.1, 100);
  }
  det = 1 / det;

  out[0] = (a11 * b11 - a12 * b10 + a13 * b09) * det;
  out[1] = (a02 * b10 - a01 * b11 - a03 * b09) * det;
  out[2] = (a31 * b05 - a32 * b04 + a33 * b03) * det;
  out[3] = (a22 * b04 - a21 * b05 - a23 * b03) * det;
  out[4] = (a12 * b08 - a10 * b11 - a13 * b07) * det;
  out[5] = (a00 * b11 - a02 * b08 + a03 * b07) * det;
  out[6] = (a32 * b02 - a30 * b05 - a33 * b01) * det;
  out[7] = (a20 * b05 - a22 * b02 + a23 * b01) * det;
  out[8] = (a10 * b10 - a11 * b08 + a13 * b06) * det;
  out[9] = (a01 * b08 - a00 * b10 - a03 * b06) * det;
  out[10] = (a30 * b04 - a31 * b02 + a33 * b00) * det;
  out[11] = (a21 * b02 - a20 * b04 - a23 * b00) * det;
  out[12] = (a11 * b07 - a10 * b09 - a12 * b06) * det;
  out[13] = (a00 * b09 - a01 * b07 + a02 * b06) * det;
  out[14] = (a31 * b01 - a30 * b03 - a32 * b00) * det;
  out[15] = (a20 * b03 - a21 * b01 + a22 * b00) * det;
  return out;
}

const MIN_DISTANCE = 4;
const MAX_DISTANCE = 900;
const MIN_PHI = 0.02;
const MAX_PHI = Math.PI - 0.02;

/**
 * Orbit camera.
 *
 * Drag rotates, wheel zooms exponentially, and both carry inertia. The
 * exponential zoom is deliberate: each notch changes the distance by a fixed
 * *fraction*, so the control feels identical whether you are 10 kpc or 500 kpc
 * from the action. A linear step would be unusably coarse at one end of that
 * range.
 *
 * Every listener is stored as a bound field and removed in `dispose`, so
 * rebuilding the renderer does not leak handlers onto the canvas.
 */
export class CameraController {
  private state: CameraState = {
    distance: 150,
    theta: 0.7,
    phi: 0.95,
    target: { x: 0, y: 0, z: 0 },
    fov: (46 * Math.PI) / 180,
  };

  /** Smoothed values actually used for rendering, easing toward `state`. */
  private smoothDistance = 150;
  private smoothTheta = 0.7;
  private smoothPhi = 0.95;
  private smoothTarget = { x: 0, y: 0, z: 0 };

  private angularVelocity = { theta: 0, phi: 0 };
  private dragging = false;
  private lastX = 0;
  private lastY = 0;
  private followTarget = true;
  private element: HTMLElement | null = null;
  private disposed = false;

  private readonly onPointerDown = (e: PointerEvent): void => {
    if (e.button !== 0 && e.button !== 2) return;
    this.dragging = true;
    this.lastX = e.clientX;
    this.lastY = e.clientY;
    // Kill inertia the moment the user grabs the scene, otherwise the camera
    // keeps drifting under the cursor and the drag feels unresponsive.
    this.angularVelocity.theta = 0;
    this.angularVelocity.phi = 0;
    this.element?.setPointerCapture(e.pointerId);
    e.preventDefault();
  };

  private readonly onPointerMove = (e: PointerEvent): void => {
    if (!this.dragging) return;
    const dx = e.clientX - this.lastX;
    const dy = e.clientY - this.lastY;
    this.lastX = e.clientX;
    this.lastY = e.clientY;

    // Slower rotation when zoomed in, so close-up framing stays controllable.
    const scale = 0.005 * clamp(this.smoothDistance / 150, 0.35, 2.2);
    this.state.theta -= dx * scale;
    this.state.phi = clamp(this.state.phi - dy * scale, MIN_PHI, MAX_PHI);
    this.angularVelocity.theta = -dx * scale;
    this.angularVelocity.phi = -dy * scale;
  };

  private readonly onPointerUp = (e: PointerEvent): void => {
    this.dragging = false;
    if (this.element?.hasPointerCapture(e.pointerId)) {
      this.element.releasePointerCapture(e.pointerId);
    }
  };

  private readonly onWheel = (e: WheelEvent): void => {
    e.preventDefault();
    const factor = Math.exp(clamp(e.deltaY, -240, 240) * 0.0016);
    this.state.distance = clamp(this.state.distance * factor, MIN_DISTANCE, MAX_DISTANCE);
  };

  private readonly onContextMenu = (e: Event): void => e.preventDefault();

  attach(element: HTMLElement): void {
    this.detach();
    this.element = element;
    this.disposed = false;
    element.addEventListener('pointerdown', this.onPointerDown);
    element.addEventListener('pointermove', this.onPointerMove);
    element.addEventListener('pointerup', this.onPointerUp);
    element.addEventListener('pointercancel', this.onPointerUp);
    element.addEventListener('wheel', this.onWheel, { passive: false });
    element.addEventListener('contextmenu', this.onContextMenu);
  }

  detach(): void {
    if (!this.element) return;
    this.element.removeEventListener('pointerdown', this.onPointerDown);
    this.element.removeEventListener('pointermove', this.onPointerMove);
    this.element.removeEventListener('pointerup', this.onPointerUp);
    this.element.removeEventListener('pointercancel', this.onPointerUp);
    this.element.removeEventListener('wheel', this.onWheel);
    this.element.removeEventListener('contextmenu', this.onContextMenu);
    this.element = null;
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    this.detach();
  }

  /** Frame the pair at a sensible starting distance. */
  frame(separation: number, radius: number): void {
    // Fit both galaxies with a margin. The 1.9x padding is aesthetic: an exact
    // fit looks cramped.
    const fit = (separation * 0.5 + radius) * 1.9;
    this.state.distance = clamp(fit, MIN_DISTANCE, MAX_DISTANCE);
    this.smoothDistance = this.state.distance;
    this.state.target = { x: 0, y: 0, z: 0 };
    this.smoothTarget = { x: 0, y: 0, z: 0 };
  }

  setFollow(enabled: boolean): void {
    this.followTarget = enabled;
  }

  get isFollowing(): boolean {
    return this.followTarget;
  }

  /** Ease the look-at point toward the pair's midpoint. */
  setFocus(x: number, y: number, z: number): void {
    if (!this.followTarget) return;
    this.state.target.x = x;
    this.state.target.y = y;
    this.state.target.z = z;
  }

  update(realDt: number): void {
    const dt = Math.min(realDt, 0.05);

    if (!this.dragging) {
      this.state.theta += this.angularVelocity.theta;
      this.state.phi = clamp(this.state.phi + this.angularVelocity.phi, MIN_PHI, MAX_PHI);
      // Exponential decay with a 33ms constant: ~92% of the velocity is gone
      // by the next frame, which reads as a short weighted glide rather than a
      // long spin.
      const decay = Math.exp(-dt / 0.033);
      this.angularVelocity.theta *= decay;
      this.angularVelocity.phi *= decay;
      if (Math.abs(this.angularVelocity.theta) < 1e-5) this.angularVelocity.theta = 0;
      if (Math.abs(this.angularVelocity.phi) < 1e-5) this.angularVelocity.phi = 0;
    }

    // Framerate-independent exponential smoothing. Each channel gets its own
    // time constant: distance is damped hardest or zooming lags, while the
    // look-at point is light so the camera tracks the merger closely.
    const kDist = 1 - Math.exp(-dt / 0.11);
    const kAng = 1 - Math.exp(-dt / 0.055);
    const kTarget = 1 - Math.exp(-dt / 0.35);

    this.smoothDistance = lerp(this.smoothDistance, this.state.distance, kDist);
    this.smoothTheta = lerp(this.smoothTheta, this.state.theta, kAng);
    this.smoothPhi = lerp(this.smoothPhi, this.state.phi, kAng);
    this.smoothTarget.x = lerp(this.smoothTarget.x, this.state.target.x, kTarget);
    this.smoothTarget.y = lerp(this.smoothTarget.y, this.state.target.y, kTarget);
    this.smoothTarget.z = lerp(this.smoothTarget.z, this.state.target.z, kTarget);
  }

  get eyePosition(): [number, number, number] {
    const d = this.smoothDistance;
    const t = this.smoothTheta;
    const p = this.smoothPhi;
    const sp = Math.sin(p);
    return [
      this.smoothTarget.x + d * sp * Math.cos(t),
      this.smoothTarget.y + d * Math.cos(p),
      this.smoothTarget.z + d * sp * Math.sin(t),
    ];
  }

  get viewMatrix(): Mat4 {
    return mat4LookAt(
      this.eyePosition,
      [this.smoothTarget.x, this.smoothTarget.y, this.smoothTarget.z],
      [0, 1, 0],
    );
  }

  get fov(): number {
    return this.state.fov;
  }

  get distance(): number {
    return this.smoothDistance;
  }

  /** Serialise, so the view can survive a scenario change. */
  snapshot(): CameraState {
    return {
      distance: this.state.distance,
      theta: this.state.theta,
      phi: this.state.phi,
      target: { ...this.state.target },
      fov: this.state.fov,
    };
  }

  restore(s: CameraState): void {
    this.state = {
      distance: s.distance,
      theta: s.theta,
      phi: clamp(s.phi, MIN_PHI, MAX_PHI),
      target: { ...s.target },
      fov: s.fov,
    };
    this.smoothDistance = s.distance;
    this.smoothTheta = s.theta;
    this.smoothPhi = s.phi;
    this.smoothTarget = { ...s.target };
  }
}
