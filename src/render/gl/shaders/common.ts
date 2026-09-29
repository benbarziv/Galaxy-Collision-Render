/**
 * Shared GLSL snippets.
 *
 * Kept as template strings rather than separate .glsl files so Vite needs no
 * GLSL plugin. Every snippet here is documented with the math it implements.
 */

/** Fullscreen triangle. No vertex buffer required; uses gl_VertexID. */
export const FULLSCREEN_VS = /* glsl */ `#version 300 es
precision highp float;
out vec2 vUv;
void main() {
  // Three vertices covering the viewport: (-1,-1), (3,-1), (-1,3).
  vec2 p = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2));
  vUv = p;
  gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
`;

/**
 * Trig-based hash, for per-star pseudo-random colour and size variation.
 *
 * We need a deterministic value per star but carry no state, and the built-in
 * hash() quality varies between drivers, so we build our own from fract(sin()).
 */
export const HASH = /* glsl */ `
float hash11(float p) {
  p = fract(p * 0.1031);
  p *= p + 33.33;
  p *= p + p;
  return fract(p);
}
vec3 hash31(float p) {
  vec3 p3 = fract(vec3(p) * vec3(0.1031, 0.1030, 0.0973));
  p3 += dot(p3, p3.yxz + 33.33);
  return fract((p3.xxy + p3.yzz) * p3.zyx);
}
`;

/**
 * Stellar colour ramp.
 *
 * Real star colours are not linear in temperature, but this is tuned by eye to
 * land on the deep orange of an early M giant through solar yellow to the
 * blue-white of an O/B star, which is the range that reads correctly against a
 * black background.
 */
export const STAR_COLOR_RAMP = /* glsl */ `
vec3 starColorRamp(float t) {
  t = clamp(t, 0.0, 1.0);
  vec3 cold = vec3(1.00, 0.60, 0.40);
  vec3 mid  = vec3(1.00, 0.94, 0.82);
  vec3 hot  = vec3(0.70, 0.82, 1.00);
  return t < 0.5
    ? mix(cold, mid, smoothstep(0.0, 0.5, t))
    : mix(mid, hot, smoothstep(0.5, 1.0, t));
}
`;

/**
 * The 3D-grid-as-2D-atlas addressing convention.
 *
 * The Poisson solver needs a 3D grid, but WebGL2 has no 3D render target, so the
 * grid is flattened into a 2D texture: N^2 texels WIDE and N texels TALL, with
 * slice z occupying rows [z*N, (z+1)*N). Encoding:
 *
 *     col = x + z*N        (long edge)
 *     row = y              (short edge)
 *
 * Every pass that reads or writes the atlas must use these two helpers. The
 * convention is genuinely easy to get subtly wrong -- the encoding is
 * asymmetric, so the two axes normalise by different extents -- and the failure
 * is silent. A mismatched decode still produces complete textures and successful
 * draw calls; the force field is simply wrong, the disks shear, and everything
 * disperses within a few Myr with no error anywhere to explain why.
 *
 * Requires `uniform float uGridSize;` in the calling shader.
 */
export const ATLAS = /* glsl */ `
/** Atlas UV -> 3D cell index (integer, centre-free). */
vec3 atlasToCell(vec2 uv, float N) {
  float col = floor(uv.x * N * N);
  float row = floor(uv.y * N);
  return vec3(mod(col, N), mod(row, N), floor(col / N));
}

/** 3D cell index -> atlas UV, sampling the texel centre. */
vec2 cellToAtlas(vec3 cell, float N) {
  return vec2(
    (cell.x + cell.z * N + 0.5) / (N * N),
    (cell.y + 0.5) / N
  );
}

/** Cell -> NDC, for the deposit pass. */
vec2 cellToClip(vec3 cell, float N) {
  return vec2(
    (cell.x + cell.z * N) / (N * N),
    cell.y / N
  ) * 2.0 - 1.0;
}
`;

/**
 * Coordinate convention shared by the solver and the integrator.
 *
 * The Poisson solve runs on 2D slices of the 3D grid, but the integrator samples
 * the finished field as a 3D texture. Both must agree on where a world position
 * lands in the grid, so the convention lives in one place.
 */
export const GRID_COORDS = /* glsl */ `
uniform float uGridSize;
uniform vec3  uBoxMin;
uniform float uBoxSize;

vec3 gridCoordFromWorld(vec3 world) {
  return (world - uBoxMin) / uBoxSize * uGridSize;
}
vec3 worldFromGridCoord(vec3 g) {
  return uBoxMin + (g / uGridSize) * uBoxSize;
}
`;

/** sRGB transfer function, for the final tonemap. */
export const COLOR_UTILS = /* glsl */ `
vec3 linearToSrgb(vec3 c) {
  c = clamp(c, 0.0, 1.0);
  return mix(c * 12.92, 1.055 * pow(c, vec3(1.0 / 2.4)) - 0.055, step(vec3(0.0031308), c));
}
`;
