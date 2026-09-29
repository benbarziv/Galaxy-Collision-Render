/**
 * The integrator: leapfrog KDK, executed on the GPU via transform feedback.
 *
 * WHY LEAPFROG
 * -------------
 *   v(n+1/2) = v(n) + a(x(n))   * dt/2
 *   x(n+1)   = x(n) + v(n+1/2) * dt
 *   v(n+1)   = v(n+1/2) + a(x(n+1)) * dt/2
 *
 * Properties that matter here, in order of importance:
 *
 *  1. It is symplectic, so energy error is bounded and oscillatory rather than
 *     secular. Over the ~2 Gyr the merger presets integrate, a non-symplectic
 *     scheme loses enough energy that galaxies visibly collapse. This is the
 *     single most important numerical choice in the project.
 *  2. One force evaluation per drift, no velocity dependence beyond the force.
 *  3. The same program can be re-run `substeps` times per frame with no CPU
 *     involvement at all.
 *
 * FIXED TIMESTEP
 * --------------
 * dt is fixed and independent of frame rate. The render loop accumulates real
 * elapsed time and consumes it in whole dt-sized substeps, capped per frame. The
 * simulation is therefore frame-rate-independent and deterministic: the same
 * wall-clock duration always produces the same number of integration steps.
 */

export const INTEGRATE_VS = /* glsl */ `#version 300 es
precision highp float;
precision highp int;

in vec3  aPosition;
in vec3  aVelocity;
in float aGalaxy;
in float aSeed;
in float aR0;
in float aRNorm;
in vec3  aBulkVel;
in float aMass;
in float aSpin;
in float aV0;

// Force field as a 2D atlas of the 3D grid: an N*N by N texture where slice z
// occupies rows [z*N, (z+1)*N).
uniform sampler2D uForce;

// The dark-matter halo is analytic rather than deposited: it is smooth and
// carries ~90% of the mass, so depositing it would spend grid resolution on a
// feature that has no structure to resolve.
uniform float uG;
uniform float uHaloMass0;
uniform float uHaloMass1;
uniform float uHaloRadius0;
uniform float uHaloRadius1;
uniform vec3  uHaloCenter0;
uniform vec3  uHaloCenter1;
uniform float uHaloSoftening;

uniform float uDt;
uniform float uGridSize;
uniform vec3  uBoxMin;
uniform float uBoxSize;

out vec3 vPosition;
out vec3 vVelocity;
out float vGalaxy;
out float vSeed;
out float vR0;
out float vRNorm;
out vec3 vBulkVel;
out float vMass;
out float vSpin;
out float vV0;

/**
 * Trilinear sample of the force atlas.
 *
 * The grid force is a 2D atlas rather than a 3D texture because every solver
 * pass is a 2D fragment pass, so the in-slice axes are filtered by the sampler
 * and only the slice axis needs manual interpolation.
 *
 * The atlas encoding is the shared one from common.ts:
 *
 *     col = x + z*N        (long edge, N^2 texels)
 *     row = y              (short edge, N texels)
 *
 * so the slice offset multiplies the *x* index, not y. Getting that backwards
 * samples the wrong column entirely: the field still looks smooth and bounded,
 * so nothing errors, but the disk feels the force from a different part of the
 * grid and spirals apart within a few orbits.
 */
vec3 sampleForce(vec3 world) {
  float N = uGridSize;
  // Clamp inside the box so a star just past the boundary feels the edge value
  // instead of wrapping to the far side of the periodic box.
  vec3 g = clamp((world - uBoxMin) / uBoxSize * N, vec3(0.0), vec3(N - 1.0));

  // Slice axis, interpolated by hand. The in-slice axes are left to the
  // sampler's bilinear filter, so sampling at the cell *centre* is what makes
  // the two halves of the interpolation line up. The +0.5 is what puts the
  // sample on a texel centre rather than a texel corner: drop it and the filter
  // blends the two neighbouring cells in equal measure, which biases the force
  // by half a cell in x and y everywhere.
  float z0 = floor(g.z);
  float z1 = min(z0 + 1.0, N - 1.0);
  float fz = g.z - z0;
  vec2 uv0 = vec2((g.x + z0 * N + 0.5) / (N * N), (g.y + 0.5) / N);
  vec2 uv1 = vec2((g.x + z1 * N + 0.5) / (N * N), (g.y + 0.5) / N);
  return mix(texture(uForce, uv0).xyz, texture(uForce, uv1).xyz, fz);
}

/**
 * Acceleration from a cored isothermal halo.
 *
 * M(<r) = M0 * x^2/(1+x^2), x = r/r_c, so a = -G M(<r) r_vec / rs^3. Falls as
 * 1/r^2 outside the core and as 1/r inside it; the core regularises the centre.
 */
vec3 haloAccel(vec3 pos, vec3 center, float mass, float coreRadius) {
  vec3 d = pos - center;
  float r2 = dot(d, d);
  float x = sqrt(r2) / coreRadius;
  float enclosed = mass * (x * x) / (1.0 + x * x);
  float rs = sqrt(r2 + uHaloSoftening * uHaloSoftening);
  return -uG * enclosed * d / (rs * rs * rs);
}

void main() {
  vec3 pos = aPosition;
  vec3 vel = aVelocity;

  vec3 acc = sampleForce(pos)
           + haloAccel(pos, uHaloCenter0, uHaloMass0, uHaloRadius0)
           + haloAccel(pos, uHaloCenter1, uHaloMass1, uHaloRadius1);
  vec3 vHalf = vel + acc * (uDt * 0.5);

  vec3 newPos = pos + vHalf * uDt;

  vec3 accNew = sampleForce(newPos)
              + haloAccel(newPos, uHaloCenter0, uHaloMass0, uHaloRadius0)
              + haloAccel(newPos, uHaloCenter1, uHaloMass1, uHaloRadius1);
  vec3 newVel = vHalf + accNew * (uDt * 0.5);

  vPosition = newPos;
  vVelocity = newVel;
  vGalaxy   = aGalaxy;
  vSeed     = aSeed;
  vR0       = aR0;
  vRNorm    = aRNorm;
  vBulkVel  = aBulkVel;
  vMass     = aMass;
  vSpin     = aSpin;
  vV0       = aV0;
}
`;

/** Fragment shader for the transform-feedback program; never rasterised. */
export const INTEGRATE_FS = /* glsl */ `#version 300 es
precision highp float;
out vec4 fragColor;
void main() { fragColor = vec4(0.0); }
`;

/** Transform feedback varying names, in buffer order. */
export const INTEGRATE_VARYINGS = [
  'vPosition',
  'vVelocity',
  'vGalaxy',
  'vSeed',
  'vR0',
  'vRNorm',
  'vBulkVel',
  'vMass',
  'vSpin',
  'vV0',
];

