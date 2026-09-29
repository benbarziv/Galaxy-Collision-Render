/**
 * Particle-Mesh gravity solver shaders.
 *
 * THE METHOD
 * ----------
 * A direct N-body sum over every particle pair is O(N^2): at 140k stars that is
 * 2e10 pair interactions per step, which no GPU will tolerate. The Particle-Mesh
 * (PM) method replaces the pairwise sum with a three-step approximation:
 *
 *   1. DEPOSIT   Each star splats its mass onto a grid (CIC).
 *   2. SOLVE     Cell densities become a gravitational potential by solving
 *                nabla^2 phi = 4*pi*G*rho with an FFT Green's function solver.
 *   3. SAMPLE    The potential is differentiated on the grid to give a force,
 *                and stars interpolate the resulting field.
 *
 * Cost is O(N) for deposit and sample and O(N_grid log N_grid) for the solve,
 * independent of particle count. The error is pair-like for N >> N_grid, which
 * is exactly our regime, so the tidal field driving the tails is near-exact.
 *
 * DISCRETE NORMALISATION
 * ----------------------
 * Let rho_n be the density in cell n. The unnormalised forward transform is
 * rho_m = sum_n rho_n exp(-i*2*pi*m*n/N), and the inverse carries 1/N^3. Since
 * each 1D axis pass already applies 1/N, three passes give exactly 1/N^3 and no
 * separate normalisation pass is needed.
 */

import { HASH, ATLAS } from './common.ts';

export { HASH };

/**
 * Pass 1: mass deposit. One point per star into a 3D grid stored as a 2D atlas
 * of size (N*N) x N, where slice z occupies rows [z*N, (z+1)*N).
 *
 * The splat is a 3x3x3 trilinear stencil emitted as nine point draws with
 * additive blending, one per corner of the cell, weighted by the CIC
 * coefficients. Linear filtering alone would only spread within a single slice
 * and would leave a visible discontinuity in z.
 */
export const DEPOSIT_VS = /* glsl */ `#version 300 es
precision highp float;

${ATLAS}

in vec3  aPosition;
in float aMass;

uniform float uGridSize;
uniform vec3  uBoxMin;
uniform float uBoxSize;
uniform float uPointSize;
uniform int   uCorner;   // 0..8, which cell corner to splat into

out float vMass;
out float vWeight;

void main() {
  vec3 g = (aPosition - uBoxMin) / uBoxSize * uGridSize;
  // Clip stars outside the box. Without this they would wrap around the
  // periodic boundary and reappear on the opposite side.
  if (any(lessThan(g, vec3(0.0))) || any(greaterThanEqual(g, vec3(uGridSize)))) {
    gl_Position = vec4(2.0, 2.0, 2.0, 1.0);
    vMass = 0.0;
    vWeight = 0.0;
    return;
  }

  // CIC weights: the fractional position within the cell, per axis.
  vec3 f = fract(g);
  vec3 base = floor(g);
  vec3 c = vec3(
    float((uCorner & 1) != 0),
    float((uCorner & 2) != 0),
    float((uCorner & 4) != 0)
  );
  vec3 w = mix(1.0 - f, f, c);

  float weight = w.x * w.y * w.z;
  vMass = aMass * weight;
  // The bare weight is carried alongside the mass for inspection only. The
  // density pass must NOT divide by it: CIC is already conservative, so the
  // mass channel is correct as deposited. See DENSITY_SCALE_FS.
  vWeight = weight;

  vec3 cell = base + c;
  gl_Position = vec4(cellToClip(cell, uGridSize), 0.0, 1.0);
  gl_PointSize = uPointSize;
}
`;

export const DEPOSIT_FS = /* glsl */ `#version 300 es
precision highp float;
in float vMass;
in float vWeight;
out vec2 fragColor;
void main() { fragColor = vec2(vMass, vWeight); }
`;

/** Straight blit, used to seed and copy solver textures. */
export const COPY_FS = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uSrc;
out vec4 fragColor;
void main() { fragColor = texture(uSrc, vUv); }
`;

/** Blit with a constant value, used to zero a target. */
export const CLEAR_FS = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
uniform float uValue;
out vec4 fragColor;
void main() { fragColor = vec4(uValue); }
`;

/**
 * Pass 2: convert deposited mass into the Poisson source term.
 * rho = mass / cellVolume, and the solver consumes 4*pi*G*rho.
 */
export const DENSITY_SCALE_FS = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uSrc;
uniform float uG;
uniform float uCellVolume;
out vec2 fragColor;
void main() {
  // Read the mass channel ONLY. The deposit also accumulates the bare CIC
  // weight, and dividing the mass by it looks like a correction for "where
  // within its cell the star happened to land". It is exactly backwards, and it
  // is the most destructive bug this solver can have.
  //
  // Cloud-in-Cell is conservative by construction: the eight corner weights sum
  // to one, so a star's mass is already split correctly across its neighbours
  // and the cell totals already add up to the true mass. Dividing each cell by
  // the weight it received instead *restores the full mass to every one of
  // those eight cells* -- one star is deposited with up to eight times its own
  // mass, and the amplification is worst where the weight is smallest, i.e. at
  // cell corners. The density field becomes far too peaked and strongly
  // position-dependent, the force near cell corners runs away, and the
  // integration diverges within a few Myr: stars are flung past the box edge
  // and every diagnostic reads 100% escaped.
  float mass = texture(uSrc, vUv).r;
  float rho = mass / uCellVolume;
  fragColor = vec2(4.0 * 3.14159265358979 * uG * rho, 0.0);
}
`;

/**
 * The separable 1D FFT, used for all three axes.
 *
 * A radix-2 decimation-in-time FFT. Pass 0 performs the bit-reversal
 * permutation; passes 1..log2(N) are the butterflies. Both stages are expressed
 * as fullscreen fragment passes that read and write whole textures, so a pass
 * costs one draw call regardless of grid size and the GPU parallelises it
 * across every cell at once.
 *
 * Stage 1 pairs index j with its bit-reversal:
 *
 *     n = bitreverse(j, lgN)
 *     if (j < n)       X[j] = (a[j] + a[n]) * w^j
 *     else             X[n] = (a[j] + a[n]) * w^j
 *
 * which is the standard DIF butterfly applied to the permuted data, with the
 * twiddle exp(-i*pi*j/2^(m+1)) for the sub-transform of width 2^(m+1).
 *
 * Subsequent butterfly stages combine adjacent cells of the current
 * sub-transform width.
 *
 * uStage: 0 = bit-reversal, 1..lgN = butterfly passes.
 * uAxis:  0 = x (contiguous), 1 = y (contiguous in-slice), 2 = z (whole slices).
 * uSign:  -1 forward, +1 inverse.
 *
 * The full stage count is 1 + lgN (one permutation plus lgN butterflies). The
 * butterfly pairs index j with j + M/2 and applies exp(sign * i * 2*pi * j/M),
 * which is the decimation-in-time form matching the adjacent-element pairing
 * below.
 *
 * The inverse differs from the forward only in the sign of the twiddle and the
 * 1/N scaling, which is folded into the final butterfly pass.
 */
export const FFT_FS = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;

${ATLAS}

uniform sampler2D uSrc;
uniform float uGridSize;
uniform int   uStage;    // 0 = bit-reversal, 1..lgN = butterflies
uniform int   uAxis;     // 0 = x, 1 = y, 2 = z
uniform float uSign;     // -1 forward, +1 inverse

out vec2 fragColor;

const float PI = 3.14159265358979;

vec3 texelToGrid(vec2 uv) { return atlasToCell(uv, uGridSize); }
vec2 gridToTexel(vec3 g) { return cellToAtlas(g, uGridSize); }

/** Reverse the low lgN bits of i. */
int bitReverse(int i, int lgN) {
  int r = 0;
  for (int b = 0; b < 16; b++) {
    if (b >= lgN) break;
    if ((i & (1 << b)) != 0) r |= 1 << (lgN - 1 - b);
  }
  return r;
}

vec2 cmul(vec2 a, vec2 b) {
  return vec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

void main() {
  float Nf = uGridSize;
  int N = int(Nf);
  int lgN = 0;
  for (int s = N; s > 1; s >>= 1) lgN++;

  vec3 g = texelToGrid(vUv);
  int idx = (uAxis == 0) ? int(g.x) : ((uAxis == 1) ? int(g.y) : int(g.z));

  if (uStage == 0) {
    // Bit-reversal permutation. A straight copy -- no twiddle, no butterfly.
    //
    // Fusing a width-2 butterfly into this pass looks like a free saving, and
    // it is wrong. The first butterfly must combine *adjacent post-permutation*
    // elements; the fused version instead combines an element with its own
    // bit-reversal, which are generally not adjacent. That computes a different
    // linear operator rather than a permuted DFT, so every later stage runs on
    // scrambled data and the result is silently wrong rather than obviously
    // broken. The permutation costs one extra pass and is worth keeping alone.
    int n = bitReverse(idx, lgN);
    vec3 gn = g;
    if (uAxis == 0) gn.x = float(n);
    else if (uAxis == 1) gn.y = float(n);
    else gn.z = float(n);
    fragColor = texture(uSrc, gridToTexel(gn)).rg;
    return;
  }

  // Butterfly stages: sub-transform width 2^uStage, pairing half a width apart.
  float M = exp2(float(uStage));
  float halfM = M * 0.5;
  float group = floor(float(idx) / M) * M;
  float within = float(idx) - group;

  vec2 a = texture(uSrc, gridToTexel(g)).rg;
  vec3 gp = g;
  float partner = group + (within < halfM ? within + halfM : within - halfM);
  if (uAxis == 0) gp.x = partner;
  else if (uAxis == 1) gp.y = partner;
  else gp.z = partner;
  vec2 b = texture(uSrc, gridToTexel(gp)).rg;

  // Twiddle for this position within the sub-transform.
  //
  // The factor is 2*PI, not PI. A decimation-in-frequency butterfly uses
  // exp(-i*pi*j/M) and pairs indices M/2 apart; a decimation-in-time butterfly
  // uses exp(-i*2*pi*j/M) and pairs adjacent elements, which is what the partner
  // offset above does. Using the DIF angle with DIT pairing silently halves
  // every frequency axis, so the solver returns a potential that looks
  // well-formed but corresponds to a box twice the intended size -- forces come
  // out far too weak at large radius, disks sag, and the pair never merges.
  float ang = uSign * 2.0 * PI * within / M;
  vec2 t = cmul(b, vec2(cos(ang), sin(ang)));

  vec2 result = within < halfM ? (a + t) : (a - t);

  // The inverse transform carries 1/N. Folding it into the final pass avoids a
  // separate normalisation sweep over the whole grid.
  if (uSign > 0.0 && uStage == lgN) result /= float(N);

  fragColor = result;
}
`;

/**
 * Pass 3: multiply by the Green's function in Fourier space.
 *
 * Wave numbers are wrapped into [-N/2, N/2) so the highest frequency is treated
 * as negative, which is what makes the inverse transform real.
 */
export const GREENS_FS = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;

${ATLAS}

uniform sampler2D uSrc;
uniform float uGridSize;
uniform float uBoxSize;

out vec2 fragColor;

const float PI = 3.14159265358979;

void main() {
  float N = uGridSize;
  // Decode through the shared helper so this pass cannot drift out of step
  // with the deposit and gradient passes. The cell indices come back in
  // [0, N) on all three axes; the frequency wrap below is what makes the
  // highest frequency negative.
  vec3 g = atlasToCell(vUv, N);
  float x = g.x;
  float y = g.y;
  float z = g.z;

  float mx = x <= N * 0.5 ? x : x - N;
  float my = y <= N * 0.5 ? y : y - N;
  float mz = z <= N * 0.5 ? z : z - N;

  vec2 val = texture(uSrc, vUv).rg;

  float k2 = mx * mx + my * my + mz * mz;
  float kFac = 2.0 * PI / uBoxSize;

  // Remove the DC mode: it is the mean potential, physically arbitrary, and
  // keeping it would let the whole field drift over long integrations.
  if (k2 == 0.0) { fragColor = vec2(0.0, 0.0); return; }

  float inv = -1.0 / (kFac * kFac * k2);
  // BOTH channels must be scaled. The source is real, so its spectrum is
  // Hermitian (real part even, imaginary part odd) and carries information in
  // both. Discarding the imaginary part keeps only the cosine half of the
  // signal and the resulting potential is visibly wrong.
  fragColor = vec2(val.r * inv, val.g * inv);
}
`;

/**
 * Pass 4: differentiate the potential to get the force.
 *
 * Central differences on the grid:
 *
 *     dphi/dx|_i = (phi_{i+1} - phi_{i-1}) / (2*h)
 *
 * The spectral derivative is more accurate in principle, but it has two
 * practical problems here. First, it requires a full complex multiply by i*k
 * with a sign convention that is easy to get subtly wrong; with the forward
 * transform defined as exp(-i*2*pi*m*n/N) the product is (-k*B, k*A), and using
 * (k*A, k*B) instead returns exactly -1/2 * grad(phi) -- a factor-of-two error
 * that makes disks rotate at half their correct rate. Second, it needs three
 * extra forward+inverse transform triples, roughly 60% more solver time.
 *
 * Central differences are unconditionally stable, cost one pass, and their
 * error is O(h^2) with h = box/N. At N = 64 over a 180 kpc box that is 2.8 kpc,
 * which is coarse enough to matter for the finest structures -- so we compensate
 * with a 7-point stencil (fourth order) rather than a 3-point one:
 *
 *     dphi/dx|_i = (phi_{i-2} - 8*phi_{i-1} + 8*phi_{i+1} - phi_{i+2}) / (12*h)
 *
 * That is accurate to O(h^4) and costs two extra taps, which is far cheaper than
 * the transforms it replaces and effectively removes the grid-scale noise that
 * the long integrations would otherwise accumulate.
 */
export const POTENTIAL_GRADIENT_FS = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;

${ATLAS}

uniform sampler2D uPhi;
uniform float uGridSize;
uniform float uCellSize;

out vec4 fragColor;

float potentialAt(vec3 cell, float N) {
  // Clamp on all three axes rather than wrapping. The FFT is periodic, so
  // wrapping would be self-consistent, but it would import the box's opposite
  // edges into the force: a star near one face would feel the mass sitting on
  // the far face. Clamping keeps the edge value constant instead, which is the
  // better-behaved choice when the box is only modestly larger than the system.
  cell = clamp(cell, vec3(0.0), vec3(N - 1.0));
  return texture(uPhi, cellToAtlas(cell, N)).r;
}

void main() {
  float N = uGridSize;
  vec3 g = atlasToCell(vUv, N);

  vec3 ex = vec3(1.0, 0.0, 0.0);
  vec3 ey = vec3(0.0, 1.0, 0.0);
  vec3 ez = vec3(0.0, 0.0, 1.0);

  // 7-point fourth-order central difference along each axis.
  float dx = (potentialAt(g - 2.0 * ex, N) - 8.0 * potentialAt(g - ex, N)
            + 8.0 * potentialAt(g + ex, N) - potentialAt(g + 2.0 * ex, N))
           / (12.0 * uCellSize);
  float dy = (potentialAt(g - 2.0 * ey, N) - 8.0 * potentialAt(g - ey, N)
            + 8.0 * potentialAt(g + ey, N) - potentialAt(g + 2.0 * ey, N))
           / (12.0 * uCellSize);
  float dz = (potentialAt(g - 2.0 * ez, N) - 8.0 * potentialAt(g - ez, N)
            + 8.0 * potentialAt(g + ez, N) - potentialAt(g + 2.0 * ez, N))
           / (12.0 * uCellSize);

  // a = -grad(phi)
  fragColor = vec4(-dx, -dy, -dz, 1.0);
}
`;
