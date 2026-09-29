/**
 * Simulation unit system and derived constants.
 *
 * WHY THESE UNITS
 * ---------------
 * Working in SI makes the numbers absurd (a parsec is 3e16 m, G is 6.67e-11, and
 * a galaxy orbit takes ~1e17 seconds), so instead we use a system where the
 * numbers are human-scale: a galaxy's orbital speed is O(100) and its
 * dynamical time is O(1).
 *
 *   length : 1 kpc
 *   mass   : 1 solar mass
 *   time   : 1 Myr
 *
 * In these units G = 4.5e-12 kpc^3/(Msun*Myr^2). That single constant is
 * what makes the rest of the simulation self-consistent: a 1e11 Msun galaxy
 * inside 8 kpc should orbit at a few hundred km/s, and it does with this G.
 *
 * The value is *derived* from the tabulated 4.30091e-6 kpc (km/s)^2 / Msun
 * rather than copied from it, because those are different units. See
 * `G_GALACTIC` below -- getting this wrong is silent, and it is the reason the
 * simulation used to fling every star out of the box on the first step.
 *
 * THE PARTICLE-MASS CONVENTION
 * ----------------------------
 * Real spirals are ~10% stars and ~90% dark matter by mass. We render only the
 * stars but give each one a full share of the combined stellar+halo mass, which
 * keeps the rotation curve flat out to the visible edge of the disk. That flat
 * curve is the single most important thing for the disks to look and behave
 * like real galaxies.
 */

import type { SimTuning, QualityProfile, QualityTier } from '../core/types.ts';

export const KPC_PER_MYR_TO_KM_PER_S = 977.79;

/**
 * G in kpc (km/s)^2 / Msun -- the value as usually tabulated.
 *
 * Note the units: this is the gravitational constant expressed with *velocity*
 * in km/s, which is how it appears in every reference source. It is NOT valid
 * for this simulator, whose time unit is the Myr.
 */
export const G_KPC_KMS2_PER_MSUN = 4.30091e-6;

/**
 * Standard gravitational constant in kpc^3 / (Msun * Myr^2).
 *
 * DERIVING THIS RATHER THAN COPYING IT
 * -----------------------------------
 * The tabulated constant above carries velocity in km/s, but everything in the
 * simulation -- initial conditions, timestep, integrator state -- carries it in
 * kpc/Myr. Substituting one for the other without converting the units makes G
 * too large by exactly (1 kpc/Myr in km/s)^2:
 *
 *     1 kpc/Myr = 977.79 km/s   =>   G is too big by 977.79^2 = 9.56e5
 *
 * That factor is the same order as the square of the conversion, so it is not
 * obvious by inspection -- nothing overflows, every intermediate value stays
 * inside float range, and the shader still produces a smooth, plausible-looking
 * force field. The simulation is simply in a universe where gravity is a
 * million times too strong.
 *
 * The symptom is distinctive: every star is unbound from the first step. The
 * pair is still moving toward itself at a few hundred km/s, but the orbital
 * timescale has collapsed to a few hundred years, so the stars are thrown off
 * almost immediately and the two galaxies separate ballistically at a
 * constant several million km/s -- faster than light -- with a 100% escaped
 * fraction and a separation that grows linearly forever.
 *
 * Correcting the units brings the circular speed at 14 kpc for a 5e11 Msun
 * galaxy to 404 km/s, which is right for an L* spiral.
 */
export const G_GALACTIC = G_KPC_KMS2_PER_MSUN / (KPC_PER_MYR_TO_KM_PER_S * KPC_PER_MYR_TO_KM_PER_S);

export const KPC_IN_KM = 3.0857e16;
export const MYR_IN_S = 3.1557e13;

/** Reference flat circular speed, km/s (typical for an L* spiral). */
export const V0_KM_S = 220;

export function defaultTuning(overrides: Partial<SimTuning> = {}): SimTuning {
  return {
    G: G_GALACTIC,
    // ~1 kpc softening: comparable to a star's mean spacing, which keeps the
    // particle-particle term finite without visibly inflating the disk.
    softening: 1.0,
    speed: 1.0,
    // Fixed base timestep. Chosen so the fastest star in a flyby
    // (v ~ 900 km/s ~ 0.92 kpc/Myr) moves <0.05 kpc per step, comfortably
    // below the PM cell size and far below the 1 kpc softening length.
    dt: 0.045,
    pmBoxScale: 2.2,
    haloFactor: 8.0,
    haloRadius: 26.0,
    v0: V0_KM_S,
    ...overrides,
  };
}

export const QUALITY_TIERS: QualityTier[] = ['low', 'medium', 'high', 'ultra'];

/**
 * Grid sizes MUST be powers of two.
 *
 * The Poisson solve is a radix-2 decimation-in-time FFT, which is only defined
 * for N = 2^k. A non-power-of-two grid size does not produce a wrong-but-close
 * answer, it produces garbage: the bit-reversal permutation is undefined and
 * the butterfly passes do not span the array. This is easy to get wrong because
 * nothing errors out -- the framebuffer stays complete and the draw calls
 * succeed, so the only symptom is a force field that is noise.
 *
 * RESOLUTION VERSUS BOX SIZE
 * --------------------------
 * The grid is stored as a (N*N) x N atlas, so N caps at sqrt(MAX_TEXTURE_SIZE):
 * 128 gives a 16384-pixel-wide texture, which every WebGL2 desktop GPU
 * accepts, and 256 would need 65536 and is not portable.
 *
 * That caps the force-field resolution, so the *box* has to be sized to suit.
 * With a 128^3 grid over the ~300 kpc box the widest preset needs, each cell is
 * 2.3 kpc across -- coarse enough to smooth over the fine spiral-arm structure
 * the integrator is supposed to feel. The trade is deliberate: PM forces are
 * inherently smoother than direct N-body, and the structures that matter for
 * the tidal tails (tens of kpc) are far above the cell size. `clampGridSize`
 * drops to a smaller power of two if a device reports a lower texture limit.
 */
export const QUALITY_PROFILES: Record<QualityTier, QualityProfile> = {
  low: {
    tier: 'low',
    label: 'Low - 24k stars',
    particles: 24000,
    gridSize: 64,
    substeps: 1,
    trailLength: 0,
    bloomIntensity: 0.75,
    bloom: false,
    resolutionScale: 1.0,
    starfield: true,
  },
  medium: {
    tier: 'medium',
    label: 'Medium - 60k stars',
    particles: 60000,
    gridSize: 128,
    substeps: 1,
    trailLength: 10,
    bloomIntensity: 1.0,
    bloom: true,
    resolutionScale: 1.0,
    starfield: true,
  },
  high: {
    tier: 'high',
    label: 'High - 140k stars',
    particles: 140000,
    gridSize: 128,
    substeps: 2,
    trailLength: 14,
    bloomIntensity: 1.15,
    bloom: true,
    resolutionScale: 1.0,
    starfield: true,
  },
  ultra: {
    tier: 'ultra',
    label: 'Ultra - 260k stars',
    particles: 260000,
    gridSize: 128,
    substeps: 2,
    trailLength: 18,
    bloomIntensity: 1.3,
    bloom: true,
    resolutionScale: 1.0,
    starfield: true,
  },
};

/**
 * Clamp a requested grid size to the largest power of two whose square still
 * fits the device's maximum texture dimension.
 */
export function clampGridSize(requested: number, maxTextureSize: number): number {
  let n = 16;
  while (n * 2 <= requested && n * 2 * n * 2 <= maxTextureSize) n *= 2;
  return n;
}
