/**
 * Initial conditions: turns a `Scenario` spec into a flat array of star states.
 *
 * This module is pure CPU, deterministic, and knows nothing about WebGL. It
 * produces the buffer that the GPU integrator consumes on every reset.
 *
 * DISK MODEL
 * ----------
 * Real exponential disks have Sigma(r) = Sigma0 * exp(-r/h). We sample r by
 * inverse-transform sampling that CDF, so the projected surface density is
 * correct without rejection sampling. Two departures from the pure exponential:
 * an inner truncation (the exponential has unphysical infinite central density)
 * and an outer taper so the disk edge is not a hard cliff.
 *
 * ORBITAL VELOCITIES
 * ------------------
 * The key line is the circular speed: v_c(r) = sqrt( G * M(<r) / r ). If this
 * were wrong the disks would shear apart within a few orbits. M(<r) is built
 * from a truncated exponential disk plus a cored isothermal halo, which
 * together produce the flat rotation curve real spirals have.
 */

import { makeRng } from '../core/math.ts';
import type { GalaxySpec, Scenario } from '../core/types.ts';
import { G_GALACTIC, KPC_PER_MYR_TO_KM_PER_S, V0_KM_S } from './constants.ts';

export const STRIDE = 16;

/**
 * Enclosed mass as a fraction of the galaxy's total, at radius `r`.
 *
 * Split the usual way: an exponential disk carrying 10% of the mass, and a cored
 * isothermal halo carrying 90%, M(<r) = M0 * x^2/(1+x^2) with x = r/r_c. The
 * sum rises steeply through the core and then flattens, which is exactly what
 * turns v_c = sqrt(G M(<r) / r) into the flat curve real spirals have.
 *
 * Returned separately from `galaxyTotalMass` because the total mass is
 * *calibrated* from this function: we choose the mass that produces the target
 * rotation speed at the disk edge, and that requires knowing the enclosed
 * fraction there.
 */
export function enclosedFraction(r: number, g: GalaxySpec): number {
  const h = g.radius * 0.32;
  const fDisk = 0.1 * (1 - Math.exp(-r / h)) / (1 - Math.exp(-1 / 0.32));
  const x = r / Math.max(g.radius * 1.9, 26);
  const fHalo = 0.9 * (x * x) / (1 + x * x);
  return Math.min(1, fDisk + fHalo);
}

/**
 * Total mass (stellar + dark halo) represented by a galaxy's stars, in Msun.
 *
 * CALIBRATED FROM THE ROTATION SPEED, NOT HARD-CODED
 * --------------------------------------------------
 * The mass is chosen so that the circular speed at the disk edge is `V0_KM_S`,
 * then scaled by the preset's `massFactor`. Calibrating like this rather than
 * writing a mass in directly matters because the rotation curve and the
 * gravity are the same fact: if the mass is picked independently, the circular
 * velocities the stars are launched with and the force field they then feel
 * can disagree, and the disk either collapses inward or expands outward instead
 * of rotating.
 *
 * `massFactor` is what makes the presets differ. It was previously accepted in
 * the scenario type and set on every galaxy, but never read, so the
 * unequal-mass merger and the flyby intruder were silently built at full mass
 * and the scenarios that exist to show *mass asymmetry* showed none.
 */
export function galaxyTotalMass(g: GalaxySpec): number {
  const fEdge = enclosedFraction(g.radius, g);
  const v0 = V0_KM_S / KPC_PER_MYR_TO_KM_PER_S; // kpc/Myr
  return ((v0 * v0 * g.radius) / (G_GALACTIC * fEdge)) * g.massFactor;
}

/**
 * Mass enclosed within radius r, in Msun.
 *
 * Used to set the initial circular velocity, so it must agree with the force
 * the PM solver will later produce from the deposited mass -- which it does,
 * because both are the same `galaxyTotalMass` spread over the same profile.
 */
export function enclosedMass(r: number, g: GalaxySpec): number {
  return galaxyTotalMass(g) * enclosedFraction(r, g);
}

/** Inverse-CDF sample of the disk radius from a truncated exponential profile. */
function sampleRadius(u: number, g: GalaxySpec, rIn: number, rOut: number): number {
  const h = g.radius * 0.32;
  const span = Math.exp(-rIn / h) - Math.exp(-rOut / h);
  const e = Math.exp(-rIn / h) - u * span;
  return -h * Math.log(Math.max(e, 1e-12));
}

export interface GalaxyInitialState {
  x: number; y: number; z: number;
  vx: number; vy: number; vz: number;
  mass: number;
}

export interface SimulationData {
  buffer: Float32Array;
  count: number;
  galaxyCenters: GalaxyInitialState[];
  totalMass: number;
}

interface GalaxyEntry {
  g: GalaxySpec;
  n: number;
  index: 0 | 1;
  cx: number; cy: number; cz: number;
  vx: number; vy: number; vz: number;
  spin: number;
  armPhase: number;
}

/**
 * Build the full star array for a scenario.
 *
 * Galaxy A sits at -separation/2 with zero bulk velocity. Galaxy B sits at
 * +separation/2 and carries the scenario's `relativeVelocity`, so the pair
 * starts on a two-body orbit whose energy is set by that single vector.
 */
export function buildInitialConditions(
  scenario: Scenario,
  totalParticles: number,
  _haloFactor: number,
): SimulationData {
  const rng = makeRng(scenario.seed);
  const [ga, gb] = scenario.galaxies;
  const scale = totalParticles / (ga.count + gb.count);
  const nA = Math.max(1, Math.round(ga.count * scale));
  const nB = Math.max(1, Math.round(gb.count * scale));
  const count = nA + nB;

  const buffer = new Float32Array(count * STRIDE);
  const halfSep = scenario.separation * 0.5;

  const entries: GalaxyEntry[] = [
    { g: ga, n: nA, index: 0, cx: -halfSep, cy: 0, cz: 0, vx: 0, vy: 0, vz: 0, spin: 1, armPhase: 0 },
    {
      g: gb, n: nB, index: 1, cx: halfSep, cy: 0, cz: 0,
      vx: scenario.relativeVelocity.x, vy: scenario.relativeVelocity.y, vz: scenario.relativeVelocity.z,
      // Counter-rotation. Prograde encounters like this one produce the longest
      // tails, because the tidal perturbation reinforces the stars' existing
      // angular motion instead of cancelling it.
      spin: -1, armPhase: Math.PI * 0.62,
    },
  ];

  const nArms = 2;
  const tanPitch = Math.tan((12 * Math.PI) / 180);

  // Running write offset into `buffer`, in stars.
  //
  // The inner loop counts `i` from 0 for *each* galaxy, so writing at `i * STRIDE`
  // makes the second galaxy overwrite the first one's records from the start of
  // the buffer. Galaxy A's stars were being replaced by galaxy B's: the array
  // ends up with the last galaxy's data in front and untouched zeros behind.
  //
  // The symptom is quiet and wrong in a specific way. The buffer length and the
  // star count are both right, the renderer draws something that looks like a
  // galaxy, and the deposit pass reads half the mass. But because the zeroed
  // records sit at the *tail* -- indices the loop never reached for the first
  // galaxy -- the centre-of-mass diagnostic averages them in: galaxy A's
  // reported position collapses to the origin while galaxy B's is correct, so
  // the HUD shows a separation of half the preset value. And the zeros carry
  // zero mass, so half the system's gravity is simply absent.
  let base = 0;

  for (const e of entries) {
    const g = e.g;
      // Each star carries a full share of the galaxy's combined stellar+halo
      // mass, so the grid-deposited force alone reproduces the flat rotation
      // curve these velocities were derived from. The analytic halo term in
      // the integrator is therefore zero: adding it here would count the same
      // mass twice, double the effective G, and disperse the pair within a few
      // tens of Myr. See `SimulationEngine.integrateOnce`.
      const starMass = galaxyTotalMass(g) / e.n;
    const rIn = g.radius * 0.05;
    const rOut = g.radius;
    // Galaxy B's disk is tilted so the pair is not perfectly coplanar, which
    // would be an artificial special case.
    const tilt = e.index === 0 ? 0.0 : 0.38;
    const ct = Math.cos(tilt);
    const st = Math.sin(tilt);

    for (let i = 0; i < e.n; i++) {
      const isBulge = rng.next() < g.bulgeFraction;
      const theta = rng.next() * Math.PI * 2;
      let r: number;
      let yThick: number;

      if (isBulge) {
        // Isotropic in r^3 within the bulge radius: correct sampling for a
        // centrally concentrated, roughly spherical population.
        r = rIn + g.radius * 0.3 * Math.cbrt(rng.next());
        yThick = rng.normal() * g.radius * 0.09;
      } else {
        r = sampleRadius(rng.next(), g, rIn, rOut);
        // Shift r onto the nearest logarithmic arm crest. The crest satisfies
        // theta - k*ln(r) = const, so an angular miss dTheta maps to a radial
        // shift of dTheta/k in log space.
        const k = nArms / tanPitch;
        const armTheta = k * Math.log(Math.max(r, rIn)) + e.armPhase;
        const dTheta = theta - armTheta;
        // Wrap into [-pi, pi] so we always shift toward the nearest arm.
        const wrapped = Math.atan2(Math.sin(dTheta), Math.cos(dTheta));
        r = Math.exp(Math.log(Math.max(r, rIn)) + wrapped / k);
        const taper = Math.exp(-Math.pow((r - rOut * 0.8) / (rOut * 0.28), 2));
        r = Math.max(rIn, r * (1 - 0.12 * taper));
        yThick = rng.normal() * g.thickness * Math.exp(-r / (g.radius * 0.5));
      }

      const vc = Math.sqrt((G_GALACTIC * enclosedMass(r, g)) / Math.max(r, rIn * 0.5));
      const vCirc = e.spin * vc;

      const tvx = -Math.sin(theta);
      const tvz = Math.cos(theta);
      const jitter = g.velocityJitter / KPC_PER_MYR_TO_KM_PER_S;
      // Dispersion falls with radius in real disks; the hotter outskirts keep
      // the arms from looking like a wireframe.
      const jitterScale = 1 - Math.min(0.8, r / g.radius) * 0.5;

      const lvx = e.vx + tvx * vCirc + rng.normal() * jitter * jitterScale;
      const lvy = e.vy + rng.normal() * jitter * jitterScale;
      const lvz = e.vz + tvz * vCirc + rng.normal() * jitter * jitterScale;

      // Rotate the disk about the x axis by `tilt`, applying the same rotation
      // to position and velocity so the disk stays internally consistent.
      const px = r * Math.cos(theta);
      const py = yThick;
      const pz = r * Math.sin(theta);

      const w = (base + i) * STRIDE;
      buffer[w + 0] = e.cx + px;
      buffer[w + 1] = e.cy + (py * ct - pz * st);
      buffer[w + 2] = e.cz + (py * st + pz * ct);
      buffer[w + 3] = lvx;
      buffer[w + 4] = lvy * ct - lvz * st;
      buffer[w + 5] = lvy * st + lvz * ct;
      buffer[w + 6] = e.index;
      buffer[w + 7] = rng.next();
      buffer[w + 8] = r;
      buffer[w + 9] = g.radius;
      buffer[w + 10] = e.vx;
      buffer[w + 11] = e.vy;
      buffer[w + 12] = e.vz;
      buffer[w + 13] = starMass;
      buffer[w + 14] = e.spin;
      buffer[w + 15] = Math.max(vCirc, 1e-3);
    }

    base += e.n;
  }

  return {
    buffer,
    count,
    galaxyCenters: entries.map((e) => ({
      x: e.cx, y: e.cy, z: e.cz,
      vx: e.vx, vy: e.vy, vz: e.vz,
      mass: galaxyTotalMass(e.g),
    })),
    totalMass: galaxyTotalMass(ga) + galaxyTotalMass(gb),
  };
}
