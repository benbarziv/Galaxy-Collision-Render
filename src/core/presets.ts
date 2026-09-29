/**
 * Collision presets.
 *
 * The numbers here are chosen from real galaxy-collision parameters (the Antennae
 * galaxies and NGC 6050 are the canonical head-on examples) and then expressed
 * in the simulation's unit system described in sim/constants.ts.
 *
 * Impact parameter is encoded implicitly: galaxy A always starts at
 * (-separation/2, 0, 0) and galaxy B at (+separation/2, 0, 0). To vary the
 * impact parameter we rotate each galaxy's *internal rotation axis* relative to
 * the separation vector, and additionally offset galaxy B along z in the grazing
 * scenarios. A "head-on" run is the one where both disks are coplanar and the
 * separation vector lies in that plane, so the two disks collide broadside.
 */

import type { Scenario } from './types.ts';

/** Cool blue-white, like an early-type or metal-poor population. */
const BLUE = { r: 0.42, g: 0.62, b: 1.0 };
/** Warm amber-orange, like a star-forming population. */
const AMBER = { r: 1.0, g: 0.62, b: 0.28 };
/** Neutral warm white. */
const IVORY = { r: 1.0, g: 0.93, b: 0.82 };
/** Deep gold core. */
const GOLD_CORE = { r: 1.0, g: 0.86, b: 0.6 };
/** Hot blue-white core. */
const BLUE_CORE = { r: 0.78, g: 0.9, b: 1.0 };

export const SCENARIOS: Scenario[] = [
  {
    id: 'headon',
    name: 'Head-on Collision',
    blurb: 'Two comparable spirals meet broadside at ~300 km/s.',
    description:
      'The classic Toomre-and-Toomre 1972 construction. Two equal-mass disks approach on a ' +
      'low-angular-momentum, nearly radial orbit, so almost no angular momentum is available to ' +
      'spin up the remnant. Expect long, thin antipodal tidal tails — the "Antennae" morphology — ' +
      'and a smooth, pressure-supported elliptical remnant rather than a rotating disk. Because ' +
      'the orbit is radial, the galaxies oscillate through each other twice before settling.',
    seed: 'headon-antennae',
    galaxies: [
      {
        id: 'a',
        label: 'Galaxy A',
        count: 22000,
        radius: 14,
        thickness: 0.55,
        massFactor: 1.0,
        color: BLUE,
        coreColor: BLUE_CORE,
        velocityJitter: 16,
        bulgeFraction: 0.18,
      },
      {
        id: 'b',
        label: 'Galaxy B',
        count: 22000,
        radius: 14,
        thickness: 0.55,
        massFactor: 1.0,
        color: AMBER,
        coreColor: GOLD_CORE,
        velocityJitter: 16,
        bulgeFraction: 0.18,
      },
    ],
    // Purely radial approach: vx drives them together, no tangential component.
    // 293 km/s against an escape speed of 385 km/s, so the pair is bound: it
    // falls through, separates, and falls back again while dynamical friction
    // drains the orbital energy, oscillating through each other before the
    // merger settles. Speeds above escape velocity produce a clean fly-apart
    // instead, which is the `flyby` scenario.
    relativeVelocity: { x: -0.3, y: 0, z: 0 },
    separation: 62,
    showDiagnostics: true,
  },
  {
    id: 'grazing',
    name: 'Grazing Collision',
    blurb: 'A slower pass with a large impact parameter — long, sweeping tails.',
    description:
      'The impact parameter is roughly one disk diameter, so the galaxies shear past one another ' +
      'instead of colliding face-first. Gravitational friction converts orbital energy into random ' +
      'stellar motion while conserving angular momentum, and the outer stars receive the largest ' +
      'kick. This is the configuration that produces the most spectacular, long-lived tidal tails ' +
      'and bridges, because the material is pulled into a prograde orbit around the pair. Watch the ' +
      'arms unwind as the disks are sheared.',
    seed: 'grazing-prograde',
    galaxies: [
      {
        id: 'a',
        label: 'Galaxy A',
        count: 22000,
        radius: 15,
        thickness: 0.6,
        massFactor: 1.0,
        color: BLUE,
        coreColor: BLUE_CORE,
        velocityJitter: 14,
        bulgeFraction: 0.16,
      },
      {
        id: 'b',
        label: 'Galaxy B',
        count: 22000,
        radius: 13,
        thickness: 0.55,
        massFactor: 0.85,
        color: IVORY,
        coreColor: GOLD_CORE,
        velocityJitter: 14,
        bulgeFraction: 0.16,
      },
    ],
    // Tangential velocity >> radial: a prograde, high-impact-parameter pass.
    // 330 km/s against an escape speed of 353 km/s -- bound, but only just, so
    // the pair shears past one another, sheds long tails, and then falls back
    // together as dynamical friction bleeds the orbital energy.
    relativeVelocity: { x: -0.06, y: 0.31, z: 0.1 },
    separation: 70,
    showDiagnostics: true,
  },
  {
    id: 'flyby',
    name: 'High-speed Flyby',
    blurb: 'A fast, unbound hyperbolic encounter — spectacular distortion, no merger.',
    description:
      'Impact parameter of several disk radii with a relative speed well above the local escape ' +
      'velocity, so the pair is not gravitationally bound. Each galaxy is tidally stretched and its ' +
      'arms are unwound as it passes, but the systems part company and the disks relax back toward ' +
      'their original form. This is the configuration used to explain the "stellar shells" seen in ' +
      'elliptical galaxies: a fast intruder sweeps a density wave through the victim and leaves a ' +
      'standing ripple in the stellar density. This scenario is where you most clearly see that ' +
      'nothing here is scripted — the shell only appears if the physics is right.',
    seed: 'flyby-hyperbolic',
    galaxies: [
      {
        id: 'a',
        label: 'Galaxy A (target)',
        count: 26000,
        radius: 16,
        thickness: 0.5,
        massFactor: 1.0,
        color: IVORY,
        coreColor: GOLD_CORE,
        velocityJitter: 10,
        bulgeFraction: 0.2,
      },
      {
        id: 'b',
        label: 'Galaxy B (intruder)',
        count: 16000,
        radius: 9,
        thickness: 0.45,
        massFactor: 0.45,
        color: BLUE,
        coreColor: BLUE_CORE,
        velocityJitter: 12,
        bulgeFraction: 0.14,
      },
    ],
    // Radial speed dominates: strongly hyperbolic, will not merge. Well above
    // the 287 km/s escape speed, so the pair parts company for good.
    relativeVelocity: { x: -0.55, y: 0.06, z: 0.04 },
    separation: 88,
    showDiagnostics: true,
  },
  {
    id: 'unequal',
    name: 'Unequal-mass Merger',
    blurb: 'A massive host absorbs a much smaller dwarf satellite.',
    description:
      'A 4:1 mass ratio, analogous to the Milky Way absorbing a large dwarf. Dynamical friction ' +
      'against the extended dark-matter halo of the host is the dominant effect: the intruder loses ' +
      'orbital energy much faster than the host, spirals inward on an ever-tightening orbit, and is ' +
      'dynamically heated into a diffuse spheroidal envelope. The host is barely disturbed. The ' +
      'satellite is tidally shredded long before it reaches the centre, which is why the merged ' +
      'system shows a smooth, extended halo of debris rather than a second bright core.',
    seed: 'unequal-4to1',
    galaxies: [
      {
        id: 'a',
        label: 'Host (massive)',
        count: 34000,
        radius: 20,
        thickness: 0.6,
        massFactor: 1.0,
        color: IVORY,
        coreColor: GOLD_CORE,
        velocityJitter: 10,
        bulgeFraction: 0.22,
      },
      {
        id: 'b',
        label: 'Dwarf satellite',
        count: 11000,
        radius: 7,
        thickness: 0.5,
        massFactor: 0.25,
        color: BLUE,
        coreColor: BLUE_CORE,
        velocityJitter: 10,
        bulgeFraction: 0.12,
      },
    ],
    // Bound infall at 275 km/s, just under the 301 km/s escape speed for the
    // combined mass. The satellite additionally feels the host's much deeper
    // potential, so dynamical friction pulls it in and it merges.
    relativeVelocity: { x: -0.1, y: 0.25, z: 0.07 },
    separation: 84,
    showDiagnostics: true,
  },
];

export const DEFAULT_SCENARIO_ID = 'headon';

export function getScenario(id: string): Scenario {
  return SCENARIOS.find((s) => s.id === id) ?? SCENARIOS[0];
}
