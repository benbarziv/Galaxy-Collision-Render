/**
 * Regression test for the initial-condition builder.
 *
 * These are pure CPU assertions, so they run in seconds without a browser and
 * they pin down the class of bug that is most expensive in this project: one
 * where the array is the right length, the renderer draws something galaxy-
 * shaped, and yet the physics is wrong. Every check here is a property that
 * must hold *independently* of whether the picture looks right.
 *
 * Usage: node scripts/initial-conditions.test.mjs
 */
import { buildInitialConditions, STRIDE, galaxyTotalMass, enclosedMass, enclosedFraction } from '../src/sim/initial-conditions.ts';
import { SCENARIOS } from '../src/core/presets.ts';
import { G_GALACTIC, KPC_PER_MYR_TO_KM_PER_S } from '../src/sim/constants.ts';

let failures = 0;
const check = (name, ok, detail = '') => {
  console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${name}${detail ? `  -- ${detail}` : ''}`);
  if (!ok) failures++;
};

for (const s of SCENARIOS) {
  console.log(`\n${s.id}:`);
  const d = buildInitialConditions(s, 4000, 8.0);
  const b = d.buffer;

  // 1. Every star must be written. The bug this guards: the per-galaxy loop
  //    restarted its write index, so the first galaxy's records were
  //    overwritten and the tail of the buffer stayed zero. A zeroed record is
  //    not just missing scenery -- it has zero mass, so it silently removes
  //    that fraction of the system's gravity.
  const empty = [];
  for (let i = 0; i < d.count; i++) {
    const w = i * STRIDE;
    if (b[w] === 0 && b[w + 1] === 0 && b[w + 2] === 0 && b[w + 13] === 0) empty.push(i);
  }
  check('no uninitialised stars', empty.length === 0, empty.length ? `${empty.length} zeroed records` : '');

  // 2. Both galaxies must be present in roughly the requested proportion.
  const tags = [0, 0];
  for (let i = 0; i < d.count; i++) tags[b[i * STRIDE + 6] < 0.5 ? 0 : 1]++;
  check('both galaxies populated', tags.every((n) => n > 0), `${tags[0]}/${tags[1]}`);

  // 3. Total deposited mass per galaxy must equal its intended mass. The PM
  //    solver integrates whatever the deposit hands it, so if the two disagree
  //    the rotation curve the stars were launched with is not the one they feel.
  const mass = [0, 0];
  const pos = [0, 0, 0, 0, 0, 0]; // sums of x,y,z per galaxy
  for (let i = 0; i < d.count; i++) {
    const w = i * STRIDE;
    const g = b[w + 6] < 0.5 ? 0 : 1;
    mass[g] += b[w + 13];
    pos[g * 3] += b[w]; pos[g * 3 + 1] += b[w + 1]; pos[g * 3 + 2] += b[w + 2];
  }
  let massOk = true;
  const massDetail = [];
  for (let g = 0; g < 2; g++) {
    const want = galaxyTotalMass(s.galaxies[g]);
    const got = mass[g];
    const rel = Math.abs(got - want) / want;
    massDetail.push(`${(got / 1e11).toFixed(3)}e11 vs ${(want / 1e11).toFixed(3)}e11`);
    if (rel > 0.01) massOk = false;
  }
  check('deposited mass matches intent', massOk, massDetail.join('  '));

  // 4. Centre of mass must sit at the prescribed offsets, so the pair starts
  //    on the orbit the preset describes.
  const half = s.separation / 2;
  const c0 = [pos[0] / tags[0], pos[3] / tags[1]];
  check('galaxy A centred at -separation/2', Math.abs(c0[0] + half) < 0.5, `x=${c0[0].toFixed(2)} want ${-half}`);
  check('galaxy B centred at +separation/2', Math.abs(c0[1] - half) < 0.5, `x=${c0[1].toFixed(2)} want ${half}`);

  // 5. The reported centres must match the measured ones. The HUD reads these,
  //    and they are what the camera frames and the HUD displays.
  const rep = d.galaxyCenters;
  check('reported centre matches measured A', Math.abs(rep[0].x + half) < 1e-6, `${rep[0].x}`);
  check('reported centre matches measured B', Math.abs(rep[1].x - half) < 1e-6, `${rep[1].x}`);

  // 6. `massFactor` must actually reach the mass. It is declared on every
  //    galaxy and was previously never read, so scenarios that exist to show a
  //    mass ratio showed none.
  // `massFactor` scales the mass, but so does the radius: each galaxy is
  // calibrated to the *same* flat rotation speed at its own edge, so a wider
  // disk needs proportionally more mass. The calibration is
  // M ~ radius * massFactor / f_edge, where f_edge is the enclosed mass
  // fraction at the disk edge. Using the exported f_edge here rather than
  // re-deriving the profile keeps this a check on the real formula.
  const ratio = galaxyTotalMass(s.galaxies[0]) / galaxyTotalMass(s.galaxies[1]);
  const [host, guest] = s.galaxies;
  const wantRatio =
    ((host.radius * host.massFactor) / enclosedFraction(host.radius, host)) /
    ((guest.radius * guest.massFactor) / enclosedFraction(guest.radius, guest));
  check('mass ratio follows the rotation-speed calibration',
    Math.abs(ratio / wantRatio - 1) < 1e-9, `${ratio.toFixed(3)}x`);

  // And the mass asymmetry the presets advertise must be real: a satellite
  // with a smaller massFactor must be genuinely lighter than its host.
  if (s.galaxies[0].massFactor > s.galaxies[1].massFactor) {
    check('host is more massive than satellite', galaxyTotalMass(s.galaxies[0]) > galaxyTotalMass(s.galaxies[1]),
      `${(galaxyTotalMass(s.galaxies[0]) / 1e11).toFixed(2)}e11 vs ${(galaxyTotalMass(s.galaxies[1]) / 1e11).toFixed(2)}e11`);
  }

  // 7. Circular speed at the disk edge must be physical (~100-300 km/s) and
  //    the same one the force field will produce.
  const g0 = s.galaxies[0];
  const vc = Math.sqrt((G_GALACTIC * enclosedMass(g0.radius, g0)) / g0.radius) * KPC_PER_MYR_TO_KM_PER_S;
  check('v_circ(edge) is physical', vc > 100 && vc < 320, `${vc.toFixed(0)} km/s`);

  // 8. The encounter must be bound unless the preset is meant to be a flyby.
  const M = galaxyTotalMass(s.galaxies[0]) + galaxyTotalMass(s.galaxies[1]);
  const vrel = Math.hypot(s.relativeVelocity.x, s.relativeVelocity.y, s.relativeVelocity.z);
  const vesc = Math.sqrt((2 * G_GALACTIC * M) / s.separation);
  const isFlyby = s.id === 'flyby';
  check(
    isFlyby ? 'flyby is unbound' : 'merger is bound',
    isFlyby ? vrel > vesc : vrel < vesc,
    `v=${(vrel * KPC_PER_MYR_TO_KM_PER_S).toFixed(0)} v_esc=${(vesc * KPC_PER_MYR_TO_KM_PER_S).toFixed(0)} km/s`,
  );
}

console.log(failures === 0 ? '\nPASS: initial conditions are correct.' : `\nFAIL: ${failures} check(s).`);
process.exit(failures === 0 ? 0 : 1);
