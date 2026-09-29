/**
 * Minimal math primitives shared by CPU and GPU.
 *
 * These are deliberately written in the same syntax as the GLSL snippets in
 * shaders/common.ts so that the identical formulas can be pasted on both sides
 * without translation. Keeping one source of truth for the physics is the
 * single most important correctness property here: the CPU initial conditions
 * and the GPU integrator must agree on units or the simulation silently explodes.
 */

export type Vec2 = { x: number; y: number };
export type Vec3 = { x: number; y: number; z: number };

export const vec2 = (x = 0, y = 0): Vec2 => ({ x, y });

export const vec3 = (x = 0, y = 0, z = 0): Vec3 => ({ x, y, z });

export const clamp = (x: number, lo: number, hi: number): number =>
  x < lo ? lo : x > hi ? hi : x;

export const lerp = (a: number, b: number, t: number): number => a + (b - a) * t;

export const smoothstep = (e0: number, e1: number, x: number): number => {
  const t = clamp((x - e0) / (e1 - e0), 0, 1);
  return t * t * (3 - 2 * t);
};

/**
 * Random number generator with an explicit, reproducible seed.
 *
 * Every scenario must be bit-for-bit repeatable when the user hits Reset, so
 * Math.random() is unusable here. xmur3 hashes the seed string into state,
 * mulberry32 expands it into a stream.
 */
export function makeRng(seed: string) {
  let h = 1779033703 ^ seed.length;
  for (let i = 0; i < seed.length; i++) {
    h = Math.imul(h ^ seed.charCodeAt(i), 3432918353);
    h = (h << 13) | (h >>> 19);
  }
  let a = (h ^= h >>> 16) >>> 0;

  const next = (): number => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };

  return {
    next,
    /** Uniform in [lo, hi). */
    range: (lo: number, hi: number) => lo + (hi - lo) * next(),
    /** Standard normal via Box-Muller, cached second sample. */
    normal(): number {
      const spare = (rng as unknown as { _spare?: number })._spare;
      if (spare !== undefined) {
        (rng as unknown as { _spare?: number })._spare = undefined;
        return spare;
      }
      let u = 0;
      let v = 0;
      let s = 0;
      do {
        u = next() * 2 - 1;
        v = next() * 2 - 1;
        s = u * u + v * v;
      } while (s === 0 || s >= 1);
      const f = Math.sqrt((-2 * Math.log(s)) / s);
      (rng as unknown as { _spare?: number })._spare = v * f;
      return u * f;
    },
  };
}

type Rng = ReturnType<typeof makeRng>;
const rng: Rng = makeRng('init');
