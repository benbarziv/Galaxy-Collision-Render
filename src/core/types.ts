/**
 * Shared type definitions.
 *
 * NOTE ON THE RENDERING LAYER
 * ---------------------------
 * Three.js stores colour components in *linear* space and applies the sRGB
 * transfer function on output. The UI, by contrast, deals in sRGB (0-255)
 * because that is what `style.background` and friends expect. Rather than
 * sprinkling conversions through the UI code, every colour crossing the module
 * boundary is normalised to 0-1 **linear** here, and the helper that hands a
 * colour to a DOM element is responsible for converting back.
 */

import type { Vec3 } from './math.ts';

export type Vec3Like = Vec3;

/** A colour in linear space, components nominally 0-1 but allowed to exceed 1 for HDR cores. */
export interface LinearColor {
  r: number;
  g: number;
  b: number;
}

export interface GalaxySpec {
  id: string;
  label: string;
  /** Number of stars the particle generator should emit for this galaxy. */
  count: number;
  /** Disk radius in simulation units (kpc). */
  radius: number;
  /** Vertical scale height of the disk in kpc. Controls how "puffy" the disk is. */
  thickness: number;
  /** Mass multiplier relative to the star count; drives the halo mass. */
  massFactor: number;
  /** Primary colour temperature of the stellar population. */
  color: LinearColor;
  /** Inner regions are shifted toward this (hotter) colour. */
  coreColor: LinearColor;
  /** Small-scale random velocity dispersion, km/s. Drives arm thickness. */
  velocityJitter: number;
  /** Fraction of the population placed in the dense central bulge. */
  bulgeFraction: number;
}

export interface Scenario {
  id: string;
  name: string;
  /** One-line summary shown in the control panel. */
  blurb: string;
  /** A longer paragraph describing the physics of this configuration. */
  description: string;
  seed: string;
  galaxies: [GalaxySpec, GalaxySpec];
  /**
   * Initial relative velocity of galaxy 2 with respect to galaxy 1, in km/s.
   * Note the sign convention: the two galaxies start on opposite sides of the
   * origin, so a *negative* vx here means "moving inward".
   */
  relativeVelocity: Vec3;
  /** Initial separation of the two galactic centres, in kpc. */
  separation: number;
  /** Renders the overlay histogram / phase diagnostics. */
  showDiagnostics: boolean;
}

export type QualityTier = 'low' | 'medium' | 'high' | 'ultra';

export interface QualityProfile {
  tier: QualityTier;
  label: string;
  /** Total star count across both galaxies. */
  particles: number;
  /** PM mesh cells per spatial axis. 64^3 = 262144 cells. */
  gridSize: number;
  /** Number of fixed-timestep integration substeps per rendered frame. */
  substeps: number;
  /** Trajectory history length, in samples. 0 disables the trail pass entirely. */
  trailLength: number;
  /** Soft-particle glow sprite size multiplier. */
  bloomIntensity: number;
  /** Additive bloom pass toggle. */
  bloom: boolean;
  /** Render at a fraction of devicePixelRatio to save fill rate. */
  resolutionScale: number;
  /** Rotate the dome background. */
  starfield: boolean;
}

export interface CameraState {
  /** Distance from the look-at target. */
  distance: number;
  /** Azimuth in radians. */
  theta: number;
  /** Elevation in radians, clamped to avoid gimbal flip. */
  phi: number;
  /** Point the camera orbits and looks at. */
  target: { x: number; y: number; z: number };
  fov: number;
}

export interface SimTuning {
  /** Gravitational constant in simulation units (see sim/constants.ts for derivation). */
  G: number;
  /** Softening length in kpc. Suppresses two-body clashes in the particle potential. */
  softening: number;
  /** Physical simulation speed multiplier. 1.0 = "real time" for the chosen unit system. */
  speed: number;
  /** Fixed integration timestep in simulation time units. */
  dt: number;
  /** Fraction of the PM mesh that is covered by the compute box. */
  pmBoxScale: number;
  /**
   * Halo mass as a multiple of the total star mass. Real galaxies are dominated
   * by dark matter; without this the disks would be unbound.
   */
  haloFactor: number;
  /** Radius of the (static, analytic) dark-matter halo, kpc. */
  haloRadius: number;
  /** Reference orbital circular speed of the galaxies, km/s. */
  v0: number;
}
