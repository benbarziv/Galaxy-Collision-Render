import { HASH } from './common.ts';

/**
 * Star rendering.
 *
 * Every star is a camera-facing quad (two triangles, expanded in the vertex
 * shader from gl_VertexID) with a procedural soft radial falloff. There is no
 * sprite atlas and no per-frame texture upload, which keeps the whole particle
 * system to two buffers and one draw call.
 *
 * The falloff is gaussian-like rather than a hard disc:
 *
 *     I(u) = exp(-k * u^2)     for u = radial distance in sprite space
 *
 * A gaussian's integral over the sprite is independent of sprite size, so a
 * star drawn larger for being brighter emits more total flux with no explicit
 * normalisation. That is what makes a dense core saturate into a smooth glow
 * instead of a pile of discrete overlapping dots.
 */

export const STAR_VS = /* glsl */ `#version 300 es
precision highp float;
precision highp int;

${HASH}

in vec3  aPosition;
in float aGalaxy;
in float aSeed;
in float aR0;
in float aRNorm;
in float aSpin;
in float aV0;

uniform mat4  uViewProjection;
uniform vec3  uCameraPos;
uniform float uPointScale;
uniform float uViewportHeight;
uniform float uSizeBoost;

out vec2  vSprite;
out float vBrightness;
out vec3  vColor;
out float vCoreBoost;
out float vDistance;

uniform vec3 uGalaxyColor0;
uniform vec3 uGalaxyColor1;
uniform vec3 uGalaxyCore0;
uniform vec3 uGalaxyCore1;

void main() {
  // Quad corners from gl_VertexID: (0,0) (1,0) (0,1) (1,0) (1,1) (0,1)
  int id = gl_VertexID;
  vec2 corner = vec2(float((id == 1 || id == 3 || id == 4) ? 1 : 0),
                     float((id == 2 || id == 4 || id == 5) ? 1 : 0));
  vSprite = corner * 2.0 - 1.0;

  vec3 world = aPosition;
  gl_Position = uViewProjection * vec4(world, 1.0);

  // Two decorrelated draws from the same seed: one sets base size, the other
  // brightness. Keeping them independent stops every star from being uniformly
  // big-and-bright, which looks mechanical.
  float hSize = hash11(aSeed * 127.1 + 3.7);
  float hBright = hash11(aSeed * 331.7 + 91.3);
  float hTemp = hash11(aSeed * 57.3 + 11.1);

  // r0 is the star's initial radius and never changes, so the hot-core to
  // cool-rim colour gradient is a property of the star itself and survives being
  // flung into a tidal tail. That is physically right: a red giant does not turn
  // blue because it is in a tail.
  float radial = clamp(aR0 / max(aRNorm, 0.001), 0.0, 1.4);

  bool isG0 = aGalaxy < 0.5;
  vec3 base = isG0 ? uGalaxyColor0 : uGalaxyColor1;
  vec3 core = isG0 ? uGalaxyCore0 : uGalaxyCore1;

  // Blend toward the core colour in the inner ~35% of the disk. The bulge
  // population has small r0, so it is strongly core-coloured, which is what
  // gives the central glow its warmer, brighter appearance.
  float coreMix = 1.0 - smoothstep(0.0, 0.35, radial);
  vColor = mix(base, core, coreMix);

  // A fraction of stars are bluer or redder than the population mean, standing
  // in for the real spread in stellar mass and temperature.
  vColor *= mix(vec3(0.82, 0.90, 1.12), vec3(1.12, 0.92, 0.78), hTemp);

  float dist = max(length(world - uCameraPos), 0.001);
  vDistance = dist;

  // Perspective-correct sprite size, in pixels.
  float pixelSize = uPointScale * uSizeBoost * (0.55 + 0.9 * hSize) / dist;
  // Taper very distant stars so the sky does not fill with sub-pixel noise from
  // escaped debris.
  float far = 1.0 - smoothstep(400.0, 1400.0, dist);
  // A floor of ~1px keeps distant stars visible as points rather than
  // flickering in and out as the camera moves.
  pixelSize = max(pixelSize, 0.7 * far);

  // A steep power law over the brightness draw. A few stars are much brighter
  // than the rest, which gives the image its sparkle; a flat distribution reads
  // as uniform noise.
  float bright = pow(hBright, 3.2);
  vCoreBoost = 0.55 + 1.5 * coreMix;
  vBrightness = (0.22 + 1.5 * bright) * vCoreBoost;

  // Offset the quad in clip space by the sprite radius, in pixels.
  vec4 clip = gl_Position;
  clip.xy += vSprite * pixelSize / uViewportHeight * clip.w * 2.0;
  gl_Position = clip;
}
`;

export const STAR_FS = /* glsl */ `#version 300 es
precision highp float;

in vec2  vSprite;
in float vBrightness;
in vec3  vColor;
in float vCoreBoost;
in float vDistance;

uniform float uExposure;
uniform float uGlowPower;

out vec4 fragColor;

void main() {
  float u2 = dot(vSprite, vSprite);
  if (u2 > 1.0) discard;

  // Gaussian falloff, renormalised so the sprite peak is always 1 regardless of
  // the glow power setting.
  float denom = 1.0 - exp(-uGlowPower * 3.0);
  float falloff = (exp(-uGlowPower * u2 * 3.0) - exp(-uGlowPower * 3.0)) / denom;

  fragColor = vec4(vColor * vBrightness * falloff * uExposure, 1.0);
}
`;

/**
 * The galactic cores.
 *
 * A dense point cloud alone gives a core that is bright but slightly lumpy.
 * Drawing an explicit core on top fixes that: large additive sprites at each
 * galaxy's centre with a two-lobe radial profile. Each centre is tracked on the
 * CPU from the mean particle position and fed in as a uniform.
 */
export const CORE_VS = /* glsl */ `#version 300 es
precision highp float;

in vec2 aCorner;   // unit quad, static buffer shared by both cores

uniform mat4  uViewProjection;
uniform vec3  uCameraPos;
uniform vec3  uCenter;
uniform float uRadius;
uniform float uViewportHeight;

out vec2 vSprite;

void main() {
  vec3 world = uCenter;
  gl_Position = uViewProjection * vec4(world, 1.0);

  float dist = max(length(world - uCameraPos), 0.001);
  float pixelSize = (uRadius / dist) * uViewportHeight * 0.5;
  vSprite = aCorner * 2.0 - 1.0;

  vec4 clip = gl_Position;
  clip.xy += vSprite * pixelSize / uViewportHeight * clip.w * 2.0;
  gl_Position = clip;
}
`;

export const CORE_FS = /* glsl */ `#version 300 es
precision highp float;
in vec2 vSprite;
uniform vec3  uColor;
uniform float uIntensity;
out vec4 fragColor;
void main() {
  float r = length(vSprite);
  if (r > 1.0) discard;
  // Tight exponential core inside a broad inverse-square envelope. The envelope
  // gives a real bulge its smooth falloff; the tight lobe is the unresolved
  // central concentration.
  float core = exp(-r * 9.0);
  float envelope = 1.0 / (1.0 + 7.0 * r * r);
  fragColor = vec4(uColor * (core * 0.75 + envelope * 0.45) * uIntensity, 1.0);
}
`;
