import { COLOR_UTILS } from './common.ts';

/**
 * Post-processing: bloom and the final tonemap.
 *
 * BLOOM
 * -----
 * A separable Gaussian blur pyramid at reduced resolution, additively
 * composited back over the scene. Bloom is what turns a field of hard white
 * points into something that reads as luminous gas: the eye interprets a bright
 * source with a soft halo as *emitting*, and without it stars look like
 * confetti.
 *
 * The bright-pass uses a soft knee rather than a hard threshold, which avoids
 * the banding a hard threshold produces as stars drift across the cutoff and
 * pop in and out of the bloom.
 */

export const BRIGHT_PASS_FS = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uSrc;
uniform float uThreshold;
uniform float uKnee;
out vec4 fragColor;
void main() {
  vec3 c = texture(uSrc, vUv).rgb;
  float L = max(c.r, max(c.g, c.b));
  float w = clamp((L - uThreshold + uKnee) / (2.0 * uKnee), 0.0, 1.0);
  fragColor = vec4(c * w * w, 1.0);
}
`;

/**
 * Separable 9-tap Gaussian, using the standard linear-sampling trick: 5 texture
 * fetches approximate a 9-tap kernel by sampling between texel centres.
 */
export const BLUR_FS = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uSrc;
uniform vec2 uDirection;
out vec4 fragColor;
void main() {
  vec2 o1 = uDirection * 1.3846153846;
  vec2 o2 = uDirection * 3.2307692308;
  vec3 c = texture(uSrc, vUv).rgb * 0.2270270270;
  c += texture(uSrc, vUv + o1).rgb * 0.3162162162;
  c += texture(uSrc, vUv - o1).rgb * 0.3162162162;
  c += texture(uSrc, vUv + o2).rgb * 0.0702702703;
  c += texture(uSrc, vUv - o2).rgb * 0.0702702703;
  fragColor = vec4(c, 1.0);
}
`;

/**
 * Plain texture blit, used to downsample and upsample the bloom pyramid.
 *
 * The pyramid levels are combined with *additive blending* rather than by
 * sampling both operands in one shader. That is not a stylistic choice: a
 * shader that reads a level while rendering into it forms a framebuffer
 * feedback loop, which WebGL rejects with GL_INVALID_OPERATION. Combining via
 * blend state means the destination texture is never bound to a sampler, so the
 * accumulation is legal.
 */
export const BLIT_FS = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uSrc;
uniform float uWeight;
out vec4 fragColor;
void main() { fragColor = vec4(texture(uSrc, vUv).rgb * uWeight, 1.0); }
`;

/**
 * The final composite and tonemap.
 *
 * The tonemap is filmic rather than a hard clamp. With additive blending,
 * galactic cores accumulate far past 1.0 in linear space; clamping there
 * flattens the core into a hard white disc with a visible edge. Reinhard-Jodie
 * blends toward the luminance-scaled value as the colour desaturates, so an
 * overexposed core stays warm rather than shifting to flat white, and a blue
 * star stays blue when it saturates.
 */
export const COMPOSITE_FS = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;

uniform sampler2D uScene;
uniform sampler2D uBloom;
uniform sampler2D uTrail;
uniform float uBloomIntensity;
uniform float uTrailIntensity;
uniform float uExposure;
uniform float uVignette;

${COLOR_UTILS}

out vec4 fragColor;

vec3 tonemapReinhardJodie(vec3 c) {
  float l = dot(c, vec3(0.2126, 0.7152, 0.0722));
  vec3 tv = c / (1.0 + c);
  return mix(c / (1.0 + l), tv, tv);
}

void main() {
  vec3 color = texture(uScene, vUv).rgb
             + texture(uBloom, vUv).rgb * uBloomIntensity
             + texture(uTrail, vUv).rgb * uTrailIntensity;

  color *= uExposure;

  vec2 d = vUv - 0.5;
  color *= 1.0 - uVignette * dot(d, d) * 2.0;

  color = tonemapReinhardJodie(color);
  fragColor = vec4(linearToSrgb(color), 1.0);
}
`;
