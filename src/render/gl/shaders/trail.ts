/**
 * Motion trails.
 *
 * Trail rendering is a feedback accumulation, not per-star history buffers. Each
 * frame we render only the current particle positions as small additive points
 * into a persistent history texture, then composite the whole history with a
 * per-frame exponential decay. A star that has not moved onto a given texel
 * this frame leaves that texel's old contribution in place, so the texture
 * accumulates a genuine motion smear.
 *
 * The cost is one small draw call per frame and no extra memory per star, which
 * matters at 260k stars. It also produces smooth sub-particle-length trails
 * rather than the faceted polyline a line strip between stored positions gives.
 */

export const TRAIL_DECAY_FS = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;
uniform sampler2D uHistory;
uniform float uDecay;
out vec4 fragColor;
void main() {
  fragColor = vec4(texture(uHistory, vUv).rgb * uDecay, 1.0);
}
`;

/** Trail point splat: one small additive quad per star into the history target. */
export const TRAIL_POINT_VS = /* glsl */ `#version 300 es
precision highp float;
precision highp int;

in vec3  aPosition;
in float aGalaxy;
in float aSeed;
in float aR0;
in float aRNorm;

uniform mat4  uViewProjection;
uniform float uPointSize;

out vec3 vColor;
out float vAlpha;

uniform vec3 uGalaxyColor0;
uniform vec3 uGalaxyColor1;

void main() {
  gl_Position = uViewProjection * vec4(aPosition, 1.0);

  float radial = clamp(aR0 / max(aRNorm, 0.001), 0.0, 1.4);
  vec3 base = aGalaxy < 0.5 ? uGalaxyColor0 : uGalaxyColor1;
  vColor = mix(base, vec3(0.55, 0.68, 1.0), 0.35);
  vColor *= 1.0 - 0.55 * smoothstep(0.0, 0.4, radial);
  vAlpha = uPointSize;
}
`;

export const TRAIL_POINT_FS = /* glsl */ `#version 300 es
precision highp float;
in vec3 vColor;
in float vAlpha;
out vec4 fragColor;
void main() { fragColor = vec4(vColor * vAlpha, 1.0); }
`;
