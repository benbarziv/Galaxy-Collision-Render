import { HASH } from './common.ts';

/**
 * Deep-space background: three procedural layers, no textures.
 *
 * The ray direction is reconstructed per-pixel from the inverse
 * view-projection matrix rather than from screen UVs, which is what makes the
 * starfield and cirrus behave like distant objects rather than a wallpaper.
 */
export const BACKGROUND_FS = /* glsl */ `#version 300 es
precision highp float;
in vec2 vUv;

uniform mat4  uInvViewProjection;
uniform vec3  uCameraPos;
uniform float uTime;
uniform float uStarfieldIntensity;

${HASH}

out vec4 fragColor;

float noise3(vec3 p) {
  vec3 i = floor(p);
  vec3 f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  float n = i.x + i.y * 57.0 + i.z * 113.0;
  return mix(
    mix(mix(hash11(n), hash11(n + 1.0), f.x),
        mix(hash11(n + 57.0), hash11(n + 58.0), f.x), f.y),
    mix(mix(hash11(n + 113.0), hash11(n + 114.0), f.x),
        mix(hash11(n + 170.0), hash11(n + 171.0), f.x), f.y), f.z);
}

float fbm(vec3 p) {
  float v = 0.0;
  float a = 0.5;
  for (int i = 0; i < 4; i++) {
    v += a * noise3(p);
    p *= 2.03;
    a *= 0.5;
  }
  return v;
}

void main() {
  vec4 world = uInvViewProjection * vec4(vUv * 2.0 - 1.0, 1.0, 1.0);
  vec3 dir = normalize(world.xyz / world.w - uCameraPos);

  // Not pure black: pure black reads as a broken panel on an OLED display,
  // whereas a very slight blue lift reads as space.
  vec3 color = vec3(0.004, 0.006, 0.013);

  float c1 = fbm(dir * 2.1 + vec3(0.0, 0.0, uTime * 0.004));
  float c2 = fbm(dir * 5.3 - vec3(uTime * 0.002, 0.0, 0.0));
  float cirrus = smoothstep(0.42, 0.95, c1 * 0.65 + c2 * 0.35);
  color += vec3(0.030, 0.036, 0.062) * cirrus;

  // Dust band along the world y = 0 plane. Fixed in world space, so it
  // parallaxes as the camera orbits.
  float band = exp(-abs(dir.y) * 7.0);
  float dust = fbm(dir * 8.0) * band;
  color += vec3(0.045, 0.042, 0.055) * smoothstep(0.35, 0.9, dust);

  if (uStarfieldIntensity > 0.0) {
    // Cube-map style parameterisation of the direction sphere, so cell areas
    // stay roughly uniform and stars do not clump at the axis directions.
    vec3 ad = abs(dir);
    vec2 uv;
    if (ad.x >= ad.y && ad.x >= ad.z) uv = dir.yz / ad.x;
    else if (ad.y >= ad.z) uv = dir.xz / ad.y;
    else uv = dir.xy / ad.z;

    vec2 cell = floor(uv * 90.0);
    vec2 f = fract(uv * 90.0) - 0.5;

    float h = hash11(cell.x + cell.y * 313.0);
    // Most cells are empty; only the top few percent of the hash hold a star.
    if (h > 0.94) {
      vec2 jitter = vec2(hash11(h * 91.0), hash11(h * 57.0 + 3.0)) - 0.5;
      float d2 = dot(f - jitter * 0.7, f - jitter * 0.7);
      float bright = pow(hash11(h * 17.0), 5.0);
      vec3 tint = mix(vec3(0.75, 0.85, 1.0), vec3(1.0, 0.88, 0.72), hash11(h * 7.0));
      color += tint * exp(-d2 * 240.0) * bright * uStarfieldIntensity;
    }
  }

  fragColor = vec4(color, 1.0);
}
`;
