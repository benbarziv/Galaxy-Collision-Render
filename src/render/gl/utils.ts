/**
 * Thin WebGL2 helpers. Nothing here knows about galaxies.
 *
 * Everything raises a descriptive error instead of silently returning null:
 * shader-compilation failures are the most common way a WebGL project turns into
 * a black screen, and surfacing the driver's info log with line numbers makes
 * them fixable in seconds.
 */

export type GL = WebGL2RenderingContext;

export interface GlCaps {
  /** Largest 2D texture edge the driver will accept. */
  maxTextureSize: number;
  /** True when 32-bit float render targets are available. */
  colorBufferFloat: boolean;
  /** True when additive blending into 32-bit float targets is available. */
  floatBlend: boolean;
}

let cachedCaps: GlCaps | null = null;

export function createContext(canvas: HTMLCanvasElement): GL {
  const gl = canvas.getContext('webgl2', {
    alpha: false,
    antialias: false,
    depth: false,
    stencil: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: false,
    powerPreference: 'high-performance',
  });
  if (!gl) {
    throw new Error(
      'WebGL2 is not available. This simulator needs WebGL2 for transform feedback and float textures.',
    );
  }

  // REQUIRED EXTENSIONS
  // -------------------
  // EXT_color_buffer_float makes R32F / RG32F / RGBA32F textures usable as
  // framebuffer colour attachments. Without it every one of those targets is
  // FRAMEBUFFER_INCOMPLETE_ATTACHMENT, which is exactly how this shader
  // pipeline fails: the whole solver silently produces nothing.
  //
  // EXT_float_blend permits additive blending into 32-bit float targets. The PM
  // mass deposit sums tens of thousands of stars into one cell, which needs
  // blending on an RG32F target; without this extension the blend is dropped
  // and every cell receives a single particle's mass instead of the sum.
  const colorBufferFloat = gl.getExtension('EXT_color_buffer_float');
  const floatBlend = gl.getExtension('EXT_float_blend');
  // OES_texture_float_linear makes 32-bit float textures filterable. Only the
  // RGBA32F force field needs this (the integrator interpolates it); if it is
  // missing the simulation still runs, just with nearest-neighbour sampling.
  const floatLinear = gl.getExtension('OES_texture_float_linear');
  void floatLinear;

  if (!colorBufferFloat) {
    throw new Error(
      'EXT_color_buffer_float is not supported by this GPU or driver. ' +
        'The gravity solver requires floating-point render targets.',
    );
  }

  cachedCaps = {
    maxTextureSize: gl.getParameter(gl.MAX_TEXTURE_SIZE) as number,
    colorBufferFloat: true,
    floatBlend: Boolean(floatBlend),
  };
  return gl;
}

/** Driver limits, queried once at context creation. */
export function glCaps(): GlCaps {
  if (!cachedCaps) {
    throw new Error('glCaps() called before createContext()');
  }
  return cachedCaps;
}

function compile(gl: GL, type: number, source: string, label: string): WebGLShader {
  const shader = gl.createShader(type);
  if (!shader) throw new Error(`Failed to create shader object for ${label}`);
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(shader) ?? '(no log)';
    const numbered = source
      .split('\n')
      .map((l, i) => `${String(i + 1).padStart(4)} | ${l}`)
      .join('\n');
    gl.deleteShader(shader);
    throw new Error(`Shader compile failed [${label}]:\n${log}\n${numbered}`);
  }
  return shader;
}

export interface CompiledProgram {
  program: WebGLProgram;
  uniforms: Record<string, WebGLUniformLocation | null>;
  attribs: Record<string, number>;
}

/**
 * Link a program and pre-resolve every active uniform and attribute location.
 *
 * Pre-resolving matters in the hot path: `getUniformLocation` is a string
 * lookup into the driver's symbol table, and the transform feedback program
 * would otherwise query it on every substep.
 */
export function createProgram(
  gl: GL,
  vsSource: string,
  fsSource: string,
  label: string,
  feedbackVaryings?: string[],
): CompiledProgram {
  const vs = compile(gl, gl.VERTEX_SHADER, vsSource, `${label}.vert`);
  const fs = compile(gl, gl.FRAGMENT_SHADER, fsSource, `${label}.frag`);
  const program = gl.createProgram();
  if (!program) throw new Error(`Failed to create program for ${label}`);
  gl.attachShader(program, vs);
  gl.attachShader(program, fs);

  if (feedbackVaryings && feedbackVaryings.length > 0) {
    gl.transformFeedbackVaryings(program, feedbackVaryings, gl.INTERLEAVED_ATTRIBS);
  }

  gl.linkProgram(program);
  gl.deleteShader(vs);
  gl.deleteShader(fs);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(program) ?? '(no log)';
    gl.deleteProgram(program);
    throw new Error(`Program link failed [${label}]: ${log}`);
  }

  const uniforms: Record<string, WebGLUniformLocation | null> = {};
  const uCount = gl.getProgramParameter(program, gl.ACTIVE_UNIFORMS) as number;
  for (let i = 0; i < uCount; i++) {
    const info = gl.getActiveUniform(program, i);
    if (!info) continue;
    const name = info.name.replace(/\[0\]$/, '');
    uniforms[name] = gl.getUniformLocation(program, name);
  }

  const attribs: Record<string, number> = {};
  const aCount = gl.getProgramParameter(program, gl.ACTIVE_ATTRIBUTES) as number;
  for (let i = 0; i < aCount; i++) {
    const info = gl.getActiveAttrib(program, i);
    if (!info) continue;
    attribs[info.name] = gl.getAttribLocation(program, info.name);
  }

  return { program, uniforms, attribs };
}

export function createBuffer(
  gl: GL,
  target: number,
  data: Float32Array | null,
  usage: number,
): WebGLBuffer {
  const buf = gl.createBuffer();
  if (!buf) throw new Error('Failed to create buffer');
  gl.bindBuffer(target, buf);
  // `bufferData(target, null, usage)` is INVALID_VALUE: the three-argument
  // overload requires an ArrayBufferView, and passing null is not the same as
  // passing "no size". A caller that wants an empty store to size later must
  // get an explicit zero-length allocation.
  if (data === null) gl.bufferData(target, 0, usage);
  else gl.bufferData(target, data, usage);
  return buf;
}

export interface RenderTarget {
  framebuffer: WebGLFramebuffer;
  texture: WebGLTexture;
  width: number;
  height: number;
}

export function createTextureTarget(
  gl: GL,
  width: number,
  height: number,
  internalFormat: number,
  filter: number,
): RenderTarget {
  // Fail loudly and specifically if the caller has not clamped to the device
  // limit. The alternative -- letting texStorage2D fail and reporting a bare
  // framebuffer error -- costs a lot of debugging time, because the real cause
  // is the allocation two steps earlier.
  const max = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;
  if (width > max || height > max) {
    throw new Error(
      `Render target ${width}x${height} exceeds MAX_TEXTURE_SIZE (${max}). ` +
        `Clamp the grid size with clampGridSize() before allocating.`,
    );
  }

  const texture = gl.createTexture();
  if (!texture) throw new Error('Failed to create texture');
  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texStorage2D(gl.TEXTURE_2D, 1, internalFormat, width, height);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, filter);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);

  const framebuffer = gl.createFramebuffer();
  if (!framebuffer) throw new Error('Failed to create framebuffer');
  gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    gl.deleteTexture(texture);
    gl.deleteFramebuffer(framebuffer);
    throw new Error(
      `Framebuffer incomplete (0x${status.toString(16)}) at ${width}x${height}. ` +
        'If this is INCOMPLETE_ATTACHMENT, EXT_color_buffer_float is missing; ' +
        'if it is INCOMPLETE_DIMENSIONS or UNSUPPORTED, the format or size is invalid.',
    );
  }
  return { framebuffer, texture, width, height };
}

export function destroyTarget(gl: GL, target: RenderTarget | null): void {
  if (!target) return;
  gl.deleteFramebuffer(target.framebuffer);
  gl.deleteTexture(target.texture);
}
