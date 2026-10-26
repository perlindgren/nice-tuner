// Batched particle rendering: every particle of one batch is one instance of
// a single quad, and the fragment shader decides what the particle looks
// like — a circle, a rectangle, or a sprite (the whole texture, centered on
// the particle) — from the batch's shape kind, the same branch pattern as
// shape.wgsl. One instanced draw_indexed per batch.
//
// The CPU packs two vec4 per particle in PIXEL space (top-left origin, y
// down): (px, py, size, life fraction) and (angle, tint.r, tint.g, tint.b),
// where size is the particle's half-width in pixels, angle is the particle's
// own rotation in pixel-space radians (the node's rotation already folded
// in), and the tint is the particle's own color. The whole batch shares the
// uniform's base color and shape; when the uniform's `lit` flag is set,
// the frame's light field (ambient + the frame's lights) multiplies every
// particle's color.
struct ParticlesUniforms {
    size: vec2<f32>, // surface size in pixels
    color: vec4<f32>, // the batch's base tint (already node-modulated)
    misc: vec2<f32>, // x: shape kind (0.0 circle, 1.0 rectangle, 2.0 sprite), y: the shape's height/width factor
    lit: f32, // 1.0 when the batch is lit by the frame's light field
    // The receiver's light-layer mask — raw bits shipped through
    // f32::from_bits on the CPU and read back with bitcast<u32>.
    lit_layers: f32,
};

struct VertexOutput {
    @builtin(position)
    position: vec4<f32>,
    @location(0)
    inst: vec4<f32>, // (px, py, size, life fraction)
    @location(1)
    extra: vec4<f32>, // (angle, tint.r, tint.g, tint.b)
};

@group(0)
@binding(0)
var<uniform>
u: ParticlesUniforms;

@group(0)
@binding(1)
var<storage, read>
instances: array<vec4<f32>>;

// The sampled texture: the batch's image for a sprite batch, and a 1x1
// placeholder for the SDF kinds (circles and rectangles never sample it).
@group(0)
@binding(2)
var tex: texture_2d<f32>;

@group(0)
@binding(3)
var samp: sampler;

@group(0)
@binding(4)
var<storage, read>
field: LightField;

@group(0)
@binding(5)
var<storage, read>
occluders: OccluderField;

@vertex
fn vs_main(
    @builtin(vertex_index) vi: u32,
    @builtin(instance_index) ii: u32,
) -> VertexOutput {
    // Two vec4 per particle; the batch's draw count is the number of
    // particles, so the instance's data sits at 2 * ii.
    let inst = instances[2 * ii];
    let extra = instances[2 * ii + 1];
    // The quad's corners in the shape's own frame, before rotation: the
    // same four corners the shared index buffer [0, 1, 2, 2, 1, 3] walks.
    let corners = array<vec2<f32>, 4>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, 1.0),
    );
    // The box's half-extents in pixels: the particle's size along x, the
    // size times the shape's height/width factor along y, plus the
    // anti-alias band the SDF shapes smooth over (a sprite's texture is its
    // own edge, so it needs no band).
    let band = mix(0.75, 0.0, step(1.5, u.misc.x));
    let half = inst.z * vec2<f32>(1.0, u.misc.y) + band;
    // Rotate the box about the particle's center by its pixel-space angle.
    let c = cos(extra.x);
    let s = sin(extra.x);
    let rot = mat2x2<f32>(c, s, -s, c);
    let p = inst.xy + rot * (corners[vi] * half);
    // The pixel space is top-left, y down; the NDC space is bottom-left, y
    // up, so the y axis flips when mapping.
    let ndc = vec2<f32>(p.x / u.size.x * 2.0 - 1.0, 1.0 - p.y / u.size.y * 2.0);
    var out: VertexOutput;
    out.position = vec4<f32>(ndc, 0.0, 1.0);
    out.inst = inst;
    out.extra = extra;
    return out;
}

// The particle's lit color: the batch's base tint times the particle's
// own tint, run through the light mix when the batch is lit. Shared by
// both emit branches below so the multiply order — and with it the exact
// emitted value — lives in one place.
fn light_rgb(tinted: vec3<f32>, at: vec2<f32>) -> vec3<f32> {
    var rgb = tinted;
    if (u.lit > 0.5) {
        rgb *= light_mix(at, bitcast<u32>(u.lit_layers));
    }
    return rgb;
}

fn shade(frag_coord: vec4<f32>, inst: vec4<f32>, extra: vec4<f32>) -> vec4<f32> {
    // The fragment's offset from the particle's center, rotated back into
    // the shape's own frame, so rotation applies for free. In that frame
    // the texture's top row always sits at the negative y edge.
    let c = cos(extra.x);
    let s = sin(extra.x);
    let unrot = mat2x2<f32>(c, -s, s, c);
    let off = unrot * (frag_coord.xy - inst.xy);
    // The particle's alpha, scaled by its remaining life fraction; the
    // lights never touch it. Resolved — with the branch's own coverage or
    // texture alpha — *before* any lighting, and a fragment whose final
    // alpha is exactly zero discards: such a fragment is the blend
    // equation's identity, and discarding it skips the whole per-fragment
    // light loop (every light's falloff, every occlusion walk) that the
    // leaf sheets and snow deck spend over their transparent margins.
    // The test is exact zero, not an epsilon — an epsilon would be an
    // image change dressed as an optimization. The sprite branch samples
    // above its discard on purpose: mip derivatives come from the
    // surviving threads of each 2x2 quad. Tile-based targets carry the
    // same caveat recorded in sprite.wgsl — drivers may drop whole-draw
    // fast paths for programs containing `discard`, and while the no-depth
    // attachment rules out the classic early-depth demotion, a measured
    // tiler regression is an answer to revert this, not to rework it.
    let alpha = u.color.a * inst.w;
    if (u.misc.x < 0.5) {
        // A circle with the particle's size as its radius.
        let d = length(off) - inst.z;
        let a = (1.0 - smoothstep(-0.75, 0.75, d)) * alpha;
        if (a == 0.0) {
            discard;
        }
        // Emit the tinted color and coverage; composited over the
        // existing attachment via alpha blending.
        return vec4<f32>(light_rgb(u.color.rgb * extra.yzw, frag_coord.xy), a);
    } else if (u.misc.x < 1.5) {
        // A rectangle with half-extents (size, size * the shape's
        // height/width factor): the same SDF as shape.wgsl's branch.
        let q = abs(off) - vec2<f32>(inst.z, inst.z * u.misc.y);
        let d = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0);
        let a = (1.0 - smoothstep(-0.75, 0.75, d)) * alpha;
        if (a == 0.0) {
            discard;
        }
        return vec4<f32>(light_rgb(u.color.rgb * extra.yzw, frag_coord.xy), a);
    }
    // A sprite: the whole texture, centered on the particle, filling
    // the quad exactly.
    let uv = off / (2.0 * inst.z * vec2<f32>(1.0, u.misc.y)) + vec2<f32>(0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return vec4<f32>(0.0);
    }
    let t = textureSampleLevel(tex, samp, uv, 0.0);
    // The texture's own alpha multiplies the particle's alpha, like a
    // Shape::Sprite.
    let a = t.a * alpha;
    if (a == 0.0) {
        discard;
    }
    return vec4<f32>(t.rgb * light_rgb(u.color.rgb * extra.yzw, frag_coord.xy), a);
}

/// The scene's two-output form: the pixel as composited, and the light the
/// surface puts out. Only `fs_emissive` writes the second one, and the engine
/// picks between the entry points by whether its frame buffer can hold a value
/// past 1.0 at all — an entry point may not call another, so both are thin and
/// the work lives in `shade`.
struct SceneOut {
    @location(0) color: vec4<f32>,
    @location(1) emissive: vec4<f32>,
};

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>, @location(0) inst: vec4<f32>, @location(1) extra: vec4<f32>,) -> @location(0) vec4<f32> {
    return shade(frag_coord, inst, extra);
}

@fragment
fn fs_emissive(@builtin(position) frag_coord: vec4<f32>, @location(0) inst: vec4<f32>, @location(1) extra: vec4<f32>,) -> SceneOut {
    let s = shade(frag_coord, inst, extra);
    // Particles carry no emission of their own yet, so one drawn over a lamp
    // dims the lamp's glow by the particle's coverage — a gap to close, not a
    // design.
    return SceneOut(s, vec4<f32>(0.0, 0.0, 0.0, s.a));
}
