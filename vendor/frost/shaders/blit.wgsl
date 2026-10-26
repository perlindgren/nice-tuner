// The fixed-render-size stretch: samples the frame's offscreen render
// buffer across the window — or, when the window's shape does not match
// the buffer's aspect, across the largest centered rectangle that does:
// the uniform scales the quad down to that letterbox rectangle and the
// cleared surface shows black bars around it. Two triangles — a quad,
// once one sample per window pixel, plain resampling, nothing smart.
@group(0) @binding(0)
var frame_buf: texture_2d<f32>;

@group(0) @binding(1)
var frame_sampler: sampler;

struct BlitUniforms {
    fit: vec2<f32>, // content extent as a fraction of the surface; (1, 1) fills it
    pad: vec2<f32>,
    // One weight per bloom slot — fine to coarse — already multiplied by the
    // user's strength, so all zero is how "no bloom" reaches this shader, and
    // the slots past the preset's length are zero the same way: bound to a
    // black texel, guarded away from ever being fetched. Two `vec4`s rather
    // than an `array<f32, 6>` because a uniform's array stride must be a
    // multiple of 16 — six floats get a vec4's stride each either way, and
    // only this shape says so honestly. Holds `bloom::MAX_LEVELS` weights,
    // at most six; `fit_bytes` asserts it.
    bloom_a: vec4<f32>, // levels 0..4, fine first
    bloom_b: vec4<f32>, // levels 4..6; z and w are the vec4's own padding
    tone: f32, // lands at 48, where the second vec4 leaves the struct
};

@group(0) @binding(2)
var<uniform> u: BlitUniforms;

// The blurred emissive levels, coarse last. One binding each rather than one
// array: a `texture_2d_array` would need all its layers the same size, and the
// whole point is that they are not.
@group(0) @binding(3)
var bloom0: texture_2d<f32>;

@group(0) @binding(4)
var bloom1: texture_2d<f32>;

@group(0) @binding(5)
var bloom2: texture_2d<f32>;

@group(0) @binding(6)
var bloom3: texture_2d<f32>;

@group(0) @binding(7)
var bloom4: texture_2d<f32>;

@group(0) @binding(8)
var bloom5: texture_2d<f32>;

@group(0) @binding(9)
var bloom_sampler: sampler;

struct VsOut {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

// The linear-surface fallback. An sRGB surface encodes gamma on
// write in hardware and this pass stands down; where the window's
// surface format is linear (a browser canvas may offer no sRGB
// format — Firefox's WebGPU offers none — and the engine's format
// preference then has nothing to prefer), the frame buffer holds raw
// linear light, exactly what every shader writes, and without
// encoding here it reaches the screen unconverted: a dark scene. The
// literal `false` below is a placeholder — the engine substitutes
// `true` when it builds this pipeline for a linear surface — and the
// false form keeps the file valid, dormant WGSL on its own.
const ENCODE_GAMMA: bool = /*@gamma@*/false;

/// The exact sRGB transfer curve (IEC 61966-2-1), per channel.
fn encode_srgb(c: vec3<f32>) -> vec3<f32> {
    let lo = c * 12.92;
    let hi = 1.055 * pow(max(c, vec3<f32>(0.0)), vec3<f32>(1.0 / 2.4)) - 0.055;
    return select(hi, lo, c < vec3<f32>(0.0031308));
}

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> VsOut {
    // The content rectangle as two triangles. It used to be the one
    // oversized full-screen triangle every pass traditionally uses — and
    // that was a lie at the corner: scaled by the fit, its hypotenuse only
    // grazes the rectangle's top-right corner tangentially, bulging above
    // the rectangle's top edge everywhere else, and the sampler's clamp
    // then paints a stretched copy of the buffer's top row across the very
    // bar the clear was meant to leave black. Measured in a browser
    // fullscreen: a flat band of stretched pixels above the picture, cut by
    // a diagonal notch near the right edge where the hypotenuse at last
    // sinks below the surface's top. A quad has four real edges; the bars
    // are the bars.
    let positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(-1.0, 1.0),
        vec2<f32>(1.0, -1.0),
        vec2<f32>(1.0, 1.0),
        vec2<f32>(-1.0, 1.0),
    );
    let p = positions[i];
    var out: VsOut;
    // Scale the quad into the letterbox rectangle: with fit (1, 1) it is
    // the full window, and with a smaller fit its edges are the rectangle
    // itself, the cleared bars outside. The uv below keeps the *unscaled*
    // vertex, so the rectangle still receives the buffer's full [0, 1]
    // range.
    out.clip = vec4<f32>(p * u.fit, 0.0, 1.0);
    // Texture v runs from the buffer's first row (its top edge) down, and
    // NDC y = +1 is that same top edge — so v is the flip of the clip y.
    out.uv = vec2<f32>((p.x + 1.0) / 2.0, (1.0 - p.y) / 2.0);
    return out;
}

/// Fold the range above the display into the top of it, so that a lamp three
/// times brighter than white arrives as a lamp and not as a white disc with
/// no colour and no gradient left in it.
///
/// `headroom` is the share of the display range handed to light above white.
/// Below the knee at `1 - headroom` the curve is the identity and its slope
/// there is 1.0, which is the whole point of that choice: an ordinary pixel,
/// authored against a screen and never meant to go past it, is left exactly
/// where the artist put it, and only the band just under white pays anything.
/// The price of reserving the top of the range is that display white itself
/// lands at `1 - headroom / 2` — the headroom is not free, and no curve makes
/// it so; a tone map trades brightness for highlight detail and that is the
/// trade, stated in one number.
///
/// Above the knee the curve is rational, approaching white as 1/x. That is
/// deliberate: an exponential shoulder — the other obvious shape — gets to
/// white so fast that a lamp at 2.0 and a lamp at 20.0 come out the same
/// pixel, and the difference between those two lamps is most of what an
/// over-bright frame has to say.
///
/// The fold is applied to the brightest channel and the ratio is what the
/// pixel is scaled by, so a light keeps its colour on the way in. Folding each
/// channel on its own drains the saturation out of every highlight in the
/// frame, which is the first thing a bad tone map shows.
fn fold_highlights(c: vec3<f32>, headroom: f32) -> vec3<f32> {
    let knee = 1.0 - headroom;
    let br = max(c.r, max(c.g, c.b));
    if (br <= knee) {
        return c;
    }
    let folded = 1.0 - (headroom * headroom) / (headroom + br - knee);
    return c * (folded / br);
}

@fragment
fn fs_main(in: VsOut) -> @location(0) vec4<f32> {
    // The buffer was fully cleared before the frame was drawn into it, so
    // the sampled color always covers the pixel: no blending needed, the
    // pass writes it straight to the window.
    var c = textureSampleLevel(frame_buf, frame_sampler, in.uv, 0.0);
    // Bloom is light, so it joins in the frame's own linear space, before any
    // encoding: added after the transfer curve it would be summed in a space
    // where doubling a number does not double its brightness, and a
    // threshold-then-encode order is how bloom ends up eating the darks.
    //
    // `SampleLevel` rather than `Sample`: each guard below is uniform because
    // it reads a uniform, but the derivative-based forms cannot know that, and
    // the level is 0 anyway — these are the chain's own resolutions, not a mip
    // chain to choose from. One guard per level rather than one for the lot:
    // a tight preset's empty slots hold zeros, and a zero's fetch is exactly
    // the work there is nothing left to save.
    if (u.bloom_a.x != 0.0) {
        c += textureSampleLevel(bloom0, bloom_sampler, in.uv, 0.0) * u.bloom_a.x;
    }
    if (u.bloom_a.y != 0.0) {
        c += textureSampleLevel(bloom1, bloom_sampler, in.uv, 0.0) * u.bloom_a.y;
    }
    if (u.bloom_a.z != 0.0) {
        c += textureSampleLevel(bloom2, bloom_sampler, in.uv, 0.0) * u.bloom_a.z;
    }
    if (u.bloom_a.w != 0.0) {
        c += textureSampleLevel(bloom3, bloom_sampler, in.uv, 0.0) * u.bloom_a.w;
    }
    if (u.bloom_b.x != 0.0) {
        c += textureSampleLevel(bloom4, bloom_sampler, in.uv, 0.0) * u.bloom_b.x;
    }
    if (u.bloom_b.y != 0.0) {
        c += textureSampleLevel(bloom5, bloom_sampler, in.uv, 0.0) * u.bloom_b.y;
    }
    // Last, after every light has been added, because it is the total that has
    // to fit the display — and before the transfer curve, which is not a
    // mapping of brightness in the sense this one is.
    if (u.tone > 0.0) {
        c = vec4<f32>(fold_highlights(c.rgb, u.tone), c.a);
    }
    if (ENCODE_GAMMA) {
        return vec4<f32>(encode_srgb(c.rgb), c.a);
    }
    return c;
}
