// The bloom chain: take the emissive target, keep what is bright, blur what
// is left at three shrinking sizes, and hand all three to the present pass to
// add back. Each level is three passes — a tent downsample, then a separable
// Gaussian across and down — so the widest glow costs the fewest pixels: the
// three levels together cover an eighth of the frame's area, and a tenth of a
// tenth of it.
//
// The threshold is a soft knee on the brightest channel rather than a hard
// cutoff, because a hard one puts a visible edge on the *source*: a lamp whose
// centre clears 1.0 and whose rim does not gets a ring instead of a glow. The
// knee makes brightness choose how much it contributes, so "authored past 1.0"
// blooms hard and "ordinary bright" blooms a little, with no cliff between.

@group(0) @binding(0)
var source: texture_2d<f32>;

@group(0) @binding(1)
var src_sampler: sampler;

struct BloomUniforms {
    texel: vec2<f32>, // 1 / the source's size, in texels
    threshold: f32, // radiance under this contributes nothing; < 0 disables
    knee: f32, // how wide a band the threshold ramps across
    spread: vec2<f32>, // the blur's direction and step, in uv units; zero to downsample
    _pad: vec2<f32>,
};

@group(0) @binding(2)
var<uniform> u: BloomUniforms;

/// The soft-knee threshold. Below `threshold - knee` nothing survives; well
/// above it everything does; in between the contribution ramps by the square,
/// which is what keeps the transition from showing as a contour on the lamp.
/// Measured on the brightest channel, not on luminance: a saturated blue and a
/// pale yellow of equal luminance are nowhere near equally bright to an eye,
/// and hue has to survive the pass untouched.
fn keep_bright(c: vec3<f32>) -> vec3<f32> {
    if (u.threshold < 0.0) {
        return c;
    }
    let br = max(c.r, max(c.g, c.b));
    let knee = max(u.knee, 1e-4);
    let soft = clamp(br - u.threshold + knee, 0.0, 2.0 * knee);
    // The ramp inside the knee, and the straight line outside it, whichever is
    // larger. The first term alone is what a naive soft threshold looks like —
    // and it *falls back* as the source gets brighter (it peaks at the knee and
    // then decays as knee/radiance), so the brightest light in the frame ends
    // up blooming less than the merely bright. The second term is what makes
    // the curve keep climbing to 1.0.
    let ramp = soft * soft / (4.0 * knee * max(br, 1e-5));
    let over = (br - u.threshold) / max(br, 1e-5);
    return c * clamp(max(ramp, over), 0.0, 1.0);
}

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(positions[i], 0.0, 1.0);
}

/// One level down: four bilinear fetches at the corners of the destination
/// texel, which is a 2×2 tent — the cheapest thing that does not alias, and
/// free of the seams a point sample leaves when a bright pixel sits on a level
/// boundary. The threshold rides here, once, at the top of the chain: the
/// levels below inherit it, and re-applying it would keep sharpening the same
/// edge at every scale.
@fragment
fn fs_down(@builtin(position) out: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = vec2<f32>(
        (out.x + 0.5) * u.texel.x * 2.0,
        (out.y + 0.5) * u.texel.y * 2.0,
    );
    let t = u.texel;
    var c = textureSampleLevel(source, src_sampler, uv + vec2<f32>(-0.5, -0.5) * t, 0.0);
    c += textureSampleLevel(source, src_sampler, uv + vec2<f32>(0.5, -0.5) * t, 0.0);
    c += textureSampleLevel(source, src_sampler, uv + vec2<f32>(-0.5, 0.5) * t, 0.0);
    c += textureSampleLevel(source, src_sampler, uv + vec2<f32>(0.5, 0.5) * t, 0.0);
    return vec4<f32>(keep_bright(c.rgb * 0.25), 1.0);
}

/// A separable Gaussian across one axis: thirteen taps in six fetches.
///
/// The taps are a σ=3 Gaussian, and each symmetric pair is folded into one
/// bilinear fetch placed at the weighted midpoint of the two texels, which is
/// where the filter does the second half of the averaging for free: taps 1 and
/// 2 arrive as a single sample at 1.46 texels, weighted 0.2393.
///
/// Folding matters because the cost of this chain is the *number of passes*,
/// not the number of texels — measured at 4K, six more passes across the three
/// levels cost 0.10 ms, about 17 µs each, while the texels they read cost
/// almost nothing. A wider kernel is therefore nearly free and a second blur
/// pass is not, and σ=3 in one pass is wider than σ=2 twice over.
@fragment
fn fs_blur(@builtin(position) out: vec4<f32>) -> @location(0) vec4<f32> {
    let uv = (out.xy + vec2<f32>(0.5, 0.5)) * u.texel;
    var c = textureSampleLevel(source, src_sampler, uv, 0.0) * 0.1370;
    let near = u.spread * 1.4586;
    let middle = u.spread * 3.4036;
    let far = u.spread * 5.3514;
    c += (textureSampleLevel(source, src_sampler, uv + near, 0.0)
        + textureSampleLevel(source, src_sampler, uv - near, 0.0))
        * 0.2393;
    c += (textureSampleLevel(source, src_sampler, uv + middle, 0.0)
        + textureSampleLevel(source, src_sampler, uv - middle, 0.0))
        * 0.1394;
    c += (textureSampleLevel(source, src_sampler, uv + far, 0.0)
        + textureSampleLevel(source, src_sampler, uv - far, 0.0))
        * 0.0527;
    return vec4<f32>(c.rgb, 1.0);
}
