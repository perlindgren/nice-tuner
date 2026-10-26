struct ShapeUniforms {
    to_local: mat2x2<f32>,
    translation: vec2<f32>,
    center: vec2<f32>,
    params: vec2<f32>,
    color: vec4<f32>,
    misc: vec2<f32>, // x: anti-alias band (local units), y: kind (0.0 circle, 1.0 rectangle)
    lit: f32, // 1.0 when the shape is lit by the frame's light field
    // The receiver's light-layer mask — raw bits shipped through
    // f32::from_bits on the CPU and read back with bitcast<u32>; the light
    // loop folds it against every light's and every occluder's set.
    lit_layers: f32,
    emission: vec4<f32>, // own light: rgb, scaled by a, added to the light mix
    // The same light as the glow sees it. Equal to `emission` unless the app
    // asked for a halo in a different hue from the surface, in which case this
    // is the one the emissive attachment carries.
    glow: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> u: ShapeUniforms;

@group(0) @binding(1)
var<storage, read>
field: LightField;

@group(0) @binding(2)
var<storage, read>
occluders: OccluderField;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    // Full-screen triangle in NDC; the scissor is set on the CPU to the
    // shape's bounding box, so fragments outside it are discarded and the
    // fragment shader only runs over the pixels the shape can write.
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(positions[i], 0.0, 1.0);
}

/// What `shade` worked out: the pixel as composited, and the light the surface
/// puts out. Carries no `@location`s because an ordinary function may return
/// this but not the IO form below, which only an entry point may.
struct Shaded {
    color: vec4<f32>,
    emissive: vec4<f32>,
};

/// The scene's two-output form: only `fs_emissive` writes the second one, and
/// the engine picks between the entry points by whether its frame buffer can
/// hold a value past 1.0 at all — an entry point may not call another, so both
/// are thin and the work lives in `shade`.
struct SceneOut {
    @location(0) color: vec4<f32>,
    @location(1) emissive: vec4<f32>,
};

fn shade(frag_coord: vec4<f32>) -> Shaded {
    // Convert the fragment's pixel coordinate into the shape's local space
    // (the inverse of its world transform), then evaluate the signed
    // distance there, so rotation and scaling apply for free. The matrix
    // is stored column-major, so it multiplies the coordinate as a column
    // vector, matching the Rust-side transform convention.
    let p = u.to_local * frag_coord.xy + u.translation;
    var d: f32;
    if (u.misc.y < 0.5) {
        d = length(p - u.center) - u.params.x;
    } else {
        // Signed distance in local units from the fragment to the
        // rectangle border: negative inside, positive outside.
        let q = abs(p - u.center) - u.params;
        d = length(max(q, vec2<f32>(0.0))) + min(max(q.x, q.y), 0.0);
    }
    let alpha = 1.0 - smoothstep(-u.misc.x, u.misc.x, d);
    // The shape's color, multiplied by the light mix when it is lit, and
    // by its own emission either way: emission is light the surface puts
    // out, not light that has to reach it, so an unlit shape glows exactly
    // as a lit one does. The lights never touch the alpha. Self-light stays
    // in the shape's own hue — rgb scaled by the emission's alpha — and
    // past 1.0 it is an overshoot the float frame buffer keeps for a later
    // pass to find.
    //
    // Two branches, not a select: select's arguments are both evaluated, so
    // folding this would make every unlit shape on screen run the light loop
    // it is not lit by. The lit branch is the old expression unchanged.
    var rgb = u.color.rgb;
    if (u.lit > 0.5) {
        rgb *= light_mix(frag_coord.xy, bitcast<u32>(u.lit_layers)) + u.emission.rgb * u.emission.a;
    } else {
        rgb += rgb * u.emission.rgb * u.emission.a;
    }
    // The shape's color and coverage, composited over the existing
    // attachment by alpha blending.
    let a = alpha * u.color.a;
    // The rule every scene shader follows, and the reason the emissive target
    // is blended like the color one: rgb is the radiance this fragment puts
    // out, alpha is the coverage the color is composited at, and the blend
    // does the arithmetic. A fragment that covers nothing (alpha 0) therefore
    // changes nothing; an opaque wall drawn over a lamp takes the lamp's glow
    // away exactly as it takes its color; a translucent one mixes. And a shape
    // with no emission writes zero radiance rather than nothing, which is what
    // makes it an occluder of glow instead of a hole in it.
    //
    // The glow is the light this surface hands out, scaled by its own color
    // exactly as the light mix scales it — so it arrives in a hue made of both,
    // and an app that wants a halo unlike the surface asks for it with `glow`.
    // The unset case never reaches here: the CPU writes the emission into that
    // slot too, so a scene that split nothing behaves as if the field never
    // existed.
    let glow = u.color.rgb * u.glow.rgb * u.glow.a;
    return Shaded(vec4<f32>(rgb, a), vec4<f32>(glow, a));
}

@fragment
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    return shade(frag_coord).color;
}

@fragment
fn fs_emissive(@builtin(position) frag_coord: vec4<f32>) -> SceneOut {
    let s = shade(frag_coord);
    return SceneOut(s.color, s.emissive);
}

