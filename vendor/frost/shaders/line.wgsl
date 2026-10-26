struct Uniforms {
    a: vec2<f32>,
    b: vec2<f32>,
    color: vec4<f32>,
    width: f32,
};

@group(0) @binding(0)
var<uniform> u: Uniforms;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    // Full-screen triangle in NDC; the scissor is set on the CPU to the
    // line's bounding box, so fragments outside it are discarded and the
    // fragment shader only runs over the pixels the line can write.
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(positions[i], 0.0, 1.0);
}

fn shade(frag_coord: vec4<f32>) -> vec4<f32> {
    let p = frag_coord.xy;
    // Distance in pixels from the fragment to the segment a-b.
    let ab = u.b - u.a;
    let t = clamp(dot(p - u.a, ab) / max(dot(ab, ab), 1e-6), 0.0, 1.0);
    let dist = length(p - (u.a + ab * t));
    let half_width = u.width * 0.5;
    let aa = 0.75; // ~1px anti-alias band.
    let alpha = 1.0 - smoothstep(half_width - aa, half_width + aa, dist);
    // Emit the line's color and coverage; composited over the existing
    // attachment via alpha blending.
    return vec4<f32>(u.color.rgb, alpha * u.color.a);
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
fn fs_main(@builtin(position) frag_coord: vec4<f32>) -> @location(0) vec4<f32> {
    return shade(frag_coord);
}

@fragment
fn fs_emissive(@builtin(position) frag_coord: vec4<f32>) -> SceneOut {
    let s = shade(frag_coord);
    // A line emits nothing. The zero rides with the line's own coverage:
    // an opaque line drawn over a lamp does take the lamp's glow away, a
    // translucent one only dims it — the rule is written out in shape.wgsl.
    return SceneOut(s, vec4<f32>(0.0, 0.0, 0.0, s.a));
}
