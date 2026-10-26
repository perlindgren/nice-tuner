struct RectUniforms {
    center: vec2<f32>,
    extent: vec2<f32>,
    color: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> u: RectUniforms;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    // Full-screen triangle in NDC; the scissor is set on the CPU to the
    // rectangle's bounding box, so fragments outside it are discarded and
    // the fragment shader only runs over the pixels it can write.
    let positions = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    return vec4<f32>(positions[i], 0.0, 1.0);
}

fn shade(frag_coord: vec4<f32>) -> vec4<f32> {
    let p = frag_coord.xy;
    // Signed distance in pixels from the fragment to the rectangle border:
    // negative inside, positive outside.
    let d = max(
        abs(p.x - u.center.x) - u.extent.x,
        abs(p.y - u.center.y) - u.extent.y,
    );
    let aa = 0.75; // ~1px anti-alias band.
    let alpha = 1.0 - smoothstep(-aa, aa, d);
    // Emit the rectangle's color and coverage; composited over the existing
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
    // Plain rectangles emit nothing, at their own coverage: an opaque rectangle
    // erases the emission it covers, a translucent one mixes with it.
    return SceneOut(s, vec4<f32>(0.0, 0.0, 0.0, s.a));
}
