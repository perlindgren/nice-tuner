struct SpriteUniforms {
    to_local: mat2x2<f32>,
    translation: vec2<f32>,
    size: vec2<f32>, // local extent of the quad in local units
    tint: vec4<f32>, // multiplied with every sampled pixel
    alpha: f32, // multiplied with the texture's own alpha
    lit: f32, // 1.0 when the sprite is lit by the frame's light field
    // The receiver's light-layer mask — raw bits shipped through
    // f32::from_bits on the CPU and read back with bitcast<u32>; the light
    // loop folds it against every light's and every occluder's set.
    lit_layers: f32,
    uv_rect: vec4<f32>, // [min_x, min_y, max_x, max_y] sub-rect of the texture
    emission: vec4<f32>, // own light: rgb, scaled by a, added to the light mix
    // The same light as the glow sees it: equal to `emission` unless the app
    // asked for a halo in a different hue from the sprite.
    glow: vec4<f32>,
};

@group(0) @binding(0)
var<uniform> u: SpriteUniforms;

@group(0) @binding(1)
var tex: texture_2d<f32>;

@group(0) @binding(2)
var samp: sampler;

@group(0) @binding(3)
var<storage, read>
field: LightField;

@group(0) @binding(4)
var<storage, read>
occluders: OccluderField;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    // Full-screen triangle in NDC; the scissor is set on the CPU to the
    // sprite's bounding box, so fragments outside it are discarded and the
    // fragment shader only runs over the pixels the sprite can write.
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
    // Convert the fragment's pixel coordinate into the sprite's local space
    // (the inverse of its world transform), where the sprite is centered at
    // the origin and extends by `size / 2` along each axis. Local +y is "up"
    // (the top row of the texture), so the v coordinate is flipped when
    // mapping to [0, 1] texture space.
    let p = u.to_local * frag_coord.xy + u.translation;
    let uv = vec2<f32>(p.x / u.size.x, -p.y / u.size.y) + vec2<f32>(0.5);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        return Shaded(vec4<f32>(0.0), vec4<f32>(0.0));
    }
    // Sample the quad's [0, 1] uv remapped into the texture sub-rectangle it
    // covers; for a whole-texture sprite the remap is the identity.
    let t = textureSampleLevel(tex, samp, mix(u.uv_rect.xy, u.uv_rect.zw, uv), 0.0);
    // The final alpha: the texture's own, scaled by the sprite's alpha and
    // the tint's. Computed here, before the lights, for one exact reason:
    // a fragment that would blend (rgb, 0) is the blend equation's
    // identity, so discarding it changes no pixel of the attachment — and
    // the whole per-fragment light loop (every light's falloff, every
    // occlusion walk) is skipped along with it. PNG margins are exactly
    // zero, so every transparent texel of a padded sprite takes this exit;
    // an epsilon would be an image change dressed as an optimization, so
    // the test is exact. Note what this does NOT recover: the texel fetch
    // and the rasterized slot are spent; only the arithmetic after them is.
    //
    // Why the fetch above names its level. `textureSample` works its mip
    // level out from screen-space derivatives, which a 2x2 quad only has
    // when all four of its threads ran the same path — so WGSL permits it
    // only in *uniform* control flow, and both this discard and the
    // out-of-rect early-out above are the opposite of uniform: they are
    // chosen per pixel. Dawn enforces the rule and rejects the module,
    // taking the pipeline, its bind group layout and every command buffer
    // touching them down with it; Naga never checks, so the same shader
    // builds and runs all day on Vulkan and Metal. Naming the level asks
    // no questions and is exactly equivalent here, because there is no
    // second level to ask about — every `mip_level_count` in the tree is
    // 1. The rule is the shaders', not this shader's: no bare
    // `textureSample` anywhere in frost.
    //
    // The sample keeps its place above the discard, which is where an
    // earlier form of this note argued it must stay so the survivors'
    // derivatives stayed defined. With the level named, that argument is
    // void and hoisting the discard over the fetch would spare the texel
    // read for every transparent texel — unmeasured, so not done. That
    // would be an optimization, and optimizations arrive with a bench.
    //
    // For tile-based targets (mobile tilers under WebGL, Apple's own
    // tiles): some drivers keep whole-draw fast paths that a program
    // containing `discard` can opt out of, fire or no fire. The classic
    // companion penalty — early-depth demotion — is a non-event here
    // because frost renders with no depth attachment at all, and desktop
    // tilers and immediate renderers run this alpha-masked pattern as
    // bread and butter; but if a tiler ever measures slower with the
    // discard than without, the answer is to revert this block, not to
    // rework it. Phone and tablet targets are outside the current
    // audience; a field report from one is the reason this note exists.
    let alpha = t.a * u.alpha * u.tint.a;
    if (alpha == 0.0) {
        discard;
    }
    // The tinted texture color, multiplied by the light mix when the
    // sprite is lit, and by its own emission either way — see the note in
    // shape.wgsl: emission is what a surface puts out, so it needs no light
    // field to be seen. The lights never touch the alpha, the self-light
    // stays in the sprite's own hue, and the branches are two because
    // select would evaluate light_mix for every unlit sprite too.
    var rgb = t.rgb * u.tint.rgb;
    if (u.lit > 0.5) {
        rgb *= light_mix(frag_coord.xy, bitcast<u32>(u.lit_layers)) + u.emission.rgb * u.emission.a;
    } else {
        rgb += rgb * u.emission.rgb * u.emission.a;
    }
    // The color, scaled by the sprite's alpha and the tint's own alpha,
    // composited over the existing attachment by alpha blending — unless the
    // pipeline is the additive one the `additive` node flag selects, which adds
    // what follows instead of covering with it. Same two channels, same meaning,
    // and this file cannot tell the two apart.
    //
    // What it puts out is the emission the light mix got, before the lights
    // touched it — the tinted texel times the light it hands out, in the hue in
    // `glow` (the emission, unless the app split the two apart, which the CPU
    // has already settled) — carried at the same coverage as the color, so the
    // blend composites both the same way. See the rule in shape.wgsl.
    let glow = t.rgb * u.tint.rgb * u.glow.rgb * u.glow.a;
    return Shaded(vec4<f32>(rgb, alpha), vec4<f32>(glow, alpha));
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

