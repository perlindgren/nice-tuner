//! The WGSL shader sources, embedded at compile time from the `.wgsl` files
//! in the crate's `shaders/` directory.
//!
//! Three of them — the pipelines that light a surface — are assembled from
//! `lighting.wgsl` plus their own file, so the shadow test and the loop over
//! the lights live in one place instead of three that can drift apart. A WGSL
//! name declared at module scope is in scope across the whole program, so the
//! shared text may read the `field` and `occluders` bindings its host declares
//! after it; those two declarations stay in each host because their `@binding`
//! indices differ per pipeline.

/// The line shader source.
pub(crate) const LINE_SHADER: &str = include_str!("../shaders/line.wgsl");

/// The shared lighting vocabulary every lit pipeline is built on: the
/// light and occluder field structs, the shadow test, and the loop over
/// the lights. The engine's lit kinds concatenate it before their own
/// source; a game's [`crate::Shape::with_program`] source gets the same
/// treatment at compile time, which is what lets a user program call
/// `light_mix` and bind the frame's fields without declaring them —
/// and lets a game test its own `prelude + source` the way `cargo test`
/// tests the engine's.
pub const LIGHTING_WGSL: &str = include_str!("../shaders/lighting.wgsl");

/// The polyline (batched line) shader source.
pub(crate) const POLYLINE_SHADER: &str = include_str!("../shaders/polyline.wgsl");

/// The circle shader source.
pub(crate) const CIRCLE_SHADER: &str = include_str!("../shaders/circle.wgsl");

/// The rectangle shader source.
pub(crate) const RECT_SHADER: &str = include_str!("../shaders/rectangle.wgsl");

/// The shape (scene circle/rectangle) shader source: the shared lighting
/// vocabulary, then the pipeline that paints a lit circle or rectangle.
pub(crate) const SHAPE_SHADER: &str = concat!(
    include_str!("../shaders/lighting.wgsl"),
    include_str!("../shaders/shape.wgsl")
);

/// The sprite (texture) shader source: the shared lighting vocabulary, then
/// the pipeline that paints textured sprites lit by the frame's fields.
pub(crate) const SPRITE_SHADER: &str = concat!(
    include_str!("../shaders/lighting.wgsl"),
    include_str!("../shaders/sprite.wgsl")
);

/// The batched particle shader source: the shared lighting vocabulary, then
/// the pipeline that paints thousands of small lit quads in one draw.
pub(crate) const PARTICLES_SHADER: &str = concat!(
    include_str!("../shaders/lighting.wgsl"),
    include_str!("../shaders/particles.wgsl")
);

/// The tile map (instanced atlas quads) shader source: the shared
/// lighting vocabulary, then the pipeline that paints a map's tiles —
/// lit by the frame's fields when its node asks, exactly as before when
/// it does not.
pub(crate) const TILEMAP_SHADER: &str = concat!(
    include_str!("../shaders/lighting.wgsl"),
    include_str!("../shaders/tilemap.wgsl")
);

/// The fixed-render-size stretch (blit) shader source.
pub(crate) const BLIT_SHADER: &str = include_str!("../shaders/blit.wgsl");

/// The bloom chain: threshold, tent downsample and separable Gaussian, one
/// level per preset weight. One file, two entry points, three passes per level.
pub(crate) const BLOOM_SHADER: &str = include_str!("../shaders/bloom.wgsl");

/// The shader table: each pipeline's source by name, and the files
/// its text is assembled from, in order — the shape of the consts
/// above, speakable at runtime. `App::wgsl` asks by these names;
/// a watched file re-assembles every source that reads from it,
/// which is how a `lighting.wgsl` edit reaches the four lit
/// pipelines at once. The live-reload table rides the filesystem:
/// a watched directory of `.wgsl` files is a development affordance
/// of a native build, and debug builds are where it lives.
#[cfg(any(test, debug_assertions))]
pub(crate) const SOURCES: &[(&str, &[&str])] = &[
    ("line", &["line"]),
    ("polyline", &["polyline"]),
    ("circle", &["circle"]),
    ("rectangle", &["rectangle"]),
    ("shape", &["lighting", "shape"]),
    ("sprite", &["lighting", "sprite"]),
    ("particles", &["lighting", "particles"]),
    ("tilemap", &["lighting", "tilemap"]),
    ("blit", &["blit"]),
];

/// The two stages wgpu runs inside `create_shader_module`: naga's
/// parse, then its validator. A source that fails either stage
/// here fails at runtime there — so the watching app catches it
/// here, where a refusal is a log line and the old shaders stand,
/// not there, where it is a dead frame. Same build gate as `Live`,
/// its only caller: `wgpu` carries naga wherever it compiles shaders
/// at runtime.
#[cfg(debug_assertions)]
pub(crate) fn validate(source: &str) -> Result<(), String> {
    let module = wgpu::naga::front::wgsl::parse_str(source).map_err(|err| format!("{err}"))?;
    wgpu::naga::valid::Validator::new(
        wgpu::naga::valid::ValidationFlags::all(),
        wgpu::naga::valid::Capabilities::all(),
    )
    .validate(&module)
    .map_err(|err| format!("{err}"))?;
    Ok(())
}

/// Check a WGSL source the way the engine checks its own: naga's parse,
/// then its validator, and the complaint handed back instead of paid
/// for with a dead frame. The runtime never calls this — a shipped
/// program travels straight to the driver, which is exactly why a game
/// should gate its own shaders on it in a test, the way `cargo test`
/// gates the engine's. Available wherever wgpu carries naga: the same
/// gate as the live shader watcher — debug builds.
#[cfg(debug_assertions)]
pub fn validate_wgsl(source: &str) -> Result<(), String> {
    validate(source)
}

/// The watched shader directory: development edits `.wgsl` files
/// under a running game, and this carries the live text that
/// shadows the embedded consts. Debug builds only, on targets with
/// a filesystem — the release world's shaders are the ones compiled
/// into the crate, and naga saw them at `cargo test`.
#[cfg(debug_assertions)]
pub(crate) struct Live {
    /// The directory of files, and one sense per file.
    dir: std::path::PathBuf,
    watches: Vec<(&'static str, crate::Watch)>,
    /// Live text per source name, swapped in only by a batch that
    /// passed validation whole.
    live: std::collections::HashMap<&'static str, String>,
}

#[cfg(debug_assertions)]
impl Live {
    /// Watch a directory of `.wgsl` files from their state right
    /// now: the first change reported is one made after the game
    /// booted.
    pub(crate) fn new(dir: std::path::PathBuf) -> Self {
        let mut watches: Vec<(&'static str, crate::Watch)> = Vec::new();
        for (_, files) in SOURCES {
            for file in *files {
                if !watches.iter().any(|(seen, _)| *seen == *file) {
                    watches.push((*file, crate::Watch::new(dir.join(format!("{file}.wgsl")))));
                }
            }
        }
        Self {
            dir,
            watches,
            live: std::collections::HashMap::new(),
        }
    }

    /// One poll: ten stats, and if any file moved — reassemble the
    /// whole table from disk, put it past naga, and swap it only if
    /// every source compiles. A broken file refuses the batch:
    /// the old sources stand, the log names the error, and the
    /// next save gets its chance. True when live text changed —
    /// the caller rebuilds the pipelines.
    pub(crate) fn poll(&mut self) -> bool {
        if !self.watches.iter_mut().any(|(_, w)| w.changed()) {
            return false;
        }
        let mut next = std::collections::HashMap::new();
        for (name, files) in SOURCES {
            let mut source = String::new();
            for file in *files {
                match std::fs::read_to_string(self.dir.join(format!("{file}.wgsl"))) {
                    Ok(text) => source.push_str(&text),
                    Err(err) => {
                        log::warn!(
                            "shader: '{file}.wgsl' unreadable ({err}) — the old shaders stand"
                        );
                        return false;
                    }
                }
            }
            if let Err(err) = validate(&source) {
                log::warn!("shader: '{name}' does not compile — the old shaders stand:\n{err}");
                return false;
            }
            next.insert(*name, source);
        }
        self.live = next;
        log::info!("shader: recompiled from disk — the pipelines rebuild next frame");
        true
    }

    /// The live text of a source, if a watched file has spoken it.
    pub(crate) fn get(&self, name: &str) -> Option<&String> {
        self.live.get(name)
    }
}

#[cfg(test)]
mod tests {
    //! Shader validation without running anything: naga's WGSL frontend and
    //! validator are the two stages wgpu executes inside
    //! `Device::create_shader_module` — parse, then validate — so a shader
    //! that fails either stage here fails at runtime there.

    use super::*;
    use crate::backend::{LIGHT_FIELD_BUFFER_MIN, OCCLUDER_FIELD_BUFFER_MIN};

    /// The ten shader sources with their file names.
    fn all_shaders() -> [(&'static str, &'static str); 10] {
        [
            ("line.wgsl", LINE_SHADER),
            ("polyline.wgsl", POLYLINE_SHADER),
            ("circle.wgsl", CIRCLE_SHADER),
            ("rectangle.wgsl", RECT_SHADER),
            ("shape.wgsl", SHAPE_SHADER),
            ("sprite.wgsl", SPRITE_SHADER),
            ("particles.wgsl", PARTICLES_SHADER),
            ("tilemap.wgsl", TILEMAP_SHADER),
            ("blit.wgsl", BLIT_SHADER),
            ("bloom.wgsl", BLOOM_SHADER),
        ]
    }

    /// All ten shaders parse as valid WGSL.
    #[test]
    fn all_shaders_parse_as_wgsl() {
        for (name, source) in all_shaders() {
            naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|err| panic!("{name} is not valid WGSL: {err}"));
        }
    }

    /// All ten shaders must pass naga's *validator*, not just its parser.
    /// Parsing only proves the WGSL is syntactically valid; the validator is
    /// the second stage wgpu runs inside `Device::create_shader_module`, and
    /// it enforces the address-space layout rules the parser never checks —
    /// e.g., in the uniform space an array member's stride must be a
    /// multiple of 16 bytes. The polyline's original
    /// `array<vec2<f32>, 256>` parsed fine but was rejected by the device
    /// for exactly this reason, after the parse-only test had passed.
    #[test]
    fn all_shaders_pass_device_side_validation() {
        for (name, source) in all_shaders() {
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|err| panic!("{name} is not valid WGSL: {err}"));
            naga::valid::Validator::new(
                naga::valid::ValidationFlags::all(),
                naga::valid::Capabilities::all(),
            )
            .validate(&module)
            .unwrap_or_else(|err| panic!("{name} fails device-side validation: {err}"));
        }
    }

    /// The seam for game-authored programs, tested from this side: a
    /// program written against the documented contract — the entry points
    /// and bindings of `sprite.wgsl`, the `SpriteUniforms` mirror, and the
    /// lighting vocabulary taken as given from the prelude — assembles and
    /// validates exactly like the engine's own lit kinds. If this test
    /// breaks, every game program in the workspace breaks with it, which
    /// is the point: the contract is only a contract if CI signs it.
    #[test]
    fn a_users_program_assembles_over_the_prelude() {
        const USER: &str = r#"
struct SpriteUniforms {
    to_local: mat2x2<f32>,
    translation: vec2<f32>,
    size: vec2<f32>,
    tint: vec4<f32>,
    alpha: f32,
    lit: f32,
    lit_layers: f32,
    uv_rect: vec4<f32>,
    emission: vec4<f32>,
    glow: vec4<f32>,
};

@group(0) @binding(0) var<uniform> u: SpriteUniforms;
@group(0) @binding(1) var tex: texture_2d<f32>;
@group(0) @binding(2) var samp: sampler;
@group(0) @binding(3) var<storage, read> field: LightField;
@group(0) @binding(4) var<storage, read> occluders: OccluderField;

@vertex
fn vs_main(@builtin(vertex_index) i: u32) -> @builtin(position) vec4<f32> {
    let p = array<vec2<f32>, 3>(vec2(-1.0, -1.0), vec2(3.0, -1.0), vec2(-1.0, 3.0));
    return vec4(p[i], 0.0, 1.0);
}

fn paint(coord: vec4<f32>) -> vec4<f32> {
    let local = u.to_local * coord.xy + u.translation;
    let uv = vec2(local.x / u.size.x, -local.y / u.size.y) + vec2(0.5);
    let t = textureSample(tex, samp, mix(u.uv_rect.xy, u.uv_rect.zw, uv));
    var rgb = t.rgb * u.tint.rgb;
    if (u.lit > 0.5) {
        rgb *= light_mix(coord.xy, bitcast<u32>(u.lit_layers));
    }
    return vec4(rgb, t.a * u.alpha * u.tint.a);
}

@fragment
fn fs_main(@builtin(position) coord: vec4<f32>) -> @location(0) vec4<f32> {
    return paint(coord);
}

struct SceneOut {
    @location(0) color: vec4<f32>,
    @location(1) emissive: vec4<f32>,
};

@fragment
fn fs_emissive(@builtin(position) coord: vec4<f32>) -> SceneOut {
    let col = paint(coord);
    return SceneOut(col, vec4(0.0));
}
"#;
        let source = format!("{LIGHTING_WGSL}{USER}");
        let module = naga::front::wgsl::parse_str(&source)
            .unwrap_or_else(|err| panic!("a contract program must parse: {err}"));
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .unwrap_or_else(|err| panic!("a contract program must validate: {err}"));
    }

    /// Every scene shader offers both fragment entry points; the blit offers
    /// one.
    ///
    /// `App::create_pipeline` picks `fs_emissive` for all eight scene
    /// pipelines when the frame carries an emissive target and `fs_main`
    /// otherwise, so a shader missing either form fails at pipeline creation
    /// on the first frame — on whichever machine happened to ask for the
    /// variant it forgot to write. The entry points are the pipeline's contract
    /// with the shader, so the contract is checked here rather than left to
    /// the driver to discover.
    #[test]
    fn every_scene_shader_offers_both_fragment_entry_points() {
        let has = |source: &str, entry: &str| {
            let module = naga::front::wgsl::parse_str(source)
                .unwrap_or_else(|err| panic!("{source} is not valid WGSL: {err}"));
            module
                .entry_points
                .iter()
                .any(|ep| ep.stage == naga::ShaderStage::Fragment && ep.name == entry)
        };
        for (name, source) in all_shaders() {
            match name {
                // Two of the ten are drawn outside the scene pass, into targets
                // of their own, so the two-entry-point rule is not theirs: they
                // need the entry points their own pipelines ask for. Named
                // rather than inferred, so a shader added to `all_shaders`
                // cannot slip through unclassified.
                "blit.wgsl" => {
                    assert!(
                        has(source, "fs_main"),
                        "{name} lost the entry point the blit pipeline asks for"
                    );
                    assert!(
                        !has(source, "fs_emissive"),
                        "{name} has an emissive output the window can never show"
                    );
                }
                "bloom.wgsl" => {
                    for entry in ["fs_down", "fs_blur"] {
                        assert!(
                            has(source, entry),
                            "{name} lost the entry point the {entry} pass asks for"
                        );
                    }
                    assert!(
                        !has(source, "fs_emissive"),
                        "{name} has an emissive output no bloom target reads"
                    );
                }
                _ => {
                    for entry in ["fs_main", "fs_emissive"] {
                        assert!(
                            has(source, entry),
                            "{name} is missing {entry}: the engine builds every \
                             scene pipeline in both forms, with one target and \
                             with two"
                        );
                    }
                }
            }
        }
    }

    /// The blit pass carries a substituted-const variant for linear
    /// presentation surfaces: `App` replaces the placeholder `false`
    /// with `true` at pipeline build time when the surface format has no
    /// hardware sRGB encoding (a browser canvas without an sRGB format),
    /// and the pass gamma-encodes instead. Both forms must be valid
    /// WGSL — and the marker must be exactly the text the engine
    /// replaces, or the substitution silently builds the dormant variant
    /// and the web scene goes dark again.
    #[test]
    fn the_blit_gamma_variant_validates() {
        const MARKER: &str = "/*@gamma@*/false";
        assert_eq!(
            BLIT_SHADER.matches(MARKER).count(),
            1,
            "the engine replaces exactly one {MARKER} placeholder"
        );
        // The same replacement the engine performs in `set_up_pipelines`.
        let gamma = BLIT_SHADER.replace(MARKER, "true");
        assert_ne!(gamma, BLIT_SHADER, "the replacement must land");
        assert!(gamma.contains("const ENCODE_GAMMA: bool = true;"));
        let module =
            naga::front::wgsl::parse_str(&gamma).expect("the gamma blit variant must parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("the gamma blit variant must pass device-side validation");
    }

    /// The lighting vocabulary is shared, not copied. Each lit pipeline
    /// carries `lighting.wgsl` once and declares the shadow test exactly
    /// once; three copies of `occluded` is what this replaced, and the shape
    /// of bug it invites is a fix landing in two of the three files, leaving
    /// one shader's shadows disagreeing with the CPU and with its neighbours.
    #[test]
    fn the_lighting_vocabulary_is_shared_not_copied() {
        const LIGHTING: &str = include_str!("../shaders/lighting.wgsl");
        for (name, source) in [
            ("shape", SHAPE_SHADER),
            ("sprite", SPRITE_SHADER),
            ("particles", PARTICLES_SHADER),
        ] {
            assert!(
                source.starts_with(LIGHTING),
                "{name} must be the shared vocabulary followed by its own code"
            );
            for declaration in [
                "struct LightField ",
                "struct OccluderField ",
                "fn occluded(",
                "fn light_mix(",
            ] {
                assert_eq!(
                    source.matches(declaration).count(),
                    1,
                    "{name} must declare {declaration}... exactly once, from the shared file"
                );
            }
        }
        // A pipeline that paints no light carries none of the vocabulary: a
        // fourth copy would compile, and nothing but this test would notice.
        for (name, source) in all_shaders() {
            if !source.starts_with(LIGHTING) {
                assert!(
                    !source.contains("fn occluded("),
                    "{name} paints no light and must not carry the shadow test"
                );
            }
        }
    }

    /// The member offsets WGSL assigns to `ShapeUniforms` must match the
    /// CPU-side uniform writer in `shape_uniform_data` (mat2x2 @0, vec2 @16,
    /// vec2 @24, vec2 @32, vec4 @48 — 16 bytes, 16-byte aligned — vec2 @64,
    /// f32 @72, vec4 @80; 96 bytes total). This is the GPU-side mirror of
    /// `shape_uniform_bytes_follow_the_wgsl_layout` in backend.rs, so a layout
    /// drift on either side fails a test.
    #[test]
    fn shape_uniform_offsets_match_the_cpu_layout() {
        let module = naga::front::wgsl::parse_str(SHAPE_SHADER)
            .expect("shape.wgsl should parse (see all_shaders_parse_as_wgsl)");
        let ty = module
            .types
            .iter()
            .find_map(|(_, ty)| match &ty.inner {
                naga::TypeInner::Struct { .. } if ty.name.as_deref() == Some("ShapeUniforms") => {
                    Some(ty)
                }
                _ => None,
            })
            .expect("shape.wgsl should declare the ShapeUniforms struct");
        let (offsets, total_size) = match &ty.inner {
            naga::TypeInner::Struct { members, span } => (
                members.iter().map(|m| m.offset).collect::<Vec<u32>>(),
                *span,
            ),
            _ => unreachable!("ShapeUniforms must be a struct"),
        };
        // `lit` rides in the 4 bytes the struct already padded after `misc`,
        // `lit_layers` takes the next 4 — the rest of that padding — and the
        // 16-byte-aligned `emission` vec4 follows at 80, `glow` at 96.
        assert_eq!(offsets, [0, 16, 24, 32, 48, 64, 72, 76, 80, 96]);
        // The struct's total size must equal the CPU-side buffer length,
        // otherwise the buffer would be too short or carry dead bytes.
        assert_eq!(total_size, 112);
    }

    /// The member offsets WGSL assigns to `SpriteUniforms` must match the
    /// CPU-side uniform writer in `sprite_uniform_data` (mat2x2 @0, vec2 @16,
    /// vec2 @24, vec4 @32 — 16 bytes, 16-byte aligned — f32 @48, f32 @52,
    /// f32 @56, vec4 @64, vec4 @80 and vec4 @96; 112 bytes total). This is the GPU-side mirror of
    /// `sprite_uniform_bytes_follow_the_wgsl_layout` in backend.rs, so a
    /// layout drift on either side fails a test.
    #[test]
    fn sprite_uniform_offsets_match_the_cpu_layout() {
        let module = naga::front::wgsl::parse_str(SPRITE_SHADER)
            .expect("sprite.wgsl should parse (see all_shaders_parse_as_wgsl)");
        let ty = module
            .types
            .iter()
            .find_map(|(_, ty)| match &ty.inner {
                naga::TypeInner::Struct { .. } if ty.name.as_deref() == Some("SpriteUniforms") => {
                    Some(ty)
                }
                _ => None,
            })
            .expect("sprite.wgsl should declare the SpriteUniforms struct");
        let (offsets, total_size) = match &ty.inner {
            naga::TypeInner::Struct { members, span } => (
                members.iter().map(|m| m.offset).collect::<Vec<u32>>(),
                *span,
            ),
            _ => unreachable!("SpriteUniforms must be a struct"),
        };
        // `lit` and `lit_layers` ride in the padding the struct already kept
        // between `alpha` and the 16-byte-aligned `uv_rect`; `emission`
        // follows the uv_rect at 80 and `glow` follows the emission at 96.
        assert_eq!(offsets, [0, 16, 24, 32, 48, 52, 56, 64, 80, 96]);
        // The struct's total size must equal the CPU-side buffer length,
        // otherwise the buffer would be too short or carry dead bytes.
        assert_eq!(total_size, 112);
    }

    /// The member offsets WGSL assigns to `ParticlesUniforms` must match the
    /// CPU-side uniform writer in `particles_uniform_data` (vec2 @0, vec4 @16
    /// — 16 bytes, 16-byte aligned — vec2 @32, f32 @40, f32 @44, 48 bytes
    /// total). This is the GPU-side mirror of the writer, so a layout drift
    /// on either side fails a test.
    #[test]
    fn particles_uniform_offsets_match_the_cpu_layout() {
        let module = naga::front::wgsl::parse_str(PARTICLES_SHADER)
            .expect("particles.wgsl should parse (see all_shaders_parse_as_wgsl)");
        let ty = module
            .types
            .iter()
            .find_map(|(_, ty)| match &ty.inner {
                naga::TypeInner::Struct { .. }
                    if ty.name.as_deref() == Some("ParticlesUniforms") =>
                {
                    Some(ty)
                }
                _ => None,
            })
            .expect("particles.wgsl should declare the ParticlesUniforms struct");
        let (offsets, total_size) = match &ty.inner {
            naga::TypeInner::Struct { members, span } => (
                members.iter().map(|m| m.offset).collect::<Vec<u32>>(),
                *span,
            ),
            _ => unreachable!("ParticlesUniforms must be a struct"),
        };
        // `lit` and `lit_layers` ride in the 8 bytes the struct already
        // padded after `misc`.
        assert_eq!(offsets, [0, 16, 32, 40, 44]);
        // The struct's total size must equal the CPU-side buffer length,
        // otherwise the buffer would be too short or carry dead bytes.
        assert_eq!(total_size, 48);
    }

    /// The member offsets WGSL assigns to `TileMapUniforms` must match the
    /// CPU-side uniform writer in `tilemap_uniform_data` (vec2 @0, lit
    /// (f32) @8 and lit_layers (f32) @12 — the bytes the padding gap
    /// already held — vec4 @16,
    /// 16-byte aligned; 32 bytes total). This is the GPU-side mirror of
    /// the writer, so a layout drift on either side fails a test.
    #[test]
    fn tilemap_uniform_offsets_match_the_cpu_layout() {
        let module = naga::front::wgsl::parse_str(TILEMAP_SHADER)
            .expect("tilemap.wgsl should parse (see all_shaders_parse_as_wgsl)");
        let ty = module
            .types
            .iter()
            .find_map(|(_, ty)| match &ty.inner {
                naga::TypeInner::Struct { .. } if ty.name.as_deref() == Some("TileMapUniforms") => {
                    Some(ty)
                }
                _ => None,
            })
            .expect("tilemap.wgsl should declare the TileMapUniforms struct");
        let (offsets, total_size) = match &ty.inner {
            naga::TypeInner::Struct { members, span } => (
                members.iter().map(|m| m.offset).collect::<Vec<u32>>(),
                *span,
            ),
            _ => unreachable!("TileMapUniforms must be a struct"),
        };
        assert_eq!(offsets, [0, 8, 12, 16]);
        // The struct's total size must equal the CPU-side buffer length,
        // otherwise the buffer would be too short or carry dead bytes.
        assert_eq!(total_size, 32);
    }

    /// The member offsets WGSL assigns to `PolylineUniforms` must match the
    /// CPU-side uniform writer in `polyline_uniform_data`:
    /// array<vec4<f32>, 128> @0 — 128 points at 16 bytes each, 2048 bytes —
    /// then vec4 @2048 (16 bytes, 16-byte aligned), u32 @2064, and f32
    /// @2068; 2080 bytes total. The points array must stay fixed at 128: the
    /// buffer size is compiled into the pipeline. Its elements must stay
    /// vec4s: the uniform address space requires an array member's stride to
    /// be a multiple of 16 bytes, and a vec2's stride is only 8 (the
    /// device-side validation rejects it even though it parses). This is the
    /// GPU-side mirror of the writer, so a layout drift on either side fails
    /// a test.
    #[test]
    fn polyline_uniform_offsets_match_the_cpu_layout() {
        let module = naga::front::wgsl::parse_str(POLYLINE_SHADER)
            .expect("polyline.wgsl should parse (see all_shaders_parse_as_wgsl)");
        let ty = module
            .types
            .iter()
            .find_map(|(_, ty)| match &ty.inner {
                naga::TypeInner::Struct { .. }
                    if ty.name.as_deref() == Some("PolylineUniforms") =>
                {
                    Some(ty)
                }
                _ => None,
            })
            .expect("polyline.wgsl should declare the PolylineUniforms struct");
        let (offsets, total_size, first) = match &ty.inner {
            naga::TypeInner::Struct { members, span } => (
                members.iter().map(|m| m.offset).collect::<Vec<u32>>(),
                *span,
                members
                    .first()
                    .expect("PolylineUniforms should declare its members"),
            ),
            _ => unreachable!("PolylineUniforms must be a struct"),
        };
        assert_eq!(offsets, [0, 2048, 2064, 2068]);
        // The struct's total size must equal the CPU-side buffer length,
        // otherwise the buffer would be too short or carry dead bytes.
        assert_eq!(total_size, 2080);
        // The points array is fixed-size (not dynamic): its 128 elements
        // are part of the pipeline's binding layout, and each element must
        // be a vec4<f32> (the uniform-space stride rule, see above).
        let points_ty = &module.types[first.ty];
        match &points_ty.inner {
            naga::TypeInner::Array {
                size: naga::ArraySize::Constant(n),
                base,
                ..
            } => {
                assert_eq!(n.get(), 128);
                let base_ty = &module.types[*base];
                assert!(
                    matches!(
                        base_ty.inner,
                        naga::TypeInner::Vector {
                            size: naga::VectorSize::Quad,
                            scalar: naga::Scalar {
                                kind: naga::ScalarKind::Float,
                                width: 4,
                            },
                        },
                    ),
                    "the points array element should be vec4<f32>, got {:?}",
                    base_ty.inner,
                );
            }
            other => panic!("the points array should be fixed at 128 elements, got {other:?}"),
        }
    }

    /// The member offsets WGSL assigns to `LightField` must match the
    /// CPU-side packer in `pack_light_field`: u32 @0, array<u32, 3> @4,
    /// vec4 @16 — the 32-byte header — and the light array tail at 32. The
    /// tail must be an unsized (dynamic) array: its length comes from the
    /// bound buffer, so the CPU can hold any light count with no
    /// MAX_LIGHTS. This is the GPU-side mirror of the packer's offset tests
    /// in backend/tests/fields.rs.
    #[test]
    fn light_field_header_offsets_match_the_packer() {
        let module = naga::front::wgsl::parse_str(SHAPE_SHADER)
            .expect("shape.wgsl should parse (see all_shaders_parse_as_wgsl)");
        let ty = module
            .types
            .iter()
            .find_map(|(_, ty)| match &ty.inner {
                naga::TypeInner::Struct { .. } if ty.name.as_deref() == Some("LightField") => {
                    Some(ty)
                }
                _ => None,
            })
            .expect("shape.wgsl should declare the LightField struct");
        let (offsets, span, last) = match &ty.inner {
            naga::TypeInner::Struct { members, span } => (
                members.iter().map(|m| m.offset).collect::<Vec<u32>>(),
                *span,
                members
                    .last()
                    .expect("LightField should declare its members"),
            ),
            _ => unreachable!("LightField must be a struct"),
        };
        assert_eq!(offsets, [0, 4, 16, 32]);
        // Naga books a dynamic array's footprint as one element (its
        // creation-time minimum), so the span is the 32-byte header plus
        // one 16-byte vec4 — wgpu's minimum binding size for the struct.
        // The real buffer length comes from the bound buffer, sized by the
        // CPU to fit the frame's lights, but never below this span.
        assert_eq!(span as usize, LIGHT_FIELD_BUFFER_MIN);
        // The tail must be a dynamic array, sized by the bound buffer.
        let last_ty = &module.types[last.ty];
        match &last_ty.inner {
            naga::TypeInner::Array {
                size: naga::ArraySize::Dynamic,
                ..
            } => {}
            other => panic!("the LightField tail should be a dynamic array, got {other:?}"),
        }
    }

    /// The member offsets WGSL assigns to `OccluderField` must match the
    /// CPU-side packer in `pack_occluder_field`: u32 @0 (the record
    /// count), u32 @4 (the light count), array<u32, 2> @8 — the 16-byte
    /// header together — and the occluder array tail at 16, where the
    /// per-light range table starts and the record blocks follow it. The
    /// tail must be an unsized (dynamic) array: its length comes from the
    /// bound buffer, so the CPU can hold any table and any block layout.
    /// This is the GPU-side mirror of the packer's offset tests in
    /// backend/tests/fields.rs.
    #[test]
    fn occluder_field_header_offsets_match_the_packer() {
        let module = naga::front::wgsl::parse_str(SHAPE_SHADER)
            .expect("shape.wgsl should parse (see all_shaders_parse_as_wgsl)");
        let ty = module
            .types
            .iter()
            .find_map(|(_, ty)| match &ty.inner {
                naga::TypeInner::Struct { .. } if ty.name.as_deref() == Some("OccluderField") => {
                    Some(ty)
                }
                _ => None,
            })
            .expect("shape.wgsl should declare the OccluderField struct");
        let (offsets, span, last) = match &ty.inner {
            naga::TypeInner::Struct { members, span } => (
                members.iter().map(|m| m.offset).collect::<Vec<u32>>(),
                *span,
                members
                    .last()
                    .expect("OccluderField should declare its members"),
            ),
            _ => unreachable!("OccluderField must be a struct"),
        };
        assert_eq!(offsets, [0, 4, 8, 16]);
        // Naga books a dynamic array's footprint as one element (its
        // creation-time minimum), so the span is the 16-byte header plus
        // one 16-byte vec4 — wgpu's minimum binding size for the struct.
        // The real buffer length comes from the bound buffer, sized by the
        // CPU to fit the frame's occluders, but never below this span.
        assert_eq!(span as usize, OCCLUDER_FIELD_BUFFER_MIN);
        // The tail must be a dynamic array, sized by the bound buffer.
        let last_ty = &module.types[last.ty];
        match &last_ty.inner {
            naga::TypeInner::Array {
                size: naga::ArraySize::Dynamic,
                ..
            } => {}
            other => panic!("the OccluderField tail should be a dynamic array, got {other:?}"),
        }
    }

    /// The CPU cull bounds a directional light's swept region with the
    /// same reach the shader runs its parallel shadow rays back by: if
    /// the two `SHADOW_RANGE`s ever drift, the packer would cull
    /// occluders the rays still reach — a lost shadow, silently. The
    /// shader's constant is parsed out of its own text rather than
    /// restated, so the pin cannot rot alongside a typo.
    #[test]
    fn shadow_range_mirrors_the_shader() {
        let text = crate::backend::SHADOW_RANGE.to_string();
        let marker = "const SHADOW_RANGE: f32 = ";
        let line = SHAPE_SHADER
            .lines()
            .find(|l| l.starts_with(marker))
            .expect("lighting.wgsl declares SHADOW_RANGE");
        let value = line
            .trim_start_matches(marker)
            .trim_end_matches(';')
            .trim_end_matches(".0")
            .parse::<f32>()
            .expect("SHADOW_RANGE parses as a float");
        assert_eq!(value.to_string(), text, "the two SHADOW_RANGEs drifted");
    }

    /// The watched table mirrors the compiled consts: a file read
    /// from disk assembles exactly the text `include_str!`
    /// assembled into the binary. This is the seam between the
    /// watched world and the shipped one — if it drifts, a game
    /// would run different shaders after a save than after a
    /// rebuild.
    #[test]
    fn the_watched_table_speaks_the_compiled_truth() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
        let compiled: Vec<(&str, &str)> = all_shaders()
            .into_iter()
            .map(|(name, source)| (name.trim_end_matches(".wgsl"), source))
            .collect();
        for (name, files) in SOURCES {
            let mut assembled = String::new();
            for file in *files {
                assembled.push_str(
                    &std::fs::read_to_string(dir.join(format!("{file}.wgsl")))
                        .unwrap_or_else(|err| panic!("{file}.wgsl: {err}")),
                );
            }
            let const_source = compiled
                .iter()
                .find(|(n, _)| *n == *name)
                .map(|(_, s)| *s)
                .unwrap_or_else(|| panic!("no compiled source named {name}"));
            assert_eq!(assembled, const_source, "{name} assembles differently");
        }
    }

    /// The watch, end to end on a scratch directory: an edit that
    /// compiles swaps in, and one that does not is refused with
    /// the good live text still standing.
    #[cfg(debug_assertions)]
    #[test]
    fn a_touched_shader_is_vetted_then_swapped() {
        let dir = std::env::temp_dir().join(format!("frost_shaders_{}", std::process::id()));
        let shipped = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders");
        std::fs::create_dir_all(&dir).unwrap();
        for entry in std::fs::read_dir(&shipped).unwrap() {
            let path = entry.unwrap().path();
            std::fs::copy(&path, dir.join(path.file_name().unwrap())).unwrap();
        }
        let mut live = Live::new(dir.clone());
        assert!(!live.poll(), "a fresh copy changes nothing");
        let mut sprite = std::fs::read(dir.join("sprite.wgsl")).unwrap();
        sprite.extend_from_slice(b"\n// touched by the test\n");
        std::fs::write(dir.join("sprite.wgsl"), &sprite).unwrap();
        assert!(live.poll());
        assert!(
            live.get("sprite")
                .is_some_and(|src| src.contains("touched by the test"))
        );
        std::fs::write(dir.join("sprite.wgsl"), b"fn broken(({{{").unwrap();
        assert!(!live.poll(), "a broken batch is refused");
        assert!(
            live.get("sprite")
                .is_some_and(|src| src.contains("touched by the test"))
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
