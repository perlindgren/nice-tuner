use std::borrow::Cow;
use std::collections::{HashMap, HashSet};

use std::num::NonZeroU64;
use std::sync::Arc;
// `block_on`'s machinery: the embedded engine asks an instance for an
// adapter once, at startup, and nothing else here ever awaits.

use wgpu::{
    Adapter, AddressMode, BindGroup, BindGroupDescriptor, BindGroupEntry, BindGroupLayout,
    BindGroupLayoutDescriptor, BindGroupLayoutEntry, BindingResource, BindingType, Buffer,
    BufferBinding, BufferBindingType, BufferDescriptor, BufferUsages, ColorTargetState,
    ColorWrites, CommandEncoderDescriptor, CurrentSurfaceTexture, Device, Extent3d, FilterMode,
    FragmentState, MipmapFilterMode, MultisampleState, PipelineCompilationOptions,
    PipelineLayout, PipelineLayoutDescriptor, PrimitiveState, Queue,
    RenderPassColorAttachment, RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor,
    Sampler, SamplerBindingType, SamplerDescriptor, ShaderModule, ShaderModuleDescriptor,
    ShaderSource, ShaderStages, StoreOp, Surface, SurfaceConfiguration, SurfaceTexture,
    TexelCopyBufferLayout, TexelCopyTextureInfo, TextureAspect, TextureDescriptor,
    TextureDimension, TextureFormat, TextureSampleType, TextureUsages, TextureView,
    TextureViewDescriptor, TextureViewDimension, VertexState,
};
use winit::dpi::PhysicalSize;
use winit::event::MouseButton;

use crate::backend::bloom::Bloom;
use crate::backend::copy::pad_rows;
use crate::backend::frame::*;
use crate::backend::windowing::*;
use crate::objects::*;
use crate::text;
use crate::{Canvas, Context, KeyCode, Process};

/// The frame's offscreen target: the texture the scene renders into,
/// paired with the pass that carries it to the window (the texture as
/// shader source, the sampler — `EmbeddedConfig::blit_filter` — that stretches it,
/// and the letterbox fit uniform), so the stretch is a `set_bind_group` and
/// three vertices. Two reasons for one: fixed render size mode (see
/// `EmbeddedConfig::render_size`), where the buffer is smaller or larger than the
/// window, and the float frame buffer ([`Frost::hdr_frames`]), where the
/// window cannot hold what the shaders write. The blit samples the buffer
/// and writes the surface, so neither end re-encodes what the other encoded.
struct BlitTarget {
    /// The size this target was built for; a surface of exactly this size
    /// renders to directly and leaves the target unused.
    size: [u32; 2],
    /// The filter this target's sampler samples with; a runtime change to
    /// the engine's filter (see [`Context::set_blit_filter`]) retires the
    /// cached target, and the next stretched frame builds a fresh sampler.
    ///
    /// [`Context::set_blit_filter`]: crate::Context::set_blit_filter
    filter: SpriteFilter,
    /// The texture behind `view`: nothing reads it after construction (the
    /// view and bind group keep the GPU resource alive on their own), but
    /// holding it makes the target's ownership obvious.
    _texture: wgpu::Texture,
    view: TextureView,
    /// The frame's emissive target, present exactly when the scene pipelines
    /// carry a second output: the same size as `view`, the same float format,
    /// and holding what each surface puts out rather than what the lights
    /// made of it. Nothing reads it yet — bloom will — but it is drawn into
    /// and cleared with every frame, because a target nobody writes is a
    /// second full-res allocation spent for a look that never arrived.
    emissive: Option<(wgpu::Texture, TextureView)>,
    /// The bloom chain: this target's emissive target, blurred once per preset
    /// level, ready for the stretch pass to add back. Dormant — one black
    /// texel and weights of zero — when there is no bloom to do; the present
    /// pass binds the plumbing's full width either way, so this is never absent.
    bloom: Bloom,
    /// The 16-byte letterbox fit (`vec2` + padding), rewritten by `render`
    /// whenever the surface's aspect may have moved.
    /// The format the frame buffer was built in, which is the format every
    /// bloom level is built in too. Stored because a chain woken later has to
    /// match it, and the answer that decides it (`float_target`) depends on the
    /// adapter, not on the frame.
    format: TextureFormat,
    /// The stretch pass's own sampler, kept for the same reason the bind group
    /// is: a chain woken mid-run rebinds it rather than making a second one.
    sampler: Sampler,
    fit_buffer: Buffer,
    bind_group: BindGroup,
}

pub(crate) struct Frost<P: Process> {
    /// Capability questions only — the host's device does the work; this
    /// adapter is what `Surface::get_capabilities` and texture-format
    /// features are asked of (see `crate::embedded::Embedded::new`).
    adapter: Adapter,
    device: Device,
    queue: Queue,
    /// The app's vsync wish, kept live for [`Context::vsync`]; an
    /// embedding presents uncapped whatever it says (the surface is always
    /// configured `AutoNoVsync`), because a plugin must not block its
    /// host's main thread.
    vsync: bool,
    /// The `EmbeddedConfig`'s fixed render size in physical pixels, or
    /// `None` to render at the surface's own resolution (see
    /// `EmbeddedConfig::render_size`).
    render_size: Option<[u32; 2]>,
    /// The filter the stretch pass samples the render buffer with when its
    /// size differs from the surface (see `EmbeddedConfig::blit_filter`).
    blit_filter: SpriteFilter,
    /// The glow the app asked for (`EmbeddedConfig::bloom`, live through
    /// [`Context::set_bloom`](crate::Context::set_bloom)): a strength, zero
    /// for none. It is read every frame, so turning a glow on or off mid-run is
    /// one assignment; what is not free is asking for a glow on a frame that
    /// was built without a chain, which the render path pays for on that frame.
    bloom: f32,
    /// The headroom the app asked for (`EmbeddedConfig::tone_map`, live through
    /// [`Context::set_tone_map`](crate::Context::set_tone_map)): the share of
    /// the display range kept for light above white, zero for none. It reaches
    /// the shader through the stretch pass's own uniform, which is rewritten
    /// every frame, so asking for a different headroom costs nothing and lands
    /// immediately — unlike [`Self::bloom`], which may have to build the chain
    /// that a glow is made of before the number means anything.
    tone_map: f32,
    /// The surface's logical size — the user space the canvas draws in,
    /// which `pixel_size` turns back into physical pixels through
    /// `scale`. The host reports the physical size; this is the same
    /// number the old window handler derived from the OS's inner size.
    logical_size: (u32, u32),
    scale: f32,
    surface: Option<Surface<'static>>,
    /// The surface format the host handed over, kept so every
    /// reconfiguration (resize, lost-and-replaced) presents in one
    /// consistent format, upgraded once through `prefer_srgb_format`.
    /// `None` until a surface is attached.
    surface_format: Option<TextureFormat>,
    /// The current spell of frames the surface would not hand over: the
    /// reason first seen, and how many frames it has cost.
    ///
    /// A surface that will not answer is ordinary — a fullscreen transition
    /// loses it for seconds at a stretch — and it is worth one line, not one
    /// per frame. Measured: three seconds of a macOS fullscreen exit printed
    /// 9,420 identical lines, which is an hour of log in an afternoon and
    /// buries the handful of lines that actually explain the pause.
    surface_gone: Option<(&'static str, u32)>,
    /// The last size the resize line spoke about, so a window told its own
    /// size twice by a chatty compositor does not report that it resized
    /// twice. The work `resize` does is not skipped, only the sentence.
    resize_logged: (u32, u32, u32, u32),
    /// A size the window asked for that the engine has not accepted yet, with
    /// the moment it was asked. See [`Frost::note_resize`].
    resize_pending: PendingResize,
    line_pipeline: Option<RenderPipeline>,
    polyline_pipeline: Option<RenderPipeline>,
    circle_pipeline: Option<RenderPipeline>,
    rect_pipeline: Option<RenderPipeline>,
    shape_pipeline: Option<RenderPipeline>,
    sprite_pipeline: Option<RenderPipeline>,
    /// The same sprite shader blended additively, for the draws whose nodes ask
    /// to add their light instead of covering with it. See
    /// [`ADDITIVE_BLEND`].
    sprite_additive_pipeline: Option<RenderPipeline>,
    /// Pipelines for game-authored programs — the WGSL a game hands to
    /// [`crate::Shape::with_program`] — one per distinct source, compiled
    /// over the sprite pipeline's own bind group layout so the arena
    /// uniform, the texture caches, and both light fields bind exactly as
    /// they do for the engine's sprites. Built at first sight of a source
    /// and dropped whenever the frame's format changes, which is the only
    /// thing any of them was ever built against.
    custom_pipelines: HashMap<Arc<str>, RenderPipeline>,
    /// The sprite pipeline's layout, kept past pipeline creation for the
    /// one reader that outlives setup: the user-program builder, which
    /// must build its pipelines on the very layout the sprite's bind
    /// groups were cut from. Every other pipeline drops its layout as
    /// soon as the pipeline exists.
    sprite_pipeline_layout: Option<PipelineLayout>,
    /// The format and emissive target the scene pipelines were built
    /// against this surface spell, remembered so a program compiled
    /// mid-frame cannot disagree with the pass it draws into.
    frame_format: Option<TextureFormat>,
    frame_emissive: Option<TextureFormat>,
    particle_pipeline: Option<RenderPipeline>,
    /// The tile map pipeline: one instanced quad per tile, sampling the
    /// batch's atlas through per-instance UV bounds.
    tilemap_pipeline: Option<RenderPipeline>,
    /// Stretches the fixed render size buffer over the window (see
    /// `EmbeddedConfig::render_size`); unused when the frame renders at the
    /// window's own resolution.
    blit_pipeline: Option<RenderPipeline>,
    /// The frame's offscreen target, built on the first frame that needs
    /// one: absent only when the frame renders at the window's own size
    /// *and* the surface can hold the frame itself.
    blit_target: Option<BlitTarget>,
    /// The index buffer for particle batches: one quad per instance
    /// (`[0, 1, 2, 2, 1, 3]`), created once and shared by every batched
    /// draw in every frame.
    particle_index_buffer: Option<Buffer>,
    /// A 1x1 white placeholder texture and its sampler, bound by every
    /// non-sprite particle batch: the particle pipeline's bind group always
    /// carries a texture (binding 2) and a sampler (binding 3), but the SDF
    /// kinds — circles and rectangles — never sample it.
    particle_placeholder: (TextureView, Sampler),
    /// The two sampling filters, one sampler each, shared by every
    /// cached texture: a tileset shown nearest on a tile map and
    /// bilinear as a sprite uploads ONE texture and picks its sampler
    /// by filter — the samplers depend on nothing else.
    sampler_linear: Sampler,
    sampler_nearest: Sampler,
    /// The frame's light field storage buffer: a 32-byte header plus one
    /// 48-byte record per light at the peak count so far. It starts at
    /// the buffer's minimum size — the header plus one vec4, the WGSL
    /// layout's minimum binding size — because a lightless frame still
    /// binds it (the shaders read the field for the ambient), and it only
    /// ever grows: the WGSL field's tail is an unsized array sized by the
    /// bound buffer, so a bigger buffer simply holds more records, and a
    /// lighter frame writes a shorter slice into the same buffer.
    field_buffer: Buffer,
    /// The frame's occluder field storage buffer: a 16-byte header plus one
    /// 48-byte record per occluder at the peak count so far. Same
    /// grow-only rationale as [`Frost::field_buffer`]; a frame with no
    /// occluders still binds it (the shaders read the field's count), and
    /// the WGSL field's tail is an unsized array sized by the bound buffer.
    occluder_buffer: Buffer,
    /// The GPU resources for each distinct sprite image, keyed by the
    /// `(pointer, generation)` of its pixel-data `Arc`: the pointer alone
    /// is not enough, because a freed buffer's address can be reused for a
    /// different buffer (an atlas repack, a rebuilt in-memory image).
    /// Sprite constructors stamp a new generation on every construction,
    /// and atlases bump theirs on a repack, so a stale texture is never
    /// served for new pixels. Sprites sharing one file share one texture,
    /// so the map stays bounded by the number of distinct images. A
    /// `TextureView` keeps its texture alive, so only the view and the
    /// sampler are stored. When an atlas repacks, its stale entry is
    /// evicted right after the text is expanded; a sprite buffer's stale
    /// entry is evicted lazily, when a new buffer recycles its address.
    sprite_resources: HashMap<(u64, u64), TextureView>,
    /// The frame's uniforms, all of them, in one buffer: every draw that has a
    /// uniform owns a run of it, the frame uploads the whole thing once, and
    /// each draw slides its pipeline's bind group window to its own run with a
    /// dynamic offset.
    ///
    /// This exists because a uniform buffer and a bind group per draw is the
    /// dearest CPU cost a frame pays: measured on an Apple M4 Max in a
    /// release build, 6.9 microseconds a draw against 0.4 with the arena — a
    /// 243-draw frame of glyph quads went from 2.1 ms to 0.3 ms. (A pool of
    /// buffers reused across frames was measured too, and lands in between
    /// at 3.4 microseconds a draw: the per-draw `write_buffer` survives, and
    /// it is half the cost. See the `draw_cost` bench.)
    ///
    /// The buffer is grow-only and reused between frames, so the only work
    /// a frame does here is one `write_buffer` of the bytes it already built.
    /// It never needs to shrink: an idle run is bytes nobody reads.
    uniform_arena: Option<Buffer>,
    /// Byte alignment of an arena run: the device's uniform-offset alignment
    /// (256 on every backend so far), since a dynamic offset must be a
    /// multiple of it. Runs are padded up to this width, not fixed at it — a
    /// polyline's 2080 bytes and a rectangle's 32 sit side by side, each
    /// paying for its own uniform rather than for the largest one.
    arena_stride: u64,
    /// Where each draw's bytes are: one offset per entry of the frame's draw
    /// list, written by the arena walk before the pass and read by the draw
    /// itself. Offsets are data rather than arithmetic repeated on both sides
    /// of the pass because the runs are not one size — only the walk that
    /// packed them can say where a draw landed.
    arena_offsets: Vec<u32>,
    /// The sprite bind groups this engine has built, one per atlas and
    /// sampling filter: the arena's window, the texture, the sampler and the
    /// two frame fields, all of which outlive a frame. A draw's own bytes
    /// arrive through the dynamic offset, so unlike the old per-draw groups
    /// these are worth keeping — a frame with three images builds three, not
    /// one per draw. Cleared when the arena or a field buffer is rebuilt.
    sprite_binds: HashMap<(u64, u64, bool), BindGroup>,
    /// The bind groups for the pipelines whose entire per-draw state is one
    /// uniform — line, polyline, circle, rectangle, and shape, which adds the
    /// two frame fields — keyed by the pipeline's label. Same reasoning as
    /// `sprite_binds`: the offset changes per draw, the group does not.
    prim_binds: HashMap<&'static str, BindGroup>,
    /// The frame's arena bytes, assembled here and uploaded once, kept
    /// between frames for its capacity alone.
    arena_staging: Vec<u8>,
    /// The rasterized glyph atlas for each distinct `(font, size, weight)`
    /// triple, keyed by the font buffer's pointer, the size's bits and the
    /// weight's bits. Kept between
    /// frames so unchanged text never re-rasterizes and its pixel buffer —
    /// and therefore the GPU texture in `sprite_resources` — keeps a stable
    /// identity. A repack (a newly packed glyph) replaces the buffer and
    /// bumps its generation; the stale texture is evicted from
    /// `sprite_resources` right after the text is expanded.
    text_atlases: HashMap<(u64, u32, u32), text::Atlas>,
    /// The surface format the current pipelines were built for; they are only
    /// rebuilt when this changes.
    format: Option<TextureFormat>,
    /// True while the presentation surface stores raw linear values —
    /// its format has no hardware sRGB encoding (a browser canvas may
    /// offer no sRGB format, leaving `prefer_srgb_format` nothing to
    /// prefer). The frame then always renders through the offscreen
    /// buffer and the blit pass gamma-encodes on the way to the
    /// window; the shaders' linear-light output has nowhere else to
    /// become the encoded bytes a display expects. An sRGB surface
    /// (`false`, every native surface so far) keeps the direct path.
    linear_surface: bool,
    /// Whether frames render into a float buffer instead of the surface.
    ///
    /// The scene already writes linear light, and an 8-bit sRGB surface
    /// rounds it on the way in: a value over 1.0 — a lamp's core, two
    /// lights landing on one pixel, an emission pushed past white — clips
    /// anything downstream can see it, and every blend composites values
    /// that were already rounded. A float buffer holds the overshoot and
    /// blends without rounding, which is what the lighting model was
    /// written assuming. The cost is one more pass over the window per
    /// frame and twice the bytes of the target, so it is taken only where
    /// the adapter offers `Rgba16Float` as an attachment; where it does
    /// not, frames go to the surface directly, exactly as before.
    hdr_frames: bool,
    /// The watched shader directory, debug builds only: `render` polls
    /// it, and a passing batch sets `format` to `None` so the pipelines
    /// rebuild by the same road a surface-format change always took.
    #[cfg(debug_assertions)]
    shaders: crate::shaders::Live,
    /// Millisecond timestamp of the previous rendered frame, used to
    /// compute `dt` (see `now_millis`).
    last_millis: Option<f64>,
    /// The CPU time in milliseconds the engine spent processing the last
    /// completed frame: from just after the frame's surface acquire (the
    /// demo's `process` callback, the scene update, the draw list, and the
    /// command encoding included) to the submission of the frame's command
    /// buffer to the queue — the surface acquire, the present handoff, and
    /// the wait for the next frame excluded. `0.0` before the first frame
    /// completes. Reported to the next frame's `Context` (see
    /// `Context::frame_processing_ms`).
    frame_processing_ms: f64,
    /// The GPU draw calls the engine issued for the last completed frame:
    /// one per `pass.draw` / `pass.draw_indexed` that actually executed —
    /// draws skipped as off-screen, background, light, or empty (a
    /// degenerate transform, an empty particle batch) are not counted.
    /// `0` before the first frame completes. Reported to the next frame's
    /// `Context` (see `Context::frame_draw_calls`).
    frame_draw_calls: u32,
    /// The GPU draw calls the last completed frame spent on the
    /// `Diagnostics` overlay's own nodes: the tagged subset of
    /// `frame_draw_calls` (see `Context::frame_diagnostic_draw_calls`).
    frame_diagnostic_draw_calls: u32,
    /// The expected frame rate in Hz. `None` for an embedding: the host's
    /// timer sets the cadence and frost is never pinned to a refresh
    /// (see `Context::expected_fps`).
    expected_fps: Option<f32>,
    /// The scene drawn every frame, after the process runs.
    scene: Scene,
    /// The physical keys currently held down, updated as the host reports
    /// presses and releases.
    keys: HashSet<KeyCode>,
    /// The character each held physical key typed as it went down, under
    /// the user's current keyboard layout — `Context::char_down` reads it.
    /// Keyed by the physical code, so a release removes exactly what its
    /// press recorded no matter which modifiers moved in between, and the
    /// character a shortcut matches never changes while the key is held.
    /// The host supplies the character with `Embedded::add_text`.
    typed: HashMap<KeyCode, char>,
    /// The key whose press the host has not yet given a character for —
    /// the key an [`Embedded::add_text`](crate::embedded::Embedded::add_text)
    /// call answers.
    last_pressed: Option<KeyCode>,
    /// Raised by the process through `Context::exit`: the frame is
    /// drawn as asked, and the host — owner of the window — is told
    /// (`Embedded::exit_requested`).
    exit_requested: bool,
    /// Whether the frame's `Context` key reads answer dead (see
    /// `Context::set_keys_frozen`).
    keys_frozen: bool,
    /// The mouse cursor's position in physical pixels, `(0, 0)` at the
    /// surface's upper-left corner, or `None` when the cursor is outside the
    /// surface. Updated as the host reports cursor motion; `render` maps it
    /// into the frame's user space (logical pixels, or the fixed render
    /// size) on every frame.
    mouse: Option<[f32; 2]>,
    /// The mouse buttons currently held down, updated as the host reports
    /// presses and releases.
    mouse_buttons: HashSet<MouseButton>,
    /// The mouse wheel's vertical movement in lines, accumulated as the
    /// host reports wheel steps since the previous frame and reset after
    /// the frame's `Context` reports it (see `Context::mouse_wheel`).
    mouse_wheel: f32,
    process: P,
}

/// A bind group layout and pipeline layout for a pipeline whose entire
/// per-draw state is one uniform: that uniform, reached by a dynamic offset
/// and bound `size` bytes wide, which is the width of the arena run the draw
/// owns — so a draw reads its own parameters and nothing past them.
///
/// Written out rather than derived for the same reason as [`sprite_layouts`],
/// and by the same argument: an auto layout reads the shader's
/// `var<uniform>` as a fixed binding and leaves a dynamic offset nowhere to
/// go. wgpu validates the pair against the shader when the pipeline is built,
/// which happens on every frame of every test, so a uniform that changes size
/// or a binding that moves fails there.
pub(crate) fn slot_layouts(device: &Device, label: &'static str, size: u64) -> PipelineLayout {
    let bind = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[BindGroupLayoutEntry {
            binding: 0,
            visibility: ShaderStages::FRAGMENT,
            ty: BindingType::Buffer {
                ty: BufferBindingType::Uniform,
                has_dynamic_offset: true,
                min_binding_size: NonZeroU64::new(size),
            },
            count: None,
        }],
    });
    device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&bind)],
        immediate_size: 0,
    })
}

/// The shape pipeline's layout: the arena window above, plus the two frame
/// fields it reads for lighting at bindings 1 and 2, which are shape.wgsl's
/// own. The fields are bound whole — a storage buffer takes no dynamic offset,
/// and there is one of each per frame — so a shape's bind group is shared by
/// every shape in the frame even though the uniform inside it is not.
pub(crate) fn shape_layouts(device: &Device, label: &'static str) -> PipelineLayout {
    let bind = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: NonZeroU64::new(SHAPE_UNIFORM_BYTES),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(&bind)],
        immediate_size: 0,
    })
}

/// The sprite pipeline's bind group layout and pipeline layout, written
/// out instead of derived. The reason is binding 0: the frame's uniform
/// arena is reached by a *dynamic offset*, and an auto layout reads the
/// shader's `var<uniform>` as a fixed one, leaving the offset nowhere to
/// go. Every other entry is sprite.wgsl's own — the image, the sampler,
/// and the two frame fields, all fragment-stage — copied one for one, so
/// the pair is only correct while it matches the file; wgpu validates
/// them against each other when the pipeline is built, which is every
/// frame of every test, and a binding that moves or changes type fails
/// there rather than painting anything wrong.
/// How an additive draw lands in the two attachments: its light is added,
/// scaled by its own alpha, and its alpha is added too.
///
/// The house rule about what the two channels mean still holds — rgb is
/// radiance, alpha is coverage, the blend does the rest — and only the last
/// clause changes its reading. Coverage here is not how much of the world
/// the fragment hides but how much light it deposits, which is what an
/// authored glow sprite is drawn out of: a ramp that fades to nothing at its
/// edge has to *add* its falloff to the wall behind it, because a wall
/// covered by a dark bell is a wall with a hole in it rather than a wall in
/// lamp-light. It applies to both attachments for the same reason the alpha
/// rule does — the frame's color and the emissive channel are composited on
/// one set of terms — so one sprite is at once the visible core of a source
/// and the light the chain goes on to blur.
///
/// Alpha adds with a factor of one rather than `SrcAlpha`, so coverage
/// accumulates in step with the light it is describing instead of being
/// squared on the way in. A frame's alpha can therefore pass 1.0 under a
/// glow; it is written to the window, where it clamps, and nothing inside
/// the engine reads it.
///
/// This is not [`wgpu::BlendState::ADDITIVE`], which adds with a source
/// factor of one and so assumes rgb that has been premultiplied by alpha
/// already. Frost's textures carry straight alpha — the ramp lives in the
/// alpha channel and the rgb beside it stays the lamp's color at full — so
/// the ramp has to scale the rgb in the blend or the transparent border of a
/// glow sprite adds a square of light around it.
pub(crate) const ADDITIVE_BLEND: wgpu::BlendState = wgpu::BlendState {
    color: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::SrcAlpha,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
    alpha: wgpu::BlendComponent {
        src_factor: wgpu::BlendFactor::One,
        dst_factor: wgpu::BlendFactor::One,
        operation: wgpu::BlendOperation::Add,
    },
};

pub(crate) fn sprite_layouts(device: &Device) -> (BindGroupLayout, PipelineLayout) {
    let bind = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
        label: Some("sprite bind group layout"),
        entries: &[
            BindGroupLayoutEntry {
                binding: 0,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    // A sprite uniform, and no less: the bound window is
                    // what the offset slides, so this is also the only
                    // part of the arena a draw can read.
                    min_binding_size: NonZeroU64::new(SPRITE_UNIFORM_BYTES),
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 1,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Texture {
                    sample_type: TextureSampleType::Float { filterable: true },
                    view_dimension: TextureViewDimension::D2,
                    multisampled: false,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 2,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Sampler(SamplerBindingType::Filtering),
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 3,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            BindGroupLayoutEntry {
                binding: 4,
                visibility: ShaderStages::FRAGMENT,
                ty: BindingType::Buffer {
                    ty: BufferBindingType::Storage { read_only: true },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline = device.create_pipeline_layout(&PipelineLayoutDescriptor {
        label: Some("sprite pipeline layout"),
        bind_group_layouts: &[Some(&bind)],
        immediate_size: 0,
    });
    (bind, pipeline)
}

impl<P: Process> Frost<P> {
    /// The engine's initial state: the host's GPU is ready, but no
    /// surface is attached yet — [`Self::attach_surface`] brings one.
    #[allow(clippy::too_many_arguments)] // the full GPU setup and config arrive at once
    pub(crate) fn new(
        adapter: Adapter,
        device: Device,
        queue: Queue,
        render_size: Option<[u32; 2]>,
        blit_filter: SpriteFilter,
        bloom: f32,
        tone_map: f32,
        scene: Scene,
        process: P,
    ) -> Self {
        // The particle batch's index buffer is surface-format independent,
        // so it is built here: every instance is one quad,
        // `[0, 1, 2, 2, 1, 3]`. It is written before any draw can run, so a
        // `write_buffer` upload has no hazard.
        let particle_index_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("particle index buffer"),
            size: 24,
            usage: BufferUsages::INDEX | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut index_bytes = [0u8; 24];
        for (i, v) in [0u32, 1, 2, 2, 1, 3].iter().enumerate() {
            index_bytes[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
        }
        queue.write_buffer(&particle_index_buffer, 0, &index_bytes);

        // The light field buffer starts at its minimum size — the header
        // plus one vec4, the WGSL layout's minimum binding size: even a
        // frame with no lights binds it (the shaders read the field to
        // fetch the ambient), and it only grows when the light count peaks
        // higher.
        let field_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("light field buffer"),
            size: LIGHT_FIELD_BUFFER_MIN as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // The occluder field buffer starts at its minimum size, the same
        // WGSL minimum binding size as the light field buffer: even a
        // frame with no occluders binds it (the shaders read the field's
        // count), and it only grows when the occluder count peaks higher.
        let occluder_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("occluder field buffer"),
            size: OCCLUDER_FIELD_BUFFER_MIN as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // A sprite uniform slot must hold the uniform (96 bytes) and start on
        // a multiple of the device's uniform-offset alignment, because a
        // dynamic offset must be aligned: 256 bytes on every backend
        // measured so far.
        let arena_stride = {
            let align = device.limits().min_uniform_buffer_offset_alignment as u64;
            SPRITE_UNIFORM_BYTES.max(align).next_multiple_of(align)
        };

        // The particle pipeline's bind group always carries a texture and a
        // sampler: a sprite batch binds its image, and the SDF kinds —
        // circles and rectangles, which never sample — bind this 1x1 white
        // placeholder. It is surface-format independent, so it is built
        // here, like the particle index buffer.
        let placeholder_texture = device.create_texture(&TextureDescriptor {
            label: Some("particle placeholder texture"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // One texel is a pitch of 4, which WebGPU refuses as readily as an
        // ant's 3864 — see [`pad_rows`]. The placeholder pays the same row
        // padding as any authored picture, because a placeholder whose
        // upload was silently refused is a white square made of garbage.
        let (placeholder_pixels, placeholder_pitch) = pad_rows(&[255, 255, 255, 255], 1, 1, 4);
        queue.write_texture(
            TexelCopyTextureInfo {
                texture: &placeholder_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &placeholder_pixels,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(placeholder_pitch),
                rows_per_image: None,
            },
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let particle_placeholder = (
            placeholder_texture.create_view(&TextureViewDescriptor::default()),
            device.create_sampler(&SamplerDescriptor {
                label: Some("particle placeholder sampler"),
                address_mode_u: AddressMode::ClampToEdge,
                address_mode_v: AddressMode::ClampToEdge,
                address_mode_w: AddressMode::ClampToEdge,
                mag_filter: FilterMode::Linear,
                min_filter: FilterMode::Linear,
                mipmap_filter: MipmapFilterMode::Nearest,
                lod_min_clamp: 0.0,
                lod_max_clamp: f32::MAX,
                compare: None,
                anisotropy_clamp: 1,
                border_color: None,
            }),
        );
        // One sampler per filter, built once for the whole app: the
        // descriptors differ only in the filter mode, so every sprite,
        // glyph atlas, particle image, and tile map of the same filter
        // shares one sampler, and every texture of the same pixels
        // shares one texture — the two axes no longer entangled.
        let sampler_linear = device.create_sampler(&SamplerDescriptor {
            label: Some("linear sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            mipmap_filter: MipmapFilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: f32::MAX,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });
        let sampler_nearest = device.create_sampler(&SamplerDescriptor {
            label: Some("nearest sampler"),
            address_mode_u: AddressMode::ClampToEdge,
            address_mode_v: AddressMode::ClampToEdge,
            address_mode_w: AddressMode::ClampToEdge,
            mag_filter: FilterMode::Nearest,
            min_filter: FilterMode::Nearest,
            mipmap_filter: MipmapFilterMode::Nearest,
            lod_min_clamp: 0.0,
            lod_max_clamp: f32::MAX,
            compare: None,
            anisotropy_clamp: 1,
            border_color: None,
        });

        Self {
            adapter,
            device,
            queue,
            // An embedding presents uncapped, always (see the field): the
            // wish starts where the swap chain will end up.
            vsync: false,
            render_size,
            blit_filter,
            bloom,
            tone_map,
            logical_size: (0, 0),
            scale: 1.0,
            surface: None,
            surface_format: None,
            surface_gone: None,
            resize_logged: (0, 0, 0, 0),
            resize_pending: PendingResize::default(),
            line_pipeline: None,
            polyline_pipeline: None,
            circle_pipeline: None,
            rect_pipeline: None,
            shape_pipeline: None,
            sprite_pipeline: None,
            sprite_additive_pipeline: None,
            custom_pipelines: HashMap::new(),
            sprite_pipeline_layout: None,
            frame_format: None,
            frame_emissive: None,
            particle_pipeline: None,
            tilemap_pipeline: None,
            blit_pipeline: None,
            blit_target: None,
            particle_index_buffer: Some(particle_index_buffer),
            particle_placeholder,
            sampler_linear,
            sampler_nearest,
            field_buffer,
            occluder_buffer,
            sprite_resources: HashMap::new(),
            uniform_arena: None,
            arena_offsets: Vec::new(),
            prim_binds: HashMap::new(),
            // A dynamic offset must be a multiple of this alignment, and a
            // slot must hold the uniform: 96 bytes rounded up to a multiple
            // of the alignment, which is 256 on every backend measured so
            // far.
            arena_stride,
            sprite_binds: HashMap::new(),
            arena_staging: Vec::new(),
            text_atlases: HashMap::new(),
            format: None,
            linear_surface: false,
            hdr_frames: false,
            #[cfg(debug_assertions)]
            shaders: crate::shaders::Live::new(
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("shaders"),
            ),
            last_millis: None,
            frame_processing_ms: 0.0,
            frame_draw_calls: 0,
            frame_diagnostic_draw_calls: 0,
            expected_fps: None,
            scene,
            keys: HashSet::new(),
            typed: HashMap::new(),
            last_pressed: None,
            mouse: None,
            mouse_buttons: HashSet::new(),
            mouse_wheel: 0.0,
            process,
            exit_requested: false,
            keys_frozen: false,
        }
    }
}

impl<P: Process> Frost<P> {
    // ==================== input, as the event loop heard it ====================
    //
    // The embeddable engine has no event loop: the host translates its OS
    // events into the calls below, which move the same state the old
    // `WindowEvent` arms moved. Each setter is one arm, lifted.

    /// A key went down or came up, by physical code (the old
    /// `KeyboardInput` arm's `event.physical_key`). Pressing records the
    /// press; the character the press types, if the host has one, arrives
    /// separately through [`Self::note_key_char`]. A release removes the
    /// key and whatever character it typed.
    pub(crate) fn set_key(&mut self, code: KeyCode, down: bool) {
        if down {
            // Debug, not info: this arrives for every press and every
            // release, and a host that forwards its whole key stream
            // drowns an info-level log.
            log::debug!("key {code:?} down");
            // The character is the press's own work — `note_key_char`
            // attaches it right after — so a fresh press starts without
            // one rather than inheriting a stale entry.
            self.typed.remove(&code);
            self.keys.insert(code);
            self.last_pressed = Some(code);
        } else {
            log::debug!("key {code:?} up");
            self.keys.remove(&code);
            self.typed.remove(&code);
            if self.last_pressed == Some(code) {
                self.last_pressed = None;
            }
        }
    }

    /// The character the most recent press typed, under the user's
    /// keyboard layout — the old arm read it off `event.logical_key`; a
    /// host reads it off its own key event. The character attaches to
    /// the last key pressed while it is still held (press, then
    /// character, then release is the order an OS speaks), which keeps
    /// the release-removes-what-the-press-typed promise the map makes.
    /// A character with no live press is the host describing a key frost
    /// does not track, and passes.
    pub(crate) fn note_key_char(&mut self, ch: char) {
        if let Some(code) = self.last_pressed {
            if self.keys.contains(&code) {
                self.typed.insert(code, ch);
            }
        }
    }

    /// The cursor moved, given in *logical* pixels from the surface's
    /// upper-left corner, y down — the unit the host's own GUI counts
    /// in. It is stored as physical pixels, the unit the old
    /// `CursorMoved` arm kept (winit's cursor position was already
    /// physical); `render` flips it to user coordinates every frame.
    pub(crate) fn set_cursor(&mut self, x: f64, y: f64) {
        let scale = (self.scale as f64).max(0.01);
        self.mouse = Some([(x * scale) as f32, (y * scale) as f32]);
    }

    /// The cursor left the surface: held buttons drop (a release outside
    /// the window may never arrive) and the position reads as outside,
    /// exactly as the old `CursorLeft` arm did.
    pub(crate) fn cursor_left(&mut self) {
        self.mouse = None;
        self.mouse_buttons.clear();
    }

    /// A mouse button went down or came up.
    pub(crate) fn set_mouse_button(&mut self, button: MouseButton, down: bool) {
        if down {
            self.mouse_buttons.insert(button);
        } else {
            self.mouse_buttons.remove(&button);
        }
    }

    /// The mouse wheel moved, `dy` in lines — the old `LineDelta` unit.
    /// The old arm divided pixel deltas by 120 before adding them
    /// (120 is one notch of a Windows classic wheel); a host that only
    /// has pixel deltas should divide the same way before calling. Frost
    /// tracks no horizontal wheel, so the old arm's horizontal line is
    /// the whole story for `dx`.
    pub(crate) fn scroll(&mut self, dy: f64) {
        self.mouse_wheel += dy as f32;
    }

    /// The surface gained or lost the keyboard focus (the old `Focused`
    /// arm): losing it clears every held control, because the releases
    /// may never arrive.
    pub(crate) fn set_focused(&mut self, focused: bool) {
        if !focused {
            self.keys.clear();
            self.typed.clear();
            self.last_pressed = None;
            self.mouse = None;
            self.mouse_buttons.clear();
        }
    }

    /// One frame's work, as the old `RedrawRequested` arm did it: settle
    /// any resize that has come to rest, then run, draw, and present.
    /// An exit request the process raised simply stands — frost no
    /// longer ends anyone's loop; the host reads it (see
    /// `Embedded::exit_requested`) and decides.
    pub(crate) fn step(&mut self) {
        self.apply_pending_resize();
        self.render();
        if self.exit_requested {
            log::debug!("process requested exit; the host decides");
        }
    }

    /// A size the host reports, taken the way a live resize was always
    /// taken: noted, and honoured once it comes to rest (see
    /// [`Self::note_resize`]).
    pub(crate) fn host_resize(&mut self, physical: (u32, u32)) {
        if physical.0 > 0 && physical.1 > 0 {
            self.note_resize(PhysicalSize::new(physical.0, physical.1));
        }
    }

    /// A scale factor the host reports, applied at once — the old
    /// `ScaleFactorChanged` path, which never waited for the settle: the
    /// surface is sized from the current scale on the next configure, and
    /// a density move is one event, not a flight of sizes.
    pub(crate) fn host_scale_factor(&mut self, scale_factor: f32) {
        if (self.scale - scale_factor).abs() <= f32::EPSILON {
            return;
        }
        self.scale = scale_factor;
        self.resize();
    }

    /// Whether the surface has stopped handing over frames — the spell
    /// `acquire_frame` counts (see [`Self::surface_gone`]). The host reads
    /// this to know a fresh surface is wanted; [`Self::attach_surface`]
    /// ends the spell.
    pub(crate) fn surface_lost(&self) -> bool {
        self.surface_gone.is_some()
    }

    /// Whether the process asked to leave (see `Context::exit`).
    pub(crate) fn exit_requested(&self) -> bool {
        self.exit_requested
    }

    /// Answers an exit request without leaving, if the host so chooses.
    pub(crate) fn clear_exit(&mut self) {
        self.exit_requested = false;
    }

    /// The scene the engine draws every frame, reachable between frames
    /// (the `Process` sees the same one through `Context::scene`).
    pub(crate) fn scene(&mut self) -> &mut Scene {
        &mut self.scene
    }
}

/// How long a window's size must hold still before the engine rebuilds for
/// it. It has to outlast the gap between two ticks of a platform animation —
/// macOS flies a window into its fullscreen Space at display rate, so its
/// intermediate sizes arrive some 16 ms apart — and stay short enough that a
/// dragged window still feels nailed to the pointer. The measured transition
/// spans under a second and its sizes arrive inside a single one.
const RESIZE_SETTLE_MS: f64 = 100.0;

/// The size a window last asked for, and when: the half of a resize that has
/// not been believed yet. See [`Frost::note_resize`] for the reason it is
/// deferred, and the tests for the sequence it has to survive.
#[derive(Default)]
struct PendingResize {
    size: Option<(u32, u32)>,
    since: f64,
}

impl PendingResize {
    /// A request arrives. It replaces whatever was pending and restarts the
    /// clock, so the size that wins is the last one the platform spoke.
    fn note(&mut self, width: u32, height: u32, now: f64) {
        self.size = Some((width, height));
        self.since = now;
    }

    /// The size worth rebuilding for, once the window has been still for
    /// [`RESIZE_SETTLE_MS`]. Consumed on delivery: one rebuild per settle.
    fn due(&mut self, now: f64) -> Option<(u32, u32)> {
        let (width, height) = self.size?;
        if now - self.since < RESIZE_SETTLE_MS {
            return None;
        }
        self.size = None;
        Some((width, height))
    }
}

/// Monotonic milliseconds since an arbitrary process-start epoch:
/// `std::time::Instant`, pinned to an epoch at the first call. Only
/// differences between two calls matter (see `Frost::last_millis`). The
/// host's timer decides *when* a frame runs; this clock only measures how
/// far apart two frames were, so any cadence — a 15 ms GUI poll, a
/// stutter, a pause — is a legitimate `dt`. Shared with the diagnostics
/// overlay, which times its own per-frame update cost with the same
/// clock (see `Diagnostics`).
pub(crate) fn now_millis() -> f64 {
    // `Instant` has no absolute epoch, so pin one at the first call;
    // `elapsed` stays monotonic.
    static EPOCH: std::sync::OnceLock<std::time::Instant> = std::sync::OnceLock::new();
    let epoch = *EPOCH.get_or_init(std::time::Instant::now);
    epoch.elapsed().as_secs_f64() * 1000.0
}

impl<P: Process> Frost<P> {
    /// Adopts a surface the host built for frost: sizes it from the
    /// physical size and scale the host reports, configures the swap
    /// chain, and builds the pipelines against its format — the surviving
    /// half of the old `resumed`/`attach_window` pair, now that the
    /// window belongs to the host.
    ///
    /// This is also the after-lost path: a surface the compositor took
    /// back is replaced by calling it again with a fresh one. The swap
    /// chain, format pipelines and the gone-counter all follow the new
    /// surface; everything else — scene, caches, arenas — rides through.
    pub(crate) fn attach_surface(
        &mut self,
        surface: Surface<'static>,
        surface_format: TextureFormat,
        physical: (u32, u32),
        scale_factor: f64,
    ) {
        let scale = scale_factor.max(0.01);
        self.scale = scale as f32;
        // The host reports *physical* pixels, so convert to logical here;
        // `pixel_size()` multiplies by the scale factor back out.
        self.logical_size = (
            (physical.0 as f64 / scale).max(1.0).round() as u32,
            (physical.1 as f64 / scale).max(1.0).round() as u32,
        );
        self.surface = Some(surface);
        self.surface_format = Some(surface_format);
        // The new chain has proven nothing yet: the format the old one
        // settled on does not carry over, and until `reconfigure_surface`
        // restates it every frame skips at the acquire guard below. A
        // same-format re-attach rebuilds its pipelines once more here —
        // the price of the honest `None` meaning "this chain has no
        // format yet", and once per lost spell is nothing.
        self.format = None;
        // A fresh chain is a fresh chance; the gone spell, if any, ends.
        self.surface_gone = None;
        log::info!(
            "surface attached ({}x{} @ {:.2}x, host format {surface_format:?})",
            self.logical_size.0,
            self.logical_size.1,
            self.scale,
        );
        if self.reconfigure_surface().is_none() {
            log::warn!(
                "surface could not be configured at {:?} pixels",
                self.pixel_size()
            );
        }
        // Draw the first frame right here: a setup that cannot present
        // should say so at construction, not at the host's first tick.
        self.render();
    }

    fn pixel_size(&self) -> (u32, u32) {
        (
            (self.logical_size.0 as f64 * self.scale as f64).max(1.0) as u32,
            (self.logical_size.1 as f64 * self.scale as f64).max(1.0) as u32,
        )
    }

    /// Reconfigures the swap chain at the surface's current size: format
    /// negotiation (the host's chosen format, upgraded to its sRGB twin
    /// where the hardware offers one), the presentation mode — always
    /// uncapped, see below — the configure itself, and the pipeline
    /// rebuild a surface-format change requires. Shared by the attach and
    /// resize paths; returns the configuration the surface now runs with,
    /// or `None` while no surface exists or none can be configured at
    /// this size.
    fn reconfigure_surface(&mut self) -> Option<SurfaceConfiguration> {
        let pixel_size = self.pixel_size();
        let config = {
            let surface = self.surface.as_ref()?;
            let mut config = surface
                .get_default_config(&self.adapter, pixel_size.0, pixel_size.1)?;
            // The host picked this format when it built the surface; keep
            // it as long as the new chain actually supports it, and say
            // plainly when it does not.
            if let Some(want) = self.surface_format {
                if surface.get_capabilities(&self.adapter).formats.contains(&want) {
                    config.format = want;
                } else {
                    log::warn!(
                        "surface does not offer the host's format {want:?}; using {:#?}",
                        config.format
                    );
                }
            }
            prefer_srgb_format(&mut config, surface, &self.adapter);
            // Never vsync, whatever the app's flag says: a blocking
            // present would stall the host's main thread — a plugin has
            // no right to hold the audio application's GUI (and on macOS,
            // its run loop) hostage for a display refresh. `AutoNoVsync`
            // falls back gracefully where the uncapped modes are absent
            // (see `present_mode_for`), so it is safe everywhere.
            config.present_mode = present_mode_for(false);
            surface.configure(&self.device, &config);
            config
        };
        log::info!(
            "surface configured at {}x{}, format {:?}, present_mode {:?}",
            config.width,
            config.height,
            config.format,
            config.present_mode
        );
        // Pipelines depend only on the surface format, not the size, so they
        // are rebuilt only if the format actually changed.
        if self.format != Some(config.format) {
            self.linear_surface = !config.format.is_srgb();
            self.hdr_frames = Self::float_target(&self.adapter).is_some();
            // The offscreen buffer carries the old format: drop it, and
            // the next frame rebuilds it against the new pipelines.
            self.blit_target = None;
            self.set_up_pipelines(config.format);
            self.format = Some(config.format);
        }
        Some(config)
    }

    /// Take a resize from the platform without honouring it yet.
    ///
    /// A resize is not one event: macOS animates a window into and out of its
    /// fullscreen Space and reports every frame of the flight as a new size,
    /// and the sizes it flies through are not sizes anything has. Measured on
    /// the Mac of `docs/benchmarking.md` §17, entering fullscreen once: four
    /// events inside a second — 2056×1286, 1329, 1286, then **1357**, a window
    /// taller than the 1329-point display it was being animated into — and
    /// 1286 again, where it sat for the next four minutes. Honouring each one
    /// means rebuilding a swap chain for pixels nobody will ever see: a
    /// 4112×2714 set of drawables, allocated because a window was in the air.
    ///
    /// So the engine accepts a size only once the platform has stopped
    /// changing its mind, on the next frame past [`RESIZE_SETTLE_MS`]. In a
    /// window drag that is the difference between following the pointer and
    /// lagging it by a tenth of a second, during which the old surface is
    /// stretched to the new window by the compositor — the same thing a
    /// dragged window already looks like for the frame the engine misses on
    /// any platform. In an animation it is the whole difference, because the
    /// flight emits nothing worth rebuilding for.
    fn note_resize(&mut self, size: PhysicalSize<u32>) {
        self.resize_pending
            .note(size.width, size.height, now_millis());
    }

    /// The deferred half of [`Self::note_resize`], asked on every frame:
    /// the surface's size changes and the swap chain follows. (The old
    /// engine also ran its aspect lock here, asking the window for a
    /// corrected shape; the host owns the window's shape now, so a fixed
    /// render size that disagrees with the surface simply letterboxes —
    /// the blit has always drawn those bars honestly.)
    fn apply_pending_resize(&mut self) {
        let Some((width, height)) = self.resize_pending.due(now_millis()) else {
            return;
        };
        // `size` is the physical client size; store the logical size so
        // `pixel_size()` reproduces the true physical pixels.
        let scale = (self.scale as f64).max(0.01);
        self.logical_size = (
            (width as f64 / scale).max(1.0).round() as u32,
            (height as f64 / scale).max(1.0).round() as u32,
        );
        self.resize();
    }

    fn resize(&mut self) {
        let Some(config) = self.reconfigure_surface() else {
            return;
        };
        let size = (
            self.logical_size.0,
            self.logical_size.1,
            config.width,
            config.height,
        );
        if size != self.resize_logged {
            self.resize_logged = size;
            log::info!(
                "window resized to {}x{} ({}x{} px)",
                self.logical_size.0,
                self.logical_size.1,
                config.width,
                config.height
            );
        }
    }

    /// A pipeline's shader text as it stands this frame: the live
    /// text if a watched file has spoken it, the const compiled
    /// into the binary otherwise. The names are the table's in
    /// `crate::shaders`.
    fn wgsl(&self, name: &str) -> &str {
        #[cfg(debug_assertions)]
        if let Some(text) = self.shaders.get(name) {
            return text;
        }
        match name {
            "line" => crate::shaders::LINE_SHADER,
            "polyline" => crate::shaders::POLYLINE_SHADER,
            "circle" => crate::shaders::CIRCLE_SHADER,
            "rectangle" => crate::shaders::RECT_SHADER,
            "shape" => crate::shaders::SHAPE_SHADER,
            "sprite" => crate::shaders::SPRITE_SHADER,
            "particles" => crate::shaders::PARTICLES_SHADER,
            "tilemap" => crate::shaders::TILEMAP_SHADER,
            _ => crate::shaders::BLIT_SHADER,
        }
    }

    /// Creates the shared pipelines. The five whose per-draw state is one
    /// uniform get hand-written layouts, because their uniforms arrive from
    /// the frame's arena by dynamic offset; the rest are derived. The layouts
    /// are dropped as soon as the pipelines exist — a pipeline hands its bind
    /// group layout back to the draws that need it.
    /// The format a frame renders into, when the machine offers one:
    /// `Rgba16Float` if the adapter will take it as a render attachment
    /// *and* let the blit sample it, `None` if not — and then frames go to
    /// the surface itself, which is what they have always done. Asked at
    /// surface configuration (to build the pipelines against the right
    /// target) and again whenever the buffer is built, so both answers come
    /// from the same question.
    /// Whether the frame carries an emissive target. It is the float frame
    /// buffer's own question, asked separately because a machine that clips at
    /// 1.0 on every store has no use for a second output whose useful range
    /// starts above 1.0 — and one more full-resolution texture is not a thing
    /// to hand out for nothing.
    fn emissive_target(adapter: &Adapter) -> Option<TextureFormat> {
        Self::float_target(adapter)
    }

    fn float_target(adapter: &Adapter) -> Option<TextureFormat> {
        let wants = TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING;
        let caps = adapter.get_texture_format_features(TextureFormat::Rgba16Float);
        caps.allowed_usages
            .contains(wants)
            .then_some(TextureFormat::Rgba16Float)
    }

    fn set_up_pipelines(&mut self, surface_format: TextureFormat) {
        let device = &self.device;
        // The scene renders into the frame buffer, so every pipeline that
        // draws the scene is built against *that* format; the blit pass is
        // the only one that writes the window, and it alone keeps the
        // surface's format. The two differ exactly when `hdr_frames` is set.
        let format = Self::float_target(&self.adapter).unwrap_or(surface_format);
        // The emissive target rides with the float frame buffer: a second
        // output whose useful range *starts* above 1.0 has nothing to say on a
        // machine that clips at 1.0 on every store, and one more full-res
        // texture is not a thing to hand out for nothing. Same question as
        // `format`, same answer — and `BlitTarget` builds the texture against
        // it, so the pipelines and the pass cannot disagree.
        let emissive = Self::emissive_target(&self.adapter);
        // Remembered for the user programs the frame may meet mid-frame:
        // they must be built against exactly this format and exactly this
        // emissive answer, and last frame's programs were built against
        // whatever the last spell answered — so their pipelines go, here,
        // with the scene's own being rebuilt.
        self.frame_format = Some(format);
        self.frame_emissive = emissive;
        self.custom_pipelines.clear();
        let line_layout = slot_layouts(device, "line pipeline", LINE_UNIFORM_BYTES);
        let polyline_layout = slot_layouts(device, "polyline pipeline", POLYLINE_UNIFORM_BYTES);
        let circle_layout = slot_layouts(device, "circle pipeline", CIRCLE_UNIFORM_BYTES);
        let rect_layout = slot_layouts(device, "rectangle pipeline", RECT_UNIFORM_BYTES);
        let shape_layout = shape_layouts(device, "shape pipeline");

        let line_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("line shaders"),
            source: ShaderSource::Wgsl(Cow::Borrowed(self.wgsl("line"))),
        });
        self.line_pipeline = Some(Self::create_pipeline(
            device,
            &line_module,
            format,
            emissive,
            Some(&line_layout),
            "line pipeline",
            Some(wgpu::BlendState::ALPHA_BLENDING),
        ));

        let polyline_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("polyline shaders"),
            source: ShaderSource::Wgsl(Cow::Borrowed(self.wgsl("polyline"))),
        });
        self.polyline_pipeline = Some(Self::create_pipeline(
            device,
            &polyline_module,
            format,
            emissive,
            Some(&polyline_layout),
            "polyline pipeline",
            Some(wgpu::BlendState::ALPHA_BLENDING),
        ));

        let circle_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("circle shaders"),
            source: ShaderSource::Wgsl(Cow::Borrowed(self.wgsl("circle"))),
        });
        self.circle_pipeline = Some(Self::create_pipeline(
            device,
            &circle_module,
            format,
            emissive,
            Some(&circle_layout),
            "circle pipeline",
            Some(wgpu::BlendState::ALPHA_BLENDING),
        ));

        let rect_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("rectangle shaders"),
            source: ShaderSource::Wgsl(Cow::Borrowed(self.wgsl("rectangle"))),
        });
        self.rect_pipeline = Some(Self::create_pipeline(
            device,
            &rect_module,
            format,
            emissive,
            Some(&rect_layout),
            "rectangle pipeline",
            Some(wgpu::BlendState::ALPHA_BLENDING),
        ));

        let shape_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("shape shaders"),
            source: ShaderSource::Wgsl(Cow::Borrowed(self.wgsl("shape"))),
        });
        self.shape_pipeline = Some(Self::create_pipeline(
            device,
            &shape_module,
            format,
            emissive,
            Some(&shape_layout),
            "shape pipeline",
            Some(wgpu::BlendState::ALPHA_BLENDING),
        ));

        let sprite_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("sprite shaders"),
            source: ShaderSource::Wgsl(Cow::Borrowed(self.wgsl("sprite"))),
        });
        // The one pipeline with a hand-written layout, for the arena's
        // dynamic offset. The bind layout is dropped once the pipeline
        // exists — `get_bind_group_layout` hands it back — but the pipeline
        // layout is kept: it is the one a game's user program must be built
        // on, so its bind groups are the sprite's, unchanged.
        let (_sprite_bind_layout, sprite_pipeline_layout) = sprite_layouts(device);
        self.sprite_pipeline = Some(Self::create_pipeline(
            device,
            &sprite_module,
            format,
            emissive,
            Some(&sprite_pipeline_layout),
            "sprite pipeline",
            Some(wgpu::BlendState::ALPHA_BLENDING),
        ));
        // The one variant the renderer carries, because it is the one thing a
        // scene cannot draw without: an authored glow. Same shader, same bind
        // group layout — a glow sprite's bind group is the ordinary sprite's,
        // which is why the two share the texture cache and the bind cache
        // below — and a different answer to how its light lands. Blend states
        // are baked into a pipeline, so opting out of coverage costs a second
        // pipeline rather than a second uniform field.
        self.sprite_additive_pipeline = Some(Self::create_pipeline(
            device,
            &sprite_module,
            format,
            emissive,
            Some(&sprite_pipeline_layout),
            "sprite additive pipeline",
            Some(ADDITIVE_BLEND),
        ));
        self.sprite_pipeline_layout = Some(sprite_pipeline_layout);

        let particles_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("particle shaders"),
            source: ShaderSource::Wgsl(Cow::Borrowed(self.wgsl("particles"))),
        });
        self.particle_pipeline = Some(Self::create_pipeline(
            device,
            &particles_module,
            format,
            emissive,
            None,
            "particle pipeline",
            Some(wgpu::BlendState::ALPHA_BLENDING),
        ));

        let tilemap_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("tilemap shaders"),
            source: ShaderSource::Wgsl(Cow::Borrowed(self.wgsl("tilemap"))),
        });
        self.tilemap_pipeline = Some(Self::create_pipeline(
            device,
            &tilemap_module,
            format,
            emissive,
            None,
            "tilemap pipeline",
            Some(wgpu::BlendState::ALPHA_BLENDING),
        ));

        // The fixed render size stretch: a full-screen sample of the
        // render target onto the surface. It replaces the surface pixel
        // instead of blending onto it — the buffer is the frame, not a
        // layer over it. On a linear surface the same pass carries the
        // gamma encoding the surface's hardware would otherwise do
        // (see `App::linear_surface`), by way of one const substituted
        // into the shader at this exact moment.
        let blit_source = match self.linear_surface {
            false => Cow::Borrowed(self.wgsl("blit")),
            true => Cow::Owned(self.wgsl("blit").replace("/*@gamma@*/false", "true")),
        };
        let blit_module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("blit shaders"),
            source: ShaderSource::Wgsl(blit_source),
        });
        self.blit_pipeline = Some(Self::create_pipeline(
            device,
            &blit_module,
            surface_format,
            // The window holds one image, not two: the emissive target stops
            // at the frame buffer.
            None,
            None,
            "blit pipeline",
            None,
        ));
    }

    /// Builds a render pipeline for a shader with no vertex buffers
    /// (full-screen triangles, or instanced quads generated in the vertex
    /// shader), one bind group at group 0, and a single color target —
    /// blended as `blend` says, or replaced outright when it is `None` (the
    /// fixed-render-size stretch, whose source is the finished frame). Scene
    /// pipelines pass [`wgpu::BlendState::ALPHA_BLENDING`] or
    /// [`ADDITIVE_BLEND`]; the two differ in one clause of one sentence,
    /// and which of them a draw gets is decided by its node's `additive` flag
    /// here rather than by anything the shader says.
    /// Builds a scene pipeline. `emissive` is the frame's emissive target
    /// format, `Some` exactly when the scene shaders are asked for their second
    /// output: a pipeline declaring one target cannot draw into a pass offering
    /// two (measured, not assumed — wgpu rejects the mismatch as an
    /// incompatible-attachment error), so the count and the entry point change
    /// together and the pass has to agree with both.
    fn create_pipeline(
        device: &Device,
        module: &ShaderModule,
        format: TextureFormat,
        emissive: Option<TextureFormat>,
        layout: Option<&PipelineLayout>,
        label: &str,
        blend: Option<wgpu::BlendState>,
    ) -> RenderPipeline {
        let blend_state = blend;
        let mut targets = vec![Some(ColorTargetState {
            format,
            blend: blend_state,
            write_mask: ColorWrites::ALL,
        })];
        if let Some(format) = emissive {
            // Blended on the same terms as the color target, which is what
            // makes the rule the shaders state work: radiance in rgb, coverage
            // in alpha, and a fragment that covers nothing changes nothing.
            targets.push(Some(ColorTargetState {
                format,
                blend: blend_state,
                write_mask: ColorWrites::ALL,
            }));
        }
        device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some(label),
            layout,
            vertex: VertexState {
                module,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module,
                entry_point: Some(if emissive.is_some() {
                    "fs_emissive"
                } else {
                    "fs_main"
                }),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &targets,
            }),
            primitive: PrimitiveState::default(),
            depth_stencil: None,
            multisample: MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        })
    }

    /// Builds the fixed render size buffer: a texture of exactly `size`
    /// pixels in the surface's own format — so the blit's samples need no
    /// re-encoding — usable as both a render attachment and a sampled
    /// texture, with its bind group pairing the view and a bilinear
    /// sampler. Built once per size change, which for a fixed size means
    /// once (the target survives resizes; only the requested size rebuilds
    /// it), and dropped whenever the surface itself becomes that size.
    /// The stretch pass's bind group: this target's frame view and sampler, its
    /// fit uniform, and the bloom levels. It has exactly one shape,
    /// dormant chain or shallow chain or not, because the weights in the fit
    /// uniform are what decide whether each sample happens — which is why
    /// [`Bloom`] hands over [`MAX_LEVELS`] views always, one black texel
    /// standing in for every slot the preset left empty. Built here rather than
    /// inline because a chain woken mid-run has to rebind its new textures into
    /// the same slot, and two copies of a dozen bindings is how that goes wrong.
    fn blit_bind_group(
        &self,
        view: &TextureView,
        sampler: &Sampler,
        fit_buffer: &Buffer,
        bloom: &Bloom,
    ) -> BindGroup {
        let mut entries = vec![
            BindGroupEntry {
                binding: 0,
                resource: BindingResource::TextureView(view),
            },
            BindGroupEntry {
                binding: 1,
                resource: BindingResource::Sampler(sampler),
            },
            BindGroupEntry {
                binding: 2,
                resource: BindingResource::Buffer(BufferBinding {
                    buffer: fit_buffer,
                    offset: 0,
                    size: None,
                }),
            },
        ];
        // The bloom levels, bound whether or not there is bloom to show, one
        // binding each: the shader's bindings are numbered from this side too.
        entries.extend(bloom.views.iter().enumerate().map(|(i, v)| BindGroupEntry {
            binding: 3 + i as u32,
            resource: BindingResource::TextureView(v),
        }));
        entries.push(BindGroupEntry {
            binding: 3 + crate::backend::bloom::MAX_LEVELS as u32,
            resource: BindingResource::Sampler(&bloom.sampler),
        });
        self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("blit bind group"),
            layout: &self
                .blit_pipeline
                .as_ref()
                .expect("the blit pipeline is built with the surface")
                .get_bind_group_layout(0),
            entries: &entries,
        })
    }

    fn new_blit_target(&mut self, size: [u32; 2]) -> BlitTarget {
        // The same question the pipelines were built against: a buffer in
        // one format with pipelines compiled for another is a validation
        // error on every frame, so the answer must come from one place.
        let format = Self::float_target(&self.adapter).unwrap_or_else(|| {
            self.format
                .expect("a surface is attached before the first frame renders")
        });
        let texture = self.device.create_texture(&TextureDescriptor {
            label: Some("render target"),
            size: Extent3d {
                width: size[0].max(1),
                height: size[1].max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        // The emissive target, on the same condition the pipelines were built
        // with: it exists where the frame buffer can hold a value past 1.0,
        // nowhere else. Sampled usage now costs nothing and is what bloom will
        // need from it later.
        let emissive = Self::emissive_target(&self.adapter).map(|format| {
            let texture = self.device.create_texture(&TextureDescriptor {
                label: Some("emissive target"),
                size: Extent3d {
                    width: size[0].max(1),
                    height: size[1].max(1),
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: TextureDimension::D2,
                format,
                usage: TextureUsages::RENDER_ATTACHMENT | TextureUsages::TEXTURE_BINDING,
                view_formats: &[],
            });
            let view = texture.create_view(&TextureViewDescriptor::default());
            (texture, view)
        });
        // The chain that turns the emissive target into a glow, sized against
        // this buffer. Built here, not lazily, because its levels are fractions
        // of this extent and a resized buffer is a new buffer: the chain's life
        // is exactly the target's.
        let bloom = Bloom::new(
            &self.device,
            &self.queue,
            format,
            size,
            emissive.as_ref().map(|(_, view)| view),
            self.bloom,
        );
        // The user's upscaling choice (`EmbeddedConfig::blit_filter`), applied to
        // both axes: bilinear smooths the stretch, nearest keeps texels
        // whole for the pixel-art look.
        let filter_mode = match self.blit_filter {
            SpriteFilter::Linear => FilterMode::Linear,
            SpriteFilter::Nearest => FilterMode::Nearest,
        };
        let label = format!("blit sampler ({:?})", self.blit_filter);
        let sampler = self.device.create_sampler(&SamplerDescriptor {
            label: Some(&label),
            mag_filter: filter_mode,
            min_filter: filter_mode,
            ..Default::default()
        });
        // The letterbox fit uniform, seeded to fill the window; `render`
        // rewrites it every frame. A zero-initialized buffer would draw the
        // quad at zero size, so the (1, 1) seed matters for the first
        // frame if it ever ran before the per-frame write.
        let fit_buffer = self.device.create_buffer(&BufferDescriptor {
            label: Some("blit fit"),
            size: 64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(
            &fit_buffer,
            0,
            &fit_bytes(
                (1.0, 1.0),
                [0.0; crate::backend::bloom::MAX_LEVELS],
                self.tone_map,
            ),
        );
        let bind_group = self.blit_bind_group(&view, &sampler, &fit_buffer, &bloom);
        log::info!(
            "fixed render size: frames render to a {}x{} buffer, stretched to the window with {:?} sampling",
            size[0],
            size[1],
            self.blit_filter
        );
        BlitTarget {
            size,
            filter: self.blit_filter,
            _texture: texture,
            view,
            emissive,
            bloom,
            format,
            sampler,
            fit_buffer,
            bind_group,
        }
    }

    /// Creates the sprite's texture, view and sampler for one image.
    ///
    /// The pixels arrive as tightly packed RGBA8 bytes (one per texel), and
    /// [`pad_rows`] gives them the row pitch every texture copy is required
    /// to carry — a multiple of 256, which a 966-texel-wide ant is not, and
    /// which wgpu-core never asks for. One `write_texture` either way. The
    /// texture is created in
    /// `Rgba8UnormSrgb` so the sample lands in linear space and the
    /// sRGB blending state produces the same colors the file was authored
    /// in. The texture itself is kept alive by the view: `TextureView`
    /// holds a reference to its texture, so storing the view in
    /// `sprite_resources` is enough.
    fn sprite_texture(&self, data: &[u8], size: [f32; 2]) -> TextureView {
        let width = (size[0] as u32).max(1);
        let height = (size[1] as u32).max(1);
        let texture = self.device.create_texture(&TextureDescriptor {
            label: Some("sprite texture"),
            size: Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format: TextureFormat::Rgba8UnormSrgb,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let view = texture.create_view(&TextureViewDescriptor::default());
        // The file's rows are tight; a texture copy's are not. Padding here
        // rather than refusing the upload is the difference between a game
        // that draws and one that draws noise, and on the native backends
        // the padding is dead weight the browser would never have asked for
        // — [`pad_rows`] borrows the bytes when the pitch already fits, so
        // an ant-sized atlas pays one copy at load and a 1024-wide one pays
        // nothing.
        let (pixels, pitch) = pad_rows(data, width, height, 4);
        self.queue.write_texture(
            TexelCopyTextureInfo {
                texture: &texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: TextureAspect::All,
            },
            &pixels,
            TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(pitch),
                rows_per_image: Some(height),
            },
            Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
        );
        view
    }

    /// The sampler a draw's filter picks: one of the two built once at
    /// startup, so a texture shared across filters stays shared.
    fn sampler_for(&self, filter: SpriteFilter) -> Sampler {
        match filter {
            SpriteFilter::Linear => self.sampler_linear.clone(),
            SpriteFilter::Nearest => self.sampler_nearest.clone(),
        }
    }

    /// Evicts a stale entry from the sprite texture cache, and the cached
    /// bind groups that bind it: two live buffers can never share an address,
    /// so a cache entry at the same pointer with a different generation
    /// belongs to a freed buffer whose address the new one has recycled, and
    /// its texture is dead weight.
    ///
    /// The bind groups are not an optional second sweep. A bind group keeps
    /// the texture it binds alive, so a cached group for an image that is gone
    /// would hold its texture — and the arena behind it — for the rest of the
    /// run. A demo that hands the engine a fresh image every frame, as the
    /// diagnostics overlay does, grows both caches one entry per frame
    /// otherwise; the eviction is what bounds them by the images in play.
    fn evict_recycled_buffer(&mut self, key: (u64, u64)) {
        if let Some(stale) = self
            .sprite_resources
            .iter()
            .find(|(k, _)| k.0 == key.0)
            .map(|(k, _)| *k)
        {
            self.sprite_resources.remove(&stale);
            self.sprite_binds.retain(|bind, _| bind.0 != stale.0);
        }
    }

    /// Creates the per-draw buffers and bind group for one particle batch:
    /// the batch's uniform (the surface size, the base color, and the shape
    /// kind and aspect) at binding 0, the packed per-particle instance data
    /// as a storage buffer read by the vertex stage at binding 1, the
    /// sampled texture and sampler — the batch's image for a sprite batch,
    /// the 1x1 placeholder for the SDF kinds — at bindings 2 and 3, the
    /// frame's light field at binding 4, and the frame's occluder field at
    /// binding 5.
    ///
    /// This batch is the frame's remaining per-draw allocation, and deliberately
    /// so: what a batch passes is a storage buffer of instance data — one record
    /// per particle, a different length every frame — and the arena that carries
    /// every *uniform* in the frame has nothing to do with it.
    // The bind-group entries are deliberately flat: each takes the
    // per-draw resources it binds, and the frame fields grow with the
    // lighting features.
    #[allow(clippy::too_many_arguments)]
    fn particle_uniform(
        &self,
        pipeline: &RenderPipeline,
        data: &[u8],
        uniform_data: &[u8],
        view: &TextureView,
        sampler: &Sampler,
        field_buffer: &Buffer,
        occluder_buffer: &Buffer,
    ) -> (Buffer, Buffer, BindGroup) {
        let device = &self.device;
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("particle uniform buffer"),
            size: uniform_data.len() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&uniform_buffer, 0, uniform_data);
        let instance_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("particle instance buffer"),
            size: data.len() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&instance_buffer, 0, data);
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("particle bind group"),
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &uniform_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &instance_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Sampler(sampler),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: field_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: occluder_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });
        (uniform_buffer, instance_buffer, bind_group)
    }

    /// Creates the uniform buffer (batch tint and surface size), the
    /// instance buffer (three `vec4`s per tile), and the four-entry bind
    /// group (uniform, instances, atlas view, sampler) for one tile-map
    /// draw call. Like the particle batch, this is one of the frame's two
    /// remaining per-draw allocations: three `vec4`s per tile ride a storage
    /// buffer, which is where the batch's bytes actually are. And, unlike the
    /// particle batch, the tile map is never lit, so no light fields join the
    /// group.
    #[allow(clippy::too_many_arguments)] // fields ride every lit pipeline alike
    fn tilemap_uniform(
        &self,
        pipeline: &RenderPipeline,
        data: &[u8],
        uniform_data: &[u8],
        view: &TextureView,
        sampler: &Sampler,
        field_buffer: &Buffer,
        occluder_buffer: &Buffer,
    ) -> (Buffer, Buffer, BindGroup) {
        let device = &self.device;
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("tilemap uniform buffer"),
            size: uniform_data.len() as u64,
            usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&uniform_buffer, 0, uniform_data);
        let instance_buffer = device.create_buffer(&BufferDescriptor {
            label: Some("tilemap instance buffer"),
            size: data.len() as u64,
            usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&instance_buffer, 0, data);
        let layout = pipeline.get_bind_group_layout(0);
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("tilemap bind group"),
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &uniform_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &instance_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::TextureView(view),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Sampler(sampler),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: field_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 5,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: occluder_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });
        (uniform_buffer, instance_buffer, bind_group)
    }

    /// The bind group for one pipeline whose per-draw state is one uniform: a
    /// window of the arena as wide as that pipeline's uniform, plus — for the
    /// shape pipeline, which lights itself — the frame's two fields.
    ///
    /// One per pipeline, built on the frame's first draw that needs it and kept
    /// for every draw after, and for the frames after: the arena, the fields and
    /// the layout all outlive a frame, and what changes per draw is the dynamic
    /// offset, which is a u32 rather than an allocation. A frame of two hundred
    /// chart segments builds this once. Cleared when the arena or a field buffer
    /// is rebuilt, because those are the buffers it holds.
    ///
    /// `label` is the pipeline's own label, which is what the cache keys on: one
    /// pipeline, one group, no key to keep in step with anything.
    #[allow(clippy::too_many_arguments)]
    fn slot_bind_group(
        &mut self,
        label: &'static str,
        pipeline: &RenderPipeline,
        uniform: u64,
        fields: bool,
        field_buffer: &Buffer,
        occluder_buffer: &Buffer,
    ) -> BindGroup {
        if let Some(bind_group) = self.prim_binds.get(label) {
            return bind_group.clone();
        }
        let layout = pipeline.get_bind_group_layout(0);
        let arena = self
            .uniform_arena
            .clone()
            .expect("a frame that draws primitives built the arena before the pass");
        // A zero-byte uniform would bind no window at all, and the draw would
        // then read the arena from its offset to its end.
        debug_assert!(uniform > 0, "{label} has no uniform to bind");
        let window = BindGroupEntry {
            binding: 0,
            // One run of the arena, not the arena: a dynamic offset slides
            // the *bound window*, so a group that bound the whole buffer
            // could only ever be offset by zero.
            resource: BindingResource::Buffer(BufferBinding {
                buffer: &arena,
                offset: 0,
                size: NonZeroU64::new(uniform),
            }),
        };
        let bind_group = if fields {
            self.device.create_bind_group(&BindGroupDescriptor {
                label: Some(label),
                layout: &layout,
                entries: &[
                    window,
                    BindGroupEntry {
                        binding: 1,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: field_buffer,
                            offset: 0,
                            size: None,
                        }),
                    },
                    BindGroupEntry {
                        binding: 2,
                        resource: BindingResource::Buffer(BufferBinding {
                            buffer: occluder_buffer,
                            offset: 0,
                            size: None,
                        }),
                    },
                ],
            })
        } else {
            self.device.create_bind_group(&BindGroupDescriptor {
                label: Some(label),
                layout: &layout,
                entries: &[window],
            })
        };
        self.prim_binds.insert(label, bind_group.clone());
        bind_group
    }

    /// The pipeline for one game-authored program: the shared lighting
    /// vocabulary, then the game's source — assembled the way the engine
    /// assembles its own lit kinds — compiled on the device and built over
    /// the sprite pipeline's very bind group layout, so the sprite's arena
    /// uniform, texture, sampler, and both light fields bind exactly as
    /// they do for the engine's own sprites, and a sprite bind group
    /// cached for the ordinary pipeline serves the custom one unchanged.
    /// Built at first sight of the source and kept until the frame's
    /// format changes; a source that fails to compile is the driver's
    /// complaint to hear — which is what [`crate::validate_wgsl`] is for.
    fn custom_pipeline(&mut self, source: &Arc<str>) -> Option<RenderPipeline> {
        if let Some(pipeline) = self.custom_pipelines.get(source.as_ref()) {
            return Some(pipeline.clone());
        }
        let format = self.frame_format?;
        let emissive = self.frame_emissive;
        let layout = self.sprite_pipeline_layout.clone()?;
        let module = self.device.create_shader_module(ShaderModuleDescriptor {
            label: Some("user program"),
            source: ShaderSource::Wgsl(
                format!("{}\n{source}", crate::shaders::LIGHTING_WGSL).into(),
            ),
        });
        let pipeline = Self::create_pipeline(
            &self.device,
            &module,
            format,
            emissive,
            Some(&layout),
            "user pipeline",
            Some(wgpu::BlendState::ALPHA_BLENDING),
        );
        self.custom_pipelines
            .insert(source.clone(), pipeline.clone());
        Some(pipeline)
    }

    /// The bind group for one sprite image at one sampling filter: the
    /// arena's 96-byte window, the image, the sampler, and the frame's two
    /// fields.
    ///
    /// One per image per filter, built on the frame's first draw that needs
    /// it and kept for the ones after — and across frames, since nothing in
    /// the group changes between them. A frame of two hundred glyph quads off
    /// one atlas builds this once; what stays per-draw is the dynamic offset
    /// that slides the uniform window, which is a u32 and not an allocation.
    /// The group is dropped when the arena or a field buffer is rebuilt,
    /// because those are the buffers it holds.
    #[allow(clippy::too_many_arguments)]
    fn sprite_bind_group(
        &mut self,
        pipeline: &RenderPipeline,
        texture: (u64, u64),
        linear: bool,
        view: &TextureView,
        sampler: &Sampler,
        field_buffer: &Buffer,
        occluder_buffer: &Buffer,
    ) -> BindGroup {
        let key = (texture.0, texture.1, linear);
        if let Some(bind_group) = self.sprite_binds.get(&key) {
            return bind_group.clone();
        }
        let layout = pipeline.get_bind_group_layout(0);
        let arena = self
            .uniform_arena
            .clone()
            .expect("a frame that draws sprites built the arena before the pass");
        let bind_group = self.device.create_bind_group(&BindGroupDescriptor {
            label: Some("sprite bind"),
            layout: &layout,
            entries: &[
                BindGroupEntry {
                    binding: 0,
                    // One slot of the arena, not the arena: a dynamic offset
                    // slides the *bound window*, so a group that bound the
                    // whole buffer could only ever be offset by zero.
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: &arena,
                        offset: 0,
                        size: NonZeroU64::new(SPRITE_UNIFORM_BYTES),
                    }),
                },
                BindGroupEntry {
                    binding: 1,
                    resource: BindingResource::TextureView(view),
                },
                BindGroupEntry {
                    binding: 2,
                    resource: BindingResource::Sampler(sampler),
                },
                BindGroupEntry {
                    binding: 3,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: field_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                BindGroupEntry {
                    binding: 4,
                    resource: BindingResource::Buffer(BufferBinding {
                        buffer: occluder_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });
        self.sprite_binds.insert(key, bind_group.clone());
        bind_group
    }

    /// The frame's light field buffer, grown to hold the header plus
    /// `count` light records when the buffer is still too small. The
    /// buffer only ever grows: its size is the peak light count so far
    /// (floored at the WGSL minimum binding size), and lighter frames
    /// reuse the bigger buffer, writing a shorter slice.
    fn field_buffer_for(&mut self, count: u32) -> Buffer {
        let size = light_field_buffer_size(count);
        if size > self.field_buffer.size() {
            self.field_buffer = self.device.create_buffer(&BufferDescriptor {
                label: Some("light field buffer"),
                size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            // The cached bind groups bind the buffer that was just replaced;
            // they go with it and are rebuilt on the next draw.
            self.sprite_binds.clear();
            self.prim_binds.clear();
        }
        self.field_buffer.clone()
    }

    /// The frame's occluder field buffer, grown to hold this frame's
    /// packed bytes when the buffer is still too small. The packer's
    /// output is the ground truth — per-light blocks, duplicated records
    /// and polygon planes all vary — so the size follows `bytes`, floored
    /// at the WGSL minimum binding size. Same grow-only rationale as
    /// [`Frost::field_buffer_for`].
    fn occluder_buffer_for(&mut self, bytes: usize) -> Buffer {
        let size = (bytes.max(OCCLUDER_FIELD_BUFFER_MIN)) as u64;
        if size > self.occluder_buffer.size() {
            self.occluder_buffer = self.device.create_buffer(&BufferDescriptor {
                label: Some("occluder field buffer"),
                size,
                usage: BufferUsages::STORAGE | BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            // As in [`Frost::field_buffer_for`]: the cached bind groups hold
            // the buffer that was just replaced.
            self.sprite_binds.clear();
            self.prim_binds.clear();
        }
        self.occluder_buffer.clone()
    }

    fn render(&mut self) {
        // The shader tier, first among equals: one poll per frame —
        // ten stats — before the surface is even asked, so a saved
        // shader lands whether or not anything is compositing. A
        // passing batch rebuilds the pipelines next frame; the
        // game's state, buffers, and textures carry across untouched.
        #[cfg(debug_assertions)]
        if self.shaders.poll() {
            self.format = None;
        }
        // The rebuild a passing shader batch signs up for: a `None`
        // format is the request, and the road is the one a surface-format
        // change always took — reconfigure compares the surface's format
        // against `format`, finds no match in `None`, and rebuilds every
        // pipeline against the sources as they now stand. One reconfigure
        // per edit, not per frame: the road ends by restating `format`.
        // A surface that has never answered a configure sits at `None`
        // too, so every frame retries it; while it keeps failing the
        // frame is skipped below rather than acquiring an unconfigured
        // surface, which wgpu would take as a panic.
        if self.format.is_none() && self.surface.is_some() {
            self.reconfigure_surface();
        }
        let (output, reconfigure) = self.acquire_frame();
        if reconfigure {
            self.resize();
        }
        let Some(output) = output else {
            return;
        };
        log::trace!("render: acquired surface texture, submitting frame");
        // The frame's processing probe: the engine's CPU time for this
        // frame, from here (the acquire is done — it may have waited on the
        // display's back buffer, and that wait is not processing) to the
        // queue submission below. The present handoff is display time.
        let t0 = now_millis();

        // Let the user update the scene and draw this frame, in their
        // coordinate system: the window's logical pixels at the display's
        // scale factor — or, in fixed render size mode (`EmbeddedConfig::render_size`),
        // the render buffer's pixels one-for-one, the stretch pass carrying
        // them to whatever the window is.
        let surface_px = self.pixel_size();
        let (user_size, user_scale) = match self.render_size {
            Some([w, h]) => ((w, h), 1.0f32),
            None => (self.logical_size, self.scale),
        };
        // The fraction of the surface the render fills once its aspect is
        // preserved: (1, 1) unless fixed render size mode meets a window of
        // a different shape, in which case the content is letterboxed and
        // both the cursor mapping and the blit pass account for it.
        let fit = content_fit(self.render_size, surface_px);
        let mut canvas = Canvas::new(user_size, user_scale);
        let now = now_millis();
        let dt = self
            .last_millis
            .map(|last| ((now - last).max(0.0) / 1000.0).min(1.0) as f32)
            .unwrap_or(0.0);
        self.last_millis = Some(now);
        // Flip the cursor from window pixels (upper-left, y-down, physical)
        // to user coordinates (origin at the window's center, y up): the
        // surface-to-user ratio maps the density away in the default mode
        // (1/scale) and the surface-to-buffer ratio in fixed render size
        // mode, and one formula covers both. Dividing by the letterbox fit
        // maps the cursor across the content rectangle, not the whole
        // surface, so a pointer over a bar reads past the play edge.
        let mouse = self.mouse.map(|[mx, my]| {
            [
                (mx - surface_px.0 as f32 / 2.0) * user_size.0 as f32
                    / (surface_px.0 as f32 * fit.0),
                (surface_px.1 as f32 / 2.0 - my) * user_size.1 as f32
                    / (surface_px.1 as f32 * fit.1),
            ]
        });
        let process = &mut self.process;
        let scene = &mut self.scene;
        let keys = &self.keys;
        let typed = &self.typed;
        let mouse_buttons = &self.mouse_buttons;
        {
            let mut ctx = Context {
                canvas: &mut canvas,
                scene,
                keys,
                typed,
                blit_filter: &mut self.blit_filter,
                vsync: &mut self.vsync,
                render_size: &mut self.render_size,
                bloom: &mut self.bloom,
                tone_map: &mut self.tone_map,
                expected_fps: self.expected_fps,
                mouse,
                mouse_buttons,
                frame_processing_ms: self.frame_processing_ms,
                frame_draw_calls: self.frame_draw_calls,
                frame_diagnostic_draw_calls: self.frame_diagnostic_draw_calls,
                mouse_wheel: self.mouse_wheel,
                exit: &mut self.exit_requested,
                keys_frozen: &mut self.keys_frozen,
            };
            process.process(&mut ctx, dt);
        }
        // The wheel delta belongs to this frame only: reset the accumulator
        // so a frame that never reads it discards the delta rather than
        // leaking it into the next.
        self.mouse_wheel = 0.0;
        // The user's process ran; now update the scene tree itself: every
        // node's `Node::process`, children before their parent.
        self.scene.visit();
        // The scene was just updated; draw it into the frame's draw list.
        canvas.draw_scene(&self.scene);
        // Expand the text into per-glyph sprite quads before the sort, so
        // each glyph keeps its node's position in the paint order.
        canvas.expand_text(&mut self.text_atlases);
        // An atlas that packed new glyphs this frame replaced its pixel
        // buffer, so the texture keyed to the old buffer is stale: drop it
        // before the sprite draws, so the fresh buffer uploads and a
        // recycled address can never serve the old texture.
        for atlas in self.text_atlases.values_mut() {
            if let Some(stale) = atlas.take_stale_key() {
                self.sprite_resources.remove(&stale);
                // And the bind groups that bind the old texture, for the same
                // reason as in [`Frost::evict_recycled_buffer`].
                self.sprite_binds.retain(|bind, _| bind.0 != stale.0);
            }
        }

        // Paint order: the draw groups (the base group and the scene's
        // layers) by ascending layer order, higher order on top; within each
        // group, ascending z, lower z behind. The sorts are stable, so
        // draws with equal keys keep call order and the last drawn is on
        // top.
        let draws = canvas.paint_order();

        // The frame's light field: the scene's lights packed in call order
        // with the scene's ambient. The field buffer is grow-only — it is
        // re-created only when this frame's light count beats the peak it
        // was last sized for.
        let field = pack_light_field(&draws, self.scene.ambient);
        let field_buffer = self.field_buffer_for(field.count);
        self.queue.write_buffer(&field_buffer, 0, &field.data);

        // The frame's occluder field: the flagged rectangle shapes packed in
        // call order, then the occluders the app declared this frame with
        // `Canvas::occluder`, cut into per-light blocks culled against the
        // user-space window — the same pixel space the shaders ray through
        // and the occluders were declared in. Same grow-only buffer as the
        // light field, sized by the packed bytes.
        let occluders = pack_occluder_field(
            &draws,
            &canvas.occluders,
            [user_size.0 as f32, user_size.1 as f32],
        );
        let occluder_buffer = self.occluder_buffer_for(occluders.data.len());
        self.queue
            .write_buffer(&occluder_buffer, 0, &occluders.data);

        // The frame's uniforms: one walk, one buffer, one upload.
        //
        // Every draw that has a uniform gets a run of the arena — its own
        // bytes, starting on the device's uniform-offset alignment, as wide as
        // that pipeline's uniform struct — plus the byte offset of its run,
        // recorded by position in the draw list so the draw can reach it with
        // nothing but its index. The offsets are data computed once here
        // rather than arithmetic repeated in the pass below, because the runs
        // are not one size: a polyline's uniform is sixty-five times a
        // rectangle's, and a fixed slot would carry two kilobytes for every
        // other draw in the frame to serve the one that needs them.
        //
        // The bytes come from [`uniform_bytes`], the one place that knows how
        // a draw becomes bytes; the arms below bind and draw, and nothing
        // else. A draw that is skipped — culled by the scissor, or collapsed
        // by a degenerate transform — keeps a run nobody reads.
        let stride = self.arena_stride;
        let mut staging = std::mem::take(&mut self.arena_staging);
        staging.clear();
        let mut offsets = std::mem::take(&mut self.arena_offsets);
        offsets.clear();
        for draw in draws.iter() {
            if let Some(bytes) = uniform_bytes(draw) {
                let at = (staging.len() as u64).next_multiple_of(stride);
                staging.resize(at as usize, 0);
                staging.extend_from_slice(&bytes);
                offsets.push(at as u32);
            } else {
                // No uniform to find: the draw carries its own parameters some
                // other way, or is not drawn at all.
                offsets.push(0);
            }
        }
        if !staging.is_empty() {
            let arena = match &self.uniform_arena {
                Some(buffer) if buffer.size() >= staging.len() as u64 => buffer.clone(),
                _ => {
                    let buffer = self.device.create_buffer(&BufferDescriptor {
                        label: Some("uniform arena"),
                        size: (staging.len() as u64).next_multiple_of(stride),
                        usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    });
                    self.uniform_arena = Some(buffer.clone());
                    // The cached bind groups bind the arena that just went
                    // away; they are built again on the next draw.
                    self.sprite_binds.clear();
                    self.prim_binds.clear();
                    buffer
                }
            };
            // One upload for the frame. Every `write_buffer` copy lands before
            // any draw executes — the fact the engine used to answer with a
            // buffer per draw — and the runs are what makes it a fact worth
            // living with: a draw can only ever be pointed at its own bytes.
            self.queue.write_buffer(&arena, 0, &staging);
        }
        self.arena_staging = staging;
        self.arena_offsets = offsets;

        // The frame's destination: the swapchain texture — or, in fixed
        // render size mode when the surface is not already exactly that
        // size, the offscreen buffer the blit pass stretches over the
        // window afterwards. A surface that already matches the render
        // size needs neither buffer nor pass. The target is taken out of
        // `self` for the frame — the draw loop needs `&mut self` — and
        // stored back after the blit. (This runs before the pipeline
        // borrows below for the same reason: building it needs `&mut self`.)
        let surface_view = output
            .texture
            .create_view(&TextureViewDescriptor::default());
        let mut target = match self.render_size {
            Some(size) if size != [surface_px.0, surface_px.1] => {
                // A cached target serves only while size and filter match;
                // a runtime filter flip rebuilds the sampler once, on the
                // frame that flips (or the next one that stretches).
                let cached = self
                    .blit_target
                    .take()
                    .filter(|t| t.size == size && t.filter == self.blit_filter);
                match cached {
                    Some(t) => Some(t),
                    None => Some(self.new_blit_target(size)),
                }
            }
            // A surface that cannot hold the frame as the shaders wrote it
            // gets the frame carried to it: a linear surface has no hardware
            // encoding, so the gamma pass supplies one, and an 8-bit surface
            // cannot hold what the lights summed past 1.0. Either way the
            // frame is parked in the offscreen buffer, at the window's own
            // size, and the blit pass writes it down. (The buffer is built
            // lazily below when no render size asked for one.)
            _ if self.linear_surface || self.hdr_frames => {
                let size = [surface_px.0, surface_px.1];
                let cached = self
                    .blit_target
                    .take()
                    .filter(|t| t.size == size && t.filter == self.blit_filter);
                match cached {
                    Some(t) => Some(t),
                    None => Some(self.new_blit_target(size)),
                }
            }
            _ => {
                self.blit_target = None;
                None
            }
        };
        // The look knobs are live, and both land on the frame that asked for
        // them: `Context::set_bloom` and `Context::set_tone_map` write these
        // two floats, the stretch pass reads them out of its uniform every
        // frame, and neither needed a pipeline to be rebuilt to get here.
        //
        // Bloom is the one with a cost attached. An app that started without a
        // glow and asks for one mid-scene wakes the chain here, on the frame
        // that asked — the alternative is a frame that quietly draws no glow
        // while waiting for someone else to notice. A chain whose frame buffer
        // clips at 1.0 has nothing to read and stays asleep whatever the
        // strength says; that is `can_bloom`, not this branch's problem.
        if let Some(t) = target.as_mut() {
            t.bloom.set_strength(self.bloom);
            if t.bloom.needs_building() {
                // `needs_building` means `can_bloom`, and `can_bloom` came from
                // this same option being `Some`, so there is no `None` case
                // here in practice — and the frame that found one anyway should
                // still be drawn, just without the glow it cannot have.
                if let Some(view) = t.emissive.as_ref().map(|(_, view)| view.clone()) {
                    let (format, size) = (t.format, t.size);
                    t.bloom
                        .build(&self.device, &self.queue, format, size, &view);
                    // Worth a line in the log: this frame is carrying something
                    // no other frame carries, and an app that wonders at the one
                    // slow frame on the moment it asks for a glow should not have
                    // to know there are chains in here to find it.
                    log::info!("bloom chain built for a {0}x{1} frame", size[0], size[1]);
                    t.bind_group =
                        self.blit_bind_group(&t.view, &t.sampler, &t.fit_buffer, &t.bloom);
                }
            }
        }
        let frame_area = match &target {
            Some(t) => [t.size[0], t.size[1]],
            None => [output.texture.width(), output.texture.height()],
        };
        let view = match &target {
            Some(t) => &t.view,
            None => &surface_view,
        };
        // The scene's second attachment, when the frame buffer has one. It is
        // cleared to black rather than to the frame's clear color: nothing
        // emits a background, and the clear color is a color, not a light.
        let emissive_attachment =
            target
                .as_ref()
                .and_then(|t| t.emissive.as_ref())
                .map(|(_, view)| RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: StoreOp::Store,
                    },
                });
        let two_attachments = emissive_attachment.is_some();

        let (
            Some(line_pipeline),
            Some(polyline_pipeline),
            Some(circle_pipeline),
            Some(rect_pipeline),
            Some(shape_pipeline),
            Some(sprite_pipeline),
            Some(sprite_additive_pipeline),
            Some(particle_pipeline),
            Some(tilemap_pipeline),
        ) = (
            // Owned handles, not borrows of `self`: a wgpu handle is an
            // `Arc`, so the clone is a refcount, and the frame needs the
            // freedom to call `&mut self` between the draws — the sprite
            // bind groups are built, and cached, from inside the loop.
            self.line_pipeline.clone(),
            self.polyline_pipeline.clone(),
            self.circle_pipeline.clone(),
            self.rect_pipeline.clone(),
            self.shape_pipeline.clone(),
            self.sprite_pipeline.clone(),
            self.sprite_additive_pipeline.clone(),
            self.particle_pipeline.clone(),
            self.tilemap_pipeline.clone(),
        )
        else {
            return;
        };
        let Some(particle_index_buffer) = self.particle_index_buffer.clone() else {
            return;
        };

        // The frame's clear color: the last background node in paint order,
        // or the default when the frame has none.
        let clear = clear_color(&draws);
        // One attachment normally, two when the frame buffer carries emissive:
        // a pipeline declaring two outputs cannot draw into a pass offering one
        // any more than the other way round, so this slice's length is decided
        // by the same flag that chose the pipelines' entry points.
        let scene_attachments = [
            Some(RenderPassColorAttachment {
                view,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: clear.r as f64,
                        g: clear.g as f64,
                        b: clear.b as f64,
                        a: clear.a as f64,
                    }),
                    store: StoreOp::Store,
                },
            }),
            emissive_attachment,
        ];

        let mut encoder = self
            .device
            .create_command_encoder(&CommandEncoderDescriptor { label: None });
        // The frame's GPU draw-call count: incremented once per
        // `pass.draw` / `pass.draw_indexed` below, so draws that are skipped
        // (off-screen, background, light, empty) never count. The second
        // counter keeps the same increments for the draws tagged by the
        // `Diagnostics` overlay — its share of the frame's draw calls.
        let mut draw_calls: u32 = 0;
        let mut diagnostic_draws: u32 = 0;
        {
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: None,
                color_attachments: &scene_attachments[..if two_attachments { 2 } else { 1 }],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            // One pass for the whole frame. The background is cleared once
            // for the full target, then each draw sets the scissor to its
            // tight bounding box, so fragments outside it are discarded and
            // the fragment shader only runs over the pixels the object can
            // write. The viewport is left at the full target, so the
            // shaders' pixel coordinates stay absolute.
            let render_area = frame_area;
            // The index is the arena slot: see the pre-pass above. The list
            // moves into the loop, as it did — by value, so the arms keep
            // owning their draw's data.
            for (i, draw) in draws.into_iter().enumerate() {
                let Some([x, y, w, h]) = draw.scissor_rect(render_area) else {
                    // Fully outside the surface, or a background (which
                    // became the clear color); nothing to draw.
                    continue;
                };
                pass.set_scissor_rect(x, y, w, h);
                // No uniform is built here: the frame packed them all before
                // the pass, and an arm that needed one takes its offset out of
                // `arena_offsets` and slides its pipeline's bind group to it.
                match draw {
                    // Every parameter of the line went into the arena with
                    // the rest of the frame; the arm only says where.
                    Draw::Line { diagnostic, .. } => {
                        let bind_group = self.slot_bind_group(
                            "line pipeline",
                            &line_pipeline,
                            LINE_UNIFORM_BYTES,
                            false,
                            &field_buffer,
                            &occluder_buffer,
                        );
                        pass.set_pipeline(&line_pipeline);
                        pass.set_bind_group(0, &bind_group, &[self.arena_offsets[i]]);
                        draw_calls += 1;
                        if diagnostic {
                            diagnostic_draws += 1;
                        }
                        pass.draw(0..3, 0..1);
                    }
                    Draw::Polyline { diagnostic, .. } => {
                        let bind_group = self.slot_bind_group(
                            "polyline pipeline",
                            &polyline_pipeline,
                            POLYLINE_UNIFORM_BYTES,
                            false,
                            &field_buffer,
                            &occluder_buffer,
                        );
                        pass.set_pipeline(&polyline_pipeline);
                        pass.set_bind_group(0, &bind_group, &[self.arena_offsets[i]]);
                        draw_calls += 1;
                        if diagnostic {
                            diagnostic_draws += 1;
                        }
                        pass.draw(0..3, 0..1);
                    }
                    Draw::Circle { diagnostic, .. } => {
                        let bind_group = self.slot_bind_group(
                            "circle pipeline",
                            &circle_pipeline,
                            CIRCLE_UNIFORM_BYTES,
                            false,
                            &field_buffer,
                            &occluder_buffer,
                        );
                        pass.set_pipeline(&circle_pipeline);
                        pass.set_bind_group(0, &bind_group, &[self.arena_offsets[i]]);
                        draw_calls += 1;
                        if diagnostic {
                            diagnostic_draws += 1;
                        }
                        pass.draw(0..3, 0..1);
                    }
                    Draw::Rectangle { diagnostic, .. } => {
                        let bind_group = self.slot_bind_group(
                            "rectangle pipeline",
                            &rect_pipeline,
                            RECT_UNIFORM_BYTES,
                            false,
                            &field_buffer,
                            &occluder_buffer,
                        );
                        pass.set_pipeline(&rect_pipeline);
                        pass.set_bind_group(0, &bind_group, &[self.arena_offsets[i]]);
                        draw_calls += 1;
                        if diagnostic {
                            diagnostic_draws += 1;
                        }
                        pass.draw(0..3, 0..1);
                    }
                    Draw::Shape {
                        world, diagnostic, ..
                    } => {
                        if world.invert().is_none() {
                            // Degenerate transform; the shape collapses to a
                            // line or a point and its inverse does not exist —
                            // which is also why [`uniform_bytes`] left this
                            // draw's run empty.
                            continue;
                        }
                        let bind_group = self.slot_bind_group(
                            "shape pipeline",
                            &shape_pipeline,
                            SHAPE_UNIFORM_BYTES,
                            true,
                            &field_buffer,
                            &occluder_buffer,
                        );
                        pass.set_pipeline(&shape_pipeline);
                        pass.set_bind_group(0, &bind_group, &[self.arena_offsets[i]]);
                        draw_calls += 1;
                        if diagnostic {
                            diagnostic_draws += 1;
                        }
                        pass.draw(0..3, 0..1);
                    }
                    Draw::Sprite {
                        world,
                        data,
                        texture_size,
                        filter,
                        generation,
                        additive,
                        diagnostic,
                        program,
                        // The rest of the sprite — its box, tint, opacity,
                        // emission and sub-rectangle — reached the arena in the
                        // pre-pass; the draw carries only the offset.
                        ..
                    } => {
                        if world.invert().is_none() {
                            // Degenerate transform; the sprite collapses to a
                            // line or a point and its inverse does not exist.
                            // The inverse itself went into the arena, and the
                            // pre-pass left this draw's slot untouched for the
                            // very same question asked of the same transform,
                            // which is why the two walks cannot drift apart.
                            continue;
                        }
                        // Look up the GPU resources for this image, creating
                        // them on first use. Two sprites from the same file
                        // share one texture, so the image is uploaded once
                        // per file; glyph quads from the same atlas share
                        // one atlas texture the same way.
                        let key = (Arc::as_ptr(&data) as *const () as u64, generation);
                        let view = match self.sprite_resources.get(&key) {
                            Some(view) => view.clone(),
                            None => {
                                // This buffer may have recycled a freed
                                // buffer's address: evict the stale entry
                                // before inserting, so it can never serve
                                // pixels to a live buffer.
                                self.evict_recycled_buffer(key);
                                let view = self.sprite_texture(
                                    &data,
                                    [texture_size[0] as f32, texture_size[1] as f32],
                                );
                                self.sprite_resources.insert(key, view.clone());
                                view
                            }
                        };
                        // The filter picks the sampler, not the cache:
                        // bilinear and nearest-neighbor draws of one
                        // image share the texture and split only here.
                        let sampler = self.sampler_for(filter);
                        // The flag chooses a pipeline, not a bind group: the two
                        // share one layout, so an additive sprite reuses the
                        // cached bind group of an ordinary one from the same
                        // image without either knowing about it. A program
                        // is a second answer to the same question — the
                        // game's own source, compiled once and cached — and
                        // it too changes only the pipeline: the layout is
                        // the sprite's, so the bind group below is the same
                        // object an ordinary sprite of this image already
                        // uses, and the additive flag stands aside, one
                        // pipeline per program being the honest amount.
                        let custom = match program.as_ref() {
                            Some(source) => self.custom_pipeline(source),
                            None => None,
                        };
                        let pipeline: &RenderPipeline = match &custom {
                            Some(pipeline) => pipeline,
                            None if additive => &sprite_additive_pipeline,
                            None => &sprite_pipeline,
                        };
                        let bind_group = self.sprite_bind_group(
                            pipeline,
                            key,
                            matches!(filter, SpriteFilter::Linear),
                            &view,
                            &sampler,
                            &field_buffer,
                            &occluder_buffer,
                        );
                        pass.set_pipeline(pipeline);
                        // The one number that is still per-draw, and it is a
                        // u32: where this draw's bytes landed in the arena.
                        pass.set_bind_group(0, &bind_group, &[self.arena_offsets[i]]);
                        draw_calls += 1;
                        if diagnostic {
                            diagnostic_draws += 1;
                        }
                        pass.draw(0..3, 0..1);
                    }
                    Draw::Particles {
                        data,
                        count,
                        color,
                        kind,
                        aspect,
                        sprite_data,
                        sprite_size,
                        sprite_generation,
                        lit,
                        lit_layers,
                        diagnostic,
                        ..
                    } => {
                        if count == 0 {
                            continue;
                        }
                        // The sampled texture: for a sprite batch, the
                        // batch's image — shared with the sprite pipeline's
                        // cache, so a particle image and a sprite from the
                        // same file upload once — and for the SDF kinds,
                        // which never sample, the 1x1 placeholder.
                        let (view, sampler) = match sprite_data.filter(|_| kind >= 1.5) {
                            Some(image) => {
                                // The image's stamped generation identifies
                                // the buffer: a rebuilt in-memory image
                                // gets a fresh one, so a recycled address
                                // can never hit a stale texture.
                                let key =
                                    (Arc::as_ptr(&image) as *const () as u64, sprite_generation);
                                let view = match self.sprite_resources.get(&key) {
                                    Some(view) => view.clone(),
                                    None => {
                                        self.evict_recycled_buffer(key);
                                        let view = self.sprite_texture(
                                            &image,
                                            [sprite_size[0] as f32, sprite_size[1] as f32],
                                        );
                                        self.sprite_resources.insert(key, view.clone());
                                        view
                                    }
                                };
                                // Particle images keep bilinear sampling:
                                // `ParticleShape` carries no filter.
                                (view, self.sampler_for(SpriteFilter::Linear))
                            }
                            None => self.particle_placeholder.clone(),
                        };
                        // The whole batch is one instanced draw: the shared
                        // index buffer expands every instance into a tight
                        // quad, and the fragment stage evaluates the batch's
                        // shape per pixel. The batch's uniform and instance
                        // data are this draw's own buffers: see
                        // [`Frost::particle_uniform`].
                        let uniform_data = particles_uniform_data(
                            [render_area[0] as f32, render_area[1] as f32],
                            color,
                            kind,
                            aspect,
                            lit,
                            lit_layers,
                        );
                        let (_uniform, _instances, bind_group) = self.particle_uniform(
                            &particle_pipeline,
                            &data,
                            &uniform_data,
                            &view,
                            &sampler,
                            &field_buffer,
                            &occluder_buffer,
                        );
                        pass.set_pipeline(&particle_pipeline);
                        pass.set_bind_group(0, &bind_group, &[]);
                        pass.set_index_buffer(
                            particle_index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        draw_calls += 1;
                        if diagnostic {
                            diagnostic_draws += 1;
                        }
                        pass.draw_indexed(0..6, 0, 0..count);
                    }
                    Draw::TileMap {
                        data,
                        count,
                        color,
                        atlas_data,
                        atlas_size,
                        atlas_generation,
                        filter,
                        lit,
                        lit_layers,
                        diagnostic,
                        ..
                    } => {
                        if count == 0 {
                            continue;
                        }
                        // The atlas rides the shared texture cache: a
                        // tileset drawn on the map and shown as a sprite
                        // (or a particle image) uploads one GPU texture.
                        let key = (
                            Arc::as_ptr(&atlas_data) as *const () as u64,
                            atlas_generation,
                        );
                        let view = match self.sprite_resources.get(&key) {
                            Some(view) => view.clone(),
                            None => {
                                self.evict_recycled_buffer(key);
                                let view = self.sprite_texture(
                                    &atlas_data,
                                    [atlas_size[0] as f32, atlas_size[1] as f32],
                                );
                                self.sprite_resources.insert(key, view.clone());
                                view
                            }
                        };
                        let sampler = self.sampler_for(filter);
                        let uniform_data = tilemap_uniform_data(
                            [render_area[0] as f32, render_area[1] as f32],
                            color,
                            lit,
                            lit_layers,
                        );
                        let (_uniform, _instances, bind_group) = self.tilemap_uniform(
                            &tilemap_pipeline,
                            &data,
                            &uniform_data,
                            &view,
                            &sampler,
                            &field_buffer,
                            &occluder_buffer,
                        );
                        pass.set_pipeline(&tilemap_pipeline);
                        pass.set_bind_group(0, &bind_group, &[]);
                        // The shared unit-quad index buffer — the very
                        // one the particle batch walks.
                        pass.set_index_buffer(
                            particle_index_buffer.slice(..),
                            wgpu::IndexFormat::Uint32,
                        );
                        draw_calls += 1;
                        if diagnostic {
                            diagnostic_draws += 1;
                        }
                        pass.draw_indexed(0..6, 0, 0..count);
                    }
                    // A background's scissor rect is `None`, so it continued
                    // above; this arm keeps the match exhaustive.
                    Draw::Background { .. } => {}
                    // A light's scissor rect is `None`, so it continued
                    // above; this arm keeps the match exhaustive.
                    Draw::Light { .. } => {}
                    // Text is expanded into glyph sprites before the render
                    // loop, so it never reaches the match; this arm keeps it
                    // exhaustive.
                    Draw::Text { .. } => {}
                }
            }
        }
        // Fixed render size mode: stretch the buffer over the window — one
        // letterbox quad, one bilinear sample per window pixel, scaled down
        // to the content rectangle when the shapes differ (black bars from
        // the clear around it). The surface needs no clear when
        // rendering direct.
        if let (Some(t), Some(blit_pipeline)) = (target.as_ref(), self.blit_pipeline.as_ref()) {
            // A dozen small passes before the frame leaves for the window: the
            // emissive target in, four blurred levels out, which the stretch
            // below adds back as it samples. A dormant chain has no levels, so
            // this draws nothing at all when the app asked for no bloom.
            t.bloom.run(&mut encoder);
            self.queue.write_buffer(
                &t.fit_buffer,
                0,
                &fit_bytes(fit, t.bloom.weights, self.tone_map),
            );
            let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                label: Some("blit pass"),
                color_attachments: &[Some(RenderPassColorAttachment {
                    view: &surface_view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });
            pass.set_pipeline(blit_pipeline);
            pass.set_bind_group(0, &t.bind_group, &[]);
            pass.draw(0..6, 0..1);
        }
        // Put the target back for the next frame — or leave `self` empty,
        // so a surface that has since resized to exactly the render size
        // finds nothing to reuse and drops the buffer for good.
        self.blit_target = target;
        self.queue.submit([encoder.finish()]);
        // Record this frame's processing time and draw-call count before the
        // present: the frame's own `Context` already went out without them,
        // so these values reach the next frame's, where they are the "last
        // frame's" values.
        self.frame_processing_ms = now_millis() - t0;
        self.frame_draw_calls = draw_calls;
        self.frame_diagnostic_draw_calls = diagnostic_draws;
        self.queue.present(output);
        log::trace!("render: frame presented");
    }

    /// Acquires the current surface texture.
    ///
    /// Returns the texture and whether the surface should be reconfigured
    /// because it is only suboptimal. A frame that gets no texture is counted
    /// rather than announced; see [`Self::surface_gone`].
    fn acquire_frame(&mut self) -> (Option<SurfaceTexture>, bool) {
        // A chain that has never answered `configure` has no drawables to
        // hand over — and wgpu takes the asking of an unconfigured surface
        // as a panic, which no guest owes the host's main thread. The
        // frame is skipped; the retry road in `render` is already working
        // on the configure.
        if self.format.is_none() {
            return (None, false);
        }
        let Some(surface) = self.surface.as_ref() else {
            return (None, false);
        };
        match surface.get_current_texture() {
            CurrentSurfaceTexture::Success(tex) => {
                self.surface_back();
                (Some(tex), false)
            }
            CurrentSurfaceTexture::Suboptimal(tex) => {
                // Its own line each time, flood-risk included: unlike the two
                // below, this one is followed by a reconfigure, so a repeat is
                // a machine doing something and not merely waiting.
                log::info!("suboptimal surface texture, reconfiguring");
                self.surface_back();
                (Some(tex), true)
            }
            CurrentSurfaceTexture::Timeout => {
                self.surface_gone("surface timeout");
                (None, false)
            }
            _ => {
                self.surface_gone("no surface texture available");
                (None, false)
            }
        }
    }

    /// Note one frame the surface would not give, and say so only if this is
    /// the spell's first: the first frame is news, the two-thousand-and-first
    /// is the same news, and the useful part — how long it lasted — is not
    /// knowable until it ends.
    fn surface_gone(&mut self, why: &'static str) {
        match &mut self.surface_gone {
            Some((_, frames)) => *frames += 1,
            None => {
                log::info!("{why}, skipping frames");
                self.surface_gone = Some((why, 1));
            }
        }
    }

    /// And note the frame that did get one, which is where the spell's length
    /// is finally worth printing.
    fn surface_back(&mut self) {
        if let Some((why, frames)) = self.surface_gone.take() {
            log::info!("surface back after {frames} skipped frames ({why})");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A macOS fullscreen transition as it was measured on the notch Mac of
    /// `docs/benchmarking.md` §17: five sizes inside a second, one of them
    /// taller than the display the window is being animated into, and the
    /// engine must rebuild once, for the size the window actually keeps.
    #[test]
    fn an_animation_through_sizes_rebuilds_once_for_the_last_one() {
        let mut pending = PendingResize::default();
        // A quiet window asks and is given nothing.
        assert_eq!(pending.due(0.0), None);

        // The flight, at display rate: 16 ms apart, and four of these five
        // sizes are positions in an animation rather than a window's home.
        for (t, (w, h)) in [
            (0.0, (4112, 2572)),
            (16.0, (4112, 2658)),
            (32.0, (4112, 2572)),
            (48.0, (4112, 2714)),
            (64.0, (4112, 2572)),
        ] {
            pending.note(w, h, t);
            assert_eq!(pending.due(t), None, "the window is still in the air");
        }

        // Frames tick past while it settles; none of them rebuilds.
        for t in [70.0, 100.0, 140.0] {
            assert_eq!(pending.due(t), None);
        }

        // A hundred ms after the last event: once, and only once.
        assert_eq!(pending.due(164.0), Some((4112, 2572)));
        assert_eq!(pending.due(200.0), None, "one rebuild per settle");
    }

    /// The other half of the promise: a dragged window lags the pointer by at
    /// most the settle window, and lands on the size the pointer was left at.
    #[test]
    fn a_drag_settles_at_the_size_it_was_left_at() {
        let mut pending = PendingResize::default();
        pending.note(1280, 720, 0.0);
        assert_eq!(pending.due(99.0), None, "just short is not settled");
        assert_eq!(pending.due(100.0), Some((1280, 720)));

        // A new drag is a new question: the clock restarts and whatever was
        // pending is forgotten rather than delivered late.
        pending.note(800, 600, 120.0);
        pending.note(640, 480, 130.0);
        assert_eq!(pending.due(230.0), Some((640, 480)));
        assert_eq!(pending.due(300.0), None);
    }
}
