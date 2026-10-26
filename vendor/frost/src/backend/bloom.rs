//! The bloom chain: the emissive target in, blurred levels out — how many
//! is the preset's length, not a fact about this file.
//!
//! Each level is half the linear size of the one above, each made of three
//! full-screen passes — a tent downsample, then a separable Gaussian across
//! and down. Each step down is a quarter of the area, so the whole chain
//! costs about a third of one full-resolution pass however big the window
//! is, and the widest glow is the cheapest of them — the opposite of what
//! one wide blur at full resolution costs. Depth is what buys width: the
//! same kernel one level down covers twice the screen, so reach grows
//! geometrically while the added level costs a quarter of the one above
//! it, down to [`MAX_LEVELS`], where the frame is small enough to blur
//! for free and the uniform stops having room.
//!
//! The levels come out *separate* rather than accumulated into one, because
//! the present pass wants them separate: a glow's width is a look, and one
//! weight per level chosen where the frame is composited is easier to reason
//! about than one blur radius chosen where it is generated. [`Bloom::weights`]
//! carries them to `blit.wgsl`, already multiplied by the user's strength —
//! and its length is the chain's depth, so a wider glow is a longer preset,
//! and levels a preset does not ask for are never built, never run, never
//! sampled.
//!
//! Nothing here knows what bloom is for. It takes a texture of radiance and
//! returns the same picture smeared at the preset's scales; the engine hands
//! it the emissive target ([`crate::SceneNode::emission`]) and the present
//! pass adds the result back. Change either end and this module does not
//! notice.
//!
//! [`crate::SceneNode::emission`]: crate::SceneNode::emission

use crate::backend::copy::pad_rows;
use crate::shaders::BLOOM_SHADER;
use std::borrow::Cow;
use wgpu::{
    BindGroup, BindGroupDescriptor, BindGroupEntry, BindingResource, Buffer, BufferBinding,
    BufferDescriptor, BufferUsages, Color, CommandEncoder, Device, Extent3d, FilterMode,
    FragmentState, LoadOp, PipelineCompilationOptions, Queue, RenderPassColorAttachment,
    RenderPassDescriptor, RenderPipeline, RenderPipelineDescriptor, Sampler, SamplerDescriptor,
    ShaderModule, ShaderModuleDescriptor, ShaderSource, StoreOp, TexelCopyBufferLayout, Texture,
    TextureDescriptor, TextureDimension, TextureFormat, TextureUsages, TextureView,
    TextureViewDescriptor, VertexState,
};

/// How wide the plumbing goes, as opposed to how wide the glow is: the
/// present pass's uniform, bindings and weight vector are all sized for this
/// many levels, and the shipped preset's length — `WEIGHTS.len()` — decides
/// how many of them are built, run and sampled. Six is where the frame runs
/// out of useful resolution to blur (a 4K frame bottoms out at 60×34), and
/// where `blit.wgsl`'s weight array stops — `fit_bytes` asserts the cap so a
/// seventh level has to widen the shader and the uniform knowingly, not by
/// overrunning them.
///
/// Depth is the chain's width dial. The other candidate — taking the blur's
/// taps further apart — was tried and measured, and it is a trap: the kernel
/// is thirteen taps in six fetches, so widening the step undersamples it, and
/// the taps show through as rings. A small bright dot's fall-off, measured
/// straight outward, is monotone at the current step, has a bump 6 pixels out
/// at 1.5× the step, two bumps at 2×, and at 3× has no shape at all — fog
/// rather than a halo. Levels are the way to go wider: each is a halving of
/// an already-averaged texture, so its taps stay well sampled however far
/// down it sits, and the reach grows geometrically — doubled by every level
/// the preset adds, whose three passes cost around a twentieth of a
/// millisecond at this adapter's fixed cost per pass. The risk of depth,
/// which [`WEIGHTS`] is what holds down, is the whole frame lifting instead
/// of the halo growing.
pub(crate) const MAX_LEVELS: usize = 6;

/// Radiance under this contributes nothing to the glow, and [`KNEE`] decides
/// how widely the band above it arrives. Below 1.0 on purpose: an emission
/// authored at exactly white is common, and a threshold at 1.0 would leave it
/// out of the effect altogether — a lamp would bloom only if someone remembered
/// that a colour is allowed to hold `2.0`.
const THRESHOLD: f32 = 0.8;

/// How wide a band the threshold ramps across, in radiance units. A hard
/// threshold draws a contour *on* the light — the lamp's rim falls below it
/// while its core stays above — so this ramp is where the behaviour the docs
/// promise lives: authored past 1.0 blooms hard, ordinary bright blooms a
/// little, and nothing jumps between the two.
const KNEE: f32 = 0.4;

/// What each level adds to the window, fine to coarse — the shipped preset,
/// whose length *is* the chain's depth: four weights, four levels, twelve
/// passes deciding how far the light travels. Summing to 1 so that a frame of
/// bloom cannot double the frame's brightness; the fine level leads because
/// it is the one that still holds the lamp's shape, and the coarse ones are
/// the haze — too much haze reads as a dirty lens.
///
/// The fine level took the whole cost of the fourth level, down from `0.5`,
/// which buys a little reach and takes back a little of the wash: a black patch
/// inside a bright plate measured +0.625 of radiance handed back into it before
/// this rebalance and +0.556 after, at the same strength. That wash — a source
/// adding its own light back onto its own face — is what reads as the glow
/// blurring the thing that glows.
///
/// Flatter balances were tried for width and are not the lever they look like.
/// `[0.32, 0.26, 0.22, 0.20]` moved a point halo's brightness 20 pixels out
/// from 10 to 11 eighths of full scale, because the pyramid *averages* on the
/// way down: a coarse level holds the mean radiance of its footprint, so a
/// small source contributes almost nothing to the wide levels however much they
/// are weighted, and a big one contributes nearly all of it. Width therefore
/// comes from source size and from how many levels there are, not from how the
/// weights are balanced — which is why a wider look is a longer preset rather
/// than a flatter one, and why a `dormant`-length `[0.0; MAX_LEVELS]` answer
/// from [`weights_for`] means "past the preset's end, nothing".
const WEIGHTS: &[f32] = &[0.40, 0.28, 0.20, 0.12];

const _: () = assert!(
    WEIGHTS.len() <= MAX_LEVELS,
    "the preset outgrew the plumbing"
);

/// The numbers the present pass reads for one request, the preset scaled by
/// strength and padded to the plumbing's width — so the levels a preset does
/// not ask for arrive as the same zeros that "no bloom" travels on, and the
/// shader's per-level guards never fetch a bound black texel by accident. Kept
/// apart from both the chain and the shader because it is the one piece of this
/// that is arithmetic rather than GPU work, and the two cases it has to get
/// right — no strength asked for, and nothing to read — are the two a caller
/// can only discover by asking for a glow it will not get.
fn weights_for(strength: f32, can_bloom: bool) -> [f32; MAX_LEVELS] {
    let mut weights = [0.0; MAX_LEVELS];
    if can_bloom && strength > 0.0 {
        for (weight, base) in weights.iter_mut().zip(WEIGHTS) {
            *weight = *base * strength;
        }
    }
    weights
}

/// One level: two textures — the one the level is read from and the scratch the
/// horizontal blur lands in — plus the three bind groups that write them.
struct Level {
    /// The level's texture: written by the downsample, written again by the
    /// vertical blur (in place, from the scratch), and sampled by the present
    /// pass. Held beside its view so the ownership is obvious, as everywhere
    /// else in the backend.
    _fine: Texture,
    fine_view: TextureView,
    /// The scratch the horizontal blur lands in, so the vertical pass never
    /// reads a texture it is also writing.
    _scratch: Texture,
    scratch_view: TextureView,
    /// Emissive target, or the level above → `fine`.
    down: BindGroup,
    /// `fine` → `scratch`, across.
    across: BindGroup,
    /// `scratch` → `fine`, down.
    along: BindGroup,
}

/// The frame's bloom chain, built for one render target's size.
///
/// It lives and dies with the target: the levels are fractions of its extent, so
/// a resize means a new chain, not a scaled one. A chain with no bloom to do is
/// still a `Bloom` ([`Bloom::dormant`]), because the present pass binds
/// [`MAX_LEVELS`] bloom textures whatever the answer is; what differs is whether
/// its weights are zero, which is what keeps the samples out of the frame.
pub(crate) struct Bloom {
    /// The levels, fine first — as many as the preset asks for. Empty when
    /// dormant, which is how `run` knows to draw nothing.
    levels: Vec<Level>,
    /// `[downsample, blur]`, in that order.
    pipelines: [RenderPipeline; 2],
    /// One sampler for the chain's fetches and the present pass's: linear,
    /// clamped. Clamping matters more than the filtering — a glow at the
    /// frame's edge samples outside it constantly, and a wrapping edge drags
    /// the far side of the picture around it.
    pub(crate) sampler: Sampler,
    /// What the present pass binds, fine first: the built levels, then one
    /// black texel for every slot the preset left empty — up to the plumbing's
    /// width, so the bind group has one shape whatever the chain decided.
    pub(crate) views: [TextureView; MAX_LEVELS],
    /// The black texel's view, kept beside the levels so a woken chain can pad
    /// its own empty slots the way [`Bloom::dormant`] pads them all.
    black: TextureView,
    /// The present pass's weights, strength already multiplied in and padded
    /// with zeros past the preset's end: all zero is how "no bloom" reaches the
    /// shader, whose per-level guards on it keep the samples off the frame.
    pub(crate) weights: [f32; MAX_LEVELS],
    /// What the app is asking for right now: `EmbeddedConfig::bloom` at startup,
    /// [`Context::set_bloom`](crate::Context::set_bloom) afterwards. Kept
    /// because the weights are derived from it, and it can change under us.
    strength: f32,
    /// Whether there is an emissive target to read at all. False on a frame
    /// buffer that clips at 1.0, and then no amount of asking builds a chain —
    /// the strength is remembered and the weights stay zero.
    can_bloom: bool,
    /// The passes' uniforms, one each, written once here and never again. Held
    /// because a bind group keeps its buffers alive but the chain wants to say
    /// how much it weighs.
    _uniforms: Vec<Buffer>,
}

impl Bloom {
    /// Builds the chain for a frame buffer of `extent` reading `emissive`.
    ///
    /// `strength` is [`EmbeddedConfig::bloom`](crate::embedded::EmbeddedConfig::bloom). Anything at or
    /// below zero — and any frame with no float target to read, which is the
    /// same answer the emissive target itself gets — gives a
    /// [dormant](Self::dormant) chain.
    pub(crate) fn new(
        device: &Device,
        queue: &Queue,
        format: TextureFormat,
        extent: [u32; 2],
        emissive: Option<&TextureView>,
        strength: f32,
    ) -> Self {
        let module = device.create_shader_module(ShaderModuleDescriptor {
            label: Some("bloom shaders"),
            source: ShaderSource::Wgsl(Cow::Borrowed(BLOOM_SHADER)),
        });
        let sampler = device.create_sampler(&SamplerDescriptor {
            label: Some("bloom sampler"),
            mag_filter: FilterMode::Linear,
            min_filter: FilterMode::Linear,
            ..Default::default()
        });
        let pipelines = [
            Self::pipeline(device, &module, format, "bloom downsample", "fs_down"),
            Self::pipeline(device, &module, format, "bloom blur", "fs_blur"),
        ];
        // The two questions are separate and stay separate from here on: is
        // there anything to read, and is the app asking. A chain with nothing
        // to read never wakes up, whatever the strength does later.
        let can_bloom = emissive.is_some();
        let mut this = Self::dormant(
            device, queue, format, pipelines, sampler, strength, can_bloom,
        );
        if let Some(emissive) = emissive.filter(|_| strength > 0.0) {
            this.build(device, queue, format, extent, emissive);
        }
        this
    }

    /// Builds the levels, or rebuilds them against a new extent. This is what
    /// an app wakes when it asks for bloom at runtime ([`Context::set_bloom`])
    /// on a frame that was built without a chain: the pipelines and the sampler
    /// already exist, so what it costs is the two textures, three bind groups
    /// and their uniforms per level the preset asks for — and the frame that
    /// pays for it is the frame that asked.
    pub(crate) fn build(
        &mut self,
        device: &Device,
        queue: &Queue,
        format: TextureFormat,
        extent: [u32; 2],
        emissive: &TextureView,
    ) {
        let depth = WEIGHTS.len();
        let mut levels: Vec<Level> = Vec::with_capacity(depth);
        let mut uniforms = Vec::with_capacity(depth * 3);
        // What the next level reads: the emissive target at the top of the
        // chain, this level's blurred result below that. Blurred, not raw —
        // the light is chosen once, at the top.
        let mut source = emissive;
        let mut source_extent = extent;
        for level_index in 0..depth {
            let extent = [(source_extent[0] / 2).max(1), (source_extent[1] / 2).max(1)];
            let new = |label: &str| {
                device.create_texture(&TextureDescriptor {
                    label: Some(label),
                    size: Extent3d {
                        width: extent[0],
                        height: extent[1],
                        depth_or_array_layers: 1,
                    },
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: TextureDimension::D2,
                    format,
                    // `COPY_SRC` for one reason: a test reads a level back to
                    // see what the chain actually made, which is the only way
                    // to tell a dim glow from an empty one.
                    usage: TextureUsages::RENDER_ATTACHMENT
                        | TextureUsages::TEXTURE_BINDING
                        | TextureUsages::COPY_SRC,
                    view_formats: &[],
                })
            };
            let fine = new("bloom level");
            let scratch = new("bloom blur scratch");
            let fine_view = fine.create_view(&TextureViewDescriptor::default());
            let scratch_view = scratch.create_view(&TextureViewDescriptor::default());
            let src_texel = [1.0 / source_extent[0] as f32, 1.0 / source_extent[1] as f32];
            let texel = [1.0 / extent[0] as f32, 1.0 / extent[1] as f32];
            let mut bind = |label: &'static str,
                            pipeline: &RenderPipeline,
                            source: &TextureView,
                            bytes: [u8; 32]|
             -> BindGroup {
                let uniform = device.create_buffer(&BufferDescriptor {
                    label: Some(label),
                    size: 32,
                    usage: BufferUsages::UNIFORM | BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                });
                queue.write_buffer(&uniform, 0, &bytes);
                let group = device.create_bind_group(&BindGroupDescriptor {
                    label: Some(label),
                    layout: &pipeline.get_bind_group_layout(0),
                    entries: &[
                        BindGroupEntry {
                            binding: 0,
                            resource: BindingResource::TextureView(source),
                        },
                        BindGroupEntry {
                            binding: 1,
                            resource: BindingResource::Sampler(&self.sampler),
                        },
                        BindGroupEntry {
                            binding: 2,
                            resource: BindingResource::Buffer(BufferBinding {
                                buffer: &uniform,
                                offset: 0,
                                size: None,
                            }),
                        },
                    ],
                });
                uniforms.push(uniform);
                group
            };
            let down = bind(
                "bloom downsample bind",
                &self.pipelines[0],
                source,
                // The threshold rides at the top of the chain only: below it
                // the bright parts have already been chosen, and thresholding
                // again would redraw the same contour at every scale.
                bloom_bytes(
                    src_texel,
                    if level_index == 0 { THRESHOLD } else { -1.0 },
                    KNEE,
                    [0.0, 0.0],
                ),
            );
            // One axis each: the step for the across pass is the horizontal
            // texel alone, and for the along pass the vertical one. Passing
            // both makes the "separable" blur a diagonal smear, which is the
            // single easiest way to get this wrong and the single easiest way
            // to see it: the glow leans.
            let across = bind(
                "bloom blur across bind",
                &self.pipelines[1],
                &fine_view,
                bloom_bytes(texel, -1.0, KNEE, [texel[0], 0.0]),
            );
            // One pass per axis: the kernel is wide enough that a second
            // iteration would only cost another pass, which is the expensive
            // thing about this chain.
            let along = bind(
                "bloom blur along bind",
                &self.pipelines[1],
                &scratch_view,
                bloom_bytes(texel, -1.0, KNEE, [0.0, texel[1]]),
            );
            levels.push(Level {
                _fine: fine,
                fine_view,
                _scratch: scratch,
                scratch_view,
                down,
                across,
                along,
            });
            source_extent = extent;
            // The borrow checker's version of "the chain feeds itself": the
            // next level reads the `fine` view that just moved into `levels`.
            source = &levels[level_index].fine_view;
        }
        self.levels = levels;
        // Built levels first, the black texel after: the bind group is always
        // `MAX_LEVELS` wide, and the weights say the same thing the views do.
        let mut views = std::array::from_fn(|_| self.black.clone());
        for (view, level) in views.iter_mut().zip(&self.levels) {
            *view = level.fine_view.clone();
        }
        self.views = views;
        self.weights = weights_for(self.strength, self.can_bloom);
        self._uniforms = uniforms;
    }

    /// Asks for a different strength. Free, because the strength reaches the
    /// shader as the preset's numbers in a uniform the present pass rewrites
    /// every frame anyway; what it cannot do is build a chain that was never
    /// possible, which is what [`Self::needs_building`] is for. Zero is not just
    /// an invisible glow: it parks [`Self::run`] as well, so an app that asks
    /// for no bloom on a frame pays no bloom passes for it.
    pub(crate) fn set_strength(&mut self, strength: f32) {
        self.strength = strength;
        self.weights = weights_for(strength, self.can_bloom);
    }

    /// Whether the app is asking for a glow this chain has no levels to make.
    /// The caller's job then is [`Self::build`], and it needs the emissive
    /// view and the extent, which belong to the frame buffer, not to us.
    pub(crate) fn needs_building(&self) -> bool {
        self.can_bloom && self.strength > 0.0 && self.levels.is_empty()
    }

    /// A chain that does nothing, for a frame with no float target or an app
    /// that asked for no bloom. The present pass still binds `MAX_LEVELS`
    /// textures, so that many exist — one black texel, `MAX_LEVELS` times — and
    /// the weights are zero, so nothing looks at them.
    fn dormant(
        device: &Device,
        queue: &Queue,
        format: TextureFormat,
        pipelines: [RenderPipeline; 2],
        sampler: Sampler,
        strength: f32,
        can_bloom: bool,
    ) -> Self {
        let texture = device.create_texture(&TextureDescriptor {
            label: Some("bloom placeholder"),
            size: Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: TextureDimension::D2,
            format,
            usage: TextureUsages::TEXTURE_BINDING | TextureUsages::COPY_DST,
            view_formats: &[],
        });
        // One black texel, actually written. A texture that exists only to be
        // bound could be left uninitialised — nothing samples it, its weights
        // are zero — but "nothing samples it" is a promise about a branch, and
        // a texel costs four bytes. Written through the same padded pitch as
        // every other upload, because a copy's row pitch is a law of the API
        // and not of the texture's size: see [`pad_rows`].
        let (pixels, pitch) = pad_rows(&[0, 0, 0, 0], 1, 1, 4);
        queue.write_texture(
            texture.as_image_copy(),
            &pixels,
            TexelCopyBufferLayout {
                bytes_per_row: Some(pitch),
                rows_per_image: Some(1),
                ..Default::default()
            },
            Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        let view = texture.create_view(&TextureViewDescriptor::default());
        if strength > 0.0 && !can_bloom {
            log::info!(
                "bloom asked for at {strength} on a frame buffer that clips at 1.0: ignored"
            );
        }
        Self {
            levels: Vec::new(),
            pipelines,
            sampler,
            views: std::array::from_fn(|_| view.clone()),
            black: view,
            weights: weights_for(strength, can_bloom),
            strength,
            can_bloom,
            _uniforms: Vec::new(),
        }
    }

    /// A full-screen pass: no vertex buffers, one float target it overwrites
    /// whole. Nothing blends here — the chain computes, it does not composite.
    fn pipeline(
        device: &Device,
        module: &ShaderModule,
        format: TextureFormat,
        label: &str,
        entry_point: &str,
    ) -> RenderPipeline {
        device.create_render_pipeline(&RenderPipelineDescriptor {
            label: Some(label),
            layout: None,
            vertex: VertexState {
                module,
                entry_point: Some("vs_main"),
                compilation_options: PipelineCompilationOptions::default(),
                buffers: &[],
            },
            fragment: Some(FragmentState {
                module,
                entry_point: Some(entry_point),
                compilation_options: PipelineCompilationOptions::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                ..Default::default()
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            cache: None,
            multiview_mask: None,
        })
    }

    /// Runs the chain: emissive in, one blurred level per preset weight out,
    /// three full-screen passes per level over about a third of one frame's
    /// pixels. Does nothing when dormant.
    pub(crate) fn run(&self, encoder: &mut CommandEncoder) {
        // Nothing to make, two ways over: no levels to make it from, and no
        // weight to put the result anywhere. The second is the app that turned
        // the glow down to zero and left the chain standing ready — and since
        // the present pass branches on those same zeros, every pass here would
        // compute a picture nothing reads. Passes are what this chain costs, so
        // skipping all of them is the entire cost of the off state.
        if self.levels.is_empty() || self.weights.iter().all(|w| *w == 0.0) {
            return;
        }
        for level in &self.levels {
            let passes = [
                (&self.pipelines[0], &level.down, &level.fine_view),
                (&self.pipelines[1], &level.across, &level.scratch_view),
                (&self.pipelines[1], &level.along, &level.fine_view),
            ];
            for (pipeline, bind_group, target) in passes {
                let mut pass = encoder.begin_render_pass(&RenderPassDescriptor {
                    label: Some("bloom pass"),
                    color_attachments: &[Some(RenderPassColorAttachment {
                        view: target,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            // The triangle covers every pixel of every level,
                            // so clearing is a tiler's convenience rather than
                            // something the frame depends on.
                            load: LoadOp::Clear(Color::BLACK),
                            store: StoreOp::Store,
                        },
                    })],
                    depth_stencil_attachment: None,
                    timestamp_writes: None,
                    occlusion_query_set: None,
                    multiview_mask: None,
                });
                pass.set_pipeline(pipeline);
                pass.set_bind_group(0, bind_group, &[]);
                pass.draw(0..3, 0..1);
            }
        }
    }
}

/// The chain's uniform: source texel size, threshold (negative to skip), knee,
/// blur step. 32 bytes, written once when the chain is built and never again —
/// which is what keeps the chain's passes a frame from costing any staging.
fn bloom_bytes(texel: [f32; 2], threshold: f32, knee: f32, spread: [f32; 2]) -> [u8; 32] {
    let mut b = [0u8; 32];
    b[0..4].copy_from_slice(&texel[0].to_le_bytes());
    b[4..8].copy_from_slice(&texel[1].to_le_bytes());
    b[8..12].copy_from_slice(&threshold.to_le_bytes());
    b[12..16].copy_from_slice(&knee.to_le_bytes());
    b[16..20].copy_from_slice(&spread[0].to_le_bytes());
    b[20..24].copy_from_slice(&spread[1].to_le_bytes());
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The uniform's layout, pinned against `bloom.wgsl`'s `BloomUniforms`:
    /// texel at 0, threshold at 8, knee at 12, spread at 16. A drift here is a
    /// chain that blurs by the wrong step or never thresholds at all, and every
    /// symptom is a picture that merely looks slightly wrong.
    #[test]
    fn the_bloom_uniform_lands_where_the_shader_reads_it() {
        let b = bloom_bytes([0.25, 0.5], 0.8, 0.4, [0.001, 0.002]);
        let f = |r: std::ops::Range<usize>| f32::from_le_bytes(b[r].try_into().unwrap());
        assert_eq!((f(0..4), f(4..8)), (0.25, 0.5));
        assert_eq!((f(8..12), f(12..16)), (0.8, 0.4));
        assert_eq!((f(16..20), f(20..24)), (0.001, 0.002));
        assert_eq!(&b[24..], &[0u8; 8]);
    }

    /// The two ways a request produces no light are the two a caller cannot see
    /// for itself: asking for nothing, and asking on a frame buffer that keeps
    /// nothing above white. Both have to come back as exact zeros across the
    /// whole plumbing, because zero is what the present pass branches on to skip
    /// its samples — anything else, a negative strength included, pays for
    /// samples of black. And the levels past the preset's end must be zero at a
    /// real strength too: they bind a black texel, and a stray nonzero there
    /// would be the one fetch the guard was built to keep off the frame.
    #[test]
    fn only_a_real_request_gets_a_weight() {
        assert_eq!(weights_for(1.0, true), [0.40, 0.28, 0.20, 0.12, 0.0, 0.0]);
        assert_eq!(weights_for(0.5, true), [0.20, 0.14, 0.10, 0.06, 0.0, 0.0]);
        assert_eq!(weights_for(0.0, true), [0.0; MAX_LEVELS]);
        assert_eq!(weights_for(-1.0, true), [0.0; MAX_LEVELS]);
        assert_eq!(weights_for(1.0, false), [0.0; MAX_LEVELS]);
        assert_eq!(weights_for(0.3, false), [0.0; MAX_LEVELS]);
        assert!(
            weights_for(1.0, true)[WEIGHTS.len()..]
                .iter()
                .all(|w| *w == 0.0)
        );
    }

    /// The weights sum to the frame they are added to: at a strength of 1.0 the
    /// chain returns roughly the light it was given, spread out, rather than
    /// tripling it. The length is the chain's depth, so it is pinned here as
    /// well as everywhere else.
    #[test]
    fn the_levels_add_up_to_one_light() {
        let sum: f32 = WEIGHTS.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-6,
            "the {} levels add to {sum}, which would change the frame's \
             total brightness at a strength of 1.0",
            WEIGHTS.len()
        );
        assert!(
            WEIGHTS.windows(2).all(|w| w[0] > w[1]),
            "the fine level should lead: {WEIGHTS:?}"
        );
        assert!(
            !WEIGHTS.is_empty() && WEIGHTS.len() <= MAX_LEVELS,
            "the preset must fit the plumbing: {} weights, {MAX_LEVELS} slots",
            WEIGHTS.len()
        );
    }
}
