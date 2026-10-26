use std::sync::Arc;

use crate::canvas::Occluder;
use crate::collision::Convex;
use crate::objects::*;
use crate::particles::Particle;

/// The frame's clear color when no [`Shape::Background`] is drawn.
pub(crate) const DEFAULT_BACKGROUND: Color = Color {
    r: 0.07,
    g: 0.09,
    b: 0.14,
    a: 1.0,
};

/// The anti-alias band in pixels, mirrored by the `aa` constant in the SDF
/// shader sources; keep the two in sync so bounding boxes cover every pixel
/// the shaders can write.
pub(crate) const AA_BAND: f32 = 0.75;

/// A drawing recorded for the current frame, in pixel space.
///
/// `z` is the draw order: lower `z` is drawn first (further back). Drawings
/// with the same `z` are drawn in call order, so the last one drawn is on top.
#[derive(Clone, Debug)]
pub(crate) enum Draw {
    Line {
        a: [f32; 2],
        b: [f32; 2],
        width: f32,
        color: Color,
        z: f32,
        /// Whether the draw was produced by a [`crate::Diagnostics`]
        /// overlay node: the overlay tags the nodes it appends to the
        /// scene, and the tag rides on every draw the node produces, so the
        /// render loop can count the overlay's share of the frame's draw
        /// calls.
        diagnostic: bool,
    },
    /// A polyline: consecutive pairs of `points` connected by straight
    /// segments of `width`, all in pixel space, drawn in one draw call
    /// regardless of its length. Points beyond the first
    /// [`POLYLINE_MAX_POINTS`] are not drawn, and a polyline with fewer
    /// than two points draws nothing.
    Polyline {
        /// The vertices in pixel space (top-left origin, y down),
        /// connected in order.
        points: Vec<[f32; 2]>,
        /// The stroke's width, in pixels.
        width: f32,
        color: Color,
        z: f32,
        /// Whether the draw was produced by a [`crate::Diagnostics`]
        /// overlay node: the overlay tags the nodes it appends to the
        /// scene, and the tag rides on every draw the node produces, so the
        /// render loop can count the overlay's share of the frame's draw
        /// calls.
        diagnostic: bool,
    },
    Circle {
        center: [f32; 2],
        radius: f32,
        color: Color,
        z: f32,
        /// Whether the draw was produced by a [`crate::Diagnostics`]
        /// overlay node: the overlay tags the nodes it appends to the
        /// scene, and the tag rides on every draw the node produces, so the
        /// render loop can count the overlay's share of the frame's draw
        /// calls.
        diagnostic: bool,
    },
    Rectangle {
        center: [f32; 2],
        extent: [f32; 2],
        color: Color,
        z: f32,
        /// Whether the draw was produced by a [`crate::Diagnostics`]
        /// overlay node: the overlay tags the nodes it appends to the
        /// scene, and the tag rides on every draw the node produces, so the
        /// render loop can count the overlay's share of the frame's draw
        /// calls.
        diagnostic: bool,
    },
    Shape {
        /// The transform from the shape's local space to pixel space.
        world: Transform,
        /// The shape center in its local space.
        center: [f32; 2],
        /// Circle: `[radius, 0.0]`; rectangle: `[half_width, half_height]`.
        params: [f32; 2],
        /// `0.0` for a circle, `1.0` for a rectangle.
        kind: f32,
        /// The anti-alias band in local units (the screen `AA_BAND` divided
        /// by the transform's scale).
        aa: f32,
        color: Color,
        /// The node's self-emission, already multiplied with the composed
        /// modulate: added to the light mix that multiplies the color, so
        /// the shape keeps a floor of light in the dark. Black = off.
        emission: Color,
        /// How the glow sees this node's light: the color the emissive channel
        /// carries instead of the emission's, when an app wants a halo in a
        /// different hue from the surface throwing it. `None` is "the emission",
        /// which is what every scene in the tree asks for today; see
        /// [`crate::SceneNode::glow`].
        glow: Option<Color>,
        /// Whether the shape is lit by the frame's light field: `1.0` when
        /// the node's `lit` flag is set, `0.0` when unlit.
        lit: f32,
        /// The light layers the shape receives from: matched per light
        /// against the light's own set, and per occluder against the
        /// occluder's (see [`Light::layers`] and
        /// [`SceneNode::occludes_layers`]).
        lit_layers: LightLayers,
        /// Whether the shape occludes the frame's lights: `1.0` when the
        /// node's `occludes` flag is set, `0.0` when it casts no shadow.
        /// Rectangles (kind `1.0`) and circles (kind `0.0`) reach the
        /// occluder field, a circle as the polygon inscribed in it.
        occludes: f32,
        /// The light layers the shape's shadow lands on, packed into its
        /// occluder record (see [`SceneNode::occludes_layers`]). Read by
        /// the field only when `occludes` is set.
        occludes_layers: LightLayers,
        z: f32,
        /// Whether the draw was produced by a [`crate::Diagnostics`]
        /// overlay node: the overlay tags the nodes it appends to the
        /// scene, and the tag rides on every draw the node produces, so the
        /// render loop can count the overlay's share of the frame's draw
        /// calls.
        diagnostic: bool,
    },
    /// A sprite: a texture sampled in the sprite's local space, which is
    /// centered on the origin and extends by `size / 2` along each axis.
    Sprite {
        /// The transform from the sprite's local space to pixel space.
        world: Transform,
        /// The RGBA8 pixel data, row by row, top row first. Two sprites
        /// created from the same file share this buffer; glyph quads
        /// expanded by [`Canvas::expand_text`] share the same atlas buffer.
        data: Arc<[u8]>,
        /// The identity of the buffer `data` points at: for sprite
        /// buffers the generation stamped by the sprite constructors
        /// (bumped on every construction, since an in-memory image may be
        /// rebuilt with new pixels at any time), and for glyph quads the
        /// [`crate::text::Atlas`] generation, which bumps whenever the
        /// atlas repacks a new glyph and replaces its buffer — so a
        /// freed-and-recycled buffer address can never hit a stale texture.
        generation: u64,
        /// The local extent of the quad in local units; the sampled region
        /// spans `uv_rect` of the texture.
        size: [f32; 2],
        /// The sampled texture's size in pixels. Normally equal to
        /// `size` (one unit per texture pixel); for glyph quads the texture
        /// is a shared atlas that is larger than the quad.
        texture_size: [u32; 2],
        /// The texture's sampling filter: it is passed on to the sampler
        /// the texture is bound with, so bilinear and nearest-neighbor
        /// sprites from the same image keep distinct samplers.
        filter: SpriteFilter,
        /// The anti-alias band in local units (the screen `AA_BAND` divided
        /// by the transform's scale).
        aa: f32,
        /// The per-pixel tint, multiplied with every sampled pixel.
        tint: Color,
        /// The sprite's opacity, multiplied with the texture's own alpha.
        alpha: f32,
        /// The sub-rectangle of the texture the quad covers, as
        /// `[min_x, min_y, max_x, max_y]` in [0, 1] texture coordinates
        /// (top-left origin). `[0.0, 0.0, 1.0, 1.0]` covers the whole
        /// texture.
        uv_rect: [f32; 4],
        /// The node's self-emission, already multiplied with the composed
        /// modulate: added to the light mix that multiplies the sampled
        /// color, so the sprite keeps a floor of light in the dark.
        /// Black = off.
        emission: Color,
        /// How the glow sees this node's light: the color the emissive channel
        /// carries instead of the emission's, when an app wants a halo in a
        /// different hue from the surface throwing it. `None` is "the emission",
        /// which is what every scene in the tree asks for today; see
        /// [`crate::SceneNode::glow`].
        glow: Option<Color>,
        /// Whether this sprite adds its light instead of covering what is
        /// behind it: the node's flag, which picks the pipeline rather than a
        /// byte of the uniform. See [`crate::SceneNode::additive`].
        additive: bool,
        /// Whether the sprite is lit by the frame's light field: `1.0` when
        /// the node's `lit` flag is set, `0.0` when unlit. Glyph quads
        /// expanded from a lit text block are lit too.
        lit: f32,
        /// The light layers the sprite receives from, as on
        /// [`Draw::Shape`]; glyph quads carry their text node's set.
        lit_layers: LightLayers,
        z: f32,
        /// Whether the draw was produced by a [`crate::Diagnostics`]
        /// overlay node: the overlay tags the nodes it appends to the
        /// scene, and the tag rides on every draw the node produces —
        /// glyph quads expanded from the overlay's text carry it too — so
        /// the render loop can count the overlay's share of the frame's
        /// draw calls.
        diagnostic: bool,
        /// The game-authored WGSL program painting this sprite, from
        /// [`crate::Shape::with_program`]: the render loop builds (and
        /// caches) a pipeline for the source over the sprite layout and
        /// draws this quad with it. `None` draws with the engine's own
        /// sprite pipelines; glyph-expanded text never carries one.
        program: Option<Arc<str>>,
    },
    /// A batch of particles drawn in one instanced draw call.
    ///
    /// `data` holds two `vec4<f32>`s per particle — `(px, py, size, life
    /// fraction)` and `(angle, tint.r, tint.g, tint.b)`, all in pixel space
    /// (top-left origin, y down; the angle in pixel-space radians) — packed
    /// as 32 bytes per particle, so `count == data.len() / 32`. The whole
    /// batch shares `color`, `kind`, and `aspect`; each particle's alpha is
    /// additionally scaled by its life fraction and its tint, so dying
    /// particles fade out.
    Particles {
        /// The packed instance data, eight floats (32 bytes) per particle.
        data: Vec<u8>,
        /// The number of particles in `data`.
        count: u32,
        /// The batch's base tint.
        color: Color,
        /// The shape kind each particle is drawn as: `0.0` circle,
        /// `1.0` rectangle, `2.0` sprite.
        kind: f32,
        /// The shape's height-to-width factor (`1.0` for a circle).
        aspect: f32,
        /// The sprite's RGBA8 pixel data, present only when `kind` is
        /// `2.0`.
        sprite_data: Option<Arc<[u8]>>,
        /// The sprite's `(width, height)` in pixels, valid when `kind` is
        /// `2.0`.
        sprite_size: [u32; 2],
        /// The generation of the `sprite_data` buffer, valid when `kind`
        /// is `2.0`: it identifies the buffer to the shared texture cache,
        /// so a freed-and-recycled buffer address can never hit a stale
        /// texture.
        sprite_generation: u64,
        /// Whether the batch is lit by the frame's light field: `1.0` when
        /// the node's `lit` flag is set, `0.0` when unlit.
        lit: f32,
        /// The light layers the batch receives from, as on [`Draw::Shape`].
        lit_layers: LightLayers,
        z: f32,
        /// Whether the draw was produced by a [`crate::Diagnostics`]
        /// overlay node: the overlay tags the nodes it appends to the
        /// scene, and the tag rides on every draw the node produces, so the
        /// render loop can count the overlay's share of the frame's draw
        /// calls.
        diagnostic: bool,
    },
    /// A tile-map batch drawn in one instanced draw call.
    ///
    /// `data` holds four `vec4<f32>`s per tile — `(px, py, ux, uy)`
    /// (pixel-space center — top-left origin, y down — and the tile's
    /// mapped x-axis half-edge), `(vx, vy, 0, 0)` (the mapped y-axis
    /// half-edge; the two edge vectors carry the node's rotation with
    /// them, the way the particle batch's packed angle does),
    /// `(min_u, min_v, max_u, max_v)` (the atlas sub-rectangle), and
    /// `(r, g, b, a)` (the tile's tint: the tile's own color, the batch
    /// color and node modulate riding the uniform) — packed as 64 bytes
    /// per tile, so `count == data.len() / 64`. The whole batch shares
    /// the atlas texture and the uniform's base `color`.
    TileMap {
        /// The packed instance data, twelve floats (48 bytes) per tile.
        data: Vec<u8>,
        /// The number of tiles in `data`.
        count: u32,
        /// The batch's base tint (already node-modulated; each tile's
        /// tint in the instance data multiplies it further).
        color: Color,
        /// The atlas's RGBA8 pixel data.
        atlas_data: Arc<[u8]>,
        /// The atlas's `(width, height)` in pixels.
        atlas_size: [u32; 2],
        /// The generation of the `atlas_data` buffer: it identifies the
        /// buffer to the shared texture cache, so a freed-and-recycled
        /// buffer address can never hit a stale texture.
        atlas_generation: u64,
        /// The atlas's sampling filter, choosing the sampler the texture
        /// is bound with.
        filter: SpriteFilter,
        /// Whether the map is lit by the frame's light field: `1.0` when
        /// the node's `lit` flag is set — its tiles then multiply the
        /// light mix and take occluders' shadows like any lit surface —
        /// `0.0` when the map stays scenery in the old sense, drawn
        /// exactly as it always was.
        lit: f32,
        /// The light layers the map receives from, as on [`Draw::Shape`]:
        /// a tilemap whose tiles must not shadow each other is lit from
        /// a layer its own occluders do not name.
        lit_layers: LightLayers,
        /// The pixel-space box over all the tile quads, as
        /// `([min_x, min_y], [max_x, max_y])` — the batch's own extent.
        bounds: ([f32; 2], [f32; 2]),
        /// The pixel-space view clip, in the same box form: the scissor
        /// intersects it with `bounds`, so a map clipped off-screen is
        /// skipped whole.
        clip: Option<([f32; 2], [f32; 2])>,
        z: f32,
        /// Whether the draw was produced by a [`crate::Diagnostics`]
        /// overlay node, as on the other variants.
        diagnostic: bool,
    },
    /// A block of text: expanded into one [`Draw::Sprite`] per glyph by
    /// [`Canvas::expand_text`] before the frame is rendered, so this
    /// variant never reaches the render loop itself.
    Text {
        /// The node's world transform in user space (not yet composed with
        /// the user-to-pixel transform).
        world: Transform,
        /// The font file's bytes.
        font: Arc<[u8]>,
        /// The string to lay out.
        text: String,
        /// The font size in pixels per em.
        size: f32,
        /// The `wght` variation axis value; 400.0 is Regular. Part of the
        /// glyph atlas's key, so the same font and size at two weights get
        /// two atlases.
        weight: f32,
        /// The glyph color.
        color: Color,
        /// The text's opacity.
        alpha: f32,
        /// The node's self-emission, already multiplied with the composed
        /// modulate; passed on to every glyph quad the block expands into.
        emission: Color,
        /// How the glow sees this node's light: the color the emissive channel
        /// carries instead of the emission's, when an app wants a halo in a
        /// different hue from the surface throwing it. `None` is "the emission",
        /// which is what every scene in the tree asks for today; see
        /// [`crate::SceneNode::glow`].
        glow: Option<Color>,
        /// Whether the text is lit by the frame's light field: `1.0` when
        /// the node's `lit` flag is set, `0.0` when unlit; the flag is
        /// passed on to every glyph quad the block expands into.
        lit: f32,
        /// The light layers the text receives from, passed on to every
        /// glyph quad as `lit` is.
        lit_layers: LightLayers,
        z: f32,
        /// Whether the text was laid out by a [`crate::Diagnostics`]
        /// overlay node: the overlay tags the nodes it appends to the
        /// scene, and the tag is passed on to every glyph quad the block
        /// expands into, so the render loop can count the overlay's share
        /// of the frame's draw calls.
        diagnostic: bool,
    },
    /// A [`Shape::Background`]: never drawn; it is promoted to the frame's
    /// clear color at render time.
    Background { color: Color },
    /// A [`Shape::Light`]: never drawn; it is promoted to the frame's light
    /// field at render time, where every lit receiver evaluates it per
    /// pixel.
    Light {
        /// The light's position, in pixel space (already through the node's
        /// composed transform).
        pos: [f32; 2],
        /// The falloff extent, in pixels: zero contribution beyond this
        /// distance from `pos`.
        radius: f32,
        /// The light's strength, multiplied with its color per pixel.
        intensity: f32,
        /// The light's color, already multiplied with the node's composed
        /// modulate.
        color: Color,
        /// The shadow penumbra radius in pixels — the light's own disk as
        /// shadow rays see it: nine taps over this disk, the unoccluded
        /// fraction lights the pixel. `0.0` is the hard point-light
        /// shadow.
        penumbra: f32,
        /// The cone's axis, a unit vector in pixel space (the light's
        /// local-space direction already through the composed
        /// transforms' linear part), or `[1.0, 0.0]` when the direction
        /// is not usable.
        dir: [f32; 2],
        /// The cosine of the cone's half opening angle: pixels `p` with
        /// `dot(p - pos, dir) >= cos_half * length(p - pos)` receive the
        /// light. `-1.0` is omni-directional (every pixel passes), `1.0`
        /// degenerates the cone to its axis.
        cos_half: f32,
        /// The edge feather's width in cosine units: the shader ramps the
        /// light from zero at the cone edge to full once `softness`
        /// radians inside it. `0.0` — and every omni light — keeps the
        /// hard gate above.
        feather: f32,
        /// The light layers this light shines on: the shader folds it
        /// against each receiver's `lit_layers`, and the light contributes
        /// only where the sets share a layer (see [`Light::layers`]).
        layers: LightLayers,
        z: f32,
    },
}

impl Draw {
    pub(crate) fn z(&self) -> f32 {
        match self {
            Draw::Line { z, .. } => *z,
            Draw::Polyline { z, .. } => *z,
            Draw::Circle { z, .. } => *z,
            Draw::Rectangle { z, .. } => *z,
            Draw::Shape { z, .. } => *z,
            Draw::Sprite { z, .. } => *z,
            Draw::Particles { z, .. } => *z,
            Draw::TileMap { z, .. } => *z,
            Draw::Text { z, .. } => *z,
            // The background is always at the very back.
            Draw::Background { .. } => f32::MIN,
            Draw::Light { z, .. } => *z,
        }
    }

    /// The tight pixel rectangle the draw can write to, including the
    /// anti-alias band the shaders render beyond each geometric edge.
    ///
    /// `area` is the render area in pixels; the result is clamped to it, or
    /// `None` if the draw is fully outside the surface or is a background or
    /// a light (which are never drawn — they become the frame's clear color
    /// and light field).
    pub(crate) fn scissor_rect(&self, area: [u32; 2]) -> Option<[u32; 4]> {
        if matches!(self, Draw::Background { .. } | Draw::Light { .. }) {
            return None;
        }
        // A polyline with fewer than two points has no segments to draw.
        if let Draw::Polyline { points, .. } = self
            && points.len() < 2
        {
            return None;
        }
        let (min, max) = match self {
            Draw::Line { a, b, width, .. } => {
                let pad = width * 0.5 + AA_BAND;
                (
                    [a[0].min(b[0]) - pad, a[1].min(b[1]) - pad],
                    [a[0].max(b[0]) + pad, a[1].max(b[1]) + pad],
                )
            }
            Draw::Polyline { points, width, .. } => {
                // The box over all the points, plus half the stroke width
                // and the anti-alias band.
                let mut min = points[0];
                let mut max = points[0];
                for p in &points[1..] {
                    min = [min[0].min(p[0]), min[1].min(p[1])];
                    max = [max[0].max(p[0]), max[1].max(p[1])];
                }
                let pad = width * 0.5 + AA_BAND;
                ([min[0] - pad, min[1] - pad], [max[0] + pad, max[1] + pad])
            }
            Draw::Circle { center, radius, .. } => {
                let pad = *radius + AA_BAND;
                (
                    [center[0] - pad, center[1] - pad],
                    [center[0] + pad, center[1] + pad],
                )
            }
            Draw::Rectangle { center, extent, .. } => (
                [
                    center[0] - extent[0] - AA_BAND,
                    center[1] - extent[1] - AA_BAND,
                ],
                [
                    center[0] + extent[0] + AA_BAND,
                    center[1] + extent[1] + AA_BAND,
                ],
            ),
            Draw::Shape {
                world,
                center,
                params,
                kind,
                aa,
                ..
            } => {
                // The shape's bounding box in its local space, including the
                // local anti-alias band. A circle's box is a square around its
                // radius; a rectangle's box is its extents.
                let (hx, hy) = if *kind < 0.5 {
                    let r = params[0].max(0.0) + *aa;
                    (r, r)
                } else {
                    (params[0].max(0.0) + *aa, params[1].max(0.0) + *aa)
                };
                aabb_of_box(world, *center, hx, hy)
            }
            Draw::Sprite {
                world, size, aa, ..
            } => {
                // The sprite's box is the texture, centered on the origin,
                // plus the local anti-alias band.
                aabb_of_box(world, [0.0, 0.0], size[0] * 0.5 + *aa, size[1] * 0.5 + *aa)
            }
            // The batch's particles can be anywhere, so the scissor is the
            // whole surface: the per-particle quads are already tight, and
            // the rasterizer discards everything else for free.
            Draw::Particles { .. } => ([0.0, 0.0], [area[0] as f32, area[1] as f32]),
            Draw::TileMap { bounds, clip, .. } => {
                // The batch's own box, cut by the view clip if it has
                // one: quads outside the clip rasterize nowhere, and a
                // clip wholly off the surface returns None below and
                // skips the draw entirely.
                let (mut min, mut max) = *bounds;
                if let Some((cmin, cmax)) = clip {
                    min = [min[0].max(cmin[0]), min[1].max(cmin[1])];
                    max = [max[0].min(cmax[0]), max[1].min(cmax[1])];
                }
                (min, max)
            }
            // A background never draws; the early return above covers it.
            Draw::Background { .. } => unreachable!(),
            // A light never draws; the early return above covers it.
            Draw::Light { .. } => unreachable!(),
            // Text is expanded into glyph sprites by `Canvas::expand_text`
            // before the render loop, so it never reaches the scissor.
            Draw::Text { .. } => unreachable!(),
        };
        clamp_box_to_area(min, max, area)
    }
}

/// The axis-aligned bounding box of the four transformed corners of a box in
/// local space centered at `center` with half extents `(hx, hy)`. The box may
/// be rotated or skewed by `world`, so the corners are transformed first and
/// the axis-aligned box of the result is taken.
pub(crate) fn aabb_of_box(
    world: &Transform,
    center: [f32; 2],
    hx: f32,
    hy: f32,
) -> ([f32; 2], [f32; 2]) {
    let corners = [
        world.apply([center[0] - hx, center[1] - hy]),
        world.apply([center[0] + hx, center[1] - hy]),
        world.apply([center[0] - hx, center[1] + hy]),
        world.apply([center[0] + hx, center[1] + hy]),
    ];
    let mut min = corners[0];
    let mut max = corners[0];
    for corner in corners.iter().skip(1) {
        min = [min[0].min(corner[0]), min[1].min(corner[1])];
        max = [max[0].max(corner[0]), max[1].max(corner[1])];
    }
    (min, max)
}

/// The tight pixel rectangle a bounding box touches, clamped to the render
/// `area`, or `None` if the box is empty or fully outside the surface.
///
/// `floor`/`ceil` pick the pixel columns and rows the box touches; the
/// clamps keep the box inside the surface (float-to-int casts saturate, so
/// out-of-range coordinates stay consistent).
fn clamp_box_to_area(min: [f32; 2], max: [f32; 2], area: [u32; 2]) -> Option<[u32; 4]> {
    let x0 = min[0].max(0.0).floor() as u32;
    let y0 = min[1].max(0.0).floor() as u32;
    let x1 = max[0].min(area[0] as f32).ceil() as u32;
    let y1 = max[1].min(area[1] as f32).ceil() as u32;
    if x1 > x0 && y1 > y0 {
        Some([x0, y0, x1 - x0, y1 - y0])
    } else {
        None
    }
}

/// The frame's clear color: the color of the last [`Draw::Background`] in
/// call order, or [`DEFAULT_BACKGROUND`] when the frame has none.
pub(crate) fn clear_color(draws: &[Draw]) -> Color {
    draws
        .iter()
        .rev()
        .find_map(|draw| match draw {
            Draw::Background { color } => Some(*color),
            _ => None,
        })
        .unwrap_or(DEFAULT_BACKGROUND)
}

/// Writes `value` as little-endian f32 bytes into `data` at byte `offset`.
fn write_f32_at(data: &mut [u8], offset: usize, value: f32) {
    data[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
}

/// Line uniform data, 48 bytes, matching the WGSL uniform-space layout of
/// the `Uniforms` Wgsl struct: `a` @ 0, `b` @ 8, `color` @ 16 (a vec4<f32>
/// is 16-byte aligned in uniform space, leaving an 8-byte gap), `width` @
/// 32 (the next member is aligned to its own alignment, so the f32 follows
/// the vec4 without a gap); the struct size rounds up to 48.
pub(crate) fn line_uniform_data(a: [f32; 2], b: [f32; 2], color: Color, width: f32) -> Vec<u8> {
    let mut data = vec![0u8; LINE_UNIFORM_BYTES as usize];
    write_f32_at(&mut data, 0, a[0]);
    write_f32_at(&mut data, 4, a[1]);
    write_f32_at(&mut data, 8, b[0]);
    write_f32_at(&mut data, 12, b[1]);
    write_f32_at(&mut data, 16, color.r);
    write_f32_at(&mut data, 20, color.g);
    write_f32_at(&mut data, 24, color.b);
    write_f32_at(&mut data, 28, color.a);
    write_f32_at(&mut data, 32, width);
    data
}

/// The most points a polyline draw carries: the shader's
/// `array<vec4<f32>, 128>` is fixed at this size, so a longer polyline
/// draws its first 128 points. The points are packed as vec4s (x, y, 0, 0)
/// because the uniform address space requires an array member's stride to
/// be a multiple of 16 bytes (a vec2's stride is only 8).
pub(crate) const POLYLINE_MAX_POINTS: usize = 128;

/// Polyline uniform data, 2080 bytes, matching the WGSL uniform-space layout
/// of the `PolylineUniforms` WGSL struct: `points` @ 0 (128 vec4<f32>s, 16
/// bytes each — 2048 bytes; each point is (x, y, 0, 0)), `color` @ 2048 (a
/// vec4<f32> is 16-byte aligned in uniform space, and 2048 already is),
/// `count` @ 2064 (a u32 follows the vec4 without a gap), `width` @ 2068;
/// the struct size rounds up to 2080. Points beyond the first
/// [`POLYLINE_MAX_POINTS`] are not packed.
pub(crate) fn polyline_uniform_data(points: &[[f32; 2]], color: Color, width: f32) -> Vec<u8> {
    let mut data = vec![0u8; POLYLINE_UNIFORM_BYTES as usize];
    let count = points.len().min(POLYLINE_MAX_POINTS);
    for (i, p) in points.iter().take(count).enumerate() {
        write_f32_at(&mut data, i * 16, p[0]);
        write_f32_at(&mut data, i * 16 + 4, p[1]);
    }
    write_f32_at(&mut data, 2048, color.r);
    write_f32_at(&mut data, 2052, color.g);
    write_f32_at(&mut data, 2056, color.b);
    write_f32_at(&mut data, 2060, color.a);
    data[2064..2068].copy_from_slice(&(count as u32).to_le_bytes());
    write_f32_at(&mut data, 2068, width);
    data
}

/// Circle uniform data, 48 bytes, matching the WGSL uniform-space layout of
/// the `CircleUniforms` Wgsl struct: `center` @ 0, `color` @ 16 (a
/// vec4<f32> is 16-byte aligned in uniform space, leaving an 8-byte gap),
/// `radius` @ 32 (the next member is aligned to its own alignment, so the
/// f32 follows the vec4 without a gap); the struct size rounds up to 48.
pub(crate) fn circle_uniform_data(center: [f32; 2], color: Color, radius: f32) -> Vec<u8> {
    let mut data = vec![0u8; CIRCLE_UNIFORM_BYTES as usize];
    write_f32_at(&mut data, 0, center[0]);
    write_f32_at(&mut data, 4, center[1]);
    write_f32_at(&mut data, 16, color.r);
    write_f32_at(&mut data, 20, color.g);
    write_f32_at(&mut data, 24, color.b);
    write_f32_at(&mut data, 28, color.a);
    write_f32_at(&mut data, 32, radius);
    data
}

/// Rectangle uniform data, matching the `RectUniforms` Wgsl struct: center,
/// extent, color. The 32 bytes of data exactly fill the struct's 32-byte
/// size, so no padding is needed.
pub(crate) fn rect_uniform_data(center: [f32; 2], extent: [f32; 2], color: Color) -> Vec<u8> {
    let mut data: Vec<u8> = center
        .iter()
        .chain(extent.iter())
        .chain(color.channels().iter())
        .flat_map(|f| f.to_le_bytes())
        .collect();
    data.resize(RECT_UNIFORM_BYTES as usize, 0);
    data
}

/// Shape uniform data, 96 bytes, matching the WGSL uniform-space layout of
/// the `ShapeUniforms` Wgsl struct. Per the WGSL memory layout rules (and
/// naga's `Layouter`), in the uniform address space a mat2x2<f32> is 16
/// bytes total with 8-byte alignment, so its two vec2 columns are packed at
/// @ 0 and @ 8; a vec3<f32> is 12 bytes with 16-byte alignment; a vec2<f32>
/// is 8 bytes with 8-byte alignment. With that rule, the struct lays out as:
/// `to_local` @ 0 (column 0 = (m[0][0], m[1][0]) at @ 0, column 1 =
/// (m[0][1], m[1][1]) at @ 8; stored column-major, so the WGSL matrix holds
/// the Rust matrix element-for-element and `m * p + t` in the shader
/// reproduces `Transform::apply`), `translation` @ 16, `center` @ 24,
/// `params` @ 32, `color` @ 48 (a vec4<f32>, spanning 48..64), `misc` @ 64
/// (aa @ 64, kind @ 68), `lit` @ 72, `lit_layers` (the receiver's layer
/// mask through `f32::from_bits`) @ 76, `emission` @ 80 (a vec4<f32>, spanning
/// 80..96), and `glow` @ 96 (a vec4<f32>, spanning 96..112); the struct size is
/// 112.
#[allow(clippy::too_many_arguments)] // uniform writer: one argument per WGSL member
pub(crate) fn shape_uniform_data(
    to_local: Transform,
    center: [f32; 2],
    params: [f32; 2],
    kind: f32,
    aa: f32,
    color: Color,
    emission: Color,
    glow: Option<Color>,
    lit: f32,
    lit_layers: LightLayers,
) -> Vec<u8> {
    let mut data = vec![0u8; SHAPE_UNIFORM_BYTES as usize];
    let m = to_local.m;
    // mat2x2: 16 bytes total, vec2 columns with an 8-byte stride.
    write_f32_at(&mut data, 0, m[0][0]);
    write_f32_at(&mut data, 4, m[1][0]);
    write_f32_at(&mut data, 8, m[0][1]);
    write_f32_at(&mut data, 12, m[1][1]);
    write_f32_at(&mut data, 16, to_local.t[0]);
    write_f32_at(&mut data, 20, to_local.t[1]);
    write_f32_at(&mut data, 24, center[0]);
    write_f32_at(&mut data, 28, center[1]);
    write_f32_at(&mut data, 32, params[0]);
    write_f32_at(&mut data, 36, params[1]);
    // vec4: 16 bytes with 16-byte alignment, so it starts at 48 and spans
    // 48..64.
    write_f32_at(&mut data, 48, color.r);
    write_f32_at(&mut data, 52, color.g);
    write_f32_at(&mut data, 56, color.b);
    write_f32_at(&mut data, 60, color.a);
    write_f32_at(&mut data, 64, aa);
    write_f32_at(&mut data, 68, kind);
    write_f32_at(&mut data, 72, lit);
    // The receiver's layer mask, raw bits in the padding slot between
    // `lit` and the 16-byte-aligned `emission`: the shader
    // `bitcast<u32>`s it back to fold against every light's and every
    // occluder's set.
    write_f32_at(&mut data, 76, f32::from_bits(lit_layers.bits()));
    // The emission vec4: 16-byte aligned, so it starts at 80 and spans 80..96,
    // and the glow vec4 follows it at 96..112.
    write_f32_at(&mut data, 80, emission.r);
    write_f32_at(&mut data, 84, emission.g);
    write_f32_at(&mut data, 88, emission.b);
    write_f32_at(&mut data, 92, emission.a);
    write_glow(&mut data, 96, glow, emission);
    data
}

/// Writes the glow vec4, resolving `None` on this side of the driver: the
/// emissive entry point in every shader reads one vec4 and knows nothing about
/// whether an app split the two colors apart, so a scene that asks for nothing
/// new pays nothing new at draw time.
fn write_glow(data: &mut [u8], at: usize, glow: Option<Color>, emission: Color) {
    let glow = glow.unwrap_or(emission);
    write_f32_at(data, at, glow.r);
    write_f32_at(data, at + 4, glow.g);
    write_f32_at(data, at + 8, glow.b);
    write_f32_at(data, at + 12, glow.a);
}

/// The uniform size of each draw kind in bytes: the size of that pipeline's
/// WGSL uniform struct, and the width of the window its bind group binds in
/// the frame's uniform arena. Each packer asserts its output is this long, so
/// the WGSL struct, the bind group layout and the packed bytes are three
/// names for one number — a window smaller than the struct is a validation
/// error when the bind group is built, and a packer longer than its slot
/// overruns into the next draw's parameters, which is what the assertions are
/// for. See [`SPRITE_UNIFORM_BYTES`], the same fact for sprites.
pub(crate) const LINE_UNIFORM_BYTES: u64 = 48;
/// See [`LINE_UNIFORM_BYTES`]. A polyline's uniform is the whole point list,
/// which is why the arena's slots are not one fixed size.
pub(crate) const POLYLINE_UNIFORM_BYTES: u64 = 2080;
/// See [`LINE_UNIFORM_BYTES`].
pub(crate) const CIRCLE_UNIFORM_BYTES: u64 = 48;
/// See [`LINE_UNIFORM_BYTES`].
pub(crate) const RECT_UNIFORM_BYTES: u64 = 32;
/// See [`LINE_UNIFORM_BYTES`].
pub(crate) const SHAPE_UNIFORM_BYTES: u64 = 112;

/// The uniform bytes one draw needs, or `None` when the draw has none — a
/// background, a light, a particle batch or tilemap that carries its
/// parameters in storage buffers — or when its transform has collapsed and
/// the draw is skipped.
///
/// Every uniform in the frame is packed here, once, before the render pass.
/// That is the point of the function: the draw arms hold no uniform code at
/// all, so the bytes a draw reads and the bytes the packer wrote cannot come
/// apart — the failure the engine's per-draw buffers used to buy their way
/// out of by construction. A kind whose uniform changes shape is a change
/// here, in the packer, and in that pipeline's layout, and the assertions
/// above and the arena's bind group layout are where getting it wrong shows.
#[allow(clippy::too_many_arguments)] // uniform writer: one argument per WGSL member
pub(crate) fn uniform_bytes(draw: &Draw) -> Option<Vec<u8>> {
    match draw {
        Draw::Line {
            a, b, width, color, ..
        } => Some(line_uniform_data(*a, *b, *color, *width)),
        Draw::Polyline {
            points,
            width,
            color,
            ..
        } => Some(polyline_uniform_data(points, *color, *width)),
        Draw::Circle {
            center,
            radius,
            color,
            ..
        } => Some(circle_uniform_data(*center, *color, *radius)),
        Draw::Rectangle {
            center,
            extent,
            color,
            ..
        } => Some(rect_uniform_data(*center, *extent, *color)),
        Draw::Shape {
            world,
            center,
            params,
            kind,
            aa,
            color,
            emission,
            glow,
            lit,
            lit_layers,
            ..
        } => world.invert().map(|inv| {
            shape_uniform_data(
                inv,
                *center,
                *params,
                *kind,
                *aa,
                *color,
                *emission,
                *glow,
                *lit,
                *lit_layers,
            )
        }),
        Draw::Sprite {
            world,
            size,
            tint,
            alpha,
            emission,
            glow,
            lit,
            lit_layers,
            uv_rect,
            ..
        } => world.invert().map(|inv| {
            sprite_uniform_data(
                inv,
                *size,
                *tint,
                *alpha,
                *emission,
                *glow,
                *lit,
                *lit_layers,
                *uv_rect,
            )
        }),
        _ => None,
    }
}

/// The size of a sprite's uniform in bytes: the `SpriteUniforms` WGSL
/// struct, and the number three places must agree on — the packer below,
/// the arena's bind group layout, and the window every sprite bind group
/// binds. It is an arena slot's payload, so a sprite uniform that grows
/// past it is caught at pipeline creation rather than as a wrong pixel.
pub(crate) const SPRITE_UNIFORM_BYTES: u64 = 112;

/// Sprite uniform data, 112 bytes, matching the WGSL uniform-space layout of
/// the `SpriteUniforms` Wgsl struct. The mat2x2 column packing is the same
/// as in [`shape_uniform_data`]; `translation` sits at @ 16, `size` (a
/// vec2) at @ 24, the tint `vec4` at @ 32 (16 bytes, 16-byte aligned,
/// spanning 32..48), the scalar `alpha` (4-byte aligned) at @ 48, the
/// scalar `lit` (4-byte aligned) at @ 52, the scalar `lit_layers` (the
/// receiver's layer mask through `f32::from_bits`) at @ 56, the `uv_rect`
/// `vec4` (16-byte
/// aligned) at @ 64, the emission `vec4` at @ 80 (spanning 80..96), and the
/// glow `vec4` at @ 96 (spanning 96..112); the struct size is 112.
#[allow(clippy::too_many_arguments)] // uniform writer: one argument per WGSL member
pub(crate) fn sprite_uniform_data(
    to_local: Transform,
    size: [f32; 2],
    tint: Color,
    alpha: f32,
    emission: Color,
    glow: Option<Color>,
    lit: f32,
    lit_layers: LightLayers,
    uv_rect: [f32; 4],
) -> Vec<u8> {
    let mut data = vec![0u8; SPRITE_UNIFORM_BYTES as usize];
    let m = to_local.m;
    // mat2x2: 16 bytes total, vec2 columns with an 8-byte stride.
    write_f32_at(&mut data, 0, m[0][0]);
    write_f32_at(&mut data, 4, m[1][0]);
    write_f32_at(&mut data, 8, m[0][1]);
    write_f32_at(&mut data, 12, m[1][1]);
    write_f32_at(&mut data, 16, to_local.t[0]);
    write_f32_at(&mut data, 20, to_local.t[1]);
    write_f32_at(&mut data, 24, size[0]);
    write_f32_at(&mut data, 28, size[1]);
    // vec4: 16 bytes with 16-byte alignment, so it starts at 32 and spans
    // 32..48. The scalar alpha is 4-byte aligned, so it follows the tint at
    // 48, and the 16-byte-aligned uv_rect starts at 64.
    write_f32_at(&mut data, 32, tint.r);
    write_f32_at(&mut data, 36, tint.g);
    write_f32_at(&mut data, 40, tint.b);
    write_f32_at(&mut data, 44, tint.a);
    write_f32_at(&mut data, 48, alpha);
    write_f32_at(&mut data, 52, lit);
    // The receiver's layer mask, raw bits in the padding before the
    // 16-byte-aligned uv_rect.
    write_f32_at(&mut data, 56, f32::from_bits(lit_layers.bits()));
    write_f32_at(&mut data, 64, uv_rect[0]);
    write_f32_at(&mut data, 68, uv_rect[1]);
    write_f32_at(&mut data, 72, uv_rect[2]);
    write_f32_at(&mut data, 76, uv_rect[3]);
    // The emission vec4: 16-byte aligned, so it follows the uv_rect at 80 and
    // spans 80..96, and the glow vec4 follows that at 96..112.
    write_f32_at(&mut data, 80, emission.r);
    write_f32_at(&mut data, 84, emission.g);
    write_f32_at(&mut data, 88, emission.b);
    write_f32_at(&mut data, 92, emission.a);
    write_glow(&mut data, 96, glow, emission);
    data
}

/// Particle batch uniform data, 48 bytes, matching the WGSL uniform-space
/// layout of the `ParticlesUniforms` Wgsl struct: `size` (a vec2) @ 0,
/// `color` (a vec4) @ 16 (16 bytes, 16-byte aligned, leaving an 8-byte
/// gap), `misc` (a vec2: the shape's kind and aspect) @ 32, `lit` @ 40, and
/// `lit_layers` (the receiver's layer mask through `f32::from_bits`) @ 44;
/// the struct size rounds up to 48.
pub(crate) fn particles_uniform_data(
    size: [f32; 2],
    color: Color,
    kind: f32,
    aspect: f32,
    lit: f32,
    lit_layers: LightLayers,
) -> Vec<u8> {
    let mut data = vec![0u8; 48];
    write_f32_at(&mut data, 0, size[0]);
    write_f32_at(&mut data, 4, size[1]);
    write_f32_at(&mut data, 16, color.r);
    write_f32_at(&mut data, 20, color.g);
    write_f32_at(&mut data, 24, color.b);
    write_f32_at(&mut data, 28, color.a);
    write_f32_at(&mut data, 32, kind);
    write_f32_at(&mut data, 36, aspect);
    write_f32_at(&mut data, 40, lit);
    write_f32_at(&mut data, 44, f32::from_bits(lit_layers.bits()));
    data
}

/// Packs one particle of a [`Draw::Particles`] batch: eight little-endian
/// floats, 32 bytes — first the pixel position `(px, py)`, the pixel size,
/// and the remaining life fraction, then the pixel-space angle and the
/// particle's tint `(r, g, b)` — in the layout `particles.wgsl` reads as
/// `array<vec4<f32>>`.
///
/// The life fraction is clamped to `0.0..=1.0` (and `0.0` for a
/// `max_life <= 0`), and a negative size is clamped to `0.0`.
pub(crate) fn particle_instance(p: &Particle, px: f32, py: f32, size: f32, angle: f32) -> [u8; 32] {
    let life = if p.max_life > 0.0 {
        (p.life / p.max_life).clamp(0.0, 1.0)
    } else {
        0.0
    };
    let mut out = [0u8; 32];
    out[0..4].copy_from_slice(&px.to_le_bytes());
    out[4..8].copy_from_slice(&py.to_le_bytes());
    out[8..12].copy_from_slice(&size.max(0.0).to_le_bytes());
    out[12..16].copy_from_slice(&life.to_le_bytes());
    out[16..20].copy_from_slice(&angle.to_le_bytes());
    out[20..24].copy_from_slice(&p.color.r.to_le_bytes());
    out[24..28].copy_from_slice(&p.color.g.to_le_bytes());
    out[28..32].copy_from_slice(&p.color.b.to_le_bytes());
    out
}

/// Tile map batch uniform data, 32 bytes, matching the WGSL uniform-space
/// layout of the `TileMapUniforms` struct: `size` (a vec2) @ 0, `lit`
/// (an f32) @ 8, `lit_layers` (the receiver's layer mask through
/// `f32::from_bits`) @ 12 — the gap the padding already held — and
/// `color` (a
/// vec4) @ 16 (16 bytes, 16-byte aligned); the struct size rounds up
/// to 32.
pub(crate) fn tilemap_uniform_data(
    size: [f32; 2],
    color: Color,
    lit: f32,
    lit_layers: LightLayers,
) -> Vec<u8> {
    let mut data = vec![0u8; 32];
    write_f32_at(&mut data, 0, size[0]);
    write_f32_at(&mut data, 4, size[1]);
    write_f32_at(&mut data, 8, lit);
    write_f32_at(&mut data, 12, f32::from_bits(lit_layers.bits()));
    write_f32_at(&mut data, 16, color.r);
    write_f32_at(&mut data, 20, color.g);
    write_f32_at(&mut data, 24, color.b);
    write_f32_at(&mut data, 28, color.a);
    data
}

/// Packs one tile of a [`Draw::TileMap`] batch: sixteen little-endian
/// floats, 64 bytes — the pixel center `(px, py)` and mapped x-axis
/// half-edge `(ux, uy)`, the mapped y-axis half-edge `(vx, vy)` padded
/// to a vec4 by the cell's own orientation — the flip bits (`x` is 1,
/// `y` is 2, both is 3) and the clockwise quarter-turn count — the
/// atlas sub-rectangle `(min_u, min_v, max_u, max_v)`, and the tint
/// `(r, g, b, a)` — in the layout `tilemap.wgsl` reads as
/// `array<vec4<f32>>` with four vec4s per instance. The quad spans
/// `center ± u ± v`, so whatever the node's transform did to the tile's
/// two axes — scale, flip, rotate — the quad follows; and whatever the
/// cell's own `flags` say, the sampling obeys.
#[allow(clippy::too_many_arguments)]
pub(crate) fn tilemap_instance(
    px: f32,
    py: f32,
    u: [f32; 2],
    v: [f32; 2],
    uv: [f32; 4],
    tint: &Color,
    flags: [f32; 2],
) -> [u8; 64] {
    let mut out = [0u8; 64];
    out[0..4].copy_from_slice(&px.to_le_bytes());
    out[4..8].copy_from_slice(&py.to_le_bytes());
    out[8..12].copy_from_slice(&u[0].to_le_bytes());
    out[12..16].copy_from_slice(&u[1].to_le_bytes());
    out[16..20].copy_from_slice(&v[0].to_le_bytes());
    out[20..24].copy_from_slice(&v[1].to_le_bytes());
    out[24..28].copy_from_slice(&flags[0].to_le_bytes());
    out[28..32].copy_from_slice(&flags[1].to_le_bytes());
    for (i, w) in uv.iter().enumerate() {
        out[32 + i * 4..36 + i * 4].copy_from_slice(&w.to_le_bytes());
    }
    out[48..52].copy_from_slice(&tint.r.to_le_bytes());
    out[52..56].copy_from_slice(&tint.g.to_le_bytes());
    out[56..60].copy_from_slice(&tint.b.to_le_bytes());
    out[60..64].copy_from_slice(&tint.a.to_le_bytes());
    out
}

/// The byte size of the light field's header: the light `count` (a u32),
/// 12 padding bytes, and the scene's ambient color (a `vec4`).
pub(crate) const LIGHT_FIELD_HEADER: usize = 32;

/// The byte size of one light record: four `vec4<f32>`s —
/// `(x, y, radius, intensity)`, `(r, g, b, penumbra)`,
/// `(dir_x, dir_y, cos_half, feather)`, and `(layers, 0, 0, 0)`, the last
/// two vec4s carrying the light's shadow disk and cone (see
/// [`Draw::Light`]). The fourth vec4's x is the light's `LightLayers` bit
/// mask, its bits read with `bitcast<u32>` by the shader (the value is the
/// mask's raw bits through `f32::from_bits`, not a number to do
/// arithmetic on); its other components are padding.
pub(crate) const LIGHT_RECORD: usize = 64;

/// The minimum size of the light field buffer in bytes: the header plus
/// one `vec4`. The WGSL field ends in an unsized `array<vec4<f32>>`, and
/// wgpu's minimum binding size for such a struct is the header plus one
/// element of that tail — so even a lightless frame, which still binds the
/// buffer (the shaders read it for the ambient), must have at least this
/// many bytes.
pub(crate) const LIGHT_FIELD_BUFFER_MIN: usize = LIGHT_FIELD_HEADER + 16;

/// The size of the light field buffer in bytes for `count` lights: the
/// header plus one record per light, never below
/// [`LIGHT_FIELD_BUFFER_MIN`].
pub(crate) fn light_field_buffer_size(count: u32) -> u64 {
    (LIGHT_FIELD_HEADER + count as usize * LIGHT_RECORD).max(LIGHT_FIELD_BUFFER_MIN) as u64
}

/// The frame's light field, packed little-endian for the GPU's storage
/// buffer.
///
/// The layout is a [`LIGHT_FIELD_HEADER`]-byte header — the light `count`
/// at bytes `0..4`, 12 padding bytes, the scene's ambient color channels at
/// bytes `16..32` — followed by one [`LIGHT_RECORD`]-byte record per light,
/// in call order: a `(x, y, radius, intensity)` `vec4`, an
/// `(r, g, b, penumbra)` `vec4`, a `(dir_x, dir_y, cos_half, feather)`
/// `vec4` carrying the cone, and a `(layers, 0, 0, 0)` `vec4` whose x is
/// the light's [`LightLayers`] bit mask through `f32::from_bits`.
/// The records line up with the unsized WGSL
/// `array<vec4<f32>>` tail of the field's storage struct, so `data` can be
/// bound as-is.
pub(crate) struct LightField {
    /// The number of lights packed — the header's `count`.
    pub count: u32,
    /// The packed bytes: a 32-byte header plus `count * 48` bytes.
    pub data: Vec<u8>,
}

/// Packs the frame's light field from its draws and the scene's ambient
/// color.
///
/// Only [`Draw::Light`]s contribute, in call order; every other draw is
/// ignored. The ambient color comes from the scene, not the draws.
pub(crate) fn pack_light_field(draws: &[Draw], ambient: Color) -> LightField {
    let count = draws
        .iter()
        .filter(|draw| matches!(draw, Draw::Light { .. }))
        .count() as u32;
    let mut data = vec![0u8; LIGHT_FIELD_HEADER + count as usize * LIGHT_RECORD];
    data[0..4].copy_from_slice(&count.to_le_bytes());
    write_f32_at(&mut data, 16, ambient.r);
    write_f32_at(&mut data, 20, ambient.g);
    write_f32_at(&mut data, 24, ambient.b);
    write_f32_at(&mut data, 28, ambient.a);
    let mut record = 0usize;
    for draw in draws {
        if let Draw::Light {
            pos,
            radius,
            intensity,
            color,
            penumbra,
            dir,
            cos_half,
            feather,
            layers,
            ..
        } = draw
        {
            let off = LIGHT_FIELD_HEADER + record * LIGHT_RECORD;
            write_f32_at(&mut data, off, pos[0]);
            write_f32_at(&mut data, off + 4, pos[1]);
            write_f32_at(&mut data, off + 8, *radius);
            write_f32_at(&mut data, off + 12, *intensity);
            write_f32_at(&mut data, off + 16, color.r);
            write_f32_at(&mut data, off + 20, color.g);
            write_f32_at(&mut data, off + 24, color.b);
            // The color vec4's w carries the shadow penumbra radius; the
            // shader never reads the color's alpha from here.
            write_f32_at(&mut data, off + 28, *penumbra);
            write_f32_at(&mut data, off + 32, dir[0]);
            write_f32_at(&mut data, off + 36, dir[1]);
            write_f32_at(&mut data, off + 40, *cos_half);
            write_f32_at(&mut data, off + 44, *feather);
            // The fourth vec4's x: the layer mask's raw bits, which the
            // shader reads back with `bitcast<u32>`. It rides as bits, not
            // as a float value: every one of the 2^32 masks, `ALL`
            // included, round-trips exact.
            data[off + 48..off + 52].copy_from_slice(&layers.bits().to_le_bytes());
            record += 1;
        }
    }
    LightField { count, data }
}

/// The byte size of the occluder field's header: the record `count` (a
/// u32 — every record the field holds, counting the per-light copies, so
/// it can exceed the number of occluders the app declared), the frame's
/// light `count` (a u32 — the length of the per-light range table), and
/// 8 padding bytes.
pub(crate) const OCCLUDER_FIELD_HEADER: usize = 16;

/// The minimum size of the occluder field buffer in bytes: the header
/// plus one `vec4`, the same WGSL minimum binding size as
/// [`LIGHT_FIELD_BUFFER_MIN`].
pub(crate) const OCCLUDER_FIELD_BUFFER_MIN: usize = OCCLUDER_FIELD_HEADER + 16;

/// The byte size of one occluder record: three `vec4<f32>`s — the inverse
/// transform's linear part, column-major; the inverse transform's
/// translation with the box's local center; and the box's local half-extents.
pub(crate) const OCCLUDER_RECORD: usize = 48;

/// The byte size of one occluder half-plane: `(n.x, n.y, c, _)`, a polygon's
/// edge with its outward unit normal and offset, following its record. A box
/// has none and its record is [`OCCLUDER_RECORD`] bytes exactly, which is what
/// lets a field packed before polygons still read correctly: the record's
/// plane count sits in a component that was padding, and zero of them is a
/// box.
pub(crate) const OCCLUDER_PLANE: usize = 16;

/// The frame's occluder field, packed little-endian for the GPU's storage
/// buffer.
///
/// The layout is a [`OCCLUDER_FIELD_HEADER`]-byte header — the record
/// `count` at bytes `0..4`, the frame's light count at `4..8`, 8 padding
/// bytes — followed by a range table of one `vec4` per light, `(start,
/// end, 0, 0)` in `vec4` units measured from the tail's start: the light's
/// own run of occluder records. Records follow the table as one contiguous
/// block per light, each block in master order (painted occluders as
/// declared, then the app's), and an occluder that can shadow more than one
/// light is copied into each of their blocks.
///
/// Each record is what it always was: a `(m00, m10, m01, m11)` `vec4`, the
/// inverse of the occluder's world transform, column-major (pixel space to
/// the occluder's local space), then a `(tx, ty, cx, cy)` `vec4`, the
/// inverse transform's translation and the box's local center, then a
/// `(hx, hy, planes, layers)` `vec4`, the box's local half-extents, the
/// number of [`OCCLUDER_PLANE`]-byte half-planes that follow it — zero for
/// a box, and for a polygon the edges that bound it — and the occluder's
/// [`LightLayers`] bit mask through `f32::from_bits`, the padding slot
/// polygon records already carried: the shader skips a record whose mask
/// shares no layer with the receiver evaluating the light.
///
/// Records are therefore variable length, `3 + planes` `vec4`s apiece, and
/// a shader walks a light's block with a running offset rather than
/// `3 * i`. A light's block holds exactly the occluders whose world bounds
/// reach its rays at all (see [`pack_occluder_field`]), so a fragment's
/// walk reads only records that could plausibly stand between its light and
/// its pixel; a block can also be empty, which answers "unoccluded" with
/// zero loads. The data line up with the unsized WGSL `array<vec4<f32>>`
/// tail of the field's storage struct, so `data` can be bound as-is.
pub(crate) struct OccluderField {
    /// The number of occluder records the field holds — every light's
    /// block included, so it can exceed the number of occluders packed.
    /// The frame path sizes buffers by the packed bytes and never reads
    /// it; the field-layout tests do.
    #[cfg_attr(not(test), allow(dead_code))]
    pub count: u32,
    /// The frame's light count — the header's table length. Read from
    /// the header bytes by the shader and the tests, from this field by
    /// the tests alone.
    #[cfg_attr(not(test), allow(dead_code))]
    pub lights: u32,
    /// The packed bytes: a 16-byte header, one `vec4` of range per light,
    /// then the per-light blocks of records, each [`OCCLUDER_RECORD`] bytes
    /// plus one [`OCCLUDER_PLANE`] per edge.
    pub data: Vec<u8>,
}

/// The reach of a directional light's parallel shadow rays, in pixels:
/// the distance the virtual source sits back along the light's axis from
/// each pixel. Mirrors `SHADOW_RANGE` in `shaders/lighting.wgsl` — the
/// CPU needs it to bound the region those rays sweep, and
/// `shadow_range_mirrors_the_shader` keeps the two values tied.
pub(crate) const SHADOW_RANGE: f32 = 1500.0;

/// One occluder's packed record with the fact the light cull needs beside
/// it: its world-space axis-aligned bounds, derived from the very same
/// inputs as its record.
struct OccRecord {
    bytes: Vec<u8>,
    bounds: [f32; 4],
}

/// The world-space AABB of the occluder whose local box (`center`, `half`)
/// rides `world`: the transformed center plus the transform's absolute
/// columns acting on the half-extents — the axis-aligned superset every
/// rotated or mirrored box lives inside.
fn occ_world_bounds(world: &Transform, center: &[f32; 2], half: &[f32; 2]) -> [f32; 4] {
    let c = world.apply(*center);
    let ex = (world.m[0][0] * half[0]).abs() + (world.m[0][1] * half[1]).abs();
    let ey = (world.m[1][0] * half[0]).abs() + (world.m[1][1] * half[1]).abs();
    [c[0] - ex, c[1] - ey, c[0] + ex, c[1] + ey]
}

/// Builds one master record — bytes, plane count, world bounds — and pushes
/// it; a transform that cannot invert collapses the box to a line or point,
/// occludes nothing, and earns no record.
fn push_master(
    master: &mut Vec<OccRecord>,
    world: &Transform,
    center: &[f32; 2],
    half: &[f32; 2],
    planes: &[[f32; 4]],
    layers: LightLayers,
) {
    let mut bytes = Vec::with_capacity(OCCLUDER_RECORD + planes.len() * OCCLUDER_PLANE);
    let mut cursor = 0usize;
    if write_occluder_record(&mut bytes, &mut cursor, world, center, half, planes, layers) != 1 {
        return;
    }
    master.push(OccRecord {
        bytes,
        bounds: occ_world_bounds(world, center, half),
    });
}

/// The region a light's shadow rays can ever traverse, given that only
/// pixels inside the viewport ask the question — and, for a ranged light,
/// only pixels the light actually reaches answer it at all. A directional
/// light's rays are parallel, sweeping the window back along the axis by
/// [`SHADOW_RANGE`]; the axis-aligned box around that parallelogram is the
/// conservative region, exactly as the whole window lights. A ranged
/// (point or cone) light is different in kind: `light_mix` rejects pixels
/// past the falloff radius before a single ray issues (the reach reject in
/// `shaders/lighting.wgsl`), so the only segments that can ever matter run
/// from a point of the light's own penumbra disk to a pixel within the
/// radius of it — and a segment between two points of a disk never leaves
/// it. The disk around the light holding every one of those segments is
/// then the whole region: radius `max(falloff, penumbra * √2)`, the
/// penumbra term sized to the corner taps of the nine-point grid. An
/// occluder whose bounds miss the region blocks nothing that any pixel
/// will show, and need never be walked.
enum RayRegion {
    Hull(Vec<[f32; 2]>),
    Bounds([f32; 4]),
    Disc { center: [f32; 2], radius: f32 },
}

fn region_hits(region: &RayRegion, bounds: &[f32; 4]) -> bool {
    match region {
        RayRegion::Bounds(r) => {
            r[0] <= bounds[2] && bounds[0] <= r[2] && r[1] <= bounds[3] && bounds[1] <= r[3]
        }
        RayRegion::Hull(hull) => aabb_hull_overlap(bounds, hull),
        RayRegion::Disc { center, radius } => {
            // Disk against axis-aligned box: the box's point nearest the
            // center, found by clamping the center into the box on each
            // axis; the distance to it, squared, against the radius
            // squared. Touching counts as hitting — the conservative
            // direction, as everywhere in this cull.
            let dx = (bounds[0] - center[0]).max(center[0] - bounds[2]).max(0.0);
            let dy = (bounds[1] - center[1]).max(center[1] - bounds[3]).max(0.0);
            dx * dx + dy * dy <= radius * radius
        }
    }
}

/// The regions one light's shadow rays sweep, as a [`RayRegion`] per
/// necessary truth: an occluder joins the light's block only when its
/// bounds hit *every* one of them, so each truth added is culling bought
/// for the price of one overlap test per record. A directional light
/// sweeps the parallelogram the window drags back along its axis, which
/// is its one truth and the whole story. A ranged light answers to two:
/// its rays still only run to visible pixels, so its bounds must hit the
/// light-window hull as ever — and since the shader asks nothing past the
/// falloff radius, they must also hit the disk of contributing rays
/// around the light, sized `max(falloff, penumbra * √2)` as
/// [`RayRegion::Disc`] argues. Which of the two bites harder depends on
/// the light: a lamp of a few hundred pixels on a busy screen is held to
/// its disk, a distant floodlight of enormous radius to its hull.
fn ray_regions(
    pos: [f32; 2],
    radius: f32,
    penumbra: f32,
    dir: [f32; 2],
    viewport: [f32; 2],
) -> Vec<RayRegion> {
    if radius < 0.0 {
        let (dx, dy) = (-dir[0] * SHADOW_RANGE, -dir[1] * SHADOW_RANGE);
        let corners = [
            [0.0, 0.0],
            [viewport[0], 0.0],
            [viewport[0], viewport[1]],
            [0.0, viewport[1]],
        ];
        let mut b = [f32::MAX, f32::MAX, f32::MIN, f32::MIN];
        for c in corners
            .into_iter()
            .chain(corners.into_iter().map(|c| [c[0] + dx, c[1] + dy]))
        {
            b[0] = b[0].min(c[0]);
            b[1] = b[1].min(c[1]);
            b[2] = b[2].max(c[0]);
            b[3] = b[3].max(c[1]);
        }
        return vec![RayRegion::Bounds(b)];
    }
    vec![
        RayRegion::Hull(light_window_hull(pos, viewport)),
        RayRegion::Disc {
            center: pos,
            // The hairline keeps touching — as opposed to cleanly
            // separated — records in, the same courtesy
            // `aabb_hull_overlap` extends, so the packer's disk and the
            // shader's rays cannot disagree across a rounding of the
            // radius alone.
            radius: radius.max(1e-4).max(penumbra * std::f32::consts::SQRT_2) + 1e-3,
        },
    ]
}

/// The convex hull of the light's position and the whole window — five
/// points, so Andrew's monotone chain is more than enough and no
/// cleverness is warranted. The window is the fragment coordinate's
/// rectangle, `(0, 0)` to the viewport's size — the space the packer's
/// lights, occluders and pixels all share by the time a `Draw` arrives.
fn light_window_hull(l: [f32; 2], viewport: [f32; 2]) -> Vec<[f32; 2]> {
    let (w, h) = (viewport[0], viewport[1]);
    let mut pts = vec![[0.0, 0.0], [w, 0.0], [w, h], [0.0, h], l];
    pts.sort_by(|a, b| {
        a[0].partial_cmp(&b[0])
            .unwrap_or(core::cmp::Ordering::Equal)
            .then(
                a[1].partial_cmp(&b[1])
                    .unwrap_or(core::cmp::Ordering::Equal),
            )
    });
    pts.dedup_by(|a, b| a[0] == b[0] && a[1] == b[1]);
    let cross = |o: &[f32; 2], a: &[f32; 2], b: &[f32; 2]| {
        (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    };
    let mut lower: Vec<[f32; 2]> = Vec::new();
    for p in &pts {
        while lower.len() >= 2 && cross(&lower[lower.len() - 2], &lower[lower.len() - 1], p) <= 0.0
        {
            lower.pop();
        }
        lower.push(*p);
    }
    let mut upper: Vec<[f32; 2]> = Vec::new();
    for p in pts.iter().rev() {
        while upper.len() >= 2 && cross(&upper[upper.len() - 2], &upper[upper.len() - 1], p) <= 0.0
        {
            upper.pop();
        }
        upper.push(*p);
    }
    lower.pop();
    upper.pop();
    lower.extend(upper);
    if lower.is_empty() {
        // Every point coincided (a zero viewport at the light): the
        // "hull" is that point, and the overlap test still reads it.
        return pts;
    }
    lower
}

/// Separating-axis overlap of an axis-aligned box and a convex polygon:
/// the box's own two axes plus every edge normal, each project-and-compare.
/// A hairline of tolerance keeps touching — as opposed to cleanly
/// separated — shapes in, which is the conservative direction: a false
/// "overlap" costs one walked record, a false "separated" loses a shadow.
fn aabb_hull_overlap(a: &[f32; 4], hull: &[[f32; 2]]) -> bool {
    if hull.is_empty() {
        return true;
    }
    let corners = [[a[0], a[1]], [a[2], a[1]], [a[2], a[3]], [a[0], a[3]]];
    let mut axes = vec![[1.0f32, 0.0], [0.0, 1.0]];
    for i in 0..hull.len() {
        let (u, v) = (hull[i], hull[(i + 1) % hull.len()]);
        let n = [-(v[1] - u[1]), v[0] - u[0]];
        if n[0] * n[0] + n[1] * n[1] > 1e-12 {
            axes.push(n);
        }
    }
    axes.iter().all(|n| {
        let mut amin = f32::MAX;
        let mut amax = f32::MIN;
        for c in &corners {
            let d = c[0] * n[0] + c[1] * n[1];
            amin = amin.min(d);
            amax = amax.max(d);
        }
        let mut hmin = f32::MAX;
        let mut hmax = f32::MIN;
        for c in hull {
            let d = c[0] * n[0] + c[1] * n[1];
            hmin = hmin.min(d);
            hmax = hmax.max(d);
        }
        amin <= hmax + 1e-6 && hmin - 1e-6 <= amax
    })
}

/// Packs the frame's occluder field: the occluding shapes it drew, then the
/// occluders the app declared, cut into per-light blocks of everything each
/// light's rays can ever touch.
///
/// Painted rectangles and circles flagged `occludes` come first, in call order
/// — a circle as the polygon cut from it, since a curved edge has to be told to
/// hardware as flat ones — followed by `declared`, the
/// [`Canvas::occluder`](crate::Canvas::occluder) and
/// [`Canvas::occluder_polygon`](crate::Canvas::occluder_polygon) calls, in
/// declaration order. Any other painted shape casts nothing: a sprite, a line
/// or a glyph has no silhouette here to pack, which is what the declare channel
/// exists for. The two channels add; neither suppresses the
/// other. A shape whose world transform cannot be inverted is skipped (it
/// collapses to a line or a point and occludes nothing), and every other draw
/// is ignored.
///
/// That master list is then culled per light: an occluder enters a light's
/// block when its world bounds hit every region that light's shadow rays
/// sweep (see [`RayRegion`] and [`ray_regions`]) — conservative by
/// construction, since every segment from the light's disk to a pixel the
/// light will ever light lies inside every one of the regions, so what the
/// cull drops could never have blocked a lit pixel. The fragment shader
/// walks only its own light's block, which turns the per-pixel cost from
/// `lights × all-occluders` into `lit-lights × near-occluders`; the safety
/// property is pinned by `the_cull_keeps_every_occluder_that_could_block`
/// in `backend/tests/fields.rs`.
///
/// The lights cut here are the `Draw::Light`s of `draws` — the frame must
/// pack the light field and the occluder field from the same draw list,
/// because the shader indexes the range table by the light field's order.
/// A pack holding no light holds no table: its tail is the master list,
/// the pre-per-light layout intact, which nothing may read while a light
/// exists anywhere.
pub(crate) fn pack_occluder_field(
    draws: &[Draw],
    declared: &[Occluder],
    viewport: [f32; 2],
) -> OccluderField {
    // The master list: every occluder once, each record's bytes with its
    // world bounds beside them.
    let mut master: Vec<OccRecord> = Vec::new();
    for draw in draws {
        let Draw::Shape {
            world,
            center,
            params,
            kind,
            occludes,
            occludes_layers,
            ..
        } = draw
        else {
            continue;
        };
        if *occludes != 1.0 {
            continue;
        }
        // The two kinds the paint pipeline draws and the field can hold. A
        // rectangle is its own record; a circle is the polygon the field
        // speaks, cut here rather than in the shape shader because the shadow
        // wants edges where the painting wants a distance, and the record's
        // inverse transform then carries the turn or mirror for free. A circle
        // without a usable radius is skipped for the very reason a collapsed
        // transform is: it occludes nothing — and no polygon can be cut from
        // it, so passing it on would trade a silent no-op for a panic in the
        // frame path, which is the worse of the two by a wide margin.
        match *kind {
            1.0 => push_master(&mut master, world, center, params, &[], *occludes_layers),
            0.0 if params[0] > 0.0 && params[0].is_finite() => {
                let polygon = Convex::disc(*center, params[0]);
                let bounds = polygon.bounds_as_box();
                push_master(
                    &mut master,
                    world,
                    &bounds.center,
                    &bounds.half,
                    polygon.planes(),
                    *occludes_layers,
                );
            }
            _ => continue,
        }
    }
    for o in declared {
        // A box brings no edges; a polygon brings its own, already in the
        // space the record's inverse transform points at.
        let planes = match &o.polygon {
            Some(polygon) => polygon.planes(),
            None => &[],
        };
        push_master(&mut master, &o.world, &o.center, &o.half, planes, o.layers);
    }
    // The lights, in the order the light field packed them — the shader
    // indexes the table by that same order.
    let lights: Vec<&Draw> = draws
        .iter()
        .filter(|draw| matches!(draw, Draw::Light { .. }))
        .collect();
    // Header and range table first, so the blocks can start at a known
    // offset; every record spans whole vec4s, so byte and vec4 addresses
    // divide exactly.
    let mut data = vec![0u8; OCCLUDER_FIELD_HEADER + lights.len() * 16];
    let mut cursor = data.len();
    let mut count = 0u32;
    // The shader's array index 0 is the tail's start — the byte right
    // past the header — so ranges are counted in vec4s from there, not
    // from the buffer's front.
    let tail = OCCLUDER_FIELD_HEADER;
    if lights.is_empty() {
        // No lights: the shader asks no occlusion questions, so there are
        // no blocks to fill and no table to index — but the field keeps
        // the master list, byte for byte as it was packed. It costs the
        // rare lightless frame what it always cost, and the field-layout
        // tests keep reading the list they always read.
        for record in &master {
            let span = record.bytes.len();
            data.resize(cursor + span, 0);
            data[cursor..cursor + span].copy_from_slice(&record.bytes);
            cursor += span;
            count += 1;
        }
    }
    for (i, light) in lights.iter().enumerate() {
        let Draw::Light {
            pos,
            radius,
            penumbra,
            dir,
            ..
        } = *light
        else {
            unreachable!("filtered to lights")
        };
        let regions = ray_regions(*pos, *radius, *penumbra, *dir, viewport);
        let start = ((cursor - tail) / 16) as f32;
        for record in &master {
            if !regions
                .iter()
                .all(|region| region_hits(region, &record.bounds))
            {
                continue;
            }
            let span = record.bytes.len();
            data.resize(cursor + span, 0);
            data[cursor..cursor + span].copy_from_slice(&record.bytes);
            cursor += span;
            count += 1;
        }
        let end = ((cursor - tail) / 16) as f32;
        let off = OCCLUDER_FIELD_HEADER + i * 16;
        write_f32_at(&mut data, off, start);
        write_f32_at(&mut data, off + 4, end);
    }
    data[0..4].copy_from_slice(&count.to_le_bytes());
    data[4..8].copy_from_slice(&(lights.len() as u32).to_le_bytes());
    OccluderField {
        count,
        lights: lights.len() as u32,
        data,
    }
}

/// Writes one occluder record at `cursor` — its half-planes included —
/// advances `cursor` past it, and returns how many occluders were written:
/// `1`, or `0` when `world` cannot be inverted — a transform that collapses a
/// box to a line or a point occludes nothing, and the field keeps no hole
/// where its record would have gone.
fn write_occluder_record(
    data: &mut Vec<u8>,
    cursor: &mut usize,
    world: &Transform,
    center: &[f32; 2],
    half: &[f32; 2],
    planes: &[[f32; 4]],
    layers: LightLayers,
) -> u32 {
    let Some(inverse) = world.invert() else {
        return 0;
    };
    let off = *cursor;
    let span = OCCLUDER_RECORD + planes.len() * OCCLUDER_PLANE;
    data.resize(off + span, 0);
    // vec4 0: the inverse transform's linear part, column-major — the
    // same layout the shape uniforms' `to_local` uses.
    write_f32_at(data, off, inverse.m[0][0]);
    write_f32_at(data, off + 4, inverse.m[1][0]);
    write_f32_at(data, off + 8, inverse.m[0][1]);
    write_f32_at(data, off + 12, inverse.m[1][1]);
    // vec4 1: the inverse transform's translation, then the box's local
    // center.
    write_f32_at(data, off + 16, inverse.t[0]);
    write_f32_at(data, off + 20, inverse.t[1]);
    write_f32_at(data, off + 24, center[0]);
    write_f32_at(data, off + 28, center[1]);
    // vec4 2: the box's local half-extents, then the edge count — the
    // component that was padding before polygons, and still zero for a box
    // — then the layer mask's raw bits, in the slot that was the vec4's
    // zero padding: the shader `bitcast<u32>`s it back and skips the
    // record for receivers whose light layers it does not name.
    write_f32_at(data, off + 32, half[0]);
    write_f32_at(data, off + 36, half[1]);
    write_f32_at(data, off + 40, planes.len() as f32);
    data[off + 44..off + 48].copy_from_slice(&layers.bits().to_le_bytes());
    // And the edges themselves, in the occluder's own space: each an outward
    // unit normal and the offset it sits at, ready to clip a ray against
    // without a normalisation or a square root on the way.
    for (k, plane) in planes.iter().enumerate() {
        let at = off + OCCLUDER_RECORD + k * OCCLUDER_PLANE;
        write_f32_at(data, at, plane[0]);
        write_f32_at(data, at + 4, plane[1]);
        write_f32_at(data, at + 8, plane[2]);
    }
    *cursor = off + span;
    1
}
