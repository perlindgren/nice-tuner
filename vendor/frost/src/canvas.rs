//! The frame's drawing surface and the scene's expansion into a draw
//! list: [`Canvas`] collects the app's immediate draws, and `draw_node`
//! walks a [`Scene`] tree, composing transforms and colors and turning
//! each shape into the draws the backend renders — the text-to-quads
//! expansion included.

use crate::backend::*;
use crate::collision::Convex;
use crate::objects::*;
use crate::particles::*;
use crate::text;
use std::collections::HashMap;
use std::sync::Arc;

/// One occluder the app declared for this frame: a box in its own space,
/// placed by a world transform, added to the frame's occluder field beside
/// the shapes flagged `occludes` (see [`Canvas::occluder`]).
pub(crate) struct Occluder {
    /// The occluder's local-to-pixel transform, the canvas scale folded in —
    /// the same space and the same meaning as a [`Draw::Shape`]'s `world`.
    pub(crate) world: Transform,
    /// The box's center in the occluder's own space.
    pub(crate) center: [f32; 2],
    /// The box's half-extents in the occluder's own space: the full size is
    /// `2 * half`, the [`Canvas::rectangle`] convention. For a polygon this is
    /// the box around it, and the field uses it to reject a ray with four
    /// comparisons before any edge is clipped.
    pub(crate) half: [f32; 2],
    /// The shape's edges, when it is a polygon rather than a box: the
    /// half-planes [`Convex`] holds, in the same local space as `center` and
    /// `half`. `None` is the box.
    pub(crate) polygon: Option<Convex>,
    /// The light layers this occluder's shadow lands on, matched against
    /// each receiver's [`SceneNode::lit_layers`]. The
    /// [`Canvas::occluder`] constructors keep it at
    /// [`LightLayers::default()`] (layer 0); [`Canvas::occluder_layers`]
    /// and [`Canvas::occluder_polygon_layers`] name the set.
    pub(crate) layers: LightLayers,
}

/// The drawing surface for a frame, reachable through the [`crate::Context`] passed
/// to [`crate::Process::process`] (which derefs to it).
///
/// Coordinates are in pixels with the origin at the window center and the y
/// axis pointing up: the top-left corner is `(-width/2, height/2)` and the
/// bottom-right corner is `(width/2, -height/2)`.
///
/// These are *logical* pixels — the OS's points, which the display's scale
/// factor turns into the physical panel pixels the frame is rendered at
/// ([`Canvas::scale_factor`], [`crate::Config::window_size`]). On a 100% display one
/// unit is one panel pixel; on a Retina Mac two. Everything the frame holds —
/// coordinates, text sizes, line widths, the mouse — shares this one unit, so
/// an app laid out in it looks the same on every display and stays sharp on
/// high-density ones: the engine rasterizes at the physical resolution and
/// bakes text at the scaled size underneath the user space.
pub struct Canvas {
    size: (f32, f32),
    /// The display's scale factor: physical panel pixels per user-space
    /// (logical) pixel. [`Canvas::user_to_pixel`] folds it into every draw,
    /// and the immediate methods multiply their scalar sizes (a line's
    /// width, a circle's radius, a light's reach) by it, so those reach the
    /// shaders in panel pixels while everything the app speaks stays in
    /// logical ones.
    scale: f32,
    /// The base draw group: the immediate draw methods and every scene's
    /// root subtree, mixed by `z` and call order.
    pub(crate) draws: Vec<Draw>,
    /// One draw list per explicit scene layer, in declaration order (the
    /// same index as in `layer_orders`).
    layer_draws: Vec<Vec<Draw>>,
    /// The order of each explicit scene layer, in declaration order.
    layer_orders: Vec<f32>,
    /// The occluders the app declared this frame with [`Canvas::occluder`],
    /// in declaration order. The frame's occluder field takes the shapes
    /// flagged `occludes` first and these after them.
    pub(crate) occluders: Vec<Occluder>,
}

impl Canvas {
    /// A canvas for a window `size` logical pixels wide and high, rendered
    /// at `scale` physical panel pixels per logical pixel.
    pub(crate) fn new(size: (u32, u32), scale: f32) -> Self {
        Self {
            size: (size.0 as f32, size.1 as f32),
            scale: scale.max(f32::EPSILON),
            draws: Vec::new(),
            layer_draws: Vec::new(),
            layer_orders: Vec::new(),
            occluders: Vec::new(),
        }
    }

    /// The window size in pixels as `(width, height)`.
    ///
    /// The unit is the user-space pixel — logical, not panel: on a display
    /// with a scale factor of 2 this is half the framebuffer's pixel count
    /// (see [`Canvas::scale_factor`]).
    pub fn size(&self) -> (f32, f32) {
        self.size
    }

    /// The display's scale factor: physical panel pixels per user-space
    /// (logical) pixel — `1.0` on a 100% display, `2.0` on a Retina Mac.
    ///
    /// The frame is already rendered at the physical resolution and all
    /// user-space draws are scaled by this factor underneath, so an app
    /// rarely needs it: it exists for the rare case that measures something
    /// in panel pixels — a fixed physical render target, say — and needs to
    /// divide the factor back out.
    pub fn scale_factor(&self) -> f32 {
        self.scale
    }

    /// Draws a line from `(x0, y0)` to `(x1, y1)` in `color` with `width` in
    /// pixels.
    ///
    /// `z` is the draw order: lower `z` is drawn first (further back). Lines
    /// with the same `z` are drawn in call order, so the last one drawn is on
    /// top.
    #[allow(clippy::too_many_arguments)] // immediate-mode draw call
    pub fn line(&mut self, x0: f32, y0: f32, x1: f32, y1: f32, color: Color, width: f32, z: f32) {
        self.draws.push(Draw::Line {
            a: self.user_to_pixels(x0, y0),
            b: self.user_to_pixels(x1, y1),
            width: width.max(0.0) * self.scale,
            color,
            z,
            // An immediate canvas draw is not owned by any scene node.
            diagnostic: false,
        });
    }

    /// Draws a filled circle centered at `(cx, cy)` with `radius` in pixels,
    /// in `color`.
    ///
    /// `z` is the draw order: lower `z` is drawn first (further back). Circles
    /// with the same `z` are drawn in call order, so the last one drawn is on
    /// top.
    pub fn circle(&mut self, cx: f32, cy: f32, radius: f32, color: Color, z: f32) {
        self.draws.push(Draw::Circle {
            center: self.user_to_pixels(cx, cy),
            radius: radius.max(0.0) * self.scale,
            color,
            z,
            // An immediate canvas draw is not owned by any scene node.
            diagnostic: false,
        });
    }

    /// Draws every particle in `particles` as circles in one batched
    /// (instanced) draw call, in `color`, at draw order `z`.
    ///
    /// Each particle is a circle at its position with its `size` as radius,
    /// and its alpha is scaled by its remaining life fraction
    /// (`life / max_life`), so the batch fades out as its particles die.
    ///
    /// The whole batch shares one base `color` and one `z`; each particle's
    /// own `color` still multiplies with it, so per-particle tinting works
    /// here too.
    ///
    /// The positions are in the canvas's user space — the window, not any
    /// node: no transform, scale, or modulate of any scene node applies to
    /// the batch. To draw particles that follow a node (transformed,
    /// scaled, and tinted by it), give the node a
    /// [`Shape::Particles`] shape instead.
    #[deprecated(
        since = "0.1.0",
        note = "give the node a `Shape::Particles` shape instead: the batch is drawn in the node's local space, with the node's world transform, scale, and composed modulate applied. Keep this method only for batches that live directly in the window's user space, with no node to attach them to"
    )]
    pub fn particles(&mut self, particles: &[Particle], color: Color, z: f32) {
        if particles.is_empty() {
            return;
        }
        let mut data = Vec::with_capacity(particles.len() * 32);
        for p in particles {
            let [px, py] = self.user_to_pixels(p.pos[0], p.pos[1]);
            // Circles are rotationally symmetric, so the pixel-space angle
            // the fragment rotates out never matters; 0.0 keeps the packing
            // trivial.
            data.extend_from_slice(&particle_instance(p, px, py, p.size * self.scale, 0.0));
        }
        self.draws.push(Draw::Particles {
            data,
            count: particles.len() as u32,
            color,
            kind: 0.0,
            aspect: 1.0,
            sprite_data: None,
            sprite_size: [0, 0],
            sprite_generation: 0,
            // Immediate canvas draws are unlit: the `lit` flag is a
            // scene-node property, and the diagnostic marker is — no
            // scene node owns this batch.
            lit: 0.0,
            lit_layers: LightLayers::default(),
            z,
            diagnostic: false,
        });
    }

    /// Draws a filled rectangle centered at `(cx, cy)` with `dx` and `dy`
    /// extents (half-width and half-height) in pixels, in `color`.
    ///
    /// `z` is the draw order: lower `z` is drawn first (further back).
    /// Rectangles with the same `z` are drawn in call order, so the last one
    /// drawn is on top.
    pub fn rectangle(&mut self, cx: f32, cy: f32, dx: f32, dy: f32, color: Color, z: f32) {
        self.draws.push(Draw::Rectangle {
            center: self.user_to_pixels(cx, cy),
            extent: [dx.max(0.0) * self.scale, dy.max(0.0) * self.scale],
            color,
            z,
            // An immediate canvas draw is not owned by any scene node.
            diagnostic: false,
        });
    }

    /// Declares a box that cuts the frame's lights, without drawing anything.
    ///
    /// `world` places the occluder the way a scene node's `transform` places
    /// its shape — the app's user space, window-centered and y-up, usually
    /// rotated about the occluder's own center and then translated — and
    /// `center` and `half` are the box in its *own* space, `half` being the
    /// [`Canvas::rectangle`] convention: the full size is `2 * half`.
    ///
    /// Shape and transform stay separate all the way to the shader, which
    /// pulls the light and the pixel into the occluder's local space rather
    /// than pushing a sheared box out into the window. That is what lets a
    /// non-uniformly scaled `world` occlude like the drawing it stands for,
    /// corners and all — and it is why a box is declared as a local shape
    /// instead of as the four world corners it happens to have this frame.
    ///
    /// Declaring is how anything that is not a painted rectangle or circle
    /// occludes. The light field takes its painted occluders from rectangle
    /// and circle shapes flagged `occludes`, so a `Sprite` — or a whole tilemap, which is one sprite
    /// with an atlas — casts no shadow however its node is flagged: an
    /// occluder that is a sprite's body, a tile, or a physics body's hit box
    /// has to be declared. Declared occluders join the field after the painted
    /// ones, in declaration order, and the two channels add: nothing here
    /// suppresses a flagged rectangle.
    ///
    /// A declared occluder casts *shadows* only. It stops no body — collision
    /// remains the app's own [`crate::Collider`], the same box built from the
    /// same transform and asked a different question.
    pub fn occluder(&mut self, world: Transform, center: [f32; 2], half: [f32; 2]) {
        self.occluder_layers(world, center, half, LightLayers::default());
    }

    /// Declares a box occluder whose shadow lands only on receivers lit
    /// from one of `layers`: the occluder cuts a lit pixel's light only
    /// when `layers` and that receiver's
    /// [`lit_layers`](SceneNode::lit_layers) share a layer. This is how a
    /// wall shadows the far background while the floor it stands on —
    /// lit from another layer — sees no shadow from it.
    ///
    /// See [`Canvas::occluder`] for everything else; the set changes only
    /// who the shadow falls on.
    pub fn occluder_layers(
        &mut self,
        world: Transform,
        center: [f32; 2],
        half: [f32; 2],
        layers: LightLayers,
    ) {
        let to_pixel = world.compose(&self.user_to_pixel());
        self.occluders.push(Occluder {
            world: to_pixel,
            center,
            half: [half[0].max(0.0), half[1].max(0.0)],
            polygon: None,
            layers,
        });
    }

    /// Declares a convex polygon occluder: [`Convex`] in the node's own space,
    /// placed by the same kind of `world` transform [`Canvas::occluder`] takes.
    ///
    /// This is how an occluder that is neither a rectangle nor a disc reaches
    /// the light field — a rock, a sloped roof, a hit box with corners — and
    /// the disc reaches it the same way, as the polygon it is cut into for the
    /// GPU, since a curved edge has to be told to hardware as flat ones.
    ///
    /// The polygon never leaves its own space. Its edges ship next to the same
    /// inverse transform a box ships, so an instance that is turned or
    /// mirrored needs no second polygon and no re-derived normals: the shader
    /// brings the light and the pixel to the shape, as it has always done for
    /// a box. That is why a convex shape's winding survives a y-flip on the
    /// way to the framebuffer without anyone noticing — the shape is asked the
    /// question in the space it was drawn in.
    ///
    /// Like the box form, this casts shadows only; it stops no body.
    pub fn occluder_polygon(&mut self, world: Transform, polygon: &Convex) {
        self.occluder_polygon_layers(world, polygon, LightLayers::default());
    }

    /// Declares a convex polygon occluder whose shadow lands only on
    /// receivers lit from one of `layers`, the polygon sibling of
    /// [`Canvas::occluder_layers`].
    pub fn occluder_polygon_layers(
        &mut self,
        world: Transform,
        polygon: &Convex,
        layers: LightLayers,
    ) {
        let to_pixel = world.compose(&self.user_to_pixel());
        let bounds = polygon.bounds_as_box();
        self.occluders.push(Occluder {
            world: to_pixel,
            center: bounds.center,
            half: bounds.half,
            polygon: Some(*polygon),
            layers,
        });
    }

    /// Draws `text` in `font` at `size` pixels per em and `weight` on the
    /// font's `wght` axis (400.0 is Regular; see [`Shape::with_weight`]),
    /// centered on `(cx, cy)`, in `color`, at draw order `z`.
    ///
    /// This is the immediate sibling of [`Shape::text`]: the same text
    /// pipeline (layout, glyph atlas, per-glyph quads) reached without a
    /// scene node, which is what overlay UIs like [`Ui`](crate::Ui) need.
    /// The text block is centered on `(cx, cy)` exactly as a text shape is
    /// centered on its node's origin. Like a text shape, `font` is the font
    /// file's bytes behind an [`Arc`]: pass the same `Arc` (or one cloned
    /// from it) to keep every block sharing one cached glyph atlas — per
    /// (font, size, weight): a font drawn at two weights caches two atlases.
    ///
    /// Immediate text is never lit and emits nothing: it always draws at
    /// full `color`, so UI text stays readable over any scene.
    /// How wide [`Canvas::text`] would lay `text`: the advance width in
    /// user-space units, through the very shaping the draw will do —
    /// so measure and paint agree. Enough to run mixed-colour pieces
    /// end to end, or right-align without guessing. Zero for a font
    /// that cannot be shaped.
    pub fn text_width(&self, font: &Arc<[u8]>, text: &str, size: f32, weight: f32) -> f32 {
        crate::text::layout(font, text, size, weight).map_or(0.0, |l| l.width)
    }

    #[allow(clippy::too_many_arguments)] // immediate-mode draw call
    pub fn text(
        &mut self,
        cx: f32,
        cy: f32,
        font: &Arc<[u8]>,
        text: impl Into<String>,
        size: f32,
        weight: f32,
        color: Color,
        z: f32,
    ) {
        self.draws.push(Draw::Text {
            // User space, translated to the text's center: the expansion
            // composes this with the user-to-pixel transform like a node's
            // world transform.
            world: Transform::translate([cx, cy]),
            font: Arc::clone(font),
            text: text.into(),
            size,
            weight,
            color,
            alpha: 1.0,
            // UI text emits nothing and takes no light: full `color`.
            emission: Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
            glow: None,
            lit: 0.0,
            lit_layers: LightLayers::default(),
            z,
            // An immediate canvas draw is not owned by any scene node.
            diagnostic: false,
        });
    }

    /// Registers an omni light at `(x, y)` in the window's user space.
    ///
    /// Lights are never drawn: they join the frame's light field, which
    /// every lit receiver ([`SceneNode::lit`]) evaluates per pixel. `color`
    /// is the light's color, `intensity` its strength (`1.0` is full
    /// strength, `0.0` is off), and `radius` its falloff extent in pixels:
    /// the light reaches exactly `radius` pixels from `(x, y)`, fading to
    /// zero at the edge.
    ///
    /// An omni light shines in every direction; use [`Canvas::light_cone`]
    /// for one that shines only into a cone.
    ///
    /// The light lives in the window's user space — not any scene node's:
    /// no node's transform, scale, or modulate applies to it. For a light
    /// that rides a node (transformed and tinted by it), give the node a
    /// [`Shape::Light`] shape instead.
    ///
    /// The frame's light field is rebuilt from that frame's lights, so to
    /// move a light, call this again each frame with the new position, and
    /// the field's ambient floor comes from the scene's
    /// [`ambient`](Scene::ambient).
    pub fn light(&mut self, x: f32, y: f32, color: Color, intensity: f32, radius: f32) {
        self.light_cone(x, y, color, intensity, radius, 0.0, std::f32::consts::TAU);
    }

    /// Registers a cone light whose apex is at `(x, y)` in the window's
    /// user space.
    ///
    /// Like [`Canvas::light`], but the light shines only into a cone:
    /// `direction` is the cone's axis in radians (counter-clockwise from
    /// the window's +x axis, y-up), and `spread` is the cone's full
    /// opening angle in radians (`PI` is a half-plane, `>= 2 * PI` shines
    /// in every direction like [`Canvas::light`]). Outside the cone the
    /// light contributes nothing.
    ///
    /// This light lives in the window's user space like
    /// [`Canvas::light`]'s, so `direction` is also the axis as seen on
    /// screen; a light riding a [`SceneNode`] picks up the node's
    /// rotation — see [`Shape::Light`].
    #[allow(clippy::too_many_arguments)] // immediate-mode draw call
    pub fn light_cone(
        &mut self,
        x: f32,
        y: f32,
        color: Color,
        intensity: f32,
        radius: f32,
        direction: f32,
        spread: f32,
    ) {
        // Fold the direction through the user transform as a vector, not
        // an angle: subtracting the image of the origin from the image of
        // the unit vector's tip cancels the translation and survives the
        // y flip (user space is y-up, pixel space y-down) without special
        // casing it. Normalizing guards against a future scaled user
        // space; the fallback keeps a degenerate direction usable.
        let tip = self.user_to_pixels(direction.cos(), direction.sin());
        let origin = self.user_to_pixels(0.0, 0.0);
        self.draws.push(Draw::Light {
            pos: self.user_to_pixels(x, y),
            radius: radius.max(0.0) * self.scale,
            intensity,
            color,
            // Immediate-mode lights are point-sized: hard shadows.
            penumbra: 0.0,
            dir: normalized_axis([tip[0] - origin[0], tip[1] - origin[1]]),
            cos_half: half_angle_cosine(spread),
            // The immediate-mode cone has no softness knob: hard edge.
            feather: 0.0,
            // Immediate-mode lights shine on layer 0: the layers a light
            // names come from its `Light` struct, and this API has none.
            layers: LightLayers::default(),
            // The light is never drawn, so its draw order is irrelevant.
            z: 0.0,
        });
    }

    /// Draws a [`Scene`] into the canvas.
    ///
    /// The scene passed to [`crate::run`] is drawn automatically every frame, after
    /// the [`crate::Process`] runs; use this method to draw additional scenes.
    ///
    /// Scene coordinates are the same user space as the direct draw methods:
    /// the origin is at the window center, y points up, and positive
    /// rotations are counterclockwise on screen.
    ///
    /// The scene is walked depth-first from the root. Each node's transform
    /// and scale are relative to its parent and compose onto the parent's
    /// (scale innermost), so a descendant's world transform is the full
    /// root-to-leaf composition. Each node's modulate multiplies into its
    /// shape's color and composes onto its children's, so a descendant's
    /// color is the channel-wise product of the modulates on the path from
    /// the root.
    /// Every node with a shape draws it (in the node's own local space)
    /// before its children — when the shape is a
    /// [`Shape::Particles`] batch, it is transformed, scaled, and tinted by
    /// the node, at the node's composed order — so parents paint under
    /// their descendants.
    ///
    /// The scene's root subtree joins the base group — the same group as
    /// the immediate draw methods — so its shapes interleave with the
    /// immediate draws by the usual stable `z` ordering, each shape carrying
    /// the `z` accumulated from its ancestors' `order` values.
    ///
    /// The scene's layers are separate groups: each layer is painted as a
    /// whole at its [`Layer::order`], after every group with a lower order
    /// and before every group with a higher one (higher order on top), and
    /// within the layer the local ordering applies.
    ///
    /// If the scene has a [`Scene::camera`], every group is drawn in the
    /// camera node's coordinate space instead of the fixed window-centered
    /// user space: the scene is transformed by the inverse of the camera's
    /// world transform (with the translation scaled per group), so the
    /// camera stays at the window origin and the scene moves and rotates
    /// around it. Each group uses the camera's translation scaled by its
    /// speed — the base group at `1.0`, each layer at its [`Layer::speed`] —
    /// so a layer with a higher speed moves faster than the camera (it reads
    /// as closer) and one with a lower speed slower (farther away).
    ///
    /// A [`Shape::Background`] is the exception in every group: it ignores
    /// its transform, never draws, and becomes the frame's clear color.
    ///
    /// A layer with a non-zero [`Layer::repeat`] component is drawn once
    /// per copy of its content that can overlap the window — the same
    /// content, re-used with just a displacement — so it can be scrolled
    /// through infinitely. With a pure-translation camera the window's box
    /// is at most one period wide, so only the few copies it and the
    /// content's span around it reach are drawn per axis; a rotated camera
    /// widens the box, and every copy it reaches is drawn. An object
    /// crossing a tile boundary is split into wrapping slices by the
    /// per-object scissor test. A period below the window size, or an
    /// object whose extent in a repeating axis exceeds its period, aborts
    /// the program with an error.
    pub fn draw_scene(&mut self, scene: &Scene) {
        let pixel = self.user_to_pixel();
        // The scene's camera, when it has one: the groups are drawn in the
        // camera node's coordinate space, so it stays at the window origin
        // and the scene moves around it. Without a camera (or with a
        // degenerate camera transform) the groups draw in the fixed
        // window-centered user space.
        let camera = scene.camera_world();
        // The base group renders from the camera at the full speed 1.0.
        let view = |speed: f32| match &camera {
            Some(c) => c
                .scaled_translation(speed)
                .invert()
                .unwrap_or(Transform::identity()),
            None => Transform::identity(),
        };
        draw_node(pixel, &scene.root, &view(1.0), 0.0, WHITE, &mut self.draws);
        for layer in &scene.layers {
            let index = self.layer_orders.len();
            self.layer_orders.push(layer.order);
            self.layer_draws.push(Vec::new());
            // A repeating layer is drawn once per tile the window overlaps,
            // each copy displaced by the tile's offset in the layer's own
            // coordinate space (before the camera view is applied).
            let view = view(layer.speed);
            for (ox, oy) in layer_repeat_offsets(layer, &view, self.size) {
                draw_node(
                    pixel,
                    &layer.root,
                    &Transform::translate([ox, oy]).compose(&view),
                    0.0,
                    WHITE,
                    &mut self.layer_draws[index],
                );
            }
        }
    }

    /// The affine transform from user coordinates (window center origin,
    /// y up, logical pixels) to the pixel coordinates the shaders use
    /// (top-left origin, y down, physical panel pixels):
    /// `(x, y) -> (s * (x + w/2), s * (h/2 - y))` with `s` the scale factor.
    ///
    /// Folding the factor in here is the whole high-density story: every
    /// scene geometry (shapes, sprites, glyph quads) composes onto this
    /// transform, so one multiply maps the logical layout onto the physical
    /// framebuffer. The scalar sizes that do not ride a transform — a
    /// stroke width, a circle's radius, a light's reach, a particle's size —
    /// multiply by `scale` where they are recorded instead.
    fn user_to_pixel(&self) -> Transform {
        let s = self.scale;
        Transform {
            m: [[s, 0.0], [0.0, -s]],
            t: [s * self.size.0 / 2.0, s * self.size.1 / 2.0],
        }
    }

    /// Expands every [`Draw::Text`] in every draw group of this frame — the
    /// base group and each layer's group — into one [`Draw::Sprite`] per
    /// shaped glyph, spliced in at the text draw's position so the stable
    /// z-sort keeps the glyph quads where the text node was. The glyphs stay
    /// in the group their text node belongs to, so a layer's text is sorted
    /// within that layer.
    ///
    /// The glyphs are laid out with `text::layout` (pen positions, y up,
    /// from the text's left baseline origin) and rasterized with
    /// `text::rasterize` into the per-`(font, size, weight)` atlas in
    /// `atlases`,
    /// which the caller keeps between frames so unchanged text never
    /// re-rasterizes and its texture buffer keeps a stable identity.
    /// Each glyph's quad is centered on its ink box; the whole text block
    /// is centered on the text node's origin.
    pub(crate) fn expand_text(&mut self, atlases: &mut HashMap<(u64, u32, u32), text::Atlas>) {
        let pixel = self.user_to_pixel();
        let scale = self.scale;
        let draws = std::mem::take(&mut self.draws);
        self.draws = expand_text_list(pixel, draws, atlases, scale);
        for list in &mut self.layer_draws {
            let draws = std::mem::take(list);
            *list = expand_text_list(pixel, draws, atlases, scale);
        }
    }

    /// Converts user coordinates (window center origin, y up) to the pixel
    /// coordinates the shaders use (top-left origin, y down).
    fn user_to_pixels(&self, x: f32, y: f32) -> [f32; 2] {
        self.user_to_pixel().apply([x, y])
    }

    /// The frame's draws in paint order.
    ///
    /// The base group and every explicit scene layer are separate draw
    /// groups. The groups are painted by ascending layer order — higher
    /// order on top — with the base group at order `0.0`, declared before
    /// every layer, so at an equal order it is painted first. Within each
    /// group, the draws are sorted by ascending `z`; the sort is stable, so
    /// draws with equal `z` keep call order and the last drawn is on top.
    pub(crate) fn paint_order(&mut self) -> Vec<Draw> {
        // The groups as (layer order, declaration index): the base group is
        // order 0.0, declared before every layer, so at an equal order it
        // is painted first.
        enum Group {
            Base,
            Layer(usize),
        }
        let mut groups: Vec<Group> = Vec::with_capacity(1 + self.layer_draws.len());
        groups.push(Group::Base);
        groups.extend((0..self.layer_draws.len()).map(Group::Layer));
        let order_of = |group: &Group| match group {
            Group::Base => (0.0f32, 0usize),
            Group::Layer(index) => (self.layer_orders[*index], *index + 1),
        };
        groups.sort_by(|a, b| {
            let (order_a, declared_a) = order_of(a);
            let (order_b, declared_b) = order_of(b);
            order_a
                .partial_cmp(&order_b)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(declared_a.cmp(&declared_b))
        });
        let mut painted = Vec::new();
        for group in groups {
            let list = match group {
                Group::Base => &mut self.draws,
                Group::Layer(index) => &mut self.layer_draws[index],
            };
            let mut draws = std::mem::take(list);
            draws.sort_by(|a, b| {
                a.z()
                    .partial_cmp(&b.z())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            painted.append(&mut draws);
        }
        painted
    }
}

/// Normalizes a pixel-space delta into a light cone's unit axis.
///
/// A delta with no usable length (zero, or a NaN from a NaN direction)
/// has no direction to speak of, so it falls back to the +x axis; the
/// cone gate then behaves normally around that axis.
fn normalized_axis(delta: [f32; 2]) -> [f32; 2] {
    let len = (delta[0] * delta[0] + delta[1] * delta[1]).sqrt();
    if len > 0.0 && len.is_finite() {
        [delta[0] / len, delta[1] / len]
    } else {
        [1.0, 0.0]
    }
}

/// The cosine of a light cone's half opening angle, from its full
/// `spread` in radians. The half angle is clamped to `0..=PI` first so
/// the cosine covers the cone's whole meaningful range: a `spread` of
/// `2 * PI` or more yields `-1.0`, which every pixel passes (an omni
/// light), while a negative `spread` yields the degenerate axis-only
/// `1.0` like `0` does. A NaN `spread` yields NaN, which fails the cone
/// gate everywhere: the light simply contributes nothing instead of
/// corrupting the frame.
fn half_angle_cosine(spread: f32) -> f32 {
    (spread * 0.5).clamp(0.0, std::f32::consts::PI).cos()
}

/// The cone's edge feather, in cosine units, from its full `spread` and
/// its `softness` in radians: the width of the band just inside each cone
/// edge across which the shader ramps the light from zero to full, as the
/// cosine difference between the band's outer and inner angles — the
/// shader gates on cosines, so the feather ships in the same units. The
/// softness is clamped to the half angle, so the feather spans at most
/// the whole cone and the axis stays fully lit. An omni light
/// (`spread` at or beyond `2 * PI`) and a `0` or negative softness both
/// yield `0.0`: the hard edge, and never a feather on an omni light,
/// whose gate must stay the all-pass it has always been.
fn feather_band(spread: f32, softness: f32) -> f32 {
    if softness <= 0.0 || spread >= std::f32::consts::TAU {
        return 0.0;
    }
    let half = (spread * 0.5).clamp(0.0, std::f32::consts::PI);
    (half - softness.clamp(0.0, half)).cos() - half.cos()
}

/// Draws one [`SceneNode`] and its subtree into `draws`, depth-first: the
/// node's shape (if any — which may itself be a [`Shape::Particles`] batch)
/// before its children, so parents paint under their descendants.
///
/// `parent` is the node's ancestors' world transform in user space, `order`
/// their accumulated draw order, and `modulate` their accumulated color
/// modulation; each is combined with the node's own value and passed on to
/// the children, so a descendant's values are the full root-to-leaf
/// composition.
fn draw_node(
    user_to_pixel: Transform,
    node: &SceneNode,
    parent: &Transform,
    order: f32,
    modulate: Color,
    draws: &mut Vec<Draw>,
) {
    // The node's effective draw order: the order inherited from its
    // ancestors plus its own, applied to the node's shape and passed on to
    // its whole subtree, just like the transform and the scale.
    let order = order + node.order;
    // The node's effective color modulation: the modulation inherited from
    // its ancestors multiplied by its own, applied to the node's shape's
    // color and passed on to its whole subtree, just like the transform and
    // the scale.
    let modulate = modulate.mul(node.modulate);
    // The node's world transform in user space: its scale first (innermost,
    // in the node's own space), then its transform (local -> parent space),
    // then the parent's world transform. Through `world` the scale
    // therefore applies to the node's shape and to its whole subtree, just
    // like the transform.
    let local = Transform::scale(node.scale).compose(&node.transform);
    let world = local.compose(parent);
    if let Some(shape) = &node.shape {
        // The node's local-to-pixel transform: the shape shaders evaluate in
        // pixel space (top-left, y down), so compose the user-to-pixel
        // transform onto the world transform, like the direct-draw methods
        // do for their arguments. The canvas's scale factor is folded into
        // `user_to_pixel`, so everything that rides this transform — and
        // every scalar sized from its scale below — lands in physical panel
        // pixels.
        let to_pixel = world.compose(&user_to_pixel);
        // Scale the anti-alias band with the transform's scale so it stays a
        // constant number of screen pixels wide under scaling — and across
        // display densities: the band is booked in panel pixels.
        let [sx, sy] = to_pixel.scales();
        let aa = AA_BAND / sx.max(sy).max(1e-9);
        // The canvas scale on its own: a light's reach and penumbra are
        // window-space sizes that never ride a node's scale, but they join
        // the light field in panel pixels like every other recorded scalar.
        let px_scale = user_to_pixel.scales()[0];
        let draw = match shape {
            // The shape shader evaluates in pixel space (top-left, y down),
            // so compose the user-to-pixel transform onto the world
            // transform, like the direct-draw methods do for their
            // arguments.
            Shape::Circle {
                center,
                radius,
                color,
            } => Some(Draw::Shape {
                world: to_pixel,
                center: *center,
                params: [(*radius).max(0.0), 0.0],
                kind: 0.0,
                aa,
                color: color.mul(modulate),
                emission: node.emission.mul(modulate),
                glow: node.glow.map(|g| g.mul(modulate)),
                lit: f32::from(node.lit),
                lit_layers: node.lit_layers,
                occludes: f32::from(node.occludes),
                occludes_layers: node.occludes_layers,
                z: order,
                diagnostic: node.diagnostic,
            }),
            Shape::Rectangle {
                center,
                extent,
                color,
            } => Some(Draw::Shape {
                world: to_pixel,
                center: *center,
                params: [extent[0].max(0.0), extent[1].max(0.0)],
                kind: 1.0,
                aa,
                color: color.mul(modulate),
                emission: node.emission.mul(modulate),
                glow: node.glow.map(|g| g.mul(modulate)),
                lit: f32::from(node.lit),
                lit_layers: node.lit_layers,
                occludes: f32::from(node.occludes),
                occludes_layers: node.occludes_layers,
                z: order,
                diagnostic: node.diagnostic,
            }),
            // A polyline: its points go through the composed
            // world-to-pixel transform and its width scales with the
            // geometric mean of the world's axis scales, like a particle's
            // size — the whole stroke becomes one draw, whatever its
            // length.
            Shape::Polyline {
                points,
                width,
                color,
            } => {
                if points.len() < 2 {
                    // Like an empty particle batch: nothing to draw.
                    None
                } else {
                    Some(Draw::Polyline {
                        points: points.iter().map(|p| to_pixel.apply(*p)).collect(),
                        width: (*width).max(0.0) * (sx * sy).sqrt(),
                        color: color.mul(modulate),
                        z: order,
                        diagnostic: node.diagnostic,
                    })
                }
            }
            // The sprite's local space is centered on the origin, one
            // texture pixel per scene pixel, so its extent is the texture
            // size.
            Shape::Sprite {
                data,
                width,
                height,
                color,
                alpha,
                filter,
                generation,
                program,
            } => Some(Draw::Sprite {
                world: to_pixel,
                data: data.clone(),
                // The buffer's stamped generation: an in-memory image may
                // be rebuilt with new pixels at any time, so its identity
                // is the constructor's stamp, not a constant.
                generation: *generation,
                size: [(*width as f32).max(0.0), (*height as f32).max(0.0)],
                texture_size: [*width, *height],
                filter: *filter,
                aa,
                tint: color.mul(modulate),
                alpha: *alpha,
                emission: node.emission.mul(modulate),
                glow: node.glow.map(|g| g.mul(modulate)),
                additive: node.additive,
                uv_rect: [0.0, 0.0, 1.0, 1.0],
                lit: f32::from(node.lit),
                lit_layers: node.lit_layers,
                z: order,
                diagnostic: node.diagnostic,
                // The program rides the shape, so every node drawing this
                // sprite paints with it; `None` is the engine's own.
                program: program.clone(),
            }),
            // Text is recorded in user space; `Canvas::expand_text` lays it
            // out and turns each glyph into a sprite quad before the frame
            // is rendered.
            Shape::Text {
                text,
                font,
                size,
                weight,
                color,
                alpha,
            } => Some(Draw::Text {
                world,
                font: font.clone(),
                text: text.clone(),
                size: *size,
                weight: *weight,
                color: color.mul(modulate),
                alpha: *alpha,
                emission: node.emission.mul(modulate),
                glow: node.glow.map(|g| g.mul(modulate)),
                lit: f32::from(node.lit),
                lit_layers: node.lit_layers,
                z: order,
                diagnostic: node.diagnostic,
            }),
            // The background ignores its transform: it is recorded in call
            // order and becomes the frame's clear color at render time.
            Shape::Background { color } => Some(Draw::Background {
                color: color.mul(modulate),
            }),
            // A light is never drawn: its position and cone axis are
            // recorded in pixel space (the node's local origin, and its
            // local cone direction, both through the composed transforms)
            // and it is promoted to the frame's light field at render time.
            Shape::Light { light } => {
                // The direction is a local-space angle, so it is folded
                // through the transform as a vector, not an angle: applying
                // the transform to the origin and to the unit vector's tip
                // and subtracting cancels the translation, carries the
                // rotation (and any mirror) into the axis, and needs no
                // atan2 that the pixel space's y flip would confuse.
                let origin = to_pixel.apply([0.0, 0.0]);
                let tip = to_pixel.apply([light.direction.cos(), light.direction.sin()]);
                Some(Draw::Light {
                    pos: origin,
                    // A negative radius is the directional sentinel
                    // (`Light::directional`): clamping it to zero would
                    // silently turn a sun into a degenerate point light,
                    // so only real radii get clamped. The reach and the
                    // penumbra are window-space pixels — they never ride a
                    // node's scale, but the canvas scale still carries them
                    // to panel pixels.
                    radius: if light.radius < 0.0 {
                        -1.0
                    } else {
                        light.radius.max(0.0) * px_scale
                    },
                    intensity: light.intensity,
                    color: light.color.mul(modulate),
                    penumbra: light.penumbra.max(0.0) * px_scale,
                    dir: normalized_axis([tip[0] - origin[0], tip[1] - origin[1]]),
                    cos_half: half_angle_cosine(light.spread),
                    feather: if light.spread >= std::f32::consts::TAU {
                        // An omni light has no cone edge to feather —
                        // the band is zero by construction — so the
                        // slot carries the penumbra's tap count to the
                        // shader instead, clamped to the grid's ends.
                        light.shadow_taps.clamp(1, 9) as f32
                    } else {
                        feather_band(light.spread, light.softness)
                    },
                    layers: light.layers,
                    z: order,
                })
            }
            // The particles ride the node's world transform and scale, are
            // tinted by `color` times the composed modulate (and each
            // particle's own color), and draw at the node's composed order —
            // one instanced draw, just after the node's own shape (if any)
            // and before its children.
            Shape::Particles {
                system,
                color,
                shape,
            } => {
                if system.is_empty() {
                    // Like an empty `Canvas::particles` call: nothing to
                    // draw.
                    None
                } else {
                    // The world scale as a uniform size factor: the
                    // geometric mean of the two axis scales of the composed
                    // local-to-pixel transform — exact under a uniform
                    // scale, area-preserving under a non-uniform one, where
                    // the shape cannot be stretched independently along the
                    // two axes and so keeps its proportions. Being composed,
                    // it carries the canvas scale too: the radius reaches
                    // the shader in panel pixels.
                    let radius_scale = (sx * sy).sqrt();
                    // The pixel-space angle of the node's local +x axis: the
                    // direction `to_pixel` maps `(1, 0)` to. A particle's own
                    // `angle` is measured in the local (y-up) frame, so it
                    // adds to that base angle when the mapping preserves
                    // orientation and subtracts from it when it flips the
                    // y axis (the usual case: the pixel space is y-down) —
                    // the flip turns a local counterclockwise turn into a
                    // pixel-space clockwise one.
                    let m = to_pixel.m;
                    let base = m[1][0].atan2(m[0][0]);
                    let flipped = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) < 0.0;
                    let (sprite_data, sprite_size, sprite_generation) = match shape {
                        ParticleShape::Sprite {
                            data,
                            width,
                            height,
                            generation,
                        } => (Some(data.clone()), [*width, *height], *generation),
                        _ => (None, [0, 0], 0),
                    };
                    let mut data = Vec::with_capacity(system.len() * 32);
                    for p in &system.particles {
                        let [px, py] = to_pixel.apply(p.pos);
                        let angle = if flipped {
                            base - p.angle
                        } else {
                            base + p.angle
                        };
                        data.extend_from_slice(&particle_instance(
                            p,
                            px,
                            py,
                            p.size * radius_scale,
                            angle,
                        ));
                    }
                    Some(Draw::Particles {
                        data,
                        count: system.len() as u32,
                        color: color.mul(modulate),
                        kind: shape.kind(),
                        aspect: shape.aspect(),
                        sprite_data,
                        sprite_size,
                        sprite_generation,
                        lit: f32::from(node.lit),
                        lit_layers: node.lit_layers,
                        z: order,
                        diagnostic: node.diagnostic,
                    })
                }
            }
            // The tile map: every placement becomes one instance of the
            // map's single instanced draw. A tile is a box, so its two
            // half-edges ride the composed transform's linear part
            // independently — scale, flip, and rotation all land on the
            // quad exactly as they would on a rectangle, where the
            // particle batch above can only carry a rotation as an
            // angle and a geometric-mean size — and the view clip rides
            // the same transform as a box, so the scissor cuts the
            // batch at the window the map is seen through.
            Shape::TileMap {
                data,
                width,
                height,
                generation,
                filter,
                tiles,
                clip,
                color,
            } => {
                if tiles.is_empty() {
                    // Like an empty particle batch: nothing to draw.
                    None
                } else {
                    let mut packed = Vec::with_capacity(tiles.len() * 64);
                    let mut box_min = [f32::MAX; 2];
                    let mut box_max = [f32::MIN; 2];
                    for tile in tiles {
                        let [cx, cy] = to_pixel.apply(tile.center);
                        // The tile's two half-edges through the linear
                        // part: each axis lands on a matrix column. An
                        // odd quarter-turn lays the cell sideways — the
                        // extents swap here, the flags tell the sampler.
                        let [sx, sy] = if tile.rot % 2 == 1 {
                            [tile.size[1], tile.size[0]]
                        } else {
                            tile.size
                        };
                        let ex = sx * 0.5;
                        let ey = sy * 0.5;
                        let u = [to_pixel.m[0][0] * ex, to_pixel.m[1][0] * ex];
                        let v = [to_pixel.m[0][1] * ey, to_pixel.m[1][1] * ey];
                        // The quad's own AABB: |u| + |v| per axis, so a
                        // rotated tile books its diamond, not its box.
                        let rx = u[0].abs() + v[0].abs();
                        let ry = u[1].abs() + v[1].abs();
                        box_min = [box_min[0].min(cx - rx), box_min[1].min(cy - ry)];
                        box_max = [box_max[0].max(cx + rx), box_max[1].max(cy + ry)];
                        packed.extend_from_slice(&tilemap_instance(
                            cx,
                            cy,
                            u,
                            v,
                            tile.uv,
                            &tile.color,
                            [
                                tile.flip_x as u8 as f32 + 2.0 * tile.flip_y as u8 as f32,
                                (tile.rot % 4) as f32,
                            ],
                        ));
                    }
                    let clip_px = clip.map(|[x0, y0, x1, y1]| {
                        aabb_of_box(
                            &to_pixel,
                            [(x0 + x1) / 2.0, (y0 + y1) / 2.0],
                            (x1 - x0) / 2.0,
                            (y1 - y0) / 2.0,
                        )
                    });
                    Some(Draw::TileMap {
                        data: packed,
                        count: tiles.len() as u32,
                        color: color.mul(modulate),
                        atlas_data: data.clone(),
                        atlas_size: [*width, *height],
                        atlas_generation: *generation,
                        filter: *filter,
                        lit: f32::from(node.lit),
                        lit_layers: node.lit_layers,
                        bounds: (box_min, box_max),
                        clip: clip_px,
                        z: order,
                        diagnostic: node.diagnostic,
                    })
                }
            }
        };
        if let Some(draw) = draw {
            draws.push(draw);
        }
    }
    for child in &node.children {
        draw_node(user_to_pixel, child, &world, order, modulate, draws);
    }
}

/// The copy offsets a repeating [`Layer`] is drawn at for this frame, in
/// the layer's own coordinate space (before the camera view is applied):
/// one offset per copy of the content that can overlap the window. A
/// non-repeating axis (`repeat` component `0.0`) has exactly one copy, at
/// offset `0.0`; a repeating axis has one copy per `abs(repeat)` period
/// that can reach the window's box — with a pure-translation camera the
/// box is at most one period wide, so only the few copies it and the
/// content's span around it reach. The offsets are ordered so `(0.0, 0.0)`
/// comes first when it is among them, so the stable z-sort keeps the base
/// content drawn before its copies at equal `z`.
///
/// Aborts the program with an error when a non-zero repeat period is
/// smaller than the window's width/height (the minimum repeat offset), or
/// when an object's extent in a repeating axis exceeds that axis's period
/// (the same object would be drawn twice).
fn layer_repeat_offsets(layer: &Layer, view: &Transform, size: (f32, f32)) -> Vec<(f32, f32)> {
    let (w, h) = size;
    // The window in the layer's coordinate space: the user-space window
    // rectangle (origin center, y up) mapped by the inverse view, as an
    // axis-aligned box.
    let to_layer = view.invert().unwrap_or(Transform::identity());
    let corners = [
        to_layer.apply([-w / 2.0, -h / 2.0]),
        to_layer.apply([w / 2.0, -h / 2.0]),
        to_layer.apply([-w / 2.0, h / 2.0]),
        to_layer.apply([w / 2.0, h / 2.0]),
    ];
    let mut min = corners[0];
    let mut max = corners[0];
    for corner in corners.iter().skip(1) {
        min = [min[0].min(corner[0]), min[1].min(corner[1])];
        max = [max[0].max(corner[0]), max[1].max(corner[1])];
    }
    // The layer's content extent in the layer's coordinate space (the union
    // of all its objects' boxes), or None when the layer has no drawable
    // objects: a copy of the content displaced by k * period spans
    // [content_min + k * period, content_max + k * period], and the content
    // need not sit inside the tile [0, period).
    let content = check_layer_repeat_extent(&layer.root, &Transform::identity(), layer.repeat);
    // The offsets of the axis: every copy of the content that can reach
    // the window's box [min_edge, max_edge], or the single copy at 0.0 for
    // a non-repeating axis. A copy displaced by k * period spans
    // [content_min + k * period, content_max + k * period], so the copies
    // that reach the box run one content span further out than the tiles
    // the box alone overlaps. With a pure-translation camera the box is
    // the window itself, at most one period wide, so this is only a few
    // copies; a rotated camera widens the box, and the extra copies cover
    // it.
    let offsets =
        |component: f32, min_edge: f32, max_edge: f32, window: f32, axis: u8| -> Vec<f32> {
            if component == 0.0 {
                return vec![0.0];
            }
            let period = component.abs();
            if period < window {
                let (name, what) = match axis {
                    0 => ("repeat_x", "the window width"),
                    _ => ("repeat_y", "the window height"),
                };
                panic!(
                    "frost: the layer's {name} {component} is smaller than {what} \
                 ({window}px); the minimum repeat offset is the window size"
                );
            }
            let (content_min, content_max) = match (content, axis) {
                (Some((cmin, cmax)), 0) => (cmin[0], cmax[0]),
                (Some((cmin, cmax)), _) => (cmin[1], cmax[1]),
                (None, _) => (0.0, 0.0),
            };
            // The copies [k * period] whose displaced content overlaps the box:
            // content_min + k * period < max_edge and content_max + k * period
            // > min_edge, a copy just touching the box (zero visible width) is
            // not drawn, matching the window-edge convention elsewhere.
            let k0 = ((min_edge - content_max) / period).floor() as i64 + 1;
            let k1 = ((max_edge - content_min) / period).ceil() as i64 - 1;
            (k0..=k1).map(|k| k as f32 * period).collect()
        };
    let xs = offsets(layer.repeat[0], min[0], max[0], w, 0);
    let ys = offsets(layer.repeat[1], min[1], max[1], h, 1);
    let mut tile_offsets = xs
        .iter()
        .flat_map(|x| ys.iter().map(move |y| (*x, *y)))
        .collect::<Vec<_>>();
    if let Some(index) = tile_offsets.iter().position(|&(x, y)| x == 0.0 && y == 0.0) {
        tile_offsets.swap(0, index);
    }
    tile_offsets
}

/// Aborts the program with an error when an object of a repeating layer has
/// an extent in a repeating axis that exceeds that axis's period: beyond
/// that, the object wraps more than once and the same object would be drawn
/// twice.
///
/// `parent` is the node's ancestors' transform in the layer's coordinate
/// space (the same accumulation as in [`draw_node`], without the camera
/// view), so each object's extent is measured in layer-space pixels — where
/// the repeat period lives. The extent is the object's axis-aligned box,
/// which is exact for the axis-parallel tile displacements.
///
/// Returns the union of all the layer's objects' boxes — the content's
/// extent in the layer's coordinate space — or `None` when the layer has
/// no drawable objects. [`layer_repeat_offsets`] uses it to reach the
/// copies of the content that can overlap the window's box: the content
/// need not sit inside the tile `[0, period)`, so a copy displaced by
/// `k * period` can reach the box even when that tile does not.
fn check_layer_repeat_extent(
    node: &SceneNode,
    parent: &Transform,
    repeat: [f32; 2],
) -> Option<([f32; 2], [f32; 2])> {
    let local = Transform::scale(node.scale).compose(&node.transform);
    let world = local.compose(parent);
    let mut extent = None;
    if let Some((kind, center, half)) = node.shape.as_ref().and_then(shape_local_box) {
        let (box_min, box_max) = aabb_of_box(&world, center, half[0], half[1]);
        let width = box_max[0] - box_min[0];
        let height = box_max[1] - box_min[1];
        if repeat[0] != 0.0 && width > repeat[0].abs() {
            panic!(
                "frost: a {kind} in the repeating layer is wider ({width:.1}px) \
                 than its repeat_x offset ({}px); the same object would be \
                 drawn twice",
                repeat[0].abs()
            );
        }
        if repeat[1] != 0.0 && height > repeat[1].abs() {
            panic!(
                "frost: a {kind} in the repeating layer is taller ({height:.1}px) \
                 than its repeat_y offset ({}px); the same object would be \
                 drawn twice",
                repeat[1].abs()
            );
        }
        extent = Some((box_min, box_max));
    }
    for child in &node.children {
        if let Some((child_min, child_max)) = check_layer_repeat_extent(child, &world, repeat) {
            extent = Some(match extent {
                Some((min, max)) => (
                    [min[0].min(child_min[0]), min[1].min(child_min[1])],
                    [max[0].max(child_max[0]), max[1].max(child_max[1])],
                ),
                None => (child_min, child_max),
            });
        }
    }
    extent
}

/// The shape's box in its node's local space — its kind, its center, and
/// its half extents — or `None` for shapes that never draw
/// (`Shape::Background`) or cannot be measured (a broken font, or a live
/// particle batch, whose extent is dynamic).
///
/// A text block is its laid-out width by its `ascent + descent`, centered
/// on the node's origin like its glyphs (see `expand_text_list`).
fn shape_local_box(shape: &Shape) -> Option<(&'static str, [f32; 2], [f32; 2])> {
    match shape {
        Shape::Circle { center, radius, .. } => Some(("circle", *center, [*radius, *radius])),
        Shape::Rectangle { center, extent, .. } => {
            Some(("rectangle", *center, [extent[0], extent[1]]))
        }
        // A sprite's local space is centered on the origin, one texture
        // pixel per scene pixel.
        Shape::Sprite { width, height, .. } => Some((
            "sprite",
            [0.0, 0.0],
            [*width as f32 / 2.0, *height as f32 / 2.0],
        )),
        Shape::Text {
            font,
            text,
            size,
            weight,
            ..
        } => {
            let layout = text::layout(font, text, *size, *weight)?;
            Some((
                "text",
                [0.0, 0.0],
                [layout.width / 2.0, (layout.ascent + layout.descent) / 2.0],
            ))
        }
        // A polyline's local box is the box over its points, grown by half
        // the stroke's width. Fewer than two points draw nothing, so there
        // is nothing to measure.
        Shape::Polyline { points, width, .. } => {
            let &first = points.first()?;
            let mut min = first;
            let mut max = first;
            for &p in &points[1..] {
                min = [min[0].min(p[0]), min[1].min(p[1])];
                max = [max[0].max(p[0]), max[1].max(p[1])];
            }
            let half = (*width).max(0.0) * 0.5;
            Some((
                "polyline",
                [(min[0] + max[0]) / 2.0, (min[1] + max[1]) / 2.0],
                [
                    (max[0] - min[0]) / 2.0 + half,
                    (max[1] - min[1]) / 2.0 + half,
                ],
            ))
        }
        // A background never draws; nothing to measure.
        Shape::Background { .. } => None,
        // A light never draws; nothing to measure.
        Shape::Light { .. } => None,
        // A particle batch has no fixed local box: its particles move,
        // spawn, and die every frame, so its extent is dynamic and cannot
        // be measured. A batch in a repeating layer therefore skips the
        // period-overflow check.
        Shape::Particles { .. } => None,
        // A tile map's box is the box over its tiles: the extent is
        // data rather than a parameter, but a bounded one.
        Shape::TileMap { tiles, .. } => {
            // The box describes what the renderer paints, so a turned
            // cell contributes its swapped extent, not its authored one.
            let half = |t: &Tile| {
                let [sx, sy] = if t.rot % 2 == 1 {
                    [t.size[1], t.size[0]]
                } else {
                    t.size
                };
                (sx / 2.0, sy / 2.0)
            };
            let first = tiles.first()?;
            let (fw, fh) = half(first);
            let mut min = [first.center[0] - fw, first.center[1] - fh];
            let mut max = [first.center[0] + fw, first.center[1] + fh];
            for tile in tiles.iter().skip(1) {
                let (w, h) = half(tile);
                min = [
                    min[0].min(tile.center[0] - w),
                    min[1].min(tile.center[1] - h),
                ];
                max = [
                    max[0].max(tile.center[0] + w),
                    max[1].max(tile.center[1] + h),
                ];
            }
            Some((
                "tilemap",
                [(min[0] + max[0]) / 2.0, (min[1] + max[1]) / 2.0],
                [(max[0] - min[0]) / 2.0, (max[1] - min[1]) / 2.0],
            ))
        }
    }
}

/// Expands every [`Draw::Text`] in the list into one [`Draw::Sprite`] per
/// shaped glyph, spliced in at the text draw's position so the stable
/// z-sort keeps the glyph quads where the text node was.
///
/// The glyphs are laid out with `text::layout` (pen positions, y up, from
/// the text's left baseline origin) and rasterized with `text::rasterize`
/// into the per-`(font, raster size, weight)` atlas in `atlases`, which the caller keeps
/// between frames so unchanged text never re-rasterizes and its texture
/// buffer keeps a stable identity. Each glyph's quad is centered on its ink
/// box; the whole text block is centered on the text node's origin.
///
/// `scale` is the display's scale factor: the text is *shaped and
/// rasterized* at the physical size `size × scale` — so the atlas holds a
/// glyph baked at the resolution the frame actually renders at, which is
/// what keeps text sharp on a high-density display — and every metric the
/// shaping yields comes back in raster pixels, so the layout divides it by
/// `scale` to return to the text's own user-space units. The atlas key
/// carries the raster size, never the requested one: the same font at two
/// display densities rasterizes two atlases, each at its own resolution,
/// and neither ever stretches the other's glyphs.
/// The atlas side for a text raster size. Font vertical metrics lie —
/// display fonts mask far taller than their claimed span — so the
/// budget comes from the em itself: four em to a side covers several
/// shelves of masks up to about 1.2 em tall, which is a dozen or more
/// glyph cells at any size. In whole 64-pixel steps, between the
/// default side and a ceiling for screen-filling text.
fn atlas_side(raster: f32) -> u32 {
    ((4.0 * raster).ceil() as u32)
        .next_multiple_of(64)
        .clamp(512, 2048)
}

fn expand_text_list(
    user_to_pixel: Transform,
    draws: Vec<Draw>,
    atlases: &mut HashMap<(u64, u32, u32), text::Atlas>,
    scale: f32,
) -> Vec<Draw> {
    // The reciprocal of the display's scale factor: metrics shaped at the
    // raster size are divided back out below, so the expansion is the
    // identity at 100% scaling and sharper above it.
    let inv_scale = 1.0 / scale;
    let mut expanded = Vec::with_capacity(draws.len());
    for draw in draws {
        match draw {
            Draw::Text {
                world,
                font,
                text: string,
                size,
                weight,
                color,
                alpha,
                emission,
                glow,
                lit,
                lit_layers,
                z,
                diagnostic,
            } => {
                // The physical raster size: what `text::layout` and
                // `text::rasterize` bake, so the atlas's texels are the
                // panel pixels a glyph will cover.
                let raster = size * scale;
                // A broken font leaves the text undrawn; `Shape::text`
                // validates the font up front, so this only guards a buffer
                // that turned out unreadable.
                let Some(layout) = text::layout(&font, &string, raster, weight) else {
                    continue;
                };
                let key = (
                    Arc::as_ptr(&font) as *const () as u64,
                    raster.to_bits(),
                    weight.to_bits(),
                );
                // Pack the missing glyphs into the map's own atlas, not a
                // clone: a clone's new cells would be dropped, because the
                // key is already registered by the time the first new glyph
                // appears (an earlier draw or frame), and the glyphs would
                // never draw. A miss builds an atlas scaled to this text's
                // raster — big text (a screen-filling splash) would have
                // every glyph skipped at the default side — and the entry
                // stays in the map even when nothing is packed (e.g.
                // all-space text) so later frames hit it.
                let atlas = atlases
                    .entry(key)
                    .or_insert_with(|| text::Atlas::sized(atlas_side(raster)));
                // Rasterize the glyphs the atlas does not have yet.
                let mut missing = Vec::new();
                for glyph in &layout.glyphs {
                    if atlas.has(glyph.id) {
                        continue;
                    }
                    if let Some(raster) = text::rasterize(&font, glyph.id, raster, weight) {
                        missing.push((glyph.id, raster));
                    }
                }
                if !missing.is_empty() {
                    atlas.insert_many(&missing);
                }
                // The band is booked against the composed (node-and-canvas)
                // scale, so it stays a constant number of panel pixels.
                let [sx, sy] = world.compose(&user_to_pixel).scales();
                let aa = AA_BAND / sx.max(sy).max(1e-9);
                // Center the text block (width by ascent + descent, baseline
                // at the pen-space origin, y up) on the node's origin. The
                // metrics come back in raster pixels; every one below is
                // divided by the scale factor back into the text's units.
                let origin = [
                    -layout.width * inv_scale / 2.0,
                    (layout.ascent - layout.descent) * inv_scale / 2.0,
                ];
                let (atlas_w, atlas_h) = (atlas.width as f32, atlas.height as f32);
                for glyph in &layout.glyphs {
                    let Some(cell) = atlas.cell(glyph.id) else {
                        continue;
                    };
                    let (gw, gh) = (cell.width as f32, cell.height as f32);
                    // The glyph's pen position in the node's local space,
                    // plus the ink box's center offset from the pen: `left`
                    // right and `top - height / 2` above the baseline — all
                    // raster pixels, divided back out.
                    let ink = [
                        origin[0] + (glyph.x + cell.left as f32 + gw / 2.0) * inv_scale,
                        origin[1] + (glyph.y + cell.top as f32 - gh / 2.0) * inv_scale,
                    ];
                    let glyph_world = Transform::translate(ink)
                        .compose(&world)
                        .compose(&user_to_pixel);
                    // Inset the cell by half a texel so the quad's edges
                    // sample the edge texels exactly instead of blending
                    // with the neighboring cell.
                    let uv_rect = [
                        (cell.x as f32 + 0.5) / atlas_w,
                        (cell.y as f32 + 0.5) / atlas_h,
                        (cell.x as f32 + cell.width as f32 - 0.5) / atlas_w,
                        (cell.y as f32 + cell.height as f32 - 0.5) / atlas_h,
                    ];
                    expanded.push(Draw::Sprite {
                        world: glyph_world,
                        data: atlas.data.clone(),
                        generation: atlas.generation(),
                        // Glyph quads are the engine's own arithmetic: the
                        // text pipeline's work, never a game program's.
                        program: None,
                        // The cell is a raster-pixel box; the quad rides the
                        // transform that multiplies by the scale factor, so
                        // its local size divides the factor back out.
                        size: [gw * inv_scale, gh * inv_scale],
                        texture_size: [atlas.width, atlas.height],
                        // Glyphs are sampled at their native size (or
                        // scaled down for subpixel layout), so bilinear is
                        // the right filter.
                        filter: SpriteFilter::Linear,
                        aa,
                        tint: color,
                        alpha,
                        uv_rect,
                        // The glyphs are the text node's own pixels: its
                        // `lit` flag, `emission`, `glow`, and diagnostic marker
                        // pass on to every quad — its light layers with them.
                        emission,
                        glow,
                        // Glyphs are coverage itself: an additive glyph would
                        // brighten the page it is written on.
                        additive: false,
                        lit,
                        lit_layers,
                        z,
                        diagnostic,
                    });
                }
            }
            other => expanded.push(other),
        }
    }
    expanded
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::*;

    /// The user-to-pixel transform of a `(100, 100)` canvas:
    /// `(x, y) -> (x + 50, 50 - y)`.
    fn pixel_map() -> Transform {
        Transform::scale([1.0, -1.0]).compose(&Transform::translate([50.0, 50.0]))
    }

    /// A particle at `pos` with a full lifetime of `life` seconds (its
    /// `max_life` is its `life`) and the given `size`; unrotated and white.
    fn particle(pos: [f32; 2], life: f32, size: f32) -> Particle {
        Particle {
            pos,
            vel: [0.0, 0.0],
            life,
            max_life: life,
            size,
            angle: 0.0,
            color: WHITE,
        }
    }

    /// The fields of a particle batch draw.
    struct Batch<'a> {
        data: &'a [u8],
        count: u32,
        color: Color,
        z: f32,
    }

    /// The node's particle draw: panics if the node drew anything but
    /// exactly one particle batch.
    fn the_batch<'a>(draws: &'a [Draw]) -> Batch<'a> {
        match draws {
            [
                Draw::Particles {
                    data,
                    count,
                    color,
                    z,
                    ..
                },
            ] => Batch {
                data,
                count: *count,
                color: *color,
                z: *z,
            },
            other => panic!("expected one particle draw, got {other:?}"),
        }
    }

    /// Reads the little-endian float at component `component` (0..4 in the
    /// first instance `vec4`: position, size, life fraction; 4..7 in the
    /// second: pixel-space angle and tint) of particle `particle` in the
    /// packed instance data of the batch.
    fn instance_f32(batch: &Batch, particle: usize, component: usize) -> f32 {
        let start = particle * 32 + component * 4;
        f32::from_le_bytes(batch.data[start..start + 4].try_into().unwrap())
    }

    /// A node with one particle at `pos`, drawn alone at the canvas origin.
    fn draw_one(node: &SceneNode, draws: &mut Vec<Draw>) {
        draw_node(pixel_map(), node, &Transform::identity(), 0.0, WHITE, draws);
    }

    #[test]
    fn node_particles_follow_the_node_world_transform() {
        // A node translated to (10, 20) with a particle at local (5, 5):
        // the particle lands at user (15, 25), pixel (65, 25).
        let node = SceneNode {
            transform: Transform::translate([10.0, 20.0]),
            shape: Some(Shape::Particles {
                system: ParticleSystem {
                    particles: vec![particle([5.0, 5.0], 4.0, 2.0)],
                },
                color: WHITE,
                shape: ParticleShape::Circle,
            }),
            ..Default::default()
        };
        let mut draws = Vec::new();
        draw_one(&node, &mut draws);
        let batch = the_batch(&draws);
        assert_eq!(batch.count, 1);
        assert_eq!(instance_f32(&batch, 0, 0), 65.0);
        assert_eq!(instance_f32(&batch, 0, 1), 25.0);
        // The size is unscaled, the life fraction is one.
        assert_eq!(instance_f32(&batch, 0, 2), 2.0);
        assert_eq!(instance_f32(&batch, 0, 3), 1.0);
    }

    #[test]
    fn node_particles_rotate_with_the_node() {
        // A node rotated a quarter turn (counter-clockwise, y up) sends
        // local (10, 0) to (0, 10), pixel (50, 40); a rotation keeps the
        // scale factors at one, so the radius is unscaled.
        let node = SceneNode {
            transform: Transform::rotate(std::f32::consts::FRAC_PI_2),
            shape: Some(Shape::Particles {
                system: ParticleSystem {
                    particles: vec![particle([10.0, 0.0], 4.0, 3.0)],
                },
                color: WHITE,
                shape: ParticleShape::Circle,
            }),
            ..Default::default()
        };
        let mut draws = Vec::new();
        draw_one(&node, &mut draws);
        let batch = the_batch(&draws);
        assert_eq!(instance_f32(&batch, 0, 0), 50.0);
        assert_eq!(instance_f32(&batch, 0, 1), 40.0);
        assert_eq!(instance_f32(&batch, 0, 2), 3.0);
    }

    #[test]
    fn node_particles_scale_the_radius() {
        // A uniform world scale of 2 doubles the radius.
        let node = SceneNode {
            scale: [2.0, 2.0],
            shape: Some(Shape::Particles {
                system: ParticleSystem {
                    particles: vec![particle([0.0, 0.0], 4.0, 3.0)],
                },
                color: WHITE,
                shape: ParticleShape::Circle,
            }),
            ..Default::default()
        };
        let mut draws = Vec::new();
        draw_one(&node, &mut draws);
        let batch = the_batch(&draws);
        assert_eq!(instance_f32(&batch, 0, 2), 6.0);
    }

    #[test]
    fn node_particles_nonuniform_scale_preserves_area() {
        // A world scale of (2, 8) has a geometric mean of 4, so the radius
        // is multiplied by 4 — the particle's area is preserved.
        let node = SceneNode {
            scale: [2.0, 8.0],
            shape: Some(Shape::Particles {
                system: ParticleSystem {
                    particles: vec![particle([0.0, 0.0], 4.0, 3.0)],
                },
                color: WHITE,
                shape: ParticleShape::Circle,
            }),
            ..Default::default()
        };
        let mut draws = Vec::new();
        draw_one(&node, &mut draws);
        let batch = the_batch(&draws);
        assert_eq!(instance_f32(&batch, 0, 2), 12.0);
    }

    #[test]
    fn node_particles_take_the_shape_color_times_the_composed_modulate() {
        // A parent with modulate (0.5, 0.5, 0.5, 1) and order 2, a child
        // with a particle shape of base color (1, 1, 0.5, 1), modulate
        // (1, 0.5, 0.25, 1), and order 3: the composed modulate is
        // (0.5, 0.25, 0.125, 1), so the batch is tinted
        // (0.5, 0.25, 0.0625, 1) and draws at order 5.
        let mut parent = SceneNode {
            modulate: Color {
                r: 0.5,
                g: 0.5,
                b: 0.5,
                a: 1.0,
            },
            order: 2.0,
            ..Default::default()
        };
        let child = SceneNode {
            modulate: Color {
                r: 1.0,
                g: 0.5,
                b: 0.25,
                a: 1.0,
            },
            order: 3.0,
            shape: Some(Shape::Particles {
                system: ParticleSystem {
                    particles: vec![particle([0.0, 0.0], 4.0, 1.0)],
                },
                color: Color {
                    r: 1.0,
                    g: 1.0,
                    b: 0.5,
                    a: 1.0,
                },
                shape: ParticleShape::Circle,
            }),
            ..Default::default()
        };
        parent.children.push(Box::new(child));
        let mut draws = Vec::new();
        draw_one(&parent, &mut draws);
        let batch = the_batch(&draws);
        assert_eq!(
            batch.color,
            Color {
                r: 0.5,
                g: 0.25,
                b: 0.0625,
                a: 1.0
            }
        );
        assert_eq!(batch.z, 5.0);
    }

    #[test]
    fn node_particles_empty_system_draws_nothing() {
        // A particle shape with no live particles produces no draw.
        let node = SceneNode {
            shape: Some(Shape::Particles {
                system: ParticleSystem::new(),
                color: WHITE,
                shape: ParticleShape::Circle,
            }),
            ..Default::default()
        };
        let mut draws = Vec::new();
        draw_one(&node, &mut draws);
        assert!(draws.is_empty());
    }

    #[test]
    fn node_particles_paint_between_shape_and_children() {
        // A grandparent with a circle, its child with a particle shape, and
        // that child's child with a rectangle: the draws land in tree order
        // — grandparent's shape, then the particle batch, then the child's
        // shape — all at the same z, so the stable z-sort keeps exactly this
        // order.
        let grandparent_child = SceneNode {
            shape: Some(Shape::Rectangle {
                center: [0.0, 0.0],
                extent: [2.0, 2.0],
                color: Color {
                    r: 0.0,
                    g: 0.0,
                    b: 1.0,
                    a: 1.0,
                },
            }),
            ..Default::default()
        };
        let mut parent = SceneNode {
            shape: Some(Shape::Particles {
                system: ParticleSystem {
                    particles: vec![particle([0.0, 0.0], 4.0, 1.0)],
                },
                color: WHITE,
                shape: ParticleShape::Circle,
            }),
            ..Default::default()
        };
        parent.children.push(Box::new(grandparent_child));
        let mut grandparent = SceneNode {
            shape: Some(Shape::Circle {
                center: [0.0, 0.0],
                radius: 4.0,
                color: Color {
                    r: 1.0,
                    g: 0.0,
                    b: 0.0,
                    a: 1.0,
                },
            }),
            ..Default::default()
        };
        grandparent.children.push(Box::new(parent));
        let mut draws = Vec::new();
        draw_one(&grandparent, &mut draws);
        assert_eq!(draws.len(), 3);
        assert!(matches!(&draws[0], Draw::Shape { .. }));
        assert!(matches!(&draws[1], Draw::Particles { .. }));
        assert!(matches!(&draws[2], Draw::Shape { .. }));
        // The whole subtree drew at the composed order 0.
        for draw in &draws {
            assert_eq!(draw.z(), 0.0);
        }
    }

    /// A node at the canvas origin whose shape is a particle batch with the
    /// given per-particle `angle` and `color`.
    fn particle_node(angle: f32, color: Color, shape: ParticleShape) -> SceneNode {
        SceneNode {
            shape: Some(Shape::Particles {
                system: ParticleSystem {
                    particles: vec![Particle {
                        pos: [0.0, 0.0],
                        vel: [0.0, 0.0],
                        life: 4.0,
                        max_life: 4.0,
                        size: 1.0,
                        angle,
                        color,
                    }],
                },
                color: WHITE,
                shape,
            }),
            ..Default::default()
        }
    }

    #[test]
    fn node_particles_pack_their_angle_and_color_in_pixel_space() {
        // The pixel space is y-down, so a particle's local (y-up,
        // counterclockwise-positive) angle packs negated: a quarter turn
        // counterclockwise on screen is -FRAC_PI_4 in pixel space, and a
        // clockwise quarter turn is +FRAC_PI_4. The particle's own color
        // packs as the tint in the second instance vec4.
        let color = Color {
            r: 0.5,
            g: 1.0,
            b: 0.25,
            a: 1.0,
        };
        for phi in [std::f32::consts::FRAC_PI_4, -std::f32::consts::FRAC_PI_4] {
            let node = particle_node(phi, color, ParticleShape::Circle);
            let mut draws = Vec::new();
            draw_one(&node, &mut draws);
            let batch = the_batch(&draws);
            assert_eq!(instance_f32(&batch, 0, 4), -phi);
            assert_eq!(instance_f32(&batch, 0, 5), color.r);
            assert_eq!(instance_f32(&batch, 0, 6), color.g);
            assert_eq!(instance_f32(&batch, 0, 7), color.b);
        }
    }

    #[test]
    fn node_particles_add_the_node_rotation_to_their_own_angle() {
        // A node rotated a quarter turn counterclockwise maps its local +x
        // axis to pixel angle -FRAC_PI_2 (y down), so a particle with its
        // own angle phi packs at -FRAC_PI_2 - phi: the node's rotation and
        // the particle's rotation compose in the local frame before the
        // y-down flip is applied.
        let phi = std::f32::consts::FRAC_PI_4;
        let mut node = particle_node(phi, WHITE, ParticleShape::Circle);
        node.transform = Transform::rotate(std::f32::consts::FRAC_PI_2);
        let mut draws = Vec::new();
        draw_one(&node, &mut draws);
        let batch = the_batch(&draws);
        let packed = instance_f32(&batch, 0, 4);
        let expected = -(std::f32::consts::FRAC_PI_2 + phi);
        assert!((packed - expected).abs() < 1e-4);
    }

    #[test]
    fn node_particles_pass_the_shape_kind_aspect_and_sprite_to_the_draw() {
        // The batch's uniform carries the shape's kind and aspect, and a
        // sprite shape hands its image and size to the draw so the pipeline
        // can bind (or cache) the texture.
        let node = particle_node(0.0, WHITE, ParticleShape::Rectangle { aspect: 2.0 });
        let mut draws = Vec::new();
        draw_one(&node, &mut draws);
        let [
            Draw::Particles {
                kind,
                aspect,
                sprite_data,
                sprite_size,
                ..
            },
        ] = &draws[..]
        else {
            panic!("expected one particle draw, got {draws:?}")
        };
        assert_eq!(*kind, 1.0);
        assert_eq!(*aspect, 2.0);
        assert!(sprite_data.is_none());
        assert_eq!(*sprite_size, [0, 0]);

        let sprite = ParticleShape::Sprite {
            data: std::sync::Arc::from([255u8, 255, 255, 255]),
            width: 2,
            height: 3,
            generation: 0,
        };
        let node = particle_node(0.0, WHITE, sprite);
        let mut draws = Vec::new();
        draw_one(&node, &mut draws);
        let [
            Draw::Particles {
                kind,
                aspect,
                sprite_data,
                sprite_size,
                ..
            },
        ] = &draws[..]
        else {
            panic!("expected one particle draw, got {draws:?}")
        };
        assert_eq!(*kind, 2.0);
        assert_eq!(*aspect, 1.5);
        assert_eq!(sprite_data.as_deref(), Some(&[255u8, 255, 255, 255][..]));
        assert_eq!(*sprite_size, [2, 3]);
    }

    #[test]
    fn canvas_light_lands_at_the_pixel_position_with_its_fields() {
        // A light registered at user (10, 20) on a 100x100 canvas lands at
        // pixel (60, 30); a negative radius clamps to zero, the intensity
        // and color pass through, and the draw order is irrelevant, so it
        // stays 0.0. The light is omni, so its cone is wide open: the
        // cosine of the half angle is -1.0 and the axis is the fallback.
        let mut canvas = Canvas::new((100, 100), 1.0);
        canvas.light(
            10.0,
            20.0,
            Color {
                r: 1.0,
                g: 0.5,
                b: 0.0,
                a: 1.0,
            },
            2.0,
            -16.0,
        );
        let [
            Draw::Light {
                pos,
                radius,
                intensity,
                color,
                penumbra,
                dir,
                cos_half,
                feather,
                layers,
                z,
            },
        ] = &canvas.draws[..]
        else {
            panic!("expected one light draw, got {:?}", canvas.draws);
        };
        assert_eq!(*pos, [60.0, 30.0]);
        assert_eq!(*radius, 0.0);
        assert_eq!(*intensity, 2.0);
        assert_eq!(
            *color,
            Color {
                r: 1.0,
                g: 0.5,
                b: 0.0,
                a: 1.0
            }
        );
        assert_eq!(*penumbra, 0.0);
        assert_eq!(*dir, [1.0, 0.0]);
        assert_eq!(*cos_half, -1.0);
        // The immediate API carries no tap knob: the slot stays zero,
        // which the shader reads as the grid's full nine.
        assert_eq!(*feather, 0.0);
        assert_eq!(*z, 0.0);
        // The immediate API carries no layers knob either: the draw
        // shines on layer 0, like every light that names none.
        assert_eq!(*layers, LightLayers::default());
    }

    #[test]
    fn immediate_and_scene_lights_pack_into_the_field_in_call_order() {
        // A light registered immediately and one on a scene node, both at z
        // 0: the immediate draw is pushed first (the process runs before
        // the scene is drawn), so the stable z-sort keeps it first in the
        // paint order, and the packed field's records follow.
        let mut canvas = Canvas::new((100, 100), 1.0);
        canvas.light(
            0.0,
            0.0,
            Color {
                r: 1.0,
                g: 0.0,
                b: 0.0,
                a: 1.0,
            },
            1.0,
            10.0,
        );
        let scene = Scene::new(SceneNode {
            transform: Transform::translate([-10.0, 0.0]),
            shape: Some(Shape::Light {
                light: Light::point(
                    Color {
                        r: 0.0,
                        g: 0.0,
                        b: 1.0,
                        a: 1.0,
                    },
                    0.5,
                    20.0,
                ),
            }),
            ..Default::default()
        });
        canvas.draw_scene(&scene);
        let field = pack_light_field(&canvas.paint_order(), AMBIENT);
        let f32_at =
            |data: &[u8], off: usize| f32::from_le_bytes(data[off..off + 4].try_into().unwrap());
        assert_eq!(field.count, 2);
        assert_eq!(field.data.len(), LIGHT_FIELD_HEADER + 2 * LIGHT_RECORD);
        // The header carries the light count and the scene's ambient.
        let count = u32::from_le_bytes(field.data[0..4].try_into().unwrap());
        assert_eq!(count, 2);
        assert_eq!(f32_at(&field.data, 16), AMBIENT.r);
        // The immediate light first: user (0, 0) -> pixel (50, 50), radius
        // 10, intensity 1, red. Both lights are omni, so the cone vec4 is
        // the fallback axis, the always-passing -1.0, and zero padding;
        // the color vec4's w is the penumbra, zero for point-sized
        // immediate lights.
        let off = LIGHT_FIELD_HEADER;
        assert_eq!(f32_at(&field.data, off), 50.0);
        assert_eq!(f32_at(&field.data, off + 4), 50.0);
        assert_eq!(f32_at(&field.data, off + 8), 10.0);
        assert_eq!(f32_at(&field.data, off + 12), 1.0);
        assert_eq!(f32_at(&field.data, off + 16), 1.0);
        assert_eq!(f32_at(&field.data, off + 28), 0.0);
        assert_eq!(f32_at(&field.data, off + 32), 1.0);
        assert_eq!(f32_at(&field.data, off + 36), 0.0);
        assert_eq!(f32_at(&field.data, off + 40), -1.0);
        // The immediate API carries no tap knob: the omni slot stays
        // zero, which the shader reads as the grid's full nine.
        assert_eq!(f32_at(&field.data, off + 44), 0.0);
        // The scene's light second: user (-10, 0) -> pixel (40, 50), radius
        // 20, intensity 0.5, blue.
        let off = LIGHT_FIELD_HEADER + LIGHT_RECORD;
        assert_eq!(f32_at(&field.data, off), 40.0);
        assert_eq!(f32_at(&field.data, off + 4), 50.0);
        assert_eq!(f32_at(&field.data, off + 8), 20.0);
        assert_eq!(f32_at(&field.data, off + 12), 0.5);
        assert_eq!(f32_at(&field.data, off + 24), 1.0);
        assert_eq!(f32_at(&field.data, off + 32), 1.0);
        assert_eq!(f32_at(&field.data, off + 36), 0.0);
        assert_eq!(f32_at(&field.data, off + 40), -1.0);
        // The omni feather slot carries the penumbra's tap count —
        // the constructor's full grid unless a light asked otherwise.
        assert_eq!(f32_at(&field.data, off + 44), 9.0);
    }

    #[test]
    fn light_node_lands_at_the_pixel_position_with_its_fields_intact() {
        // A light node translated to (10, 20): the light sits at the node's
        // local origin, which lands at user (10, 20), pixel (60, 30) on a
        // 100x100 canvas; the radius, intensity, and color pass through
        // untinted (the modulate is white), the point light's cone is wide
        // open, and the node's order becomes the draw's z.
        let node = SceneNode {
            transform: Transform::translate([10.0, 20.0]),
            order: 3.0,
            shape: Some(Shape::Light {
                light: Light::point(
                    Color {
                        r: 1.0,
                        g: 1.0,
                        b: 1.0,
                        a: 1.0,
                    },
                    2.0,
                    16.0,
                ),
            }),
            ..Default::default()
        };
        let mut draws = Vec::new();
        draw_one(&node, &mut draws);
        let [
            Draw::Light {
                pos,
                radius,
                intensity,
                color,
                penumbra,
                dir,
                cos_half,
                feather,
                layers,
                z,
            },
        ] = &draws[..]
        else {
            panic!("expected one light draw, got {draws:?}")
        };
        assert_eq!(*pos, [60.0, 30.0]);
        assert_eq!(*radius, 16.0);
        assert_eq!(*intensity, 2.0);
        assert_eq!(
            *color,
            Color {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0
            }
        );
        assert_eq!(*penumbra, 0.0);
        assert_eq!(*dir, [1.0, 0.0]);
        assert_eq!(*cos_half, -1.0);
        // Omni node light: the feather slot is the tap grid's nine.
        assert_eq!(*feather, 9.0);
        // A `Light::point` names no layers: the draw shines on layer 0.
        assert_eq!(*layers, LightLayers::default());
        assert_eq!(*z, 3.0);
    }

    #[test]
    fn light_color_is_tinted_by_the_composed_modulate() {
        // A red light under a half-strength green parent modulate: the
        // light's color is multiplied channel by channel, and a negative
        // radius is clamped to zero.
        let node = SceneNode {
            transform: Transform::identity(),
            shape: Some(Shape::Light {
                light: Light::point(
                    Color {
                        r: 1.0,
                        g: 1.0,
                        b: 0.0,
                        a: 1.0,
                    },
                    1.0,
                    -4.0,
                ),
            }),
            ..Default::default()
        };
        let mut draws = Vec::new();
        draw_node(
            pixel_map(),
            &node,
            &Transform::identity(),
            5.0,
            Color {
                r: 1.0,
                g: 0.5,
                b: 0.0,
                a: 1.0,
            },
            &mut draws,
        );
        let [
            Draw::Light {
                radius, color, z, ..
            },
        ] = &draws[..]
        else {
            panic!("expected one light draw, got {draws:?}")
        };
        assert_eq!(*radius, 0.0);
        assert_eq!(
            *color,
            Color {
                r: 1.0,
                g: 0.5,
                b: 0.0,
                a: 1.0
            }
        );
        assert_eq!(*z, 5.0);
    }

    #[test]
    fn a_cone_light_records_its_axis_and_half_angle() {
        // An immediate cone aimed at user 45 degrees with a 60-degree
        // opening: the axis is folded through the canvas transform, so the
        // y-up user vector (cos 45, sin 45) becomes the y-down pixel vector
        // (0.707, -0.707), and the gate carries the cosine of the half
        // opening, cos(30 degrees).
        let mut canvas = Canvas::new((100, 100), 1.0);
        canvas.light_cone(
            0.0,
            0.0,
            Color {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
            1.0,
            10.0,
            std::f32::consts::FRAC_PI_4,
            std::f32::consts::FRAC_PI_3,
        );
        let [Draw::Light { dir, cos_half, .. }] = &canvas.draws[..] else {
            panic!("expected one light draw, got {:?}", canvas.draws);
        };
        let k = std::f32::consts::FRAC_1_SQRT_2;
        assert!((dir[0] - k).abs() < 1e-6);
        assert!((dir[1] + k).abs() < 1e-6);
        assert!((cos_half - (std::f32::consts::FRAC_PI_6).cos()).abs() < 1e-6);
    }

    #[test]
    fn an_omni_light_carries_its_taps_where_a_cone_keeps_its_feather() {
        // The omni feather slot is free by construction, so it carries
        // the penumbra's tap count; a cone's slot belongs to its edge
        // band and ignores taps entirely.
        let white = Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        };
        let omni = SceneNode {
            shape: Some(Shape::Light {
                light: Light::point(white, 1.0, 10.0)
                    .with_penumbra(4.0)
                    .with_shadow_taps(4),
            }),
            ..Default::default()
        };
        let mut draws = Vec::new();
        draw_one(&omni, &mut draws);
        let [
            Draw::Light {
                feather,
                penumbra,
                cos_half,
                ..
            },
        ] = &draws[..]
        else {
            panic!("expected one light draw");
        };
        assert_eq!(*feather, 4.0, "the omni slot carries the taps");
        assert!(*cos_half <= -1.0, "and the light is still omni");
        assert!((*penumbra - 4.0).abs() < 1e-6, "penumbra rides through");

        let half = std::f32::consts::FRAC_PI_4;
        let cone = SceneNode {
            shape: Some(Shape::Light {
                light: Light::cone(white, 1.0, 10.0, 0.0, 2.0 * half, half * 0.5)
                    .with_shadow_taps(4),
            }),
            ..Default::default()
        };
        let mut draws = Vec::new();
        draw_one(&cone, &mut draws);
        let [Draw::Light { feather, .. }] = &draws[..] else {
            panic!("expected one light draw");
        };
        assert!(*feather != 4.0, "a cone's feather is its own band");
    }

    #[test]
    fn a_directional_light_records_the_sentinel_and_its_travel() {
        // A `Light::directional` aimed at user PI — travelling toward
        // -x: the record keeps the negative radius sentinel exactly (the
        // zero-clamp that tames stray point radii must not demote a sun
        // to a zero-size point), the travel direction folds through the
        // canvas transform into the pixel-space vector (-1, 0), and the
        // gate opens fully like an omni light's, so nothing but the
        // shader's directional branch distinguishes it downstream.
        let node = SceneNode {
            shape: Some(Shape::Light {
                light: Light::directional(
                    Color {
                        r: 1.0,
                        g: 0.5,
                        b: 0.2,
                        a: 1.0,
                    },
                    1.25,
                    std::f32::consts::PI,
                ),
            }),
            ..Default::default()
        };
        let mut draws = Vec::new();
        draw_node(
            pixel_map(),
            &node,
            &Transform::identity(),
            0.0,
            Color {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
            &mut draws,
        );
        let [
            Draw::Light {
                radius,
                intensity,
                dir,
                cos_half,
                feather,
                ..
            },
        ] = &draws[..]
        else {
            panic!("expected one light draw, got {draws:?}")
        };
        assert_eq!(*radius, -1.0);
        assert_eq!(*intensity, 1.25);
        assert!((dir[0] + 1.0).abs() < 1e-6);
        assert!(dir[1].abs() < 1e-6);
        assert!((cos_half + 1.0).abs() < 1e-6);
        // Omni by construction: the slot is the tap grid's full nine.
        assert_eq!(*feather, 9.0);
    }

    #[test]
    fn a_parent_rotation_turns_a_cone_axis_like_any_other_shape() {
        // A cone node aimed at local 0 degrees under a parent rotated 90
        // degrees: the axis is carried by the composed transform, so the
        // world-space direction is the node transform applied to (1, 0),
        // rotated into the canvas's y-down pixel space. With a quarter turn
        // this is exactly what a point's displacement would do.
        let node = SceneNode {
            transform: Transform::rotate(std::f32::consts::FRAC_PI_2),
            shape: Some(Shape::Light {
                light: Light::cone(
                    Color {
                        r: 1.0,
                        g: 1.0,
                        b: 1.0,
                        a: 1.0,
                    },
                    1.0,
                    10.0,
                    0.0,
                    std::f32::consts::FRAC_PI_2,
                    0.0,
                ),
            }),
            ..Default::default()
        };
        let mut draws = Vec::new();
        draw_one(&node, &mut draws);
        let [Draw::Light { dir, cos_half, .. }] = &draws[..] else {
            panic!("expected one light draw, got {draws:?}")
        };
        assert!(dir[0].abs() < 1e-5);
        assert!((dir[1] + 1.0).abs() < 1e-5);
        assert!((cos_half - (std::f32::consts::FRAC_PI_4).cos()).abs() < 1e-6);
    }

    #[test]
    fn a_degenerate_cone_falls_back_to_a_usable_axis() {
        // A degenerate axis - the delta the fold produces is zero-length -
        // must not reach the shader as NaN, which would poison the angular
        // gate for every pixel. The axis falls back to +x; the wide-open
        // spread passes everywhere regardless.
        let mut canvas = Canvas::new((100, 100), 1.0);
        canvas.light_cone(
            0.0,
            0.0,
            Color {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
            1.0,
            10.0,
            f32::NAN,
            std::f32::consts::TAU,
        );
        let [Draw::Light { dir, cos_half, .. }] = &canvas.draws[..] else {
            panic!("expected one light draw, got {:?}", canvas.draws);
        };
        assert_eq!(*dir, [1.0, 0.0]);
        assert_eq!(*cos_half, -1.0);
    }

    #[test]
    fn feather_band_measures_the_soft_cone_edge() {
        // The feather is the cosine difference between the band's inner
        // and outer angles: full softness (the half angle) feathers the
        // whole cone, half softness feathers half the cosine range, and
        // zero — or a clamped negative — keeps the hard edge.
        let half = std::f32::consts::FRAC_PI_4; // spread = PI/2.
        let spread = 2.0 * half;
        assert_eq!(feather_band(spread, 0.0), 0.0);
        assert_eq!(feather_band(spread, -1.0), 0.0);
        let full = feather_band(spread, half);
        assert!((full - (1.0 - half.cos())).abs() < 1e-6);
        // Softness beyond the half angle clamps to the whole cone.
        assert_eq!(feather_band(spread, half * 10.0), full);
        // Half the cone feathered: cos(PI/8) - cos(PI/4), strictly
        // between zero and the full feather.
        let soft = feather_band(spread, half * 0.5);
        assert!((soft - (std::f32::consts::PI / 8.0).cos() + half.cos()).abs() < 1e-6);
        assert!(soft > 0.0 && soft < full);
        // Omni lights never feather, whatever the softness says.
        assert_eq!(feather_band(std::f32::consts::TAU, 1.0), 0.0);
        assert_eq!(feather_band(std::f32::consts::TAU * 2.0, 1.0), 0.0);
    }

    #[test]
    fn a_nodes_emission_reaches_its_shape_draw() {
        // The node's emission rides to the draw multiplied by the composed
        // modulate, exactly like the shape's own color.
        let node = SceneNode {
            lit: true,
            emission: Color {
                r: 0.5,
                g: 0.2,
                b: 0.0,
                a: 1.0,
            },
            shape: Some(Shape::Circle {
                center: [0.0, 0.0],
                radius: 4.0,
                color: Color {
                    r: 1.0,
                    g: 1.0,
                    b: 1.0,
                    a: 1.0,
                },
            }),
            ..Default::default()
        };
        let mut draws = Vec::new();
        draw_node(
            pixel_map(),
            &node,
            &Transform::identity(),
            0.0,
            Color {
                r: 0.5,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
            &mut draws,
        );
        let [Draw::Shape { emission, .. }] = &draws[..] else {
            panic!("expected one shape draw, got {draws:?}")
        };
        assert_eq!(
            *emission,
            Color {
                r: 0.25,
                g: 0.2,
                b: 0.0,
                a: 1.0,
            }
        );
    }

    #[test]
    fn a_nodes_glow_rides_the_modulate_and_can_stay_unset() {
        // The second color of the pair composes exactly as the emission does,
        // so a node fading out takes its halo with it — and left unset it stays
        // unset down to the draw, which is where the uniform writer decides the
        // halo is the emission after all.
        let white = Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        };
        let node_with = |g: Option<Color>| SceneNode {
            emission: Color {
                r: 1.0,
                g: 0.4,
                b: 0.0,
                a: 1.0,
            },
            glow: g,
            shape: Some(Shape::Circle {
                center: [0.0, 0.0],
                radius: 4.0,
                color: white,
            }),
            ..Default::default()
        };
        let half_cyan = Color {
            r: 0.5,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        };
        let mut draws = Vec::new();
        draw_node(
            pixel_map(),
            &node_with(Some(Color {
                r: 0.2,
                g: 0.4,
                b: 1.0,
                a: 1.0,
            })),
            &Transform::identity(),
            0.0,
            half_cyan,
            &mut draws,
        );
        let [Draw::Shape { glow, .. }] = &draws[..] else {
            panic!("expected one shape draw, got {draws:?}")
        };
        assert_eq!(
            *glow,
            Some(Color {
                r: 0.1,
                g: 0.4,
                b: 1.0,
                a: 1.0,
            }),
            "the glow should ride the composed modulate, as the emission does"
        );

        let mut unset = Vec::new();
        draw_node(
            pixel_map(),
            &node_with(None),
            &Transform::identity(),
            0.0,
            half_cyan,
            &mut unset,
        );
        let [Draw::Shape { glow, .. }] = &unset[..] else {
            panic!("expected one shape draw, got {unset:?}")
        };
        assert_eq!(*glow, None, "an unset glow must stay unset");
    }

    #[test]
    fn a_soft_cone_carries_its_feather_and_penumbra_into_the_draw() {
        // The scene path feeds the light's softness into the draw's
        // feather band and its penumbra through untouched (negatives
        // clamp to zero); a hard cone (and the immediate-mode API, which
        // has no knobs) stay at zero.
        let half = std::f32::consts::FRAC_PI_4;
        let node = SceneNode {
            shape: Some(Shape::Light {
                light: Light::cone(
                    Color {
                        r: 1.0,
                        g: 1.0,
                        b: 1.0,
                        a: 1.0,
                    },
                    1.0,
                    10.0,
                    0.0,
                    2.0 * half,
                    half * 0.5,
                )
                .with_penumbra(4.0),
            }),
            ..Default::default()
        };
        let mut draws = Vec::new();
        draw_one(&node, &mut draws);
        let [
            Draw::Light {
                feather, penumbra, ..
            },
        ] = &draws[..]
        else {
            panic!("expected one light draw, got {draws:?}")
        };
        assert_eq!(*feather, feather_band(2.0 * half, half * 0.5));
        assert!(*feather > 0.0);
        assert_eq!(*penumbra, 4.0);
        // The builder clamps a negative radius to the hard shadow.
        let light = Light::point(
            Color {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
            1.0,
            10.0,
        )
        .with_penumbra(-3.0);
        assert_eq!(light.penumbra, 0.0);
    }

    /// The user-to-pixel transform of a `(100, 100)` canvas at a display
    /// scale factor of 2: `(x, y) -> (2x + 100, 100 - 2y)` — the same
    /// center-origin map, with every user unit paying two panel pixels.
    fn pixel_map_2() -> Transform {
        Transform::scale([2.0, -2.0]).compose(&Transform::translate([100.0, 100.0]))
    }

    #[test]
    fn a_scale_2_canvas_maps_user_space_to_panel_pixels() {
        // The user space stays logical: the size is the window's logical
        // size, the center maps to the panel's center (which is at panel
        // (100, 100) of a 200x200 panel), and the user corners reach the
        // panel corners, doubled.
        let canvas = Canvas::new((100, 100), 2.0);
        assert_eq!(canvas.size(), (100.0, 100.0));
        assert_eq!(canvas.scale_factor(), 2.0);
        let to_pixel = canvas.user_to_pixel();
        assert_eq!(to_pixel.apply([0.0, 0.0]), [100.0, 100.0]);
        assert_eq!(to_pixel.apply([-50.0, -50.0]), [0.0, 200.0]);
        assert_eq!(to_pixel.apply([50.0, 50.0]), [200.0, 0.0]);
    }

    #[test]
    fn a_scale_2_canvas_doubles_the_scalar_sizes_it_records() {
        // A width-2 line, a radius-5 circle and a light of reach 10
        // recorded in logical units at density 2 reach the shaders in
        // panel pixels: width 4, radius 10, reach 20 — and the endpoints
        // land doubled like every other coordinate.
        let mut canvas = Canvas::new((100, 100), 2.0);
        canvas.line(-10.0, 0.0, 10.0, 0.0, WHITE, 2.0, 0.0);
        canvas.circle(0.0, 0.0, 5.0, WHITE, 0.0);
        canvas.light(0.0, 0.0, WHITE, 1.0, 10.0);
        let [
            Draw::Line { a, b, width, .. },
            Draw::Circle { center, radius, .. },
            Draw::Light {
                pos, radius: reach, ..
            },
        ] = &canvas.draws[..]
        else {
            panic!(
                "expected a line, a circle and a light, got {:?}",
                canvas.draws
            );
        };
        assert_eq!(*a, [80.0, 100.0]);
        assert_eq!(*b, [120.0, 100.0]);
        assert_eq!(*width, 4.0);
        assert_eq!(*center, [100.0, 100.0]);
        assert_eq!(*radius, 10.0);
        assert_eq!(*pos, [100.0, 100.0]);
        assert_eq!(*reach, 20.0);
    }

    #[test]
    fn a_scale_2_pixel_map_scales_polyline_particle_and_light_scalars() {
        // The scene path books its transform-free scalars — a polyline's
        // width, a particle's size, a light's reach and penumbra — in
        // panel pixels too: with the density folded into the user-to-pixel
        // transform, width 1.5 strokes 3, size 3 inflates to 6, and a
        // reach-10 penumbra-4 point light covers 20 with a penumbra of 8.
        let map = pixel_map_2();
        let mut draws = Vec::new();
        draw_node(
            map,
            &SceneNode {
                shape: Some(Shape::Polyline {
                    points: vec![[-10.0, 0.0], [10.0, 0.0]],
                    width: 1.5,
                    color: WHITE,
                }),
                ..Default::default()
            },
            &Transform::identity(),
            0.0,
            WHITE,
            &mut draws,
        );
        let [Draw::Polyline { points, width, .. }] = &draws[..] else {
            panic!("expected one polyline draw, got {draws:?}")
        };
        assert_eq!(*width, 3.0);
        assert_eq!(points[0], [80.0, 100.0]);
        assert_eq!(points[1], [120.0, 100.0]);

        let mut draws = Vec::new();
        draw_node(
            map,
            &SceneNode {
                shape: Some(Shape::Particles {
                    system: ParticleSystem {
                        particles: vec![particle([5.0, 5.0], 4.0, 3.0)],
                    },
                    color: WHITE,
                    shape: ParticleShape::Circle,
                }),
                ..Default::default()
            },
            &Transform::identity(),
            0.0,
            WHITE,
            &mut draws,
        );
        let batch = the_batch(&draws);
        assert_eq!(instance_f32(&batch, 0, 0), 110.0);
        assert_eq!(instance_f32(&batch, 0, 1), 90.0);
        assert_eq!(instance_f32(&batch, 0, 2), 6.0);

        let mut draws = Vec::new();
        draw_node(
            map,
            &SceneNode {
                shape: Some(Shape::Light {
                    light: Light::point(WHITE, 1.0, 10.0).with_penumbra(4.0),
                }),
                ..Default::default()
            },
            &Transform::identity(),
            0.0,
            WHITE,
            &mut draws,
        );
        let [
            Draw::Light {
                radius, penumbra, ..
            },
        ] = &draws[..]
        else {
            panic!("expected one light draw, got {draws:?}")
        };
        assert_eq!(*radius, 20.0);
        assert_eq!(*penumbra, 8.0);
    }

    // Tile maps: the collection tests the same way the particle batch
    // is tested — one node in, one draw out, and the packed instance
    // bytes read back float by float.

    /// The fields of a tile-map draw that the tests read.
    struct Map<'a> {
        data: &'a [u8],
        count: u32,
        color: Color,
        bounds: ([f32; 2], [f32; 2]),
        clip: Option<([f32; 2], [f32; 2])>,
        z: f32,
    }

    /// The node's tile-map draw: panics if the node drew anything but
    /// exactly one tile map.
    fn the_map(draws: &[Draw]) -> Map<'_> {
        match draws {
            [
                Draw::TileMap {
                    data,
                    count,
                    color,
                    bounds,
                    clip,
                    z,
                    ..
                },
            ] => Map {
                data,
                count: *count,
                color: *color,
                bounds: *bounds,
                clip: *clip,
                z: *z,
            },
            other => panic!("expected one tile map draw, got {other:?}"),
        }
    }

    /// Reads the little-endian float at component `component` (0..16 in
    /// the four instance vec4s: center and x-half-edge; y-half-edge and
    /// the orientation flags — flips, then quarter-turns; UV bounds;
    /// tint — so uv lives at 8..12 and the tint at 12..16) of tile
    /// `tile` in the packed instance data of the map.
    fn tile_f32(map: &Map, tile: usize, component: usize) -> f32 {
        let start = tile * 64 + component * 4;
        f32::from_le_bytes(map.data[start..start + 4].try_into().unwrap())
    }

    /// A full-opacity color for tile tests.
    fn rgba(r: f32, g: f32, b: f32, a: f32) -> Color {
        Color { r, g, b, a }
    }

    /// A tile map shape over the 4x4-byte test atlas with the given
    /// placements.
    fn tile_map_full(tiles: Vec<Tile>, clip: Option<[f32; 4]>, color: Color) -> Shape {
        Shape::TileMap {
            data: vec![0u8; 16].into(),
            width: 2,
            height: 2,
            generation: 1,
            filter: SpriteFilter::Nearest,
            tiles,
            clip,
            color,
        }
    }

    /// A plain tile map: the given placements, no clip, white batch.
    fn tile_map(tiles: Vec<Tile>) -> Shape {
        tile_map_full(tiles, None, WHITE)
    }

    #[test]
    fn node_tilemap_places_tiles_in_pixel_space() {
        // A node translated to (10, 20) with a 32x32 tile at local
        // (5, 5): the center lands at user (15, 25), pixel (65, 25);
        // the half-extents ride the canvas scale untouched (1x here).
        let node = SceneNode {
            transform: Transform::translate([10.0, 20.0]),
            shape: Some(tile_map(vec![Tile::new(
                [5.0, 5.0],
                [32.0, 32.0],
                [0.25, 0.5, 0.5, 1.0],
            )])),
            ..Default::default()
        };
        let mut draws = Vec::new();
        draw_one(&node, &mut draws);
        let map = the_map(&draws);
        assert_eq!(map.count, 1);
        assert_eq!(tile_f32(&map, 0, 0), 65.0);
        assert_eq!(tile_f32(&map, 0, 1), 25.0);
        // The half-edges: the x-axis stays rightward, the y-axis rides
        // the pixel space's flip and points down-screen.
        assert_eq!(tile_f32(&map, 0, 2), 16.0);
        assert_eq!(tile_f32(&map, 0, 3), 0.0);
        assert_eq!(tile_f32(&map, 0, 4), 0.0);
        assert_eq!(tile_f32(&map, 0, 5), -16.0);
        // The UV bounds pass through untouched, in order (slot two,
        // after the padded edge vec4).
        assert_eq!(tile_f32(&map, 0, 8), 0.25);
        assert_eq!(tile_f32(&map, 0, 9), 0.5);
        assert_eq!(tile_f32(&map, 0, 10), 0.5);
        assert_eq!(tile_f32(&map, 0, 11), 1.0);
    }

    #[test]
    fn a_transformed_tile_swaps_its_edges_and_rides_its_flags() {
        // A cell turned one quarter lies sideways on the quad: the
        // half-edges carry the swapped extents, the edge vec4's last
        // two floats carry the flip bits and the turn, and the batch's
        // AABB books the sideways footprint.
        let node = SceneNode {
            shape: Some(tile_map(vec![
                Tile::new([0.0, 0.0], [32.0, 16.0], [0.0, 0.0, 0.5, 0.5])
                    .transformed(1, true, false),
            ])),
            ..Default::default()
        };
        let mut draws = Vec::new();
        draw_one(&node, &mut draws);
        let map = the_map(&draws);
        // Swapped half-edges: the x-axis now carries the 16-tall
        // dimension's half, the y-axis the 32-wide one's.
        assert_eq!(tile_f32(&map, 0, 2), 8.0);
        assert_eq!(tile_f32(&map, 0, 5), -16.0);
        // The flags: flip bits (x only), then one clockwise turn.
        assert_eq!(tile_f32(&map, 0, 6), 1.0);
        assert_eq!(tile_f32(&map, 0, 7), 1.0);
        // And the batch's box is the sideways box, about the pixel
        // center wherever the canvas parks it.
        let (cx, cy) = (tile_f32(&map, 0, 0), tile_f32(&map, 0, 1));
        assert_eq!(map.bounds.0, [cx - 8.0, cy - 16.0]);
        assert_eq!(map.bounds.1, [cx + 8.0, cy + 16.0]);
    }

    #[test]
    fn node_tilemap_rotates_with_its_node() {
        // A 45-degree node rotation lands on both half-edges: the tile
        // becomes a diamond whose edges are the rotated axes, and the
        // batch's AABB books the diamond's reach, not the tile's box.
        let node = SceneNode {
            transform: Transform::rotate(std::f32::consts::FRAC_PI_4),
            shape: Some(tile_map(vec![Tile::new(
                [0.0, 0.0],
                [32.0, 32.0],
                [0.0, 0.0, 1.0, 1.0],
            )])),
            ..Default::default()
        };
        let mut draws = Vec::new();
        draw_one(&node, &mut draws);
        let map = the_map(&draws);
        let c = 16.0 * std::f32::consts::FRAC_PI_4.cos();
        let near = |component: usize, want: f32| {
            assert!(
                (tile_f32(&map, 0, component) - want).abs() < 1e-4,
                "component {component}: {} != {want}",
                tile_f32(&map, 0, component)
            );
        };
        // Center unchanged; u = 16 * rotated x-axis, v = 16 * rotated
        // y-axis (through the pixel space's y-flip).
        assert_eq!(tile_f32(&map, 0, 0), 50.0);
        assert_eq!(tile_f32(&map, 0, 1), 50.0);
        near(2, c);
        near(3, -c);
        near(4, -c);
        near(5, -c);
        // Both boxes reach 2c = |u| + |v| on each axis.
        let reach = 2.0 * c;
        let close = |got: [f32; 2], want: [f32; 2]| {
            assert!(
                (got[0] - want[0]).abs() < 1e-3 && (got[1] - want[1]).abs() < 1e-3,
                "{got:?} != {want:?}"
            );
        };
        close(map.bounds.0, [50.0 - reach, 50.0 - reach]);
        close(map.bounds.1, [50.0 + reach, 50.0 + reach]);
    }

    #[test]
    fn node_tilemap_scales_each_axis_apart() {
        // A tile is a box: under a (2, 3) node scale its two axes
        // stretch independently — the exact answer, not the particle
        // batch's geometric mean.
        let node = SceneNode {
            scale: [2.0, 3.0],
            shape: Some(tile_map(vec![Tile::new(
                [10.0, 10.0],
                [32.0, 32.0],
                [0.0, 0.0, 1.0, 1.0],
            )])),
            ..Default::default()
        };
        let mut draws = Vec::new();
        draw_one(&node, &mut draws);
        let map = the_map(&draws);
        // user center (20, 30) -> pixel (70, 20); half-edges (32, 0)
        // and (0, -48): each axis stretched by its own scale.
        assert_eq!(tile_f32(&map, 0, 0), 70.0);
        assert_eq!(tile_f32(&map, 0, 1), 20.0);
        assert_eq!(tile_f32(&map, 0, 2), 32.0);
        assert_eq!(tile_f32(&map, 0, 3), 0.0);
        assert_eq!(tile_f32(&map, 0, 4), 0.0);
        assert_eq!(tile_f32(&map, 0, 5), -48.0);
    }

    #[test]
    fn node_tilemap_bounds_union_and_view_clip() {
        // Two tiles far apart: the batch box is their union; the view
        // clip rides the node's transform as a box of its own.
        let node = SceneNode {
            shape: Some(tile_map_full(
                vec![
                    Tile::new([-30.0, 0.0], [20.0, 20.0], [0.0, 0.0, 0.5, 0.5]),
                    Tile::new([50.0, 30.0], [20.0, 20.0], [0.5, 0.5, 1.0, 1.0]),
                ],
                Some([-10.0, -20.0, 30.0, 20.0]),
                WHITE,
            )),
            ..Default::default()
        };
        let mut draws = Vec::new();
        draw_one(&node, &mut draws);
        let map = the_map(&draws);
        assert_eq!(map.count, 2);
        // The local box spans x -40..60, y -10..40; pixel space shifts
        // x by 50 and flips y, so it lands at ([10, 10], [110, 60]) —
        // and the pixel-space y *min* comes from the local y *max*.
        assert_eq!(map.bounds.0, [10.0, 10.0]);
        assert_eq!(map.bounds.1, [110.0, 60.0]);
        let clip = map.clip.expect("the clip rides the draw");
        assert_eq!(clip.0, [40.0, 30.0]);
        assert_eq!(clip.1, [80.0, 70.0]);
    }

    #[test]
    fn empty_tilemap_draws_nothing() {
        // Like an empty particle batch.
        let node = SceneNode {
            shape: Some(tile_map(Vec::new())),
            ..Default::default()
        };
        let mut draws = Vec::new();
        draw_one(&node, &mut draws);
        assert!(draws.is_empty(), "an empty map draws nothing: {draws:?}");
    }

    #[test]
    fn node_tilemap_multiplies_tints() {
        // Like the particle batch, modulate rides the draw's own color
        // once — the packed per-tile tint stays the tile's raw color,
        // and the shader's product of the two carries every factor.
        let node = SceneNode {
            modulate: rgba(0.5, 0.5, 0.5, 1.0),
            shape: Some(tile_map_full(
                vec![Tile {
                    center: [0.0, 0.0],
                    size: [10.0, 10.0],
                    uv: [0.0, 0.0, 1.0, 1.0],
                    color: rgba(1.0, 1.0, 0.5, 0.8),
                    rot: 0,
                    flip_x: false,
                    flip_y: false,
                }],
                None,
                rgba(1.0, 0.0, 0.0, 1.0),
            )),
            ..Default::default()
        };
        let mut draws = Vec::new();
        draw_one(&node, &mut draws);
        let map = the_map(&draws);
        assert_eq!(tile_f32(&map, 0, 12), 1.0);
        assert_eq!(tile_f32(&map, 0, 13), 1.0);
        assert_eq!(tile_f32(&map, 0, 14), 0.5);
        assert_eq!(tile_f32(&map, 0, 15), 0.8);
        assert_eq!(map.color, rgba(0.5, 0.0, 0.0, 1.0));
    }

    #[test]
    fn node_tilemap_draws_at_the_node_order() {
        let node = SceneNode {
            order: 7.0,
            shape: Some(tile_map(vec![Tile::new(
                [0.0, 0.0],
                [8.0, 8.0],
                [0.0, 0.0, 1.0, 1.0],
            )])),
            ..Default::default()
        };
        let mut draws = Vec::new();
        draw_one(&node, &mut draws);
        assert_eq!(the_map(&draws).z, 7.0);
    }

    #[test]
    fn a_declared_occluder_is_stored_in_pixel_space_with_its_shape_local() {
        // A 100 x 100 window at scale 2: user space is window-centered and
        // y-up, pixels are top-left and y-down. The declared occluder's
        // transform crosses into pixel space at the declare call, exactly as
        // a draw's arguments do, while its center and half-extents stay in the
        // occluder's own space for the shader to work in.
        let mut canvas = Canvas::new((100, 100), 2.0);
        canvas.occluder(Transform::translate([10.0, 20.0]), [1.0, 2.0], [3.0, -4.0]);
        assert_eq!(canvas.occluders.len(), 1);
        let o = &canvas.occluders[0];
        // The transform carries the occluder's origin wherever the canvas
        // puts that user-space point, and a unit step is now two panel pixels
        // down the y axis.
        assert_eq!(o.world.apply([0.0, 0.0]), canvas.user_to_pixels(10.0, 20.0));
        assert_eq!(o.world.m, [[2.0, 0.0], [0.0, -2.0]]);
        assert_eq!(
            o.world.apply([1.0, 0.0]),
            [o.world.t[0] + 2.0, o.world.t[1]]
        );
        // The shape never left its own space — and a negative half-extent is
        // clamped to zero rather than turning the box inside out, as a
        // rectangle's extent is.
        assert_eq!(o.center, [1.0, 2.0]);
        assert_eq!(o.half, [3.0, 0.0]);
    }
}
