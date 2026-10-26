//! frost — a minimal wgpu 2D drawing library, embeddable in a host application.
//!
//! Provide a [`Scene`] and a [`Process`]: a function called once per frame
//! with a [`Context`] and the delta time in seconds since the previous
//! frame. The scene is drawn every frame, *after* the process runs and the
//! scene's tree has been updated by a visit (every node's
//! [`Node::process`], children before their parent), so both can mutate it
//! in place to animate it. Draw with window-centered
//! pixel coordinates: the origin is the window center, y points up, so the
//! top-left corner is `(-width/2, height/2)`. The unit is the logical
//! pixel — the OS's window units: one panel pixel each at 100% scaling,
//! more per unit on a high-density display (see [`Canvas::scale_factor`]) —
//! so a layout sized in these units keeps its physical size on every
//! screen, while text and shapes rasterize at the panel's own resolution.
//!
//! The frame loop lives in [`embedded::Embedded`]: the host owns the
//! window, the event pump, and the wgpu device and surface, and calls
//! [`Embedded::frame`](embedded::Embedded::frame) once per redraw —
//! typically from a ~15 ms GUI timer. OS input is translated into frost's
//! input setters (`set_key`, `set_cursor`, …) as it arrives.
//!
//! ```no_run
//! # fn example(
//! #     device: wgpu::Device,
//! #     queue: wgpu::Queue,
//! #     adapter: wgpu::Adapter,
//! #     surface: wgpu::Surface<'static>,
//! # ) {
//! let scene = frost::Scene::new(frost::SceneNode {
//!     shape: Some(frost::Shape::Background {
//!         color: frost::Color { r: 0.05, g: 0.06, b: 0.12, a: 1.0 },
//!     }),
//!     ..Default::default()
//! });
//! let mut app = frost::embedded::Embedded::new(
//!     device,
//!     queue,
//!     adapter,
//!     surface,
//!     wgpu::TextureFormat::Bgra8UnormSrgb,
//!     (800, 600),
//!     1.0,
//!     scene,
//!     |ctx: &mut frost::Context, _dt: f32| {
//!         let (w, h) = ctx.size();
//!         ctx.line(
//!             -w / 2.0, h / 2.0, w / 2.0, -h / 2.0,
//!             frost::Color { r: 1.0, g: 1.0, b: 1.0, a: 1.0 },
//!             2.0,
//!             0.0,
//!         );
//!         ctx.circle(
//!             0.0, 0.0, h / 4.0,
//!             frost::Color { r: 0.9, g: 0.4, b: 0.2, a: 1.0 },
//!             1.0,
//!         );
//!     },
//!     frost::embedded::EmbeddedConfig::default(),
//! );
//! app.frame().unwrap();
//! # }
//! ```
//!
//! Draw order is set by each object's `z`: lower `z` is drawn first (further
//! back). Objects with the same `z` are drawn in call order, so the last one
//! drawn is on top.
//!
//! A [`Scene`] can additionally define rendering [`Layer`]s: hard draw
//! partitions, each with its own order, its own parallax speed, and its own
//! root node. Groups are painted by ascending layer order — higher order
//! closer to the camera, drawn later, on top — and the scene's root subtree
//! is the group at the implicit order `0.0`, declared before the explicit
//! layers. Within a group the local `z` ordering applies. A layer's
//! [`Layer::repeat`] tiles it in one or both axes with a period of at least
//! the window size, so it can be scrolled through infinitely — an object
//! crossing a tile boundary is split into wrapping slices, and any setup
//! that would draw the same object twice aborts with an error.
//!
//! A [`Scene`] can also designate a camera node ([`Scene::camera`]): the
//! scene is then drawn in that node's coordinate space, so the node stays at
//! the window origin while the rest of the scene moves around it — hang the
//! camera under a moving node and it follows. Each group renders from the
//! camera scaled by its speed: the base group at the full speed `1.0`, each
//! [`Layer`] at its [`Layer::speed`].
//!
//! Each object is clipped to its tight bounding box (the scissor test), so a
//! frame's cost scales with the objects' on-screen areas, not the window size.
//!
//! Beyond the immediate draws, a [`Scene`] is a tree of [`SceneNode`]s where
//! each node holds a [`Transform`], a `scale`, a `modulate`, and an `order`,
//! all relative to its parent, plus its own optional [`Shape`]; the scale
//! and transform apply to the node's shape and compose onto its children,
//! the modulate multiplies into the node's shape color and composes onto
//! its children's, and the order does the same for the subtree's draw
//! order. Every [`SceneNode`] is a [`Node`]: its [`Node::visit`] walks its
//! children before itself (post-order), so the whole tree updates with a
//! single [`Scene::visit`]. A [`Shape::Background`]
//! node fills the whole window with its color, ignoring its transform, and
//! is drawn at the very back.
//! The scene passed to [`Embedded::new`](embedded::Embedded::new) is drawn
//! every frame; use [`Canvas::draw_scene`] to draw additional scenes.
//!
//! A [`ParticleSystem`] is a plain collection of [`Particle`]s for
//! effects like smoke or sparks. It is pure simulation, independent of the
//! renderer: spawn particles in [`Process::process`] and advance them each
//! frame with [`ParticleSystem::update`] under a constant gravity. Draw
//! them by putting the system in the node's [`Shape::Particles`] shape —
//! the batch is then drawn in the node's local space, transformed,
//! scaled, and tinted by the node — or, for batches that live directly in
//! the window's user space, with the deprecated immediate
//! [`Canvas::particles`] draw; either way each particle fades by its
//! remaining lifetime.
//!
//! A [`Shape::Polyline`] connects a list of points with straight segments
//! of a fixed width and draws the whole chain in one draw call, whatever
//! its length — the batched alternative to a run of thin rectangles or a
//! chain of [`Canvas::line`] draws.
//!
//! Randomness comes from [`Rng`], a small seeded splitmix64 generator:
//! seed it from the clock for a different stream on each run, or fix the
//! seed for a reproducible one. It is pure integer arithmetic, so it runs
//! identically on every platform, and keeps the crate free of a `rand`
//! dependency.
//!
//! The host feeds input through the [`embedded::Embedded`] setters: the
//! currently held keys are reported by [`Context::key_down`], the mouse
//! cursor's position by [`Context::mouse_position`], the held mouse buttons
//! by [`Context::mouse_button_down`], and the mouse wheel's movement since
//! the previous frame by [`Context::mouse_wheel`].
//!
//! A [`Diagnostics`] overlay reports the window size and, for each
//! statistic enabled in its [`DiagnosticsFlags`], a left-aligned line in
//! the window's top-left corner: the smoothed frame rate, the current
//! frame time, the last frame's total processing time (the engine's own
//! probe of its per-frame CPU work), that time excluding the overlay's own
//! update cost (the overlay times itself and subtracts), and the last
//! frame's draw-call count (the engine's own probe of how many GPU draw
//! calls it issued) — with a scrolling ten-second strip chart beneath each
//! line: the frame rate and the frame time in one series each, and the
//! processing time and the draw-call count folded into one chart each —
//! the total, the app's share, and the overlay's own share. Hold one in
//! the demo state and call its [`Process::process`] each frame; its parts
//! can be toggled at runtime — Alt-0 the overlay as a whole, Alt-1..Alt-4
//! the charts, Alt-T the readout lines — with the layout reflowing around
//! whatever is hidden, and Alt+'+' / Alt+'-' grow and shrink the whole
//! overlay, font included.
//!
//! Presentation cadence belongs to the host: frost never blocks on a
//! vertical blank (the host's surface is configured
//! [`AutoNoVsync`](wgpu::PresentMode::AutoNoVsync)), so a frame returns as
//! soon as it is submitted.

use std::collections::{HashMap, HashSet};

mod backend;
pub use backend::embed as embedded;
/// Scope only: the `Context` documentation below refers to the startup
/// settings by name.
#[allow(unused_imports)]
use backend::embed::EmbeddedConfig;
#[allow(unused_imports)]
use backend::*;

mod objects;
pub use objects::*;

pub use shaders::LIGHTING_WGSL;
#[cfg(debug_assertions)]
pub use shaders::validate_wgsl;

mod canvas;
pub use canvas::Canvas;

/// The physical keyboard keys used by [`Context::key_down`], such as
/// `KeyCode::KeyW`. Physical keys identify the key's position on the
/// keyboard (its scancode), independent of the active layout, which is what
/// game controls like WASD want.
pub use winit::keyboard::KeyCode;

/// The mouse buttons used by [`Context::mouse_button_down`], such as
/// `MouseButton::Left`.
pub use winit::event::MouseButton;

/// A physical-pixel size, for hosts that speak the frame's buffer size in
/// the unit the OS reports it in.
pub use winit::dpi::PhysicalSize;
/// The fullscreen state a host window may carry. `Fullscreen::Borderless(None)`
/// — the game's usual choice — fills the monitor the window sits on; on
/// macOS that is the platform's native fullscreen Space. frost draws the
/// same frame either way; it is the host that swaps one in and out.
pub use winit::window::Fullscreen;

mod particles;
pub use particles::*;

mod shaders;

mod text;

mod tween;
pub use tween::*;

mod collision;
pub use collision::*;

mod physics;
pub use physics::*;

mod bake;
pub use bake::*;

mod shapes;
pub use shapes::*;

pub mod assets;
/// Text kept under a name until the next launch: a file beside the
/// launch folder, as in [`assets`].
pub mod store;
pub mod tell;
pub mod watch;
pub use assets::{Assets, Table};
pub use tell::{Announcement, Teller, Tiding};
pub use watch::Watch;
pub mod map;
/// The crate's own RON subset: a dependency-free parser, tree model and
/// lossless writer (comments and verbatim atoms ride through the tree),
/// for anything that must read a `.ron` file back out unchanged. Kept
/// namespaced because its names (`Val`, `Item`, `Doc`, `parse`…) are a
/// format's vocabulary, not the engine's.
pub mod ron;
pub mod tileset;

mod diagnostics;
pub use diagnostics::{Diagnostics, DiagnosticsFlags};

mod ui;
pub use ui::{
    Align, Answer, Col, ColSize, ItemKind, Menu, MenuEvent, MenuItem, TreeEvent, TreeLine, TreeOut,
    TreeSpec, TreeState, TreeStyle, Ui, UiStyle,
};

mod rng;
pub use rng::*;

/// The per-frame context handed to [`Process::process`].
///
/// Derefs to the frame's [`Canvas`] (immediate draws) and
/// [`scene`](Context::scene) gives mutable access to the [`Scene`] passed to
/// [`Embedded::new`](embedded::Embedded::new): mutate it here to animate
/// it, and it is drawn after the process returns.
pub struct Context<'c> {
    canvas: &'c mut Canvas,
    scene: &'c mut Scene,
    keys: &'c HashSet<KeyCode>,
    /// The character each held key typed as it went down, under the user's
    /// keyboard layout (see [`Context::char_down`]).
    typed: &'c HashMap<KeyCode, char>,
    /// The engine's live stretch filter, mutable so a running app can flip
    /// it (see [`Context::set_blit_filter`]).
    blit_filter: &'c mut SpriteFilter,
    /// The engine's live vsync state, read by the app (see
    /// [`Context::set_vsync`]); an embedding always presents uncapped, so
    /// the engine itself never acts on it.
    vsync: &'c mut bool,
    /// The engine's live fixed render size, mutable so a running app can
    /// change the quality of the stretch pass (see
    /// [`Context::set_render_size`]).
    render_size: &'c mut Option<[u32; 2]>,
    /// The engine's live bloom strength, mutable so a running app can turn the
    /// glow up, down, or on (see [`Context::set_bloom`]).
    bloom: &'c mut f32,
    /// The engine's live tone-map headroom, mutable so a running app can change
    /// what it does with light above white (see [`Context::set_tone_map`]).
    tone_map: &'c mut f32,
    expected_fps: Option<f32>,
    /// The cursor's position in user coordinates, or `None` when the cursor
    /// is outside the window (or has never moved into it).
    mouse: Option<[f32; 2]>,
    /// The mouse buttons currently held down.
    mouse_buttons: &'c HashSet<MouseButton>,
    /// The mouse wheel's vertical movement since the previous frame, in
    /// lines (see [`Context::mouse_wheel`]).
    mouse_wheel: f32,
    /// The CPU time in milliseconds the engine spent processing the last
    /// completed frame (see [`Context::frame_processing_ms`]).
    frame_processing_ms: f64,
    /// The GPU draw calls the engine issued for the last completed frame
    /// (see [`Context::frame_draw_calls`]).
    frame_draw_calls: u32,
    /// The GPU draw calls the last completed frame spent on the
    /// [`Diagnostics`] overlay's own nodes (see
    /// [`Context::frame_diagnostic_draw_calls`]).
    frame_diagnostic_draw_calls: u32,
    /// The engine's exit flag, raised by [`Context::exit`]: the frame
    /// completes as drawn, and the host — which owns the window's
    /// lifetime — reads it (see [`Embedded::exit_requested`]) and decides.
    ///
    /// [`Embedded::exit_requested`]: embedded::Embedded::exit_requested
    exit: &'c mut bool,
    /// The engine's key freeze, read live (see [`Context::set_keys_frozen`]).
    keys_frozen: &'c mut bool,
}

impl Context<'_> {
    /// Mutable access to the scene owned by [`Embedded::new`](embedded::Embedded::new).
    pub fn scene(&mut self) -> &mut Scene {
        &mut *self.scene
    }

    /// Whether the physical `key` is currently held down.
    ///
    /// The state is updated as keyboard events arrive, so it reflects every
    /// press and release since the previous frame.
    pub fn key_down(&self, key: KeyCode) -> bool {
        !*self.keys_frozen && self.keys.contains(&key)
    }

    /// [`Context::key_down`], unread by the key freeze: the app's modal
    /// reads its own keys through here while every other key stands
    /// dead (see [`Context::set_keys_frozen`]).
    pub fn key_down_raw(&self, key: KeyCode) -> bool {
        self.keys.contains(&key)
    }

    /// Whether a key held down right now typed the character `ch` as it
    /// went down, under the user's current keyboard layout.
    ///
    /// This is the layout-following sibling of [`Context::key_down`]:
    /// physical codes name US-ANSI key positions, so the key that types
    /// `'+'` lives wherever the user's layout puts it — left of `= ¨ ´`
    /// on a Swedish keyboard, say, where the US layout has its `-` key —
    /// and `char_down('+')` finds that user's `+` key wherever it is. The
    /// character is the one the press produced, so a modifier involved in
    /// the press (a macOS Option, which rewrites characters) shows up in
    /// it. Multi-character presses match on their first character.
    pub fn char_down(&self, ch: char) -> bool {
        !*self.keys_frozen && self.typed.values().any(|&c| c == ch)
    }

    /// [`Context::char_down`], unread by the key freeze (see
    /// [`Context::key_down_raw`]).
    pub fn char_down_raw(&self, ch: char) -> bool {
        self.typed.values().any(|&c| c == ch)
    }

    /// Asks the engine to end the application when this frame is drawn.
    /// The frame completes — whatever the process draws last is what the
    /// window shows last — and the flag stands until the host reads it
    /// (see [`Embedded::exit_requested`](embedded::Embedded::exit_requested))
    /// and acts; frost itself never closes anything.
    pub fn exit(&mut self) {
        *self.exit = true;
    }

    /// Freezes the keyboard reads: while frozen, [`Context::key_down`]
    /// and [`Context::char_down`] answer false for every key, so the
    /// game's keyboard controls stand down while a modal dialog holds
    /// the keys. The freeze is read live, effective the moment it is
    /// set; the dialog itself hears the keys through
    /// [`Context::key_down_raw`] and [`Context::char_down_raw`]. Held
    /// keys are remembered, not dropped: unfreezing mid-press finds
    /// them still down, and a key released while frozen shows no edge
    /// at all.
    pub fn set_keys_frozen(&mut self, frozen: bool) {
        *self.keys_frozen = frozen;
    }

    /// Whether the keyboard reads are frozen now (see
    /// [`Context::set_keys_frozen`]).
    pub fn keys_frozen(&self) -> bool {
        *self.keys_frozen
    }

    /// The filter that stretches the fixed [`EmbeddedConfig::render_size`] buffer
    /// over the window — the upscaling filter (see
    /// [`EmbeddedConfig::blit_filter`]).
    pub fn blit_filter(&self) -> SpriteFilter {
        *self.blit_filter
    }

    /// Sets the stretch filter at runtime: the mid-session sibling of
    /// setting [`EmbeddedConfig::blit_filter`] at startup, effective from this
    /// frame's stretch pass on. The live change rebuilds the blit sampler
    /// and its bind group once, on the first frame that stretches with the
    /// new setting. Without [`EmbeddedConfig::render_size`] there is no stretch to
    /// filter — the setting records faithfully and shows its effect the
    /// moment a fixed buffer ever runs.
    pub fn set_blit_filter(&mut self, filter: SpriteFilter) {
        *self.blit_filter = filter;
    }

    /// Whether the app believes frames are presented in sync with the
    /// display's vertical blank ("vsync") — the live state
    /// [`Self::set_vsync`] flips. An embedding presents uncapped
    /// whatever this says: presentation cadence is the host's, and frost
    /// must not block its caller.
    pub fn vsync(&self) -> bool {
        *self.vsync
    }

    /// Records a vsync wish on the engine's flag; see [`Self::vsync`] for
    /// what an embedding does with it — which is nothing at the swap
    /// chain. The overlay's Alt-V and the app's own readouts keep working;
    /// the host's timer owns the pacing either way.
    pub fn set_vsync(&mut self, on: bool) {
        *self.vsync = on;
    }

    /// The fixed render size this frame is being drawn into — the live
    /// state behind [`EmbeddedConfig::render_size`] (which [`Self::set_render_size`]
    /// changes at runtime). `None` means the frame renders at the window's
    /// own resolution.
    pub fn render_size(&self) -> Option<[u32; 2]> {
        *self.render_size
    }

    /// How much the frame's bright light bleeds into its neighbours: the live
    /// strength behind [`EmbeddedConfig::bloom`], which [`Self::set_bloom`] changes at
    /// runtime. `0.0` is no glow.
    pub fn bloom(&self) -> f32 {
        *self.bloom
    }

    /// Turns the glow up, down, or on, mid-run: the mid-session sibling of
    /// setting [`EmbeddedConfig::bloom`] at startup.
    ///
    /// The strength itself costs nothing: it is the preset's weights in a uniform
    /// the stretch pass rewrites every frame, so a change lands on the frame that
    /// asks for it. What does cost is asking for a glow *at all* on a run that
    /// started without one. If the frame buffer can hold a value past 1.0 — the
    /// same condition that gives it an emissive target — then the first frame to
    /// see a non-zero strength builds the chain, six textures and nine bind
    /// groups, and pays for that on its own frame; §11 of the benchmarking notes
    /// measures the hitch. Where the buffer clips at 1.0 there is nothing for a
    /// chain to read: the strength records faithfully, says so in the log once,
    /// and shows its effect the moment a float buffer ever runs.
    pub fn set_bloom(&mut self, strength: f32) {
        *self.bloom = strength;
    }

    /// How much of the display's range is reserved for light above white: the
    /// live value behind [`EmbeddedConfig::tone_map`], which [`Self::set_tone_map`]
    /// changes at runtime. `0.0` clips.
    pub fn tone_map(&self) -> f32 {
        *self.tone_map
    }

    /// Reserves room above display white, or takes it away: the mid-session
    /// sibling of setting [`EmbeddedConfig::tone_map`] at startup.
    ///
    /// This one has no cost and no delay — the headroom is a single float in a
    /// uniform the stretch pass rewrites every frame — which is what makes it
    /// sane to drive from the frame's own content: a shot with a sun in it and
    /// the menu that follows it can ask for different headrooms on consecutive
    /// frames, and each gets what it asked for.
    ///
    /// It is one number for the whole window, and there is deliberately no
    /// per-layer version of it: the fold happens once, over the composed frame,
    /// so an interface and the world behind it are folded together. Keeping them
    /// apart needs a subtree that renders into a buffer of its own, which the
    /// scene tree cannot express today.
    pub fn set_tone_map(&mut self, headroom: f32) {
        *self.tone_map = headroom;
    }

    /// Fixes the render size — or lifts the fix — mid-run: the runtime
    /// sibling of [`EmbeddedConfig::render_size`], and the lever behind a
    /// "quality" setting. `Some([w, h])` renders every frame into a
    /// buffer of exactly those physical pixels and stretches it over the
    /// window through the blit filter ([`Self::set_blit_filter`]); `None`
    /// returns to rendering at the window's own resolution.
    ///
    /// The change lands the way a window resize lands, one frame late:
    /// this frame is already measured against the old size, and the next
    /// renders through the new one — its offscreen buffer is built on
    /// demand, the old one retires, and [`Canvas::size`] starts reporting
    /// the new user space. An app that lays itself out from `ctx.size()`
    /// each frame needs no notice of the switch; a window shaped unlike
    /// the buffer letterboxes, as it always does under a fixed size.
    pub fn set_render_size(&mut self, size: Option<[u32; 2]>) {
        *self.render_size = size;
    }

    /// The frame rate this app is expected to run at, in frames per second.
    ///
    /// `None` for an embedding: presentation cadence belongs to the host's
    /// timer, and frost's surface is never pinned to a vertical blank, so
    /// there is no expected rate the engine could honestly report. The
    /// [`Diagnostics`] overlay copes — it measures its own frame rate from
    /// the frames it is actually shown.
    pub fn expected_fps(&self) -> Option<f32> {
        self.expected_fps
    }

    /// The CPU time, in milliseconds, the engine spent processing the last
    /// completed frame.
    ///
    /// The backend probes its own frame pipeline: the measurement starts
    /// just after the frame's surface acquire — so it includes this frame's
    /// [`Process::process`] call, the scene update, the draw list, and the
    /// command encoding — and stops when the frame's command buffer is
    /// submitted to the queue. The surface acquire, the present handoff to
    /// the display, and the wait for the next frame are excluded, so the
    /// value measures the CPU work of a frame rather than its on-screen
    /// duration — that is the `dt` [`Process::process`] receives. Before
    /// the first frame completes it is `0.0`.
    ///
    /// Compare it with `1000.0 / fps`: as the two approach each other the
    /// app becomes CPU-bound, and the gap between them is the headroom the
    /// [`Diagnostics`] overlay's processing-time chart shows.
    pub fn frame_processing_ms(&self) -> f64 {
        self.frame_processing_ms
    }

    /// The GPU draw calls the engine issued for the last completed frame.
    ///
    /// The backend counts one draw call per `pass.draw` / `pass.draw_indexed`
    /// it actually submits into the frame's render pass — a shape, a line, a
    /// sprite, a glyph quad, a chart polyline, a particle batch, and so on.
    /// Draws the engine skips are not counted: a shape fully outside the
    /// window, a background (which became the clear color), a light (which
    /// became part of the light field), or a particle batch with no live
    /// particles. Before the first frame completes it is `0`.
    ///
    /// This is the per-frame draw-call load of the app. In this engine each
    /// draw call also costs CPU work (a fresh uniform buffer, bind group,
    /// and scissor/pipeline state per draw), so a rising count shows up in
    /// the [`Diagnostics`] overlay's processing-time chart as well.
    pub fn frame_draw_calls(&self) -> u32 {
        self.frame_draw_calls
    }

    /// The GPU draw calls the last completed frame spent on the
    /// [`Diagnostics`] overlay's own nodes: the subset of
    /// [`frame_draw_calls`](Self::frame_draw_calls) whose draws the overlay
    /// produced — the text lines, the graph panels, the reference lines,
    /// and the chart polylines, one per glyph quad in the text.
    ///
    /// The overlay tags the scene nodes it creates with the
    /// [`SceneNode::diagnostic`] engine-managed marker, the marker rides
    /// onto every draw the nodes produce (the glyphs included), and the
    /// backend counts a tagged draw wherever it
    /// would count the total: the same skips (off-screen, background,
    /// light, empty) apply, so the value is a true subset of the total.
    /// Before the first frame completes it is `0`, like the total.
    pub fn frame_diagnostic_draw_calls(&self) -> u32 {
        self.frame_diagnostic_draw_calls
    }

    /// The mouse cursor's position in user coordinates (origin at the
    /// window's center, y up), or `None` when the cursor is outside the
    /// window.
    ///
    /// The position is updated as cursor events arrive, so it reflects the
    /// cursor's latest position rather than its position at frame start.
    /// While the cursor is outside the window — after it has left, or
    /// before it has first moved in — the last known position is no longer
    /// reported, so keep one yourself if you want the pointer to stick.
    pub fn mouse_position(&self) -> Option<[f32; 2]> {
        self.mouse
    }

    /// Whether the `button` mouse button is currently held down.
    ///
    /// The state is updated as mouse input events arrive, so it reflects
    /// every press and release since the previous frame. When the cursor
    /// leaves the window or the window loses focus, the held state is
    /// dropped, so a button can never appear stuck down.
    pub fn mouse_button_down(&self, button: MouseButton) -> bool {
        self.mouse_buttons.contains(&button)
    }

    /// The mouse wheel's vertical movement since the previous frame, in
    /// lines: positive when the wheel moves up (away from the user),
    /// negative when it moves down.
    ///
    /// The delta sums every wheel event that arrived since the previous
    /// frame — several notches may accumulate in one frame — and is consumed
    /// per frame: a frame that never reads it discards the delta. `0.0`
    /// when the wheel did not move.
    pub fn mouse_wheel(&self) -> f32 {
        self.mouse_wheel
    }
}

impl std::ops::Deref for Context<'_> {
    type Target = Canvas;

    fn deref(&self) -> &Canvas {
        self.canvas
    }
}

impl std::ops::DerefMut for Context<'_> {
    fn deref_mut(&mut self) -> &mut Canvas {
        self.canvas
    }
}

/// Called once per frame; mutate the scene and draw into the canvas.
///
/// `ctx` gives mutable access to the scene owned by the
/// [`Embedded`](embedded::Embedded) engine (via [`Context::scene`]) and
/// derefs to the frame's [`Canvas`] for the
/// immediate draw methods. `dt` is the time in seconds
/// since the previous frame (`0.0` on the first frame, clamped to at most
/// `1.0`s to absorb stalls). Use it to advance animation state such as a
/// [`Tween`].
pub trait Process {
    fn process(&mut self, ctx: &mut Context, dt: f32);
}

/// Any closure `FnMut(&mut Context, f32)` is a [`Process`].
impl<F> Process for F
where
    F: FnMut(&mut Context, f32),
{
    fn process(&mut self, ctx: &mut Context, dt: f32) {
        self(ctx, dt);
    }
}
