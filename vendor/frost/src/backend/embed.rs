//! The embeddable engine: frost as a guest in someone else's
//! application.
//!
//! The host — a plugin GUI, an editor, any app that already owns a
//! window, a wgpu device, and a way to be woken — hands frost a surface
//! and calls [`Embedded::frame`] once per redraw. Everything the event
//! loop used to do between waking and presenting happens there: the
//! [`Process`] runs with a delta time measured from the previous frame,
//! the scene tree visits, the draw list is rebuilt, and the surface
//! presents. Everything the event loop used to *own* — window creation,
//! OS events, the cadence of waking — stays outside this module: the
//! host translates its events into the input setters below, at whatever
//! rate it likes. A ~15 ms GUI timer is the intended rhythm, and any
//! other rhythm is legal: `dt` is measured, never assumed.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use wgpu::{Adapter, Device, Queue, Surface, TextureFormat};

use crate::backend::app::Frost;
use crate::objects::{Scene, SpriteFilter};
use crate::{KeyCode, MouseButton, Process};

/// Startup settings for [`Embedded`]: the frame's rendering choices,
/// all of them live afterwards through the matching [`Context`] getters
/// and setters.
///
/// [`Context`]: crate::Context
#[derive(Clone, Copy, Debug)]
pub struct EmbeddedConfig {
    /// A fixed render size in physical pixels, or `None` to render at
    /// the surface's own resolution (see [`Context::render_size`]).
    /// The frame is rendered into a buffer of exactly this size and
    /// stretched over the surface; a surface of a different shape
    /// letterboxes — the largest centered rectangle of the buffer's
    /// aspect, bars included — and the mouse maps through that
    /// rectangle. Unlike a standalone window there is no aspect lock to
    /// fight the host's shape: the host asked for this rectangle, so the
    /// bars are the honest answer.
    ///
    /// [`Context::render_size`]: crate::Context::render_size
    pub render_size: Option<[u32; 2]>,
    /// The filter that resamples the fixed buffer when it is stretched
    /// (see [`Context::blit_filter`]); without [`Self::render_size`]
    /// there is no stretch to filter.
    ///
    /// [`Context::blit_filter`]: crate::Context::blit_filter
    pub blit_filter: SpriteFilter,
    /// The initial glow strength (see [`Context::set_bloom`]).
    ///
    /// [`Context::set_bloom`]: crate::Context::set_bloom
    pub bloom: f32,
    /// The initial tone-map headroom (see [`Context::set_tone_map`]).
    ///
    /// [`Context::set_tone_map`]: crate::Context::set_tone_map
    pub tone_map: f32,
}

impl Default for EmbeddedConfig {
    /// No fixed render size, bilinear upscaling, no bloom, no tone map.
    fn default() -> Self {
        Self {
            render_size: None,
            blit_filter: SpriteFilter::Linear,
            bloom: 0.0,
            tone_map: 0.0,
        }
    }
}

/// The reason [`Embedded::frame`] refused to render. Ordinary skipped
/// frames — a surface mid-resize, a texture the compositor would not
/// hand over — are not errors: the frame returns `Ok(())` having drawn
/// nothing, and [`Embedded::surface_lost`] says whether a fresh surface
/// is wanted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum FrameError {
    /// The GPU device is gone; no frame will render again until the
    /// host builds a new device (and hands frost a new `Embedded`).
    DeviceLost,
    /// The host last reported a zero-sized surface; there is nothing to
    /// draw into. A later [`Embedded::resize`] with a real size resumes
    /// rendering — the error stands for every frame called meanwhile,
    /// so a host that polls blindly notices without tracking anything.
    SizeInvalid,
}

/// The device limits a host should request so every frost pipeline fits.
///
/// frost's shape pipeline binds storage buffers visible to the fragment
/// stage; downlevel/WebGL2 limit profiles zero those out, and wgpu then
/// aborts with a validation panic the first time the engine builds a
/// bind group layout — inside *someone else's* host process. This is the
/// single source of truth: request the device with exactly
/// `frost::embedded::device_limits(&adapter)` (uplevel defaults — what
/// frost asks for its own device — with resolution limits clamped to the
/// adapter so the request itself cannot fail on sizes).
pub fn device_limits(adapter: &wgpu::Adapter) -> wgpu::Limits {
    wgpu::Limits::defaults().using_resolution(adapter.limits())
}

/// The embeddable engine: the scene graph, the frame pipeline, and the
/// input state, driven one call per frame by the host.
///
/// All wgpu resources come from the caller — the device, the queue, and
/// the surface are the host's; frost only records, submits, and presents
/// through them. The crate's windowing days are over: nothing here
/// creates or owns a window, and nothing blocks: the surface is
/// configured `PresentMode::AutoNoVsync` forever, because a plugin must
/// never hold the host's main thread across a display refresh.
pub struct Embedded<P: Process> {
    frost: Frost<P>,
    device: Device,
    queue: Queue,
    /// The size and scale the host last reported, kept in the host's
    /// own units (physical pixels, raw scale): `Frost` derives a logical
    /// size from them, and `resize` compares and forwards the raw pair.
    physical: (u32, u32),
    scale_factor: f64,
    /// Raised by the device-lost callback installed in [`Self::new`]:
    /// the honest answer to "is this device dead", wgpu's own verdict
    /// rather than a guess from failed calls.
    device_lost: Arc<AtomicBool>,
}

impl<P: Process> Embedded<P> {
    /// Wraps a scene and process around the host's GPU and surface, and
    /// draws the first frame before returning — a setup that cannot
    /// present says so here, at construction, not at the host's first
    /// timer tick.
    ///
    /// `physical` is the surface's size in physical pixels and
    /// `scale_factor` the host window's scale (Retina: 2.0);
    /// `surface_format` is the format the host chose for the surface —
    /// frost configures the swap chain with it (upgraded to its sRGB
    /// twin where the hardware offers one, the choice frost has always
    /// made) and never with a surprise instead.
    ///
    /// vsync is never used: every configure forces
    /// [`wgpu::PresentMode::AutoNoVsync`], which presents as soon as a
    /// frame is rendered and falls back gracefully where the uncapped
    /// modes are absent — a blocking `Fifo` present could stall the
    /// host's main thread for a refresh, and no guest in someone else's
    /// GUI has that right. The cadence belongs to the host's timer; the
    /// [`crate::Context::set_vsync`] flag still exists for apps that read it,
    /// but it moves no swap chain here.
    ///
    /// The host moves in the very [`Adapter`] it requested its device
    /// from — a capability *query* partner only (surface formats, texture
    /// features): nothing is ever created from it, and the device that
    /// does the work is the host's. It must come from the same
    /// [`Instance`] as the device and the surface: wgpu ids are scoped to
    /// their instance's registry, and an adapter from anywhere else is a
    /// dangling number by the first capability question (this replaced a
    /// private-instance adapter, which panicked exactly that way under
    /// wgpu 30's per-instance globals).
    pub fn new(
        device: Device,
        queue: Queue,
        adapter: Adapter,
        surface: Surface<'static>,
        surface_format: TextureFormat,
        physical: (u32, u32),
        scale_factor: f64,
        scene: Scene,
        process: P,
        config: EmbeddedConfig,
    ) -> Self {
        // Fail fast, at construction, with an actionable message — rather
        // than later inside `Device::create_bind_group_layout` ("limit is 0,
        // count was 2"), which is where a host that requested downlevel /
        // WebGL2 limits would otherwise trip over the shape pipeline.
        let limits = device.limits();
        assert!(
            limits.max_storage_buffers_per_shader_stage >= 2,
            "frost's shape pipeline binds 2 storage buffers visible to the \
             fragment stage, but the host device was requested with \
             max_storage_buffers_per_shader_stage = {}. Request the device \
             with frost::embedded::device_limits(&adapter).",
            limits.max_storage_buffers_per_shader_stage
        );
        let device_lost = Arc::new(AtomicBool::new(false));
        {
            let flag = Arc::clone(&device_lost);
            device.set_device_lost_callback(move |reason, message| {
                log::error!("wgpu device lost ({reason:?}): {message}");
                flag.store(true, Ordering::Relaxed);
            });
        }
        let mut frost = Frost::new(
            adapter,
            device.clone(),
            queue.clone(),
            config.render_size,
            config.blit_filter,
            config.bloom,
            config.tone_map,
            scene,
            process,
        );
        frost.attach_surface(surface, surface_format, physical, scale_factor);
        Self {
            frost,
            device,
            queue,
            physical,
            scale_factor: scale_factor.max(0.01),
            device_lost,
        }
    }

    /// The surface changed shape: the host reports its new physical size
    /// and scale. Exactly the way a live window resize used to arrive —
    /// the size is noted and honoured once it stops moving for a tenth
    /// of a second, so a dragged plugin panel rebuilds one swap chain at
    /// the end of the drag rather than one per frame of it; the old
    /// surface stretches meanwhile. A scale-factor change is the
    /// exception, applied at once, because a density move is one event
    /// rather than a flight.
    pub fn resize(&mut self, physical: (u32, u32), scale_factor: f64) {
        let scale_changed = (scale_factor - self.scale_factor).abs() > f64::EPSILON;
        self.physical = physical;
        if scale_changed {
            self.scale_factor = scale_factor.max(0.01);
            self.frost.host_scale_factor(self.scale_factor as f32);
            // The scale path sizes from the stored logical size, which
            // still describes the *old* rectangle; take the new size the
            // ordinary way too so the two agree as soon as it settles.
            self.frost.host_resize(physical);
        } else {
            self.frost.host_resize(physical);
        }
    }

    /// Hands frost a fresh surface after the old one was lost — the
    /// compositor took it back (the host's `surface_lost` check says
    /// when), and the host has built another over the same window with
    /// the same device. The swap chain and the pipelines that were built
    /// against its format follow the new surface; the scene, the texture
    /// caches, and the uniform arena ride through untouched.
    pub fn replace_surface(&mut self, surface: Surface<'static>, surface_format: TextureFormat) {
        self.frost
            .attach_surface(surface, surface_format, self.physical, self.scale_factor);
    }

    /// Whether the surface stopped handing over frames: the current
    /// frame returned without drawing, and the host should build a new
    /// surface and [`Self::replace_surface`] it. This is how wgpu tells
    /// a plugin that its drawable layer moved (a window dragged to
    /// another Retina screen, a macOS Space transition) without ever
    /// dropping an error on the host's thread.
    pub fn surface_lost(&self) -> bool {
        self.frost.surface_lost()
    }

    /// One frame: exactly what the event loop used to do per redraw —
    /// settle a pending resize, run the [`Process`] with the `dt` since
    /// the previous frame, visit the scene, rebuild the draw list, and
    /// present. Non-blocking: the call returns when the commands are
    /// submitted, never when they retire on the GPU.
    ///
    /// A frame with nothing to draw (no surface, a texture mid-flight)
    /// is a skip, not an error: `Ok(())` and no work, with
    /// [`Self::surface_lost`] as the reason. `Err` means the call is
    /// pointless until something outside changes:
    pub fn frame(&mut self) -> Result<(), FrameError> {
        if self.device_lost.load(Ordering::Relaxed) {
            return Err(FrameError::DeviceLost);
        }
        if self.physical.0 == 0 || self.physical.1 == 0 {
            return Err(FrameError::SizeInvalid);
        }
        self.frost.step();
        if self.device_lost.load(Ordering::Relaxed) {
            return Err(FrameError::DeviceLost);
        }
        Ok(())
    }

    // ==================== input ====================
    //
    // The host translates its OS events into the calls below; each is
    // the event-loop arm it replaces, with the same state as the
    // destination.

    /// A physical key went down (`down == true`) or up. The character
    /// the press types is a separate call, [`Self::add_text`] — call it
    /// right after a press, with the character the host's event carries
    /// (its layout-translated logical key), or not at all for keys that
    /// type nothing.
    pub fn set_key(&mut self, code: KeyCode, down: bool) {
        self.frost.set_key(code, down);
    }

    /// Reports the modifier *state*, for hosts that track a bitmask
    /// rather than individual key events. frost has no modifier state of
    /// its own — a Shift is a key like any other — so this is shorthand
    /// for the four left-hand modifier codes, each moved only if its
    /// state changed. A host that forwards raw key events already has
    /// the true left/right codes; it should call [`Self::set_key`] and
    /// leave this alone. One source of truth, whichever the host has.
    pub fn set_modifiers(&mut self, shift: bool, ctrl: bool, alt: bool, logo: bool) {
        self.set_key(KeyCode::ShiftLeft, shift);
        self.set_key(KeyCode::ControlLeft, ctrl);
        self.set_key(KeyCode::AltLeft, alt);
        self.set_key(KeyCode::SuperLeft, logo);
    }

    /// The cursor moved to `(x, y)` in **logical** (CSS / point) pixels
    /// from the surface's upper-left corner, y down — the unit the
    /// host's own widgets count in, at the `scale_factor` reported to
    /// [`Self::new`] or [`Self::resize`]. frost multiplies by the scale
    /// internally: the old cursor arm stored physical pixels, and a
    /// host holding logical ones should not have to know that.
    /// Coordinates outside the surface are legal and mean what they say
    /// — over the next window over, still "inside" until the host says
    /// [`Self::cursor_left`].
    pub fn set_cursor(&mut self, x: f64, y: f64) {
        self.frost.set_cursor(x, y);
    }

    /// The cursor left the surface: the position reads as absent
    /// ([`Context::mouse_position`] answers `None`) and held buttons are
    /// dropped, because a release outside the window may never arrive.
    ///
    /// [`Context::mouse_position`]: crate::Context::mouse_position
    pub fn cursor_left(&mut self) {
        self.frost.cursor_left();
    }

    /// A mouse button went down (`down == true`) or up.
    pub fn set_mouse_button(&mut self, btn: MouseButton, down: bool) {
        self.frost.set_mouse_button(btn, down);
    }

    /// The mouse wheel moved. `dy` is in lines — one unit per notch,
    /// the way the old line-delta events spoke; a host holding pixel
    /// deltas should divide by 120 first, the same conversion the old
    /// arm applied. `dx` is received and unused: frost has never tracked
    /// a horizontal wheel.
    pub fn scroll(&mut self, dx: f64, dy: f64) {
        let _ = dx;
        self.frost.scroll(dy);
    }

    /// The character typed by the key of the most recent
    /// [`Self::set_key`] press that is still held — the layout-following
    /// half of key input (see [`Context::char_down`]), what lets the
    /// diagnostics overlay find Alt+'+' on a keyboard whose `+` is
    /// nowhere near the US position. A character with no live press
    /// passes: frost's state only ever asks "which held key typed
    /// this".
    ///
    /// [`Context::char_down`]: crate::Context::char_down
    pub fn add_text(&mut self, ch: char) {
        self.frost.note_key_char(ch);
    }

    /// The surface gained or lost keyboard focus. Losing it clears every
    /// held key and button — their releases may never arrive — exactly
    /// as the old focus event did; regaining it remembers nothing.
    pub fn set_focused(&mut self, focused: bool) {
        self.frost.set_focused(focused);
    }

    // ==================== state ====================

    /// Mutable access to the scene the engine draws every frame — the
    /// same [`Scene`] the [`Process`] reaches through
    /// [`Context::scene`], reachable between frames too.
    ///
    /// [`Context::scene`]: crate::Context::scene
    pub fn scene(&mut self) -> &mut Scene {
        self.frost.scene()
    }

    /// The host's device, as handed to [`Self::new`].
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// The host's queue, as handed to [`Self::new`].
    pub fn queue(&self) -> &Queue {
        &self.queue
    }

    /// Whether the [`Process`] asked to leave this frame or an earlier
    /// one (see [`Context::exit`]). frost drew its last frame and keeps
    /// drawing; the flag stands until [`Self::clear_exit`] — the host
    /// owns the window's lifetime and decides what an exit means.
    ///
    /// [`Context::exit`]: crate::Context::exit
    pub fn exit_requested(&self) -> bool {
        self.frost.exit_requested()
    }

    /// Answers an exit request by keeping the engine alive; the flag is
    /// free to be raised again by a stubborn process.
    pub fn clear_exit(&mut self) {
        self.frost.clear_exit();
    }
}
