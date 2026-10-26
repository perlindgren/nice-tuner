//! STAGING (M4): replaces the `TunerWindow` half of src/lib.rs once
//! `frost::embedded::Embedded` (M3) lands. See stage/integration_notes.md.
//!
//! Responsibilities:
//! - own the wgpu surface on the plugin's own `wgpu::Instance`
//! - build the frost static scene + `Embedded` engine once
//! - drive `frame()` from baseview's host-side timer (`on_frame`)
//! - reconfigure on `resized`, recreate on surface `Lost`
//! - translate `baseview::Event` → frost input injection (docs/SCENE.md table)

use crate::{TunerParams, TunerSharedState};
use baseview::{EventStatus, HandlerError, WindowContext, WindowHandler};
use frost::embedded::{Embedded, EmbeddedConfig, FrameError};
#[cfg(all(windows, feature = "standalone"))]
use std::cell::Cell;
use std::cell::RefCell;
use std::sync::{Arc, atomic::AtomicBool};

/// Standalone window placement (Windows only). baseview 0.3.3 has no
/// position API and exposes no NSWindow/X11 handle, so only the Win32
/// top-level can be positioned and measured — exactly the platform the
/// standalone lives on in practice. Size alone is restored everywhere
/// (baseview does that via `with_size`), macOS cascades new windows.
#[cfg(all(windows, feature = "standalone"))]
mod place {
    use baseview::WindowContext;
    use raw_window_handle::HasWindowHandle as _;
    use std::ffi::c_void;
    use windows_sys::Win32::Foundation::{HWND, POINT, RECT};
    use windows_sys::Win32::Graphics::Gdi::{MONITOR_DEFAULTTONULL, MonitorFromPoint};
    use windows_sys::Win32::UI::WindowsAndMessaging::{
        GetWindowRect, SPI_GETWORKAREA, SWP_NOACTIVATE, SWP_NOSIZE, SWP_NOZORDER, SetWindowPos,
        SystemParametersInfoW,
    };

    pub fn hwnd_of(ctx: &WindowContext) -> Option<isize> {
        match ctx.platform_handle().window_handle().ok()?.as_raw() {
            raw_window_handle::RawWindowHandle::Win32(w) => Some(w.hwnd.get()),
            _ => None,
        }
    }

    fn work_area() -> Option<RECT> {
        let mut r = RECT {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        };
        (unsafe { SystemParametersInfoW(SPI_GETWORKAREA, 0, &mut r as *mut _ as *mut c_void, 0) }
            != 0)
            .then_some(r)
    }

    fn rect_of(hwnd: isize) -> Option<RECT> {
        let mut r = RECT {
            left: 0,
            top: 0,
            right: 0,
            bottom: 0,
        };
        (unsafe { GetWindowRect(hwnd as HWND, &mut r) } != 0).then_some(r)
    }

    /// Restore the saved spot if that screen still exists (a monitor
    /// unplugged since the save must not strand the window offscreen),
    /// else centre in the primary work area — the no-settings-file case.
    /// Size is deliberately untouched: baseview already sized it.
    pub fn restore(ctx: &WindowContext, saved: Option<(i32, i32)>) {
        let Some(hwnd) = hwnd_of(ctx) else { return };
        let Some(work) = work_area() else { return };
        let Some(r) = rect_of(hwnd) else { return };
        let (w, h) = (r.right - r.left, r.bottom - r.top);
        let center = (
            work.left + (work.right - work.left - w) / 2,
            work.top + (work.bottom - work.top - h) / 2,
        );
        let (x, y) = match saved {
            Some(p) => {
                let probe = POINT {
                    x: p.0 + w / 2,
                    y: p.1 + 16,
                };
                let live = unsafe { MonitorFromPoint(probe, MONITOR_DEFAULTTONULL) } != 0;
                if live { p } else { center }
            }
            None => center,
        };
        unsafe {
            SetWindowPos(
                hwnd as HWND,
                0, // insert-after: NULL (handles are isize aliases in windows-sys 0.52)
                x,
                y,
                0,
                0,
                SWP_NOSIZE | SWP_NOZORDER | SWP_NOACTIVATE,
            );
        }
    }

    pub fn top_left(hwnd: isize) -> Option<(i32, i32)> {
        let r = rect_of(hwnd)?;
        Some((r.left, r.top))
    }
}

/// OFL font shipped with the frost fork; `Arc<[u8]>` for cheap shape sharing.
pub fn font() -> Arc<[u8]> {
    use std::sync::OnceLock;
    static BYTES: &[u8] =
        include_bytes!("../vendor/frost/assets/fonts/FiraCode-VariableFont_wght.ttf");
    static FONT: OnceLock<Arc<[u8]>> = OnceLock::new();
    FONT.get_or_init(|| Arc::from(BYTES)).clone()
}

pub struct TunerWindow {
    window: WindowContext,

    engine: RefCell<Option<TunerEngine>>,
    /// Kept for upcoming interactive controls (a4 nudge); the readout itself
    /// flows through the shared atomics, mirrored by the audio thread.
    #[allow(dead_code)]
    params: Arc<TunerParams>,
    shared: Arc<TunerSharedState>,
    _redraw_requested: Arc<AtomicBool>,

    /// Win32 handle of the standalone top-level (geometry APIs).
    #[cfg(all(windows, feature = "standalone"))]
    hwnd: Option<isize>,
    /// Geometry write-back: debounce — persist 2 s after the window
    /// stops moving, and flush on WillClose.
    #[cfg(all(windows, feature = "standalone"))]
    geom_seen: Cell<Option<(i32, i32, f32, f32)>>,
    #[cfg(all(windows, feature = "standalone"))]
    geom_due_ms: Cell<u64>,
    #[cfg(all(windows, feature = "standalone"))]
    geom_saved: Cell<Option<(i32, i32, f32, f32)>>,
}

struct TunerEngine {
    /// The instance that the surface, adapter, and device were all made
    /// from. wgpu 30 scopes resource ids to their instance's registry, so
    /// a surface-lost repair MUST re-create the surface on this very
    /// instance — a fresh one would make every stored id cross registries
    /// (regression: "Cannot get non-existent resource AdapterId").
    instance: wgpu::Instance,
    /// Owns the `wgpu::Surface` internally; resize handled by `Embedded`.
    /// A *lost* surface is detected via `Embedded::surface_lost()` and
    /// repaired with `replace_surface(fresh, format)`.
    embedded: Embedded<crate::tuner_ui::TunerUi>,
    format: wgpu::TextureFormat,
}

impl TunerWindow {
    pub fn new(
        window: WindowContext,
        _gui_context: nice_plug::context::gui::GuiContext,
        params: Arc<TunerParams>,
        shared: Arc<TunerSharedState>,
        redraw: Arc<AtomicBool>,
    ) -> Result<Self, HandlerError> {
        // Capture the Win32 handle before `window` moves into the struct.
        #[cfg(all(windows, feature = "standalone"))]
        let hwnd = place::hwnd_of(&window);
        let this = Self {
            window,
            engine: RefCell::new(None),
            params,
            shared,
            _redraw_requested: redraw,
            #[cfg(all(windows, feature = "standalone"))]
            hwnd,
            #[cfg(all(windows, feature = "standalone"))]
            geom_seen: Cell::new(None),
            #[cfg(all(windows, feature = "standalone"))]
            geom_due_ms: Cell::new(0),
            #[cfg(all(windows, feature = "standalone"))]
            geom_saved: Cell::new(None),
        };
        this.spawn_engine()?;
        // Standalone top-level: honour the saved spot, or centre when no
        // geometry was ever saved. A host-embedded child is left exactly
        // where the host put it (STANDALONE_MODE gate inside).
        #[cfg(all(windows, feature = "standalone"))]
        if crate::STANDALONE_MODE.load(std::sync::atomic::Ordering::Relaxed) {
            let saved = *this.shared.pending_geom.lock().unwrap();
            place::restore(&this.window, saved);
        }
        Ok(this)
    }

    /// Standalone geometry write-back (Windows): watch the top-left each
    /// frame — one syscall, nanoseconds — debounce two seconds past the
    /// last move/resize, then merge into the settings file. An abrupt
    /// quit loses at most the last two seconds of dragging.
    #[cfg(all(windows, feature = "standalone"))]
    fn track_geometry(&self) {
        if !crate::STANDALONE_MODE.load(std::sync::atomic::Ordering::Relaxed) {
            return;
        }
        let Some(hwnd) = self.hwnd else { return };
        let Some((x, y)) = place::top_left(hwnd) else {
            return;
        };
        *self.shared.window_pos.lock().unwrap() = Some((x, y));
        let ls = self.shared.logical_size();
        let geom = (x, y, ls.width, ls.height);
        if Some(geom) == self.geom_saved.get() {
            self.geom_due_ms.set(0);
            self.geom_seen.set(Some(geom));
            return;
        }
        if self.geom_seen.get() != Some(geom) {
            self.geom_seen.set(Some(geom));
            self.geom_due_ms.set(crate::now_epoch_ms() + 2_000);
        }
        let due = self.geom_due_ms.get();
        if due != 0 && crate::now_epoch_ms() >= due {
            self.geom_due_ms.set(0);
            self.persist_geom();
        }
    }

    /// Flush a debounced geometry save (close, or anything else ending
    /// the session before the debounce elapses).
    #[cfg(all(windows, feature = "standalone"))]
    fn flush_geom(&self) {
        if self.geom_due_ms.get() != 0 {
            self.geom_due_ms.set(0);
            self.persist_geom();
        }
    }

    /// Merge-only write: never touch a user's other settings here.
    #[cfg(all(windows, feature = "standalone"))]
    fn persist_geom(&self) {
        use crate::settings::{Settings, WindowGeom};
        let Some((x, y, width, height)) = self.geom_seen.get() else {
            return;
        };
        let (mut s, _) = Settings::load();
        s.window = Some(WindowGeom {
            x: Some(x),
            y: Some(y),
            width,
            height,
        });
        if s.save().is_ok() {
            self.geom_saved.set(Some((x, y, width, height)));
        }
    }

    fn spawn_engine(&self) -> Result<(), HandlerError> {
        let size = self.window.size();
        let scale = size.scale_factor;

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let surface = pollster::block_on(async {
            let surface = instance.create_surface(self.window.platform_handle())?;
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower,
                    force_fallback_adapter: false,
                    compatible_surface: Some(&surface),
                    ..Default::default()
                })
                .await
                .map_err(|e| HandlerError::from_boxed(format!("{e}").into()))?;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("nice-tuner"),
                    required_features: wgpu::Features::empty(),
                    required_limits: frost::embedded::device_limits(&adapter),
                    memory_hints: wgpu::MemoryHints::MemoryUsage,
                    ..Default::default()
                })
                .await
                .map_err(|e| HandlerError::from_boxed(e.to_string().into()))?;
            Ok::<_, HandlerError>((surface, adapter, device, queue))
        })?;
        let (surface, adapter, device, queue) = surface;

        let mut surface_config = surface
            .get_default_config(
                &adapter,
                size.physical.width.max(1),
                size.physical.height.max(1),
            )
            .ok_or_else(|| HandlerError::from_boxed("no surface config".into()))?;
        // Plugins must never block the host main thread:
        surface_config.present_mode = wgpu::PresentMode::AutoNoVsync;
        surface.configure(&device, &surface_config);
        let format = surface_config.format;

        let ui = crate::tuner_ui::TunerUi::new(Arc::clone(&self.shared), font());
        let scene = crate::tuner_ui::static_scene();

        // The adapter moves in: frost keeps it for capability queries, and
        // it must be the very adapter this device (and surface) came from.
        let embedded = Embedded::new(
            device.clone(),
            queue.clone(),
            adapter,
            surface,
            format,
            (size.physical.width, size.physical.height),
            scale,
            scene,
            ui,
            EmbeddedConfig::default(),
        );

        *self.engine.borrow_mut() = Some(TunerEngine {
            instance,
            embedded,
            format,
        });
        Ok(())
    }
}

impl WindowHandler for TunerWindow {
    fn on_frame(&self) -> Result<(), HandlerError> {
        #[cfg(all(windows, feature = "standalone"))]
        self.track_geometry();
        // Settings-page "Apply & restart" (standalone): unwind the editor
        // with a deliberate close; nice-plug's wrapper tears down cleanly on
        // the way out and main() respawns us with the saved settings.
        if crate::restart_requested() {
            return Err(HandlerError::from_boxed("restart requested".into()));
        }
        let mut guard = self.engine.borrow_mut();
        match guard.as_mut() {
            Some(engine) => match engine.embedded.frame() {
                // SizeInvalid = zero-sized panel mid-collapse: nothing to
                // draw, `resize` resumes. Never surface it to baseview — a
                // HandlerError would tear the window down.
                // frame() may return Ok while having drawn nothing: the
                // honest loss signal is surface_lost(). Repair in place with
                // a fresh surface — on the SAME instance as device/adapter
                // (see TunerEngine::instance).
                Ok(()) => {
                    if engine.embedded.surface_lost() {
                        let fresh = engine
                            .instance
                            .create_surface(self.window.platform_handle())
                            .map_err(|e| HandlerError::from_boxed(e.to_string().into()))?;
                        engine.embedded.replace_surface(fresh, engine.format);
                    }
                    Ok(())
                }
                // Zero-size panel mid-collapse: nothing was ever presented,
                // so there is nothing "lost"; `resize` resumes. Never
                // surface it to baseview — a HandlerError tears the window
                // down.
                Err(FrameError::SizeInvalid) => Ok(()),
                // The device is gone: rebuild device+surface+scene wholesale;
                // the baseview window itself is still alive.
                Err(FrameError::DeviceLost) | Err(_) => {
                    drop(guard);
                    *self.engine.borrow_mut() = None;
                    self.spawn_engine()
                }
            },
            None => self.spawn_engine(),
        }
    }

    fn on_event(&self, event: baseview::Event) -> EventStatus {
        use baseview::{Event as B, MouseEvent as M, WindowEvent as W};
        use keyboard_types::KeyState;
        let mut guard = self.engine.borrow_mut();
        if let Some(engine) = guard.as_mut() {
            let size = self.window.size();
            let scale = size.scale_factor;
            match event {
                B::Keyboard(ke) => {
                    let down = ke.state == KeyState::Down;
                    if let Some(code) = crate::tuner_ui::map_key(&ke.key) {
                        engine.embedded.set_key(code, down);
                    }
                    // keyboard_types 0.8: typed characters ride in
                    // `key: Key::Character` on key-down events
                    if down && let keyboard_types::Key::Character(s) = &ke.key {
                        for ch in s.chars() {
                            engine.embedded.add_text(ch);
                        }
                    }
                    use keyboard_types::Modifiers;
                    engine.embedded.set_modifiers(
                        ke.modifiers.contains(Modifiers::SHIFT),
                        ke.modifiers.contains(Modifiers::CONTROL),
                        ke.modifiers.contains(Modifiers::ALT),
                        ke.modifiers.contains(Modifiers::META),
                    );
                }
                B::Mouse(mouse) => match mouse {
                    // baseview 0.3.3 macOS delivers PHYSICAL px (view.rs
                    // to_physical); frost's set_cursor wants logical px,
                    // y-down from the surface top-left -> just un-scale.
                    M::CursorMoved { position, .. } => {
                        engine
                            .embedded
                            .set_cursor(position.x / scale, position.y / scale);
                    }
                    M::ButtonPressed { button, .. } => {
                        if let Some(btn) = crate::tuner_ui::map_mouse(button) {
                            engine.embedded.set_mouse_button(btn, true);
                        }
                    }
                    M::ButtonReleased { button, .. } => {
                        if let Some(btn) = crate::tuner_ui::map_mouse(button) {
                            engine.embedded.set_mouse_button(btn, false);
                        }
                    }
                    M::WheelScrolled { delta, .. } => {
                        // frost's scroll dy is in LINES (raw); pixels /120
                        let (dx, dy) = match delta {
                            baseview::ScrollDelta::Lines { y, .. } => (0.0, y as f64),
                            baseview::ScrollDelta::Pixels { y, .. } => (0.0, y as f64 / 120.0),
                        };
                        engine.embedded.scroll(dx, dy);
                    }
                    M::CursorLeft => engine.embedded.cursor_left(),
                    _ => {}
                },
                B::Window(W::Focused) => engine.embedded.set_focused(true),
                B::Window(W::Unfocused) => engine.embedded.set_focused(false),
                // window lifecycle stays with the host
                B::Window(W::WillClose) => {
                    // Flush a pending (debounced) geometry save on the way
                    // out, so a quit right after a move still remembers.
                    #[cfg(all(windows, feature = "standalone"))]
                    self.flush_geom();
                }
                _ => {}
            }
            // No explicit redraw call: baseview's frame timer drives `on_frame`
            // continuously (~64 Hz on macOS), and the tuner animates anyway.
        }
        baseview::EventStatus::Captured
    }

    fn resized(&self, new_size: baseview::WindowSize) -> Result<(), HandlerError> {
        self.shared.logical_size.store(new_size.logical.cast());
        self.shared.scale_factor.store(new_size.scale_factor);

        let mut guard = self.engine.borrow_mut();
        if let Some(engine) = guard.as_mut() {
            // Embedded debounces swap-chain rebuilds internally (settles 100ms
            // after the last size event; scale changes apply instantly).
            engine.embedded.resize(
                (new_size.physical.width, new_size.physical.height),
                new_size.scale_factor,
            );
        }
        Ok(())
    }
}
