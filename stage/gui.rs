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
use baseview::{EventStatus, HandlerError, WindowContext, WindowHandler, WindowSettings};
use frost::embedded::{Embedded, EmbeddedConfig, FrameError};
use std::cell::RefCell;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

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
    /// Stashed inputs while the engine is not yet up (or after a surface loss).
    params: Arc<TunerParams>,
    shared: Arc<TunerSharedState>,
    _redraw_requested: Arc<AtomicBool>,
}

struct TunerEngine {
    /// Owns the `wgpu::Surface` internally; recreate via `replace_surface`.
    embedded: Embedded<crate::tuner_ui::TunerUi>,
    surface_config: wgpu::SurfaceConfiguration,
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
        let this = Self {
            window,
            engine: RefCell::new(None),
            params,
            shared,
            _redraw_requested: redraw,
        };
        this.spawn_engine()?;
        Ok(this)
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
                .ok_or_else(|| HandlerError::from_boxed("no wgpu adapter".into()))?;
            let (device, queue) = adapter
                .request_device(&wgpu::DeviceDescriptor {
                    label: Some("nice-tuner"),
                    required_features: wgpu::Features::empty(),
                    required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                        .using_resolution(adapter.limits()),
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

        let embedded = Embedded::new(
            device.clone(),
            queue.clone(),
            surface,
            format,
            (size.physical.width, size.physical.height),
            scale,
            scene,
            ui,
            EmbeddedConfig::default(),
        );

        *self.engine.borrow_mut() = Some(TunerEngine {
            embedded,
            surface_config,
            format,
        });
        Ok(())
    }
}

impl WindowHandler for TunerWindow {
    fn on_frame(&self) -> Result<(), HandlerError> {
        let mut guard = self.engine.borrow_mut();
        match guard.as_mut() {
            Some(engine) => match engine.embedded.frame() {
                Ok(()) | Err(FrameError::Skipped) => Ok(()),
                Err(FrameError::SurfaceLost) => {
                    // Rebuild the surface from the still-live baseview window.
                    let instance =
                        wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
                    let fresh = instance
                        .create_surface(self.window.platform_handle())
                        .map_err(|e| HandlerError::from_boxed(e.to_string().into()))?;
                    engine.embedded.replace_surface(fresh, engine.format);
                    Ok(())
                }
                Err(e) => Err(HandlerError::from_boxed(format!("frost frame: {e}").into())),
            },
            None => {
                self.spawn_engine()?;
                Ok(())
            }
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
                    if let Some(code) = crate::tuner_ui::map_key(&ke.key) {
                        engine.embedded.set_key(code, ke.state == KeyState::Down);
                    }
                    // text input rides along with key events in baseview
                    if let Some(text) = &ke.text {
                        for ch in text.chars() {
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
                    // position arrives in PHYSICAL px (view.rs converts logical
                    // NSView points via to_physical); frost wants logical, y-up.
                    M::CursorMoved { position, .. } => {
                        let h = size.logical.height as f64;
                        engine
                            .embedded
                            .set_cursor(position.x / scale, h - position.y / scale);
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
                        let (dx, dy) = match delta {
                            baseview::ScrollDelta::Lines { x, y } => {
                                (x as f64 * 40.0, y as f64 * 40.0)
                            }
                            baseview::ScrollDelta::Pixels { x, y } => (x as f64, y as f64),
                        };
                        engine.embedded.scroll(dx, dy);
                    }
                    _ => {}
                },
                B::Window(W::Focused) => engine.embedded.set_focused(true),
                B::Window(W::Unfocused) => engine.embedded.set_focused(false),
                // window lifecycle stays with the host
                B::Window(W::WillClose) => {}
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
            engine.embedded.resize(
                (new_size.physical.width, new_size.physical.height),
                new_size.scale_factor,
            );
            engine.surface_config.width = new_size.physical.width.max(1);
            engine.surface_config.height = new_size.physical.height.max(1);
        }
        Ok(())
    }
}
