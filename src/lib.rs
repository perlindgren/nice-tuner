//! Nice Tuner — a stringed-instrument tuner plugin.
//!
//! Host plumbing follows nice-plug's `byo_gui_wgpu` example (tag v0.4.2):
//! a `baseview` child window embedded in the host, rendered with wgpu.
//! The triangle placeholder renderer is replaced by the frost-derived
//! embedded engine in M3/M4 (see docs/VIABILITY.md).

use baseview::{EventStatus, HandlerError, WindowContext, WindowSettings};
use crossbeam::atomic::AtomicCell;
use nice_plug::context::gui::GuiContext;
use nice_plug::editor::dpi::{LogicalSize, NativeSize};
use nice_plug::editor::{EditorHandle, HostMethods, SpawnedEditor};
use nice_plug::prelude::*;
use std::cell::RefCell;
use std::error::Error;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::{borrow::Cow, sync::atomic::AtomicU64};

const MIN_SIZE: LogicalSize<f32> = LogicalSize::new(480.0, 320.0);
const DEFAULT_SIZE: LogicalSize<f32> = LogicalSize::new(640.0, 400.0);
const RESIZE_HINT: ResizeHint = ResizeHint::resizable().with_min_logical_size(MIN_SIZE);

// ---------------------------------------------------------------------------------------------------
// GUI <-> audio-thread shared state

/// State shared between the audio thread (`process`) and the GUI.
#[derive(Debug)]
pub struct TunerSharedState {
    /// Logical window size, kept up to date by the editor.
    logical_size: AtomicCell<LogicalSize<f32>>,
    /// Whether the editor's window is currently open. The audio thread uses
    /// this to gate expensive detection work.
    open: AtomicBool,
    scale_factor: AtomicCell<f64>,
    /// Latest pitch estimate in Hz (0.0 = no detection yet).
    pub freq_hz: AtomicCell<f64>,
    /// Confidence of the latest estimate, 0.0..=1.0.
    pub clarity: AtomicCell<f64>,
    /// RMS level of the analysed window (for the input meter).
    pub rms: AtomicCell<f32>,
    /// Wall-clock ms (UNIX epoch) of the last confident detection; the UI
    /// treats the signal as absent when this goes stale (the detector only
    /// reports confident windows).
    pub last_detection_ms: AtomicU64,
    /// A4 reference frequency in Hz, mirrored from the param so the GUI can
    /// label notes with the user's tuning standard. Stored as bits of f64.
    pub a4_ref_bits: AtomicU64,
    /// Gate threshold in dB, mirrored from the param for the GUI's
    /// signal-present decision.
    pub gate_db: AtomicCell<f32>,
}

impl TunerSharedState {
    pub fn new() -> Self {
        Self {
            logical_size: AtomicCell::new(DEFAULT_SIZE),
            open: AtomicBool::new(false),
            scale_factor: AtomicCell::new(1.0),
            freq_hz: AtomicCell::new(0.0),
            clarity: AtomicCell::new(0.0),
            rms: AtomicCell::new(0.0),
            last_detection_ms: AtomicU64::new(0),
            a4_ref_bits: AtomicU64::new(440.0f64.to_bits()),
            gate_db: AtomicCell::new(-60.0),
        }
    }

    pub fn logical_size(&self) -> LogicalSize<f32> {
        self.logical_size.load()
    }

    pub fn scale_factor(&self) -> f64 {
        self.scale_factor.load()
    }

    pub fn is_open(&self) -> bool {
        self.open.load(Ordering::Acquire)
    }

    pub fn a4_ref(&self) -> f64 {
        f64::from_bits(self.a4_ref_bits.load(Ordering::Relaxed))
    }
}

// ---------------------------------------------------------------------------------------------------
// The embedded window (baseview handler)

struct Surface {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline: wgpu::RenderPipeline,
    surface: wgpu::Surface<'static>,
    surface_config: wgpu::SurfaceConfiguration,
}

pub struct TunerWindow {
    _gui_context: GuiContext,
    window: WindowContext,

    surface: RefCell<Surface>,

    #[allow(unused)]
    params: Arc<TunerParams>,
    shared: Arc<TunerSharedState>,
    redraw_requested: Arc<AtomicBool>,
}

impl TunerWindow {
    fn new(
        window: WindowContext,
        gui_context: GuiContext,
        params: Arc<TunerParams>,
        shared: Arc<TunerSharedState>,
        redraw: Arc<AtomicBool>,
    ) -> Result<Self, HandlerError> {
        pollster::block_on(Self::create(
            window,
            gui_context,
            params,
            shared,
            redraw,
        ))
    }

    async fn create(
        window: WindowContext,
        gui_context: GuiContext,
        params: Arc<TunerParams>,
        shared: Arc<TunerSharedState>,
        redraw: Arc<AtomicBool>,
    ) -> Result<Self, HandlerError> {
        let size = window.size();

        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

        let surface = instance.create_surface(window.platform_handle())?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::LowPower,
                force_fallback_adapter: false,
                // Request an adapter which can render to our surface
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
                memory_hints: wgpu::MemoryHints::MemoryUsage,
                ..Default::default()
            })
            .await?;

        // M1 placeholder: a triangle. Replaced by the frost scene in M4.
        const SHADER: &str = "
            const VERTS = array(
                vec2<f32>(0.5, 1.0),
                vec2<f32>(0.0, 0.0),
                vec2<f32>(1.0, 0.0)
            );

            struct VertexOutput {
                @builtin(position) clip_position: vec4<f32>,
                @location(0) position: vec2<f32>,
            };

            @vertex
            fn vs_main(
                @builtin(vertex_index) in_vertex_index: u32,
            ) -> VertexOutput {
                var out: VertexOutput;
                out.position = VERTS[in_vertex_index];
                out.clip_position = vec4<f32>(out.position - 0.5, 0.0, 1.0);
                return out;
            }

            @fragment
            fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
                return vec4<f32>(in.position, 0.5, 1.0);
            }
            ";

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(SHADER)),
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[],
            immediate_size: 0,
        });

        let swapchain_capabilities = surface.get_capabilities(&adapter);
        let swapchain_format = swapchain_capabilities.formats[0];

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vs_main"),
                buffers: &[],
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_main"),
                compilation_options: Default::default(),
                targets: &[Some(swapchain_format.into())],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });

        let mut surface_config = surface
            .get_default_config(&adapter, size.physical.width, size.physical.height)
            .unwrap(); // TODO: handle None

        // IMPORTANT! Plugins should not use vsync as it blocks the host's main thread.
        surface_config.present_mode = wgpu::PresentMode::AutoNoVsync;

        surface.configure(&device, &surface_config);

        Ok(Self {
            _gui_context: gui_context,
            window,
            surface: RefCell::new(Surface {
                device,
                queue,
                pipeline,
                surface,
                surface_config,
            }),
            params,
            shared,
            redraw_requested: redraw,
        })
    }

    fn configure_surface(&self, width: u32, height: u32) {
        let mut surface = self.surface.borrow_mut();
        surface.surface_config.width = width;
        surface.surface_config.height = height;
        surface
            .surface
            .configure(&surface.device, &surface.surface_config);
    }
}

impl baseview::WindowHandler for TunerWindow {
    fn on_frame(&self) -> Result<(), HandlerError> {
        if !self.redraw_requested.swap(false, Ordering::Relaxed) {
            return Ok(());
        }

        let mut surface = self.surface.borrow_mut();
        let Surface {
            device,
            queue,
            pipeline,
            surface,
            surface_config,
        } = &mut *surface;

        let mut recreate_surface = false;
        let frame = match surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(texture) => Some(texture),
            wgpu::CurrentSurfaceTexture::Occluded | wgpu::CurrentSurfaceTexture::Timeout => {
                return Ok(());
            }
            wgpu::CurrentSurfaceTexture::Suboptimal(_) | wgpu::CurrentSurfaceTexture::Outdated => {
                None
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                unreachable!("No error scope registered, so validation errors will panic")
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                recreate_surface = true;
                None
            }
        };

        let Some(frame) = frame else {
            if recreate_surface {
                let instance =
                    wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());

                *surface = instance
                    .create_surface(self.window.platform_handle())
                    .unwrap();
            }

            surface.configure(device, surface_config);
            return Ok(());
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        let mut encoder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });

        {
            let mut rpass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
                multiview_mask: None,
            });

            rpass.set_pipeline(pipeline);
            rpass.draw(0..3, 0..1);
        }

        queue.submit(Some(encoder.finish()));
        queue.present(frame);

        Ok(())
    }

    fn on_event(&self, event: baseview::Event) -> EventStatus {
        if let baseview::Event::Window(baseview::WindowEvent::Focused) = &event {
            self.redraw_requested.store(true, Ordering::Relaxed);
        }

        baseview::EventStatus::Captured
    }

    fn resized(&self, new_size: baseview::WindowSize) -> Result<(), HandlerError> {
        self.shared.scale_factor.store(new_size.scale_factor);
        self.shared.logical_size.store(new_size.logical.cast());

        self.configure_surface(new_size.physical.width, new_size.physical.height);

        self.redraw_requested.store(true, Ordering::Relaxed);

        Ok(())
    }
}

// ---------------------------------------------------------------------------------------------------
// Editor + handle (nice-plug integration)

pub struct TunerEditor {
    params: Arc<TunerParams>,
    shared: Arc<TunerSharedState>,
}

impl Editor for TunerEditor {
    type Handle = TunerEditorHandle;

    fn spawn(
        &self,
        parent: Option<ParentWindowHandle>,
        wait_for_parent: bool,
        fallback_scale_factor: Option<f64>,
        gui_context: GuiContext,
        host: Option<HostMethods>,
    ) -> Result<SpawnedEditor<Self::Handle>, Box<dyn Error>> {
        let params = Arc::clone(&self.params);
        let shared = Arc::clone(&self.shared);

        let redraw_requested = Arc::new(AtomicBool::new(true));
        let redraw_requested_2 = Arc::clone(&redraw_requested);

        // The host is a re-implementation of baseview's `HostCallbacks` /
        // `HostMainThreadCaller` to keep `baseview` out of nice-plug-core.
        // Adapt it to baseview's traits (same adapter as byo_gui_wgpu v0.4.2).
        let host = {
            struct HostCallbackAdapter {
                host: Box<dyn nice_plug::editor::HostCallbacks>,
            }
            impl baseview::host::HostCallbacks for HostCallbackAdapter {
                fn request_resize(
                    &mut self,
                    new_size: baseview::WindowSize,
                ) -> Result<(), HandlerError> {
                    self.host
                        .request_resize(new_size.physical.into(), new_size.scale_factor)
                        .map_err(HandlerError::from_boxed)
                }

                fn destroyed(&mut self) {
                    self.host.destroyed();
                }
            }
            struct HostMainThreadCallerAdapter {
                host: Box<dyn nice_plug::editor::HostMainThreadCaller>,
            }
            impl baseview::host::HostMainThreadCaller for HostMainThreadCallerAdapter {
                fn call_main_thread(&mut self) {
                    self.host.call_main_thread();
                }
            }
            host.map(|host| {
                baseview::host::Host::new()
                    .with_callbacks(HostCallbackAdapter {
                        host: host.callbacks,
                    })
                    .with_main_thread(HostMainThreadCallerAdapter {
                        host: host.main_thread_caller,
                    })
            })
        };

        let window = baseview::Window::create_with_host(
            WindowSettings::new()
                .with_title("Nice Tuner")
                .with_size(self.shared.logical_size())
                .with_min_size::<LogicalSize<f32>>(Some(MIN_SIZE))
                .with_resizable(RESIZE_HINT.can_resize)
                .with_parent(parent.as_ref())
                .with_wait_for_parent(wait_for_parent)
                .with_fallback_scale_factor(fallback_scale_factor),
            move |window: WindowContext| -> Result<TunerWindow, HandlerError> {
                TunerWindow::new(window, gui_context, params, shared, redraw_requested_2)
            },
            host,
        )?;

        self.shared.open.store(true, Ordering::Release);

        Ok(SpawnedEditor {
            handle: TunerEditorHandle {
                shared: Arc::clone(&self.shared),
                redraw_requested,
            },
            window,
        })
    }

    fn size(&self) -> NativeSize<u32> {
        NativeSize::from_size(
            self.shared.logical_size().into(),
            self.shared.scale_factor(),
        )
    }

    fn resize_hint(&self) -> ResizeHint {
        RESIZE_HINT
    }
}

/// A handle to a spawned instance of our editor.
pub struct TunerEditorHandle {
    shared: Arc<TunerSharedState>,
    redraw_requested: Arc<AtomicBool>,
}

impl EditorHandle for TunerEditorHandle {
    type Window = baseview::Window;
    type Error = baseview::Error;

    fn run_until_closed(window: Self::Window) -> Result<(), Self::Error> {
        window.run_until_closed()
    }

    fn set_parent(
        &self,
        parent: ParentWindowHandle,
        window: &Self::Window,
    ) -> Result<(), Self::Error> {
        window.set_parent(&parent)
    }

    fn show(&self, window: &Self::Window) -> Result<(), Self::Error> {
        let res = window.show();

        // baseview quirk: `WindowEvent::Focused` does not always fire after
        // `show()` (see byo_gui_wgpu v0.4.2) — force a redraw manually.
        self.redraw_requested.store(true, Ordering::Relaxed);

        res
    }

    fn hide(&self, window: &Self::Window) -> Result<(), Self::Error> {
        window.hide()
    }

    fn host_main_thread_callback(&self, window: &Self::Window) {
        window.host_main_thread_callback();
    }

    fn set_size(
        &self,
        new_size: NativeSize<u32>,
        window: &Self::Window,
    ) -> Result<(), Self::Error> {
        window.resize(new_size)
    }

    fn set_fallback_scale_factor(
        &self,
        scale_factor: f64,
        window: &Self::Window,
    ) -> Result<(), Self::Error> {
        window.suggest_fallback_scale_factor(scale_factor)
    }

    /// Return the closest supported size.
    fn adjust_size(
        &self,
        new_size: NativeSize<u32>,
        window: &Self::Window,
    ) -> Option<NativeSize<u32>> {
        let current_size = window.size();
        Some(RESIZE_HINT.adjust_size(
            new_size,
            NativeSize::from_logical_or_physical(current_size.logical, current_size.physical),
            current_size.scale_factor,
        ))
    }

    fn state_changed(&self) {
        self.redraw_requested.store(true, Ordering::Relaxed);
    }

    fn param_value_changed(&self, _id: &str, _normalized_value: f32) {
        self.redraw_requested.store(true, Ordering::Relaxed);
    }

    fn param_modulation_changed(&self, _id: &str, _modulation_offset: f32) {
        self.redraw_requested.store(true, Ordering::Relaxed);
    }
}

impl Drop for TunerEditorHandle {
    fn drop(&mut self) {
        self.shared.open.store(false, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------------------------------
// Plugin

/// The tuner plugin. Audio is passed through; the value it adds is analysis.
pub struct TunerPlugin {
    params: Arc<TunerParams>,
    shared: Arc<TunerSharedState>,
    detector: Option<pitch::PitchDetector>,
}

#[derive(Params)]
pub struct TunerParams {
    /// A4 concert pitch reference, in Hz.
    #[id = "a4_ref"]
    pub a4_ref: FloatParam,
    /// Input gain feeding the detector only (audio passthrough is unaffected).
    #[id = "gate"]
    pub gate_db: FloatParam,
}

impl Default for TunerPlugin {
    fn default() -> Self {
        Self {
            params: Arc::new(TunerParams::default()),
            shared: Arc::new(TunerSharedState::new()),
            detector: None,
        }
    }
}

impl Default for TunerParams {
    fn default() -> Self {
        Self {
            a4_ref: FloatParam::new(
                "A4 Reference",
                440.0,
                FloatRange::Linear {
                    min: 415.0,
                    max: 466.0,
                },
            )
            .with_unit(" Hz")
            .with_step_size(0.1),
            gate_db: FloatParam::new(
                "Gate",
                -60.0,
                FloatRange::Linear {
                    min: -80.0,
                    max: -20.0,
                },
            )
            .with_unit(" dB")
            .with_value_to_string(formatters::v2s_f32_gain_to_db(1))
            .with_string_to_value(formatters::s2v_f32_gain_to_db()),
        }
    }
}

impl Plugin for TunerPlugin {
    const NAME: &'static str = "Nice Tuner";
    const VENDOR: &'static str = "nice-tuner developers";
    const URL: &'static str = "https://example.invalid/nice-tuner";
    const EMAIL: &'static str = "";

    const VERSION: &'static str = env!("CARGO_PKG_VERSION");

    const AUDIO_IO_LAYOUTS: &'static [AudioIOLayout] = &[
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(2),
            main_output_channels: NonZeroU32::new(2),
            ..AudioIOLayout::const_default()
        },
        AudioIOLayout {
            main_input_channels: NonZeroU32::new(1),
            main_output_channels: NonZeroU32::new(1),
            ..AudioIOLayout::const_default()
        },
    ];

    type Editor = TunerEditor;
    type SysExMessage = ();
    type BackgroundTask = ();

    fn params(&self) -> Arc<dyn Params> {
        self.params.clone()
    }

    fn editor(&mut self, _async_executor: AsyncExecutor<Self>) -> Option<Self::Editor> {
        Some(TunerEditor {
            params: Arc::clone(&self.params),
            shared: Arc::clone(&self.shared),
        })
    }

    fn activate(
        &mut self,
        _audio_io_layout: &AudioIOLayout,
        buffer_config: &BufferConfig,
        _context: &mut impl ActivateContext<Self>,
    ) -> bool {
        self.detector = Some(pitch::PitchDetector::new(pitch::Config {
            sample_rate: buffer_config.sample_rate as f64,
            min_freq: 30.0,   // 5-string bass B0
            max_freq: 2100.0, // harmonics up past the guitar's E7 territory
            ..Default::default()
        }));
        true
    }

    fn process(
        &mut self,
        buffer: &mut Buffer,
        _aux: &mut AuxiliaryBuffers,
        _context: &mut impl ProcessContext<Self>,
    ) -> ProcessStatus {
        // Param mirrors for the GUI thread (cheap, once per block).
        self.shared
            .a4_ref_bits
            .store((self.params.a4_ref.value() as f64).to_bits(), Ordering::Relaxed);
        self.shared.gate_db.store(self.params.gate_db.value());

        if let Some(detector) = self.detector.as_mut() {
            if self.shared.is_open() {
                // Buffer is in-place (the framework copied host input into it);
                // we never write to it — passthrough — and only read it here.
                // Stack-only downmix, allocation-free (assert_process_allocs
                // guards this in debug builds).
                const CHUNK: usize = 4096;
                let mut mono = [0f32; CHUNK];
                for (_start, blk) in buffer.iter_blocks(CHUNK) {
                    let n = blk.samples().min(CHUNK);
                    let nch = blk.channels().max(1) as f32;
                    for i in 0..n {
                        let mut acc = 0.0f32;
                        for c in 0..blk.channels() {
                            if let Some(ch) = blk.get(c) {
                                acc += ch[i];
                            }
                        }
                        mono[i] = acc / nch;
                    }
                    if let Some(d) = detector.push_samples(&mono[..n]) {
                        self.shared.freq_hz.store(d.freq_hz);
                        self.shared.clarity.store(d.clarity);
                        self.shared.rms.store(d.rms);
                        self.shared
                            .last_detection_ms
                            .store(now_epoch_ms(), Ordering::Relaxed);
                    }
                }
            } else {
                detector.reset();
            }
        }

        ProcessStatus::Normal
    }
}

fn now_epoch_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

impl ClapPlugin for TunerPlugin {
    const CLAP_ID: &'static str = "com.nice-tuner.nice-tuner";
    const CLAP_DESCRIPTION: Option<&'static str> =
        Some("A stringed-instrument chromatic tuner");
    const CLAP_MANUAL_URL: Option<&'static str> = None;
    const CLAP_SUPPORT_URL: Option<&'static str> = None;
    const CLAP_FEATURES: &'static [ClapFeature] = &[
        ClapFeature::AudioEffect,
        ClapFeature::NoteDetector,
        ClapFeature::Stereo,
        ClapFeature::Mono,
        ClapFeature::Utility,
    ];
}

impl Vst3Plugin for TunerPlugin {
    const VST3_CLASS_ID: [u8; 16] = *b"NiceTunerGtrTnr1";
    const VST3_SUBCATEGORIES: &'static [Vst3SubCategory] =
        &[Vst3SubCategory::Fx, Vst3SubCategory::Tools];
}

nice_export_clap!(TunerPlugin);
nice_export_vst3!(TunerPlugin);
