//! Nice Tuner — a stringed-instrument tuner plugin.
//!
//! Host plumbing follows nice-plug's `byo_gui_wgpu` example (tag v0.4.2):
//! a `baseview` child window embedded in the host, rendered with wgpu.
//! The triangle placeholder renderer is replaced by the frost-derived
//! embedded engine in M3/M4 (see docs/VIABILITY.md).

use baseview::{HandlerError, WindowContext, WindowSettings};
use crossbeam::atomic::AtomicCell;
use nice_plug::context::gui::GuiContext;
use nice_plug::editor::dpi::{LogicalSize, NativeSize};
use nice_plug::editor::{EditorHandle, HostMethods, SpawnedEditor};
use nice_plug::prelude::*;
use std::error::Error;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};
use std::sync::atomic::AtomicU64;

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
// The embedded window: frost renderer hosted by baseview (src/gui.rs),
// tuner scene + UI logic in src/tuner_ui.rs.
mod gui;
mod tuner_ui;

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
            move |window: WindowContext| -> Result<gui::TunerWindow, HandlerError> {
                gui::TunerWindow::new(window, gui_context, params, shared, redraw_requested_2)
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
