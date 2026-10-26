//! STAGING (M4/M5): the tuner UI as a frost `Process`.
//! Becomes `src/tuner_ui.rs` at integration. Design: docs/SCENE.md.
//!
//! Layout (logical px, origin center, y up), designed for 640×400 and scaled
//! by `fit = min(w/640, h/400)`:
//!
//! ```text
//!        "A2"            note,  size 96, y +95
//!   ├───────┼────┤        gauge ticks at y +25, x = cents·2.4 (±120 px)
//!        ▲              needle line at x = smoothed cents·2.4
//!       +4.1¢            cents, size 28, y −5
//!   440.0 Hz · A4=440    freq + ref, size 22 dim, y −45
//!   ▂▃▅ level ▅▃▂        dBFS meter, y −150, halfwidth 240 (0 to −48 dB)
//! ```

use crate::TunerSharedState;
use crate::settings::Settings;
use frost::{Color, Context, Process, Scene, SceneNode, Shape};
use pitch::freq_to_note;
use std::sync::{Arc, atomic::Ordering};

// Clarity gating lives in the detector (Config::min_clarity, audio thread);
// the UI only freshness-gates + level-gates (see process()).
/// cents → px on the gauge (±50 ¢ spans ±120 px)
/// Pixels per cent on the needle gauge. The ±50¢ sweep covers
/// ±50 × this in fit units (±240 at 4.8, inside the ±320 half-width);
/// the ±3¢ green band therefore draws ±14.4 units wide.
const CENTS_PX: f32 = 4.8;
/// |cents| inside which we call it in tune, when latched
/// Half-width of the in-tune band, in cents — drives both the green zone
/// behind the needle and the color lock. Strict on purpose; the band's
/// *pixel* width comes from CENTS_PX, so widening the draw never widens
/// the verdict.
const IN_TUNE_CENTS: f64 = 3.0;
/// ms of continuous same-note stability before latching
const LATCH_MS: f32 = 400.0;

const fn rgb(r: u8, g: u8, b: u8) -> Color {
    Color {
        r: r as f32 / 255.0,
        g: g as f32 / 255.0,
        b: b as f32 / 255.0,
        a: 1.0,
    }
}
const BG: Color = rgb(0x17, 0x12, 0x26); // deep indigo, frost-style
const AMBER: Color = rgb(0xe8, 0xa3, 0x3d);
const GREEN: Color = rgb(0x4a, 0xde, 0x80);
const DIM: Color = rgb(0x6b, 0x72, 0x80);
const TRACK: Color = rgb(0x2a, 0x24, 0x3d);
/// Dropdown well: distinctly darker than the page so the open list reads as
/// a cut-in surface; a near-page tint was invisible against the background.
const MENU_BG: Color = rgb(0x0d, 0x0a, 0x16);
const HOVER: Color = rgb(0x3a, 0x33, 0x4f);
const FG: Color = rgb(0xe6, 0xe6, 0xea);

/// Static chrome: background and gauge tick marks. Rebuilt never;
/// everything that moves is drawn immediately in [`TunerUi::process`].
pub fn static_scene() -> Scene {
    let mut ticks = Vec::new();
    for cents in (-50..=50).step_by(10) {
        let x = cents as f32 * CENTS_PX;
        let long = cents % 50 == 0;
        let half = if long { 9.0 } else { 4.5 };
        ticks.push(Box::new(SceneNode {
            shape: Some(Shape::Rectangle {
                center: [x, -2.0], // on the needle band (moved from 25)
                extent: [0.75, half],
                color: if long { DIM } else { TRACK },
            }),
            ..Default::default()
        }));
    }
    Scene::new(SceneNode {
        shape: Some(Shape::Background { color: BG }),
        children: ticks.into_iter().collect(),
        ..Default::default()
    })
}

/// The UI state machine + immediate-mode drawing, run inside frost's frame.
pub struct TunerUi {
    shared: Arc<TunerSharedState>,
    font: Arc<[u8]>,
    // smoothed display state
    needle_cents: f32, // EMA'd cents shown on the gauge
    latch_midi: i16,   // -1 = nothing latched
    /// Meter: current input level (bar), RMS-based, EMA-smoothed every
    /// frame (fast attack, moderate release).
    meter_level_db: f32,
    /// Meter: max peak (marker). Sticks for 1 s, timer resets whenever a
    /// frame beats it; afterwards eases toward the current level.
    meter_peak_db: f32,
    /// Wall-clock ms until the current peak stops holding (fps-independent).
    peak_hold_until_ms: u64,
    latch_name: &'static str,
    latch_oct: i8,
    cents_shown: f64,
    freq_shown: f64,
    stable_ms: f32,
    green_t: f32, // 0 = amber, 1 = green (lerped) // decayed meter fill, 0..1
    // ---- settings page (standalone-only UI; inert state elsewhere) ----
    page: Page,
    prev_mouse_down: bool,
    settings: Settings,
    settings_loaded: bool,
    open_dropdown: Option<Dropdown>,
    list_scroll: f32,
    backend_opts: Vec<(String, String)>,
    input_opts: Vec<String>,
    output_opts: Vec<String>,
    rate_opts: Vec<String>,
    buffer_opts: Vec<String>,
    save_note: Option<String>,
    // ---- analysis. This is the UI thread (nice-plug's main thread, pumped
    // by frost frames): detection runs HERE, never in the audio callback.
    // The consumer half is borrowed from the shared state on open and
    // returned on close (Drop); TunerPlugin owns the producer half. ----
    rb: Option<rtrb::Consumer<f32>>,
    detector: Option<pitch::PitchDetector>,
    det_rate: f64,
}

/// Return the analysis feed so the next editor session can take it.
/// Main thread: frost tears the UI down inside the window handler.
impl Drop for TunerUi {
    fn drop(&mut self) {
        if let Some(rb) = self.rb.take() {
            self.shared.put_consumer(rb);
        }
    }
}

impl TunerUi {
    pub fn new(shared: Arc<TunerSharedState>, font: Arc<[u8]>) -> Self {
        let mut rb = shared.take_consumer();
        // A reopened editor analyzes fresh audio, not last session's backlog.
        if let Some(rb) = rb.as_mut() {
            while rb.pop().is_ok() {}
        }
        Self {
            shared,
            font,
            needle_cents: 0.0,
            latch_midi: -1,
            meter_level_db: -120.0,
            meter_peak_db: -120.0,
            peak_hold_until_ms: 0,
            latch_name: "–",
            latch_oct: 0,
            cents_shown: 0.0,
            freq_shown: 0.0,
            stable_ms: 0.0,
            green_t: 0.0,
            page: Page::Main,
            prev_mouse_down: false,
            settings: Settings::default(),
            settings_loaded: false,
            open_dropdown: None,
            list_scroll: 0.0,
            backend_opts: Vec::new(),
            input_opts: Vec::new(),
            output_opts: Vec::new(),
            rate_opts: Vec::new(),
            buffer_opts: Vec::new(),
            save_note: None,
            rb,
            detector: None,
            det_rate: 0.0,
        }
    }

    /// Drain the RT feed and run pitch detection — on this (UI) thread, so
    /// the bursty window search can at worst delay the needle, never glitch
    /// audio. Called once per frost frame, including while the settings page
    /// is up (state keeps warming behind it).
    fn analyze(&mut self) {
        let Some(rb) = self.rb.as_mut() else { return };
        let rate = self.shared.sample_rate_hz.load();
        if rate > 0.0 && (self.detector.is_none() || self.det_rate != rate) {
            self.detector = Some(pitch::PitchDetector::new(pitch::Config {
                sample_rate: rate,
                min_freq: 30.0,   // 5-string bass B0
                max_freq: 2100.0, // harmonics up past the guitar's E7 territory
                ..Default::default()
            }));
            self.det_rate = rate;
        }
        let Some(detector) = self.detector.as_mut() else {
            return;
        };
        // Catch-up cap: after a long pump stall, analyze the most recent
        // audio rather than grinding through a stale backlog. Older samples
        // are dropped from the head (Consumer side: legal in SPSC).
        const MAX_DRAIN: usize = 16384; // ~0.34 s
        if rb.slots() > MAX_DRAIN {
            let skip = rb.slots() - MAX_DRAIN;
            for _ in 0..skip {
                let _ = rb.pop();
            }
        }
        const CHUNK: usize = 4096;
        let mut buf = [0f32; CHUNK];
        // Continuous input level for the dB meter: mean square over
        // everything drained this frame, independent of detections (the
        // detector's own rms only moves when a pitch is confident).
        let mut ss: f64 = 0.0;
        let mut n: usize = 0;
        let mut peak: f64 = 0.0;
        let mut drained = false;
        while rb.slots() > 0 {
            let want = rb.slots().min(CHUNK);
            if rb.pop_entire_slice(&mut buf[..want]).is_err() {
                break;
            }
            drained = true;
            for &x in &buf[..want] {
                ss += (x as f64) * (x as f64);
                let ax = if x < 0.0 { -(x as f64) } else { x as f64 };
                if ax > peak {
                    peak = ax;
                }
            }
            n += want;
            if let Some(d) = detector.push_samples(&buf[..want]) {
                self.shared.freq_hz.store(d.freq_hz);
                self.shared.clarity.store(d.clarity);
                self.shared.rms.store(d.rms);
                self.shared
                    .last_detection_ms
                    .store(crate::now_epoch_ms(), Ordering::Relaxed);
            }
        }
        // Two-layer metering, both updated every frame:
        //   bar   — current level (RMS), EMA with fast attack / moderate
        //           release, always chasing the ongoing signal;
        //   peak  — the max of frame peaks, STICKS for 1 s (wall-clock, so
        //           the hold is fps-independent); any frame that beats the
        //           held peak jumps to it and RESETS the 1 s timer; once
        //           expired it eases down toward the current bar level.
        let db_of = |a: f64| -> f32 {
            if a > 1e-7 {
                (20.0 * a.log10()) as f32
            } else {
                -120.0
            }
        };
        let rms_db = if drained && n > 0 {
            db_of((ss / n as f64).sqrt())
        } else {
            -120.0
        };
        let frame_peak_db = db_of(peak);
        {
            let k = if rms_db > self.meter_level_db {
                0.5 // fast attack
            } else {
                0.08 // moderate release
            };
            self.meter_level_db += (rms_db - self.meter_level_db) * k;
        }
        if frame_peak_db >= self.meter_peak_db {
            self.meter_peak_db = frame_peak_db;
            self.peak_hold_until_ms = crate::now_epoch_ms() + 1000;
        } else if crate::now_epoch_ms() >= self.peak_hold_until_ms {
            self.meter_peak_db += (self.meter_level_db - self.meter_peak_db) * 0.06;
        }
    }
}

impl Process for TunerUi {
    fn process(&mut self, ctx: &mut Context, dt: f32) {
        let dt = dt.clamp(0.0, 0.25); // tolerate host pauses
        self.analyze();

        // ---- read shared state (updated by analyze(), above) --------------
        // The detector already applied its clarity gate
        // (push_samples only yields confident windows), so "signal present"
        // here = a *fresh* detection + level above the user gate.
        let freq = self.shared.freq_hz.load();
        let rms = self.shared.rms.load() as f64;
        let a4 = self.shared.a4_ref();
        let gate_db = self.shared.gate_db.load() as f64;
        let rms_db = if rms > 1e-7 {
            20.0 * rms.log10()
        } else {
            -120.0
        };
        let last_ms = self.shared.last_detection_ms.load(Ordering::Relaxed);
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0);
        let fresh = last_ms != 0 && now_ms.saturating_sub(last_ms) < 500;
        let signal = fresh && rms_db >= gate_db;

        // ---- note latch / stability ---------------------------------------
        if let Some(note) = signal.then(|| freq_to_note(freq, a4)) {
            if note.midi == self.latch_midi {
                self.stable_ms += dt * 1000.0;
            } else {
                self.latch_midi = note.midi;
                self.latch_name = note.name;
                self.latch_oct = note.octave;
                self.stable_ms = 0.0;
            }
            self.cents_shown = note.cents;
            self.freq_shown = freq;
        } else {
            self.stable_ms = (self.stable_ms - dt * 500.0).max(0.0);
        }
        let stable = self.stable_ms >= LATCH_MS;
        let in_tune = signal && stable && self.cents_shown.abs() <= IN_TUNE_CENTS;

        // ---- color lerp amber→green (≈200 ms) ------------------------------
        let target = if in_tune { 1.0 } else { 0.0 };
        let k = 1.0 - (-dt / 0.08).exp(); // EMA factor
        self.green_t += (target - self.green_t) * k;
        let main = Color {
            r: AMBER.r + (GREEN.r - AMBER.r) * self.green_t,
            g: AMBER.g + (GREEN.g - AMBER.g) * self.green_t,
            b: AMBER.b + (GREEN.b - AMBER.b) * self.green_t,
            a: 1.0,
        };
        let main = if signal { main } else { DIM };

        // ---- needle: EMA toward clamped cents, spring to 0 when silent ----
        let wanted = if signal {
            (self.cents_shown.clamp(-50.0, 50.0) as f32) * 1.0
        } else {
            0.0
        };
        let nk = if signal {
            1.0 - (-dt / 0.08).exp()
        } else {
            1.0 - (-dt / 0.15).exp()
        };
        self.needle_cents += (wanted - self.needle_cents) * nk;

        // ---- draw (Context derefs to Canvas) -------------------------------
        let (w, h) = ctx.size();
        let fit = (w / 640.0).min(h / 400.0).max(0.35);

        // Mouse edge + page routing. The settings page exists only in the
        // standalone app: inside a host, the host owns the audio system.
        let down = ctx.mouse_button_down(frost::MouseButton::Left);
        let clicked = down && !self.prev_mouse_down;
        self.prev_mouse_down = down;
        if self.page == Page::Settings {
            self.process_settings_page(ctx, clicked);
            return;
        }
        if crate::STANDALONE_MODE.load(Ordering::Relaxed) {
            let gx = w / 2.0 - 62.0 * fit;
            let gy = h / 2.0 - 18.0 * fit;
            let hov = point_in(ctx.mouse_position(), gx, gy, 50.0 * fit, 11.0 * fit);
            ctx.rectangle(
                gx,
                gy,
                50.0 * fit,
                11.0 * fit,
                if hov { HOVER } else { TRACK },
                0.4,
            );
            ctx.text(
                gx,
                gy,
                &self.font,
                "SETTINGS",
                19.8 * fit,
                650.0,
                if hov { AMBER } else { DIM },
                0.41,
            );
            if hov && clicked {
                self.open_settings_page();
                return;
            }
        }
        if !crate::STANDALONE_MODE.load(Ordering::Relaxed) {
            // Inside a host there is no settings page on purpose: the host
            // owns the device, a per-plugin device picker would lie.
            ctx.text(
                -w / 2.0 + 78.0 * fit,
                h / 2.0 - 16.0 * fit,
                &self.font,
                "audio: host-managed",
                18.0 * fit,
                550.0,
                DIM,
                0.4,
            );
        }
        let font = &self.font;
        let x = self.needle_cents * CENTS_PX * fit;

        // in-tune zone behind the needle
        ctx.rectangle(
            0.0,
            -2.0 * fit,
            IN_TUNE_CENTS as f32 * CENTS_PX * fit,
            9.0 * fit,
            Color { a: 0.15, ..GREEN },
            0.05,
        );
        // needle
        ctx.line(
            x,
            (-2.0 - 16.0) * fit,
            x,
            (-2.0 + 16.0) * fit,
            main,
            2.5 * fit,
            0.2,
        );
        // note label
        // Nothing latched yet: a plain dash — not the octave-0 placeholder
        // that used to leak out as "–0".
        let label = if self.latch_midi < 0 {
            "–".to_string()
        } else {
            format!("{}{}", self.latch_name, self.latch_oct)
        };
        ctx.text(0.0, 95.0 * fit, font, label, 86.4 * fit, 700.0, main, 0.2);
        // cents readout (one decimal, sign fixed for stable layout)
        // No signal, no reading: a bare dash, not a confident "+0.0¢".
        let cents_s = if signal {
            let cents = self.cents_shown;
            format!(
                "{}{:.1}¢",
                if cents >= 0.0 { "+" } else { "−" },
                cents.abs()
            )
        } else {
            "—".to_string()
        };
        ctx.text(
            0.0,
            -52.0 * fit,
            font,
            cents_s,
            50.4 * fit,
            600.0,
            main,
            0.2,
        );
        // frequency + reference
        let freq_s = if signal {
            format!("{:.1} Hz  ·  A4={:.0}", self.freq_shown, a4)
        } else {
            format!("———  ·  A4={:.0}", a4)
        };
        ctx.text(0.0, -104.0 * fit, font, freq_s, 39.6 * fit, 550.0, DIM, 0.2);
        // ---- input meter with a real dBFS scale ----------------------------
        // 0 dBFS right, -48 left, ticks every 6 dB, as wide as the needle
        // sweep. Driven by meter_db (continuous), so it shows input level
        // always — silent room reads silence, no detection required.
        {
            let half = 240.0 * fit;
            let y = -150.0 * fit;
            ctx.rectangle(0.0, y, half, 2.5 * fit, Color { a: 0.5, ..TRACK }, 0.05);
            // Bar: current level.
            let m = self.meter_level_db.clamp(-48.0, 0.0);
            let xv = half * (1.0 + m / 24.0);
            if xv > -half + 0.5 {
                ctx.rectangle(
                    (-half + xv) / 2.0,
                    y,
                    (xv + half) / 2.0,
                    2.5 * fit,
                    // Same amber→green state as note and needle (main lerps).
                    main,
                    0.07,
                );
            }
            // Peak marker: a thin bright line at the held peak, above the
            // bar in z; sits at the bar while rising, sticks after it falls.
            let p = self.meter_peak_db.clamp(-48.0, 0.0);
            let px = half * (1.0 + p / 24.0);
            if p > -47.5 {
                ctx.rectangle(px, y, 1.5 * fit, 6.0 * fit, Color { a: 0.9, ..FG }, 0.08);
            }
        }
    }
}

// ---- input mapping (baseview 0.3.3 / keyboard_types 0.8 → frost) ----------

pub fn map_key(key: &keyboard_types::Key) -> Option<frost::KeyCode> {
    use keyboard_types::{Key, NamedKey};
    match key {
        Key::Character(s) => match s.as_str() {
            "a" | "A" => Some(frost::KeyCode::KeyA),
            "b" | "B" => Some(frost::KeyCode::KeyB),
            "c" | "C" => Some(frost::KeyCode::KeyC),
            "d" | "D" => Some(frost::KeyCode::KeyD),
            "e" | "E" => Some(frost::KeyCode::KeyE),
            "f" | "F" => Some(frost::KeyCode::KeyF),
            "g" | "G" => Some(frost::KeyCode::KeyG),
            "h" | "H" => Some(frost::KeyCode::KeyH),
            "i" | "I" => Some(frost::KeyCode::KeyI),
            "j" | "J" => Some(frost::KeyCode::KeyJ),
            "k" | "K" => Some(frost::KeyCode::KeyK),
            "l" | "L" => Some(frost::KeyCode::KeyL),
            "m" | "M" => Some(frost::KeyCode::KeyM),
            "n" | "N" => Some(frost::KeyCode::KeyN),
            "o" | "O" => Some(frost::KeyCode::KeyO),
            "p" | "P" => Some(frost::KeyCode::KeyP),
            "q" | "Q" => Some(frost::KeyCode::KeyQ),
            "r" | "R" => Some(frost::KeyCode::KeyR),
            "s" | "S" => Some(frost::KeyCode::KeyS),
            "t" | "T" => Some(frost::KeyCode::KeyT),
            "u" | "U" => Some(frost::KeyCode::KeyU),
            "v" | "V" => Some(frost::KeyCode::KeyV),
            "w" | "W" => Some(frost::KeyCode::KeyW),
            "x" | "X" => Some(frost::KeyCode::KeyX),
            "y" | "Y" => Some(frost::KeyCode::KeyY),
            "z" | "Z" => Some(frost::KeyCode::KeyZ),
            "0" => Some(frost::KeyCode::Digit0),
            "1" => Some(frost::KeyCode::Digit1),
            "2" => Some(frost::KeyCode::Digit2),
            "3" => Some(frost::KeyCode::Digit3),
            "4" => Some(frost::KeyCode::Digit4),
            "5" => Some(frost::KeyCode::Digit5),
            "6" => Some(frost::KeyCode::Digit6),
            "7" => Some(frost::KeyCode::Digit7),
            "8" => Some(frost::KeyCode::Digit8),
            "9" => Some(frost::KeyCode::Digit9),
            " " => Some(frost::KeyCode::Space),
            _ => None,
        },
        Key::Named(named) => Some(match named {
            NamedKey::Escape => frost::KeyCode::Escape,
            NamedKey::Enter => frost::KeyCode::Enter,
            NamedKey::Tab => frost::KeyCode::Tab,
            NamedKey::Backspace => frost::KeyCode::Backspace,
            NamedKey::Delete => frost::KeyCode::Delete,
            NamedKey::ArrowLeft => frost::KeyCode::ArrowLeft,
            NamedKey::ArrowRight => frost::KeyCode::ArrowRight,
            NamedKey::ArrowUp => frost::KeyCode::ArrowUp,
            NamedKey::ArrowDown => frost::KeyCode::ArrowDown,
            _ => return None,
        }),
    }
}

pub fn map_mouse(button: baseview::MouseButton) -> Option<frost::MouseButton> {
    Some(match button {
        baseview::MouseButton::Left => frost::MouseButton::Left,
        baseview::MouseButton::Right => frost::MouseButton::Right,
        baseview::MouseButton::Middle => frost::MouseButton::Middle,
        _ => return None,
    })
}

// ============================================================================
// Settings page — backend/device/buffer configuration for the standalone app.
// Hand-rolled widgets on frost's immediate canvas (rectangles + text + mouse
// edges); the static scene underneath is page-agnostic.
// ============================================================================

#[derive(Clone, Copy, PartialEq)]
enum Page {
    Main,
    Settings,
}

#[derive(Clone, Copy, PartialEq)]
enum Dropdown {
    Backend,
    Rate,
    Buffer,
    Input,
    Output,
    /// Plugin-param pickers. In a host the host renders these controls;
    /// standalone has none, so the panel plays that role — same atomics the
    /// `process()` mirror feeds in host mode (see lib.rs).
    A4,
    Gate,
}

/// Sensible tuning-standards list; 440 sits mid-list.
const A4_OPTIONS: [f32; 9] = [
    432.0, 438.0, 439.0, 440.0, 441.0, 442.0, 443.0, 444.0, 446.0,
];
const GATE_OPTIONS: [f32; 7] = [-80.0, -70.0, -60.0, -50.0, -40.0, -30.0, -20.0];

fn point_in(p: Option<[f32; 2]>, cx: f32, cy: f32, hw: f32, hh: f32) -> bool {
    p.is_some_and(|[x, y]| (x - cx).abs() <= hw && (y - cy).abs() <= hh)
}

fn clip(s: &str, n: usize) -> String {
    if s.chars().count() <= n {
        s.to_owned()
    } else {
        format!("{}…", s.chars().take(n - 1).collect::<String>())
    }
}

fn fmt_rate(r: f32) -> String {
    if r.fract() == 0.0 {
        format!("{} Hz", r as u32)
    } else {
        format!("{r:.1} Hz")
    }
}

const VISIBLE_ITEMS: usize = 8;

impl TunerUi {
    fn open_settings_page(&mut self) {
        if !self.settings_loaded {
            let (settings, warning) = Settings::load();
            self.settings = settings;
            self.save_note = warning;
            #[cfg(feature = "standalone")]
            {
                self.backend_opts = crate::devices::available_backends()
                    .into_iter()
                    .map(|b| (b.id, b.label))
                    .collect();
            }
            self.refresh_devices();
            self.settings_loaded = true;
        }
        self.open_dropdown = None;
        self.page = Page::Settings;
    }

    fn backend_key(&self) -> &str {
        self.settings.backend.as_deref().unwrap_or("auto")
    }

    /// (Re-)enumerate devices for the selected backend; selections that the
    /// new backend cannot offer fall back to "(system default)".
    fn refresh_devices(&mut self) {
        // Owned: the refresh below mutates self while enumerating from b.
        let b = self.backend_key().to_owned();
        #[cfg(not(feature = "standalone"))]
        let _ = &b; // plugin build has no devices module to feed
        #[cfg(feature = "standalone")]
        {
            self.input_opts = crate::devices::list_input_devices(&b);
            self.output_opts = crate::devices::list_output_devices(&b);
        }
        if let Some(d) = &self.settings.input_device
            && !self.input_opts.contains(d)
        {
            self.settings.input_device = None;
        }
        if let Some(d) = &self.settings.output_device
            && !self.output_opts.contains(d)
        {
            self.settings.output_device = None;
        }
        self.refresh_stream_options();
    }

    fn refresh_stream_options(&mut self) {
        #[cfg(feature = "standalone")]
        {
            let backend = self.backend_key().to_owned();
            let input = self.settings.input_device.clone();
            let output = self.settings.output_device.clone();
            let (rates, buffers) =
                crate::devices::stream_options(&backend, input.as_deref(), output.as_deref());
            self.rate_opts = rates.iter().map(|r| fmt_rate(*r)).collect();
            self.buffer_opts = buffers.iter().map(|b| b.to_string()).collect();
        }
    }

    fn dropdown_items(&self, dd: Dropdown) -> Vec<String> {
        let with_default = |v: &Vec<String>| {
            std::iter::once("(default)".to_owned())
                .chain(v.iter().cloned())
                .collect::<Vec<_>>()
        };
        match dd {
            Dropdown::Backend => self.backend_opts.iter().map(|(_, l)| l.clone()).collect(),
            Dropdown::Input => with_default(&self.input_opts),
            Dropdown::Output => with_default(&self.output_opts),
            Dropdown::Rate => with_default(&self.rate_opts),
            Dropdown::Buffer => with_default(&self.buffer_opts),
            Dropdown::A4 => A4_OPTIONS.iter().map(|v| format!("{v:.0} Hz")).collect(),
            Dropdown::Gate => GATE_OPTIONS.iter().map(|v| format!("{v:.0} dB")).collect(),
        }
    }

    fn selected_index(&self, dd: Dropdown) -> usize {
        let pick_opt = |list: &Vec<String>, saved: &Option<String>| {
            saved
                .as_ref()
                .and_then(|d| list.iter().position(|x| x == d))
                .map_or(0, |i| i + 1)
        };
        match dd {
            Dropdown::Backend => self
                .backend_opts
                .iter()
                .position(|(id, _)| id == self.backend_key())
                .unwrap_or(0),
            Dropdown::Input => pick_opt(&self.input_opts, &self.settings.input_device),
            Dropdown::Output => pick_opt(&self.output_opts, &self.settings.output_device),
            Dropdown::Rate => self
                .settings
                .sample_rate
                .and_then(|r| self.rate_opts.iter().position(|s| *s == fmt_rate(r)))
                .map_or(0, |i| i + 1),
            Dropdown::Buffer => self
                .settings
                .period_size
                .and_then(|p| self.buffer_opts.iter().position(|s| *s == p.to_string()))
                .map_or(0, |i| i + 1),
            Dropdown::A4 => A4_OPTIONS
                .iter()
                .position(|v| (*v as f64 - self.shared.a4_ref()).abs() < 0.05)
                .unwrap_or(0),
            Dropdown::Gate => GATE_OPTIONS
                .iter()
                .position(|v| (*v - self.shared.gate_db.load()).abs() < 0.05)
                .unwrap_or(0),
        }
    }

    fn pick(&mut self, dd: Dropdown, idx: usize) {
        match dd {
            Dropdown::Backend => {
                if let Some((id, _)) = self.backend_opts.get(idx) {
                    let id = id.clone();
                    self.settings.backend = if id == "auto" { None } else { Some(id) };
                    self.refresh_devices();
                }
            }
            Dropdown::Input => {
                self.settings.input_device = if idx == 0 {
                    None
                } else {
                    self.input_opts.get(idx - 1).cloned()
                };
                self.refresh_stream_options();
            }
            Dropdown::Output => {
                self.settings.output_device = if idx == 0 {
                    None
                } else {
                    self.output_opts.get(idx - 1).cloned()
                };
                // Duplex constraint: the output device narrows the buffer
                // range too — recompute.
                self.refresh_stream_options();
            }
            Dropdown::Rate => {
                self.settings.sample_rate = if idx == 0 {
                    None
                } else {
                    self.rate_opts
                        .get(idx - 1)
                        .and_then(|s| s.split(' ').next())
                        .and_then(|t| t.parse().ok())
                };
            }
            Dropdown::Buffer => {
                self.settings.period_size = if idx == 0 {
                    None
                } else {
                    self.buffer_opts.get(idx - 1).and_then(|s| s.parse().ok())
                };
            }
            // Params are live the moment they are picked (no restart needed);
            // APPLY only decides whether they persist to the next launch.
            Dropdown::A4 => {
                if let Some(v) = A4_OPTIONS.get(idx) {
                    self.shared
                        .a4_ref_bits
                        .store((*v as f64).to_bits(), Ordering::Relaxed);
                    self.settings.a4_ref = Some(*v);
                }
            }
            Dropdown::Gate => {
                if let Some(v) = GATE_OPTIONS.get(idx) {
                    self.shared.gate_db.store(*v);
                    self.settings.gate_db = Some(*v);
                }
            }
        }
        self.open_dropdown = None;
        self.list_scroll = 0.0;
    }

    fn current_value_label(&self, dd: Dropdown) -> String {
        match dd {
            Dropdown::Backend => self
                .backend_opts
                .iter()
                .find(|(id, _)| id == self.backend_key())
                .map(|(_, l)| l.clone())
                .unwrap_or_else(|| self.backend_key().to_owned()),
            Dropdown::Rate => self
                .settings
                .sample_rate
                .map(fmt_rate)
                .unwrap_or_else(|| "(default)".into()),
            Dropdown::Buffer => self
                .settings
                .period_size
                .map(|p| p.to_string())
                .unwrap_or_else(|| "(default)".into()),
            Dropdown::Input => self
                .settings
                .input_device
                .as_deref()
                .map(|d| clip(d, 26))
                .unwrap_or_else(|| "(system default)".into()),
            Dropdown::Output => self
                .settings
                .output_device
                .as_deref()
                .map(|d| clip(d, 26))
                .unwrap_or_else(|| "(system default)".into()),
            Dropdown::A4 => format!("{:.1} Hz", self.shared.a4_ref()),
            Dropdown::Gate => format!("{:.0} dB", self.shared.gate_db.load()),
        }
    }

    fn process_settings_page(&mut self, ctx: &mut Context, clicked: bool) {
        let (w, h) = ctx.size();
        let fit = (w / 720.0).min(h / 440.0).max(0.45);
        let mouse = ctx.mouse_position();
        let font = std::sync::Arc::clone(&self.font);
        let z = 0.4f32;

        const ROWS: [(&str, Dropdown); 7] = [
            ("Backend", Dropdown::Backend),
            ("Sample rate", Dropdown::Rate),
            ("Buffer (frames)", Dropdown::Buffer),
            ("Input device", Dropdown::Input),
            ("Output device", Dropdown::Output),
            ("A4 reference", Dropdown::A4),
            ("Input gate", Dropdown::Gate),
        ];
        // Rows 5+ (A4, gate) are the param block: an extra gap drops them
        // below the audio block instead of running one uniform ladder.
        let row_y = |i: usize| {
            h / 2.0 - 46.0 * fit - i as f32 * 38.0 * fit - if i >= 5 { 26.0 } else { 0.0 } * fit
        };
        let bx = w / 2.0 - 185.0 * fit; // value-button column
        let bhw = 165.0 * fit;
        let bhh = 14.0 * fit;

        // Wheel scrolls the open dropdown.
        if let Some(dd) = self.open_dropdown {
            let n = self.dropdown_items(dd).len();
            let max_scroll = n.saturating_sub(VISIBLE_ITEMS) as f32;
            self.list_scroll = (self.list_scroll + ctx.mouse_wheel() * 0.5).clamp(0.0, max_scroll);
        }

        for (i, (label, dd)) in ROWS.iter().enumerate() {
            let y = row_y(i);
            let open = self.open_dropdown == Some(*dd);
            let hov = point_in(mouse, bx, y, bhw, bhh);
            ctx.text(
                -w / 2.0 + 105.0 * fit,
                y,
                &font,
                *label,
                25.2 * fit,
                550.0,
                DIM,
                z,
            );
            ctx.rectangle(bx, y, bhw, bhh, if open || hov { HOVER } else { TRACK }, z);
            ctx.text(
                bx,
                y,
                &font,
                self.current_value_label(*dd),
                23.4 * fit,
                550.0,
                if hov || open { AMBER } else { FG },
                z + 0.01,
            );
        }

        // Bottom actions.
        let by = -h / 2.0 + 30.0 * fit;
        let back = point_in(mouse, -w / 2.0 + 90.0 * fit, by, 65.0 * fit, 15.0 * fit);
        let apply = point_in(mouse, w / 2.0 - 120.0 * fit, by, 110.0 * fit, 15.0 * fit);
        ctx.rectangle(
            -w / 2.0 + 90.0 * fit,
            by,
            65.0 * fit,
            15.0 * fit,
            if back { HOVER } else { TRACK },
            z,
        );
        ctx.text(
            -w / 2.0 + 90.0 * fit,
            by,
            &font,
            "BACK",
            23.4 * fit,
            650.0,
            if back { AMBER } else { FG },
            z + 0.01,
        );
        ctx.rectangle(
            w / 2.0 - 120.0 * fit,
            by,
            110.0 * fit,
            15.0 * fit,
            if apply { HOVER } else { TRACK },
            z,
        );
        ctx.text(
            w / 2.0 - 120.0 * fit,
            by,
            &font,
            "APPLY & RESTART",
            23.4 * fit,
            700.0,
            if apply { AMBER } else { GREEN },
            z + 0.01,
        );

        match &self.save_note {
            Some(note) => ctx.text(
                0.0,
                by + 34.0 * fit,
                &font,
                clip(note, 90),
                21.6 * fit,
                550.0,
                AMBER,
                z,
            ),
            None => ctx.text(
                0.0,
                by + 34.0 * fit,
                &font,
                format!(
                    "saved to {} · audio: Apply restarts · A4/gate: live, saved on Apply",
                    crate::settings::FILE_NAME,
                ),
                19.8 * fit,
                550.0,
                DIM,
                z,
            ),
        }

        // Open dropdown: drawn last (on top), items hit first.
        if let Some(dd) = self.open_dropdown {
            let items = self.dropdown_items(dd);
            let row = ROWS.iter().position(|(_, d)| *d == dd).unwrap_or(0);
            let ih = 13.0 * fit;
            let step = ih * 2.0 + 5.0 * fit;
            let skip = self.list_scroll as usize;
            let shown = items.len().saturating_sub(skip).min(VISIBLE_ITEMS);
            let slot_y =
                |i: usize| row_y(row) - (i as f32 - self.list_scroll) * step - ih - 8.0 * fit;

            // Backing panel: opaque, padded past the item stack on every
            // side, with a one-unit lighter rim. The list must read as its
            // own surface lifted off the page — not rows floating across the
            // tuner chrome. Drawn under the items (z), over everything else.
            if shown > 0 {
                let pad = 9.0 * fit;
                let top = slot_y(skip) + ih;
                let bottom = slot_y(skip + shown - 1) - ih;
                let py = (top + bottom) * 0.5;
                let ph = (top - bottom) * 0.5 + pad;
                let pw = bhw + pad;
                ctx.rectangle(bx, py, pw + 1.5 * fit, ph + 1.5 * fit, HOVER, 0.44);
                ctx.rectangle(bx, py, pw, ph, MENU_BG, 0.445);
            }

            for (idx, label) in items
                .iter()
                .enumerate()
                .skip(skip)
                .take(VISIBLE_ITEMS)
                .collect::<Vec<_>>()
            {
                let iy = slot_y(idx);
                let sel = self.selected_index(dd) == idx;
                let hov = point_in(mouse, bx, iy, bhw, ih);
                ctx.rectangle(
                    bx,
                    iy,
                    bhw,
                    ih,
                    if hov {
                        HOVER
                    } else if sel {
                        TRACK
                    } else {
                        Color {
                            a: 0.0,
                            r: 0.0,
                            g: 0.0,
                            b: 0.0,
                        }
                    },
                    z + 0.05,
                );
                ctx.text(
                    bx,
                    iy,
                    &font,
                    clip(label, 36),
                    18.7 * fit,
                    if sel { 700.0 } else { 550.0 },
                    if sel { GREEN } else { FG },
                    z + 0.06,
                );
                if clicked && hov {
                    self.pick(dd, idx);
                    return;
                }
            }
        }

        if clicked {
            if self.open_dropdown.is_some() {
                self.open_dropdown = None;
                self.list_scroll = 0.0;
            } else {
                for (i, (_, dd)) in ROWS.iter().enumerate() {
                    if point_in(mouse, bx, row_y(i), bhw, bhh) {
                        self.open_dropdown = Some(*dd);
                        self.list_scroll = 0.0;
                        break;
                    }
                }
                if back {
                    self.page = Page::Main;
                    self.save_note = None;
                }
                if apply {
                    match self.settings.save() {
                        Ok(()) => crate::request_restart(),
                        Err(e) => self.save_note = Some(format!("save failed: {e}")),
                    }
                }
            }
        }
    }
}
