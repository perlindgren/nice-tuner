//! STAGING (M4/M5): the tuner UI as a frost `Process`.
//! Becomes `src/tuner_ui.rs` at integration. Design: docs/SCENE.md.
//!
//! Layout (logical px, origin center, y up), designed for 640×400 and scaled
//! by `fit = min(w/640, h/400)`:
//!
//! ```text
//!        "A2"            note,  size 96, y +95
//!   ├───────┼────┤        gauge ticks on the needle band, asinh scale
//!        ▲              needle at x = cents_to_units(smoothed cents)
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
/// Edge scale factor: ±50 ¢ spans ±240 fit units (the ±320 half-width
/// keeps margins). NOTE: cents→position is NOT linear — see
/// [`cents_to_units`]; CENTS_PX only fixes the outer scale.
const CENTS_PX: f32 = 4.8;
/// Half-width of the gauge, in fit units (50 ¢ × CENTS_PX).
const GAUGE_HALF: f32 = 50.0 * CENTS_PX;
/// Default curvature knee of the gauge scale, in cents (the panel can
/// change it live: `Settings::gauge_knee_cents`).
const GAUGE_KNEE_CENTS: f32 = 5.0;
/// Gauge mapping: sign-preserving asinh with a knee (default ±5 ¢, panel
/// param) — near-linear and readable inside a few cents, progressively
/// compressing beyond it. At the default knee the ±3 ¢ verdict spans
/// ~91 of 480 units, 3.2× the linear size; edges still reach ±50 ¢.
/// asinh over a power curve: finite slope at 0, so needle jitter near
/// perfect pitch stays calm even at steep settings.
fn cents_to_units(cents: f32, knee: f32) -> f32 {
    let k = knee.max(1.0);
    let c = cents.clamp(-50.0, 50.0) / k;
    GAUGE_HALF * c.asinh() / (50.0 / k).asinh()
}
/// Default half-width of the in-tune band and verdict, in cents (the
/// panel can change it live: `Settings::in_tune_cents`). Drives both the
/// green zone behind the needle and the color lock; the band's *pixel*
/// width comes from [`cents_to_units`], so scale changes never alter the
/// verdict's meaning in cents.
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

/// Static chrome: the background only. Gauge ticks are NOT here — they
/// follow the panel-adjustable curve and are drawn per frame in
/// [`TunerUi::process`] together with the needle they share a mapping
/// with, so chrome and reading cannot disagree about the scale.
pub fn static_scene() -> Scene {
    Scene::new(SceneNode {
        shape: Some(Shape::Background { color: BG }),
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
    /// Input meter ballistics (bar + sticky peak marker), fully unit-tested
    /// in [`crate::meter`].
    meter: crate::meter::PeakMeter,
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
    /// Rising-edge memory for Escape on the settings page.
    esc_held: bool,
    /// The audio-class settings this PROCESS actually booted with —
    /// "restart required" is exactly: saved differs from these. (Params
    /// outside this class apply live and never mark a restart.)
    applied: Settings,
    /// The settings exactly as last SAVED — boot file, APPLY, or RESET.
    /// APPLY is clickable only while the draft differs from this:
    /// when there is nothing to write, there is nothing to press.
    /// when there is nothing to write, there is nothing to press.
    saved: Settings,
    /// True when NO settings file existed at startup: a cold start is
    /// shown as a clean slate — every unset audio row a dash, nothing
    /// narrated. Applies to the whole session (choosing values lights
    /// those rows; APPLY ends coldness for good, next boot is warm).
    cold: bool,
    settings_loaded: bool,
    devices_loaded: bool,
    open_dropdown: Option<Dropdown>,
    list_scroll: f32,
    backend_opts: Vec<(String, String)>,
    input_opts: Vec<String>,
    output_opts: Vec<String>,
    rate_opts: Vec<String>,
    /// System-default device names for the current backend: the concrete
    /// value "(default)" hides, shown instead in the device rows.
    default_in: Option<String>,
    default_out: Option<String>,
    /// Rate the driver itself is set to, if known — what "(default)"
    /// resolves to at startup (see main.rs / devices::driver_default_rate).
    driver_rate: Option<f32>,
    /// The backend's last read — devices, defaults, rates, buffers —
    /// taken in-process, or through a streamless child when this one
    /// is ASIO-blind while streaming (devices::snapshot_via_child).
    #[cfg(feature = "standalone")]
    snap: Option<crate::devices::Snapshot>,
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
            meter: crate::meter::PeakMeter::new(crate::now_epoch_ms()),
            latch_name: "–",
            latch_oct: 0,
            cents_shown: 0.0,
            freq_shown: 0.0,
            stable_ms: 0.0,
            green_t: 0.0,
            page: Page::Main,
            prev_mouse_down: false,
            settings: Settings::default(),
            esc_held: false,
            applied: Settings::default(),
            saved: Settings::default(),
            cold: false,
            settings_loaded: false,
            devices_loaded: false,
            open_dropdown: None,
            list_scroll: 0.0,
            backend_opts: Vec::new(),
            input_opts: Vec::new(),
            output_opts: Vec::new(),
            rate_opts: Vec::new(),
            default_in: None,
            default_out: None,
            driver_rate: None,
            #[cfg(feature = "standalone")]
            snap: None,
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
        // Meter levels come from the audio thread (pre-mix, hottest
        // channel RMS + sample peak): averaging the mono downmix would
        // understate one live channel beside a silent sibling by 6 dB.
        let mut drained = false;
        while rb.slots() > 0 {
            let want = rb.slots().min(CHUNK);
            if rb.pop_entire_slice(&mut buf[..want]).is_err() {
                break;
            }
            drained = true;
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
        let (rms_db, frame_peak_db) = if drained {
            (
                db_of(self.shared.input_rms.load() as f64),
                db_of(self.shared.input_peak.load() as f64),
            )
        } else {
            (-120.0, -120.0)
        };
        self.meter
            .tick(rms_db, frame_peak_db, crate::now_epoch_ms());
    }
}

impl Process for TunerUi {
    fn process(&mut self, ctx: &mut Context, dt: f32) {
        let dt = dt.clamp(0.0, 0.25); // tolerate host pauses
        // First frame: read the settings file. Look-and-judgement values
        // (verdict window, gauge curve) apply in hosts too — nothing RT
        // consumes them; audio settings stay standalone-only by design.
        if !self.settings_loaded {
            let (settings, _) = Settings::load();
            self.cold = !Settings::exists();
            self.applied = settings.clone();
            self.saved = settings.clone();
            self.settings = settings;
            self.settings_loaded = true;
        }
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
        let in_tune = signal
            && stable
            && self.cents_shown.abs()
                <= self.settings.in_tune_cents.unwrap_or(IN_TUNE_CENTS as f32) as f64;

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
        let knee = self.settings.gauge_knee_cents.unwrap_or(GAUGE_KNEE_CENTS);
        let in_tune_cents = self.settings.in_tune_cents.unwrap_or(IN_TUNE_CENTS as f32);
        let x = cents_to_units(self.needle_cents, knee) * fit;

        // Gauge ticks: dynamic (they follow the panel-adjustable curve),
        // sharing the needle's mapping so they can never lie about it.
        for cents in (-50..=50).step_by(10) {
            let tx = cents_to_units(cents as f32, knee) * fit;
            let long = cents % 50 == 0;
            ctx.rectangle(
                tx,
                -2.0 * fit,
                0.75 * fit,
                if long { 9.0 } else { 4.5 } * fit,
                if long { DIM } else { TRACK },
                0.04,
            );
        }

        // in-tune zone behind the needle — panel-adjustable width, same
        // nonlinear mapping, so its edges sit exactly where the needle
        // reads ±the configured window
        ctx.rectangle(
            0.0,
            -2.0 * fit,
            cents_to_units(in_tune_cents, knee) * fit,
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
        // cents readout: TABULAR on a fixed pitch — every glyph lands in
        // its own digit-width slot (see draw_tabular). Constant glyph
        // count alone was not enough: `ctx.text` centers the INK box,
        // and `+` versus `−` carry visibly different ink widths, so the
        // whole line shifted a hair whenever the sign flipped. Slotted
        // layout cannot: nothing depends on what the glyphs look like.
        // No signal, no reading: a bare dash, not a confident "+00.0¢".
        let cents_s = if signal {
            let cents = self.cents_shown;
            trim_zeros(&format!(
                "{}{:04.1}¢",
                if cents >= 0.0 { "+" } else { "−" },
                cents.abs()
            ))
        } else {
            "—".to_string()
        };
        self.draw_tabular(
            ctx,
            0.0,
            -52.0 * fit,
            &cents_s,
            50.4 * fit,
            600.0,
            main,
            0.2,
        );
        // frequency + reference — same tabular rule: always exactly six
        // glyphs before " Hz" (zero-padded width incl. the dot), and the
        // idle bar spans the same six, so even the "· A4=" tail cannot
        // shift.
        let freq_s = if signal {
            format!(
                "{} Hz  ·  A4={:.0}",
                trim_zeros(&format!("{:06.1}", self.freq_shown)),
                a4
            )
        } else {
            format!("——————  ·  A4={:.0}", a4)
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
            // dBFS scale: 0 at the right end, -48 at the left, every 6 dB.
            // Short ticks (±4) with numerals well below — digit tops must
            // not crowd the ticks.
            for i in 0..=8u32 {
                let xv = half - i as f32 * (half / 4.0);
                ctx.rectangle(xv, y, 1.0 * fit, 4.0 * fit, Color { a: 0.6, ..DIM }, 0.06);
                ctx.text(
                    xv,
                    y - 26.0 * fit,
                    font,
                    format!("{}", -(6 * i as i32)),
                    16.0 * fit,
                    550.0,
                    DIM,
                    0.06,
                );
            }
            // Bar: current level.
            let m = self.meter.level_db().clamp(-48.0, 0.0);
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
            let p = self.meter.peak_db().clamp(-48.0, 0.0);
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
    InputCh,
    A4,
    Gate,
    /// Editor look/judgement params (verdict window, gauge steepness).
    InTune,
    GaugeCurve,
}

/// Sensible tuning-standards list; 440 sits mid-list.
const A4_OPTIONS: [f32; 9] = [
    432.0, 438.0, 439.0, 440.0, 441.0, 442.0, 443.0, 444.0, 446.0,
];
const GATE_OPTIONS: [f32; 7] = [-80.0, -70.0, -60.0, -50.0, -40.0, -30.0, -20.0];
/// Verdict / green-band half-width choices, cents (3.0 = the original).
const IN_TUNE_OPTIONS: [f32; 7] = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0];
/// Gauge curvature knee choices, cents (5.0 = the original steep middle).
const GAUGE_KNEE_OPTIONS: [f32; 6] = [3.0, 4.0, 5.0, 7.0, 10.0, 20.0];

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

/// Pad-with-zeros output, pruned to ONE anchor zero: "0055.0" → " 055.0",
/// "+00.7¢" → "+ 0.7¢", while "0440.0" → " 440.0" and a lone "0.7" keeps
/// its integer zero. The glyph COUNT is untouched, so the slotted
/// (draw_tabular) layout stays as immovable as with full zero padding —
/// only the ink changes: leading zeros that merely fill space become
/// blanks, the last zero stays as the digit the eye anchors on.
fn trim_zeros(s: &str) -> String {
    let mut c: Vec<char> = s.chars().collect();
    // Skip a leading sign so the zero run is found where it lives.
    let start = usize::from(matches!(
        c.first(),
        Some('+') | Some('-') | Some('\u{2212}') | Some(' ')
    ));
    let mut end = start;
    while end < c.len() && c[end] == '0' {
        end += 1;
    }
    for x in &mut c[start..end.saturating_sub(1).max(start)] {
        *x = ' ';
    }
    c.into_iter().collect()
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
        // File contents arrive on the first frame (see process); opening
        // the page additionally pays for device enumeration, which can
        // load ASIO drivers and must stay lazy.
        if !self.devices_loaded {
            let (_, warning) = Settings::load();
            self.save_note = warning;
            #[cfg(feature = "standalone")]
            {
                self.backend_opts = crate::devices::available_backends()
                    .into_iter()
                    .map(|b| (b.id, b.label))
                    .collect();
            }
            self.refresh_devices();
            self.devices_loaded = true;
        }
        self.open_dropdown = None;
        self.page = Page::Settings;
    }

    /// nice-plug's backend key for the devices module (standalone only;
    /// the plugin build has no devices to enumerate).
    #[cfg(feature = "standalone")]
    fn backend_key(&self) -> &str {
        self.settings.backend.as_deref().unwrap_or("auto")
    }

    /// (Re-)enumerate devices for the selected backend; selections that the
    /// new backend cannot offer fall back to "(system default)".
    /// Read the current backend ONCE: devices, defaults, rates, buffers.
    ///
    /// ASIO must be asked through a child process while this one
    /// streams: asio-sys keeps exactly ONE driver loaded (the live
    /// stream's) and refuses every other load, so in-process
    /// enumeration here sees only the driver already in use — device
    /// lists come back near-empty and stream options fall through to
    /// canned guesses (a hard-wired 64 …). A child opens no stream, so
    /// it sees every registered driver and can ask the SELECTED device
    /// for its real rates and buffer ranges. The live audio is never
    /// touched: nothing is closed, nothing restarts. The parent's own
    /// (partial) view is unioned in, so neither side's blind spot wins.
    #[cfg(feature = "standalone")]
    fn take_snapshot(&mut self) {
        let backend = self.backend_key().to_owned();
        let input = self.settings.input_device.clone();
        let output = self.settings.output_device.clone();
        let in_proc = || crate::devices::snapshot(&backend, input.as_deref(), output.as_deref());
        self.snap = if backend == "asio" {
            match crate::devices::snapshot_via_child(&backend, input.as_deref(), output.as_deref())
            {
                Some(mut s) => {
                    let p = in_proc();
                    s.inputs.extend(p.inputs);
                    s.inputs.sort();
                    s.inputs.dedup();
                    s.outputs.extend(p.outputs);
                    s.outputs.sort();
                    s.outputs.dedup();
                    s.rates.extend(p.rates);
                    s.rates.sort_by(f32::total_cmp);
                    s.rates.dedup();
                    s.buffers.extend(p.buffers);
                    s.buffers.sort_unstable();
                    s.buffers.dedup();
                    if s.def_in.is_none() {
                        s.def_in = p.def_in;
                    }
                    if s.def_out.is_none() {
                        s.def_out = p.def_out;
                    }
                    if s.driver_rate.is_none() {
                        s.driver_rate = p.driver_rate;
                    }
                    if s.buffer_pref.is_none() {
                        s.buffer_pref = p.buffer_pref;
                    }
                    Some(s)
                }
                None => Some(in_proc()),
            }
        } else {
            Some(in_proc())
        };
    }

    /// The snapshot's stream side onto the option lists, with the
    /// stored-value keep rule (a value the settings file holds is
    /// never dropped just because this moment's enumeration missed it).
    #[cfg(feature = "standalone")]
    fn apply_stream_snap(&mut self) {
        let Some(s) = self.snap.clone() else {
            return;
        };
        self.rate_opts = s.rates.iter().map(|r| fmt_rate(*r)).collect();
        self.buffer_opts = s.buffers.iter().map(|b| b.to_string()).collect();
        // QUERIED, REMEMBERED: a driver rate once learned is never
        // cleared by a later blind refresh (driver busy, enumeration
        // hiccup) — the page would forget the truth in front of the
        // user's eyes. New answers overwrite; silence keeps the last.
        self.driver_rate = s.driver_rate.or(self.driver_rate);
        if let Some(p) = self.settings.period_size
            && !self.buffer_opts.contains(&p.to_string())
        {
            self.buffer_opts.insert(0, p.to_string());
        }
        if let Some(r) = self.settings.sample_rate
            && !self.rate_opts.contains(&fmt_rate(r))
        {
            self.rate_opts.insert(0, fmt_rate(r));
        }
    }

    /// COLD CASCADE — choosing a BACKEND adopts that whole world: the
    /// input and output devices fall to their system defaults, then
    /// the stream values are QUERIED for the adopted device and set
    /// to its defaults (the rate the driver runs at; the buffer the
    /// driver prefers — where such defaults are knowable at all).
    /// One human decision, spelled out fully on the page; APPLY only
    /// has to persist it. Warm sessions never adopt: there "follow
    /// the default" is the state, and states are not overwritten.
    #[cfg(feature = "standalone")]
    fn adopt_cold_backend(&mut self) {
        if !self.cold {
            return;
        }
        // Devices fall to the probed defaults — for ASIO the first
        // driver with at least one input channel (devices.rs filters
        // the nomination by capability, registry order).
        self.settings.input_device.clone_from(&self.default_in);
        self.settings.output_device = if self.backend_key() == "asio" {
            // One ASIO driver per process is not policy but asio-sys
            // law: input and output must be the SAME interface, and
            // the input-capable one owns that choice.
            self.settings.input_device.clone()
        } else {
            self.default_out.clone()
        };
        // The tuner's channel adopts 1 explicitly — a cold choice is
        // a choice, so the row shows "channel 1*" rather than hiding
        // behind a dash.
        if self.settings.input_device.is_some() && self.settings.input_channel.is_none() {
            self.settings.input_channel = Some(1);
        }
        // The adopted devices are new stream truth: re-read with them
        // in hand, then adopt the device's stream defaults too.
        self.take_snapshot();
        self.apply_stream_snap();
        self.adopt_cold_stream();
    }

    /// COLD CASCADE — the devices changed (adopted with a backend, or
    /// overridden by hand): buffer and sample rate follow the NOW
    /// selected device's queried defaults, whenever one is known.
    #[cfg(feature = "standalone")]
    fn adopt_cold_stream(&mut self) {
        if !self.cold {
            return;
        }
        if let Some(r) = self.driver_rate_known() {
            self.settings.sample_rate = Some(r);
        }
        if let Some(b) = self.snap.as_ref().and_then(|s| s.buffer_pref) {
            self.settings.period_size = Some(b);
        }
    }

    fn refresh_devices(&mut self) {
        #[cfg(feature = "standalone")]
        {
            self.take_snapshot();
            if let Some(s) = self.snap.clone() {
                self.input_opts = s.inputs;
                self.output_opts = s.outputs;
                self.default_in = s.def_in;
                self.default_out = s.def_out;
            }
            // The same read carries the stream side — applying it here
            // means ONE enumeration pass per backend switch; a second
            // child spin-up would just pay the same cost again.
            self.apply_stream_snap();
        }
        // A saved device the current enumeration does not list is KEPT,
        // not dropped: ASIO enumeration is driver-flaky (cold starts,
        // driver-busy moments), and silently nulling the user's choice
        // here persisted "back to default" files on the next Apply.
        // Instead the saved name joins the option list, so it stays
        // visible and re-selectable even while enumeration is blind.
        if let Some(d) = &self.settings.input_device
            && !self.input_opts.contains(d)
        {
            self.input_opts.insert(0, d.clone());
        }
        if let Some(d) = &self.settings.output_device
            && !self.output_opts.contains(d)
        {
            self.output_opts.insert(0, d.clone());
        }
    }

    fn refresh_stream_options(&mut self) {
        #[cfg(feature = "standalone")]
        {
            self.take_snapshot();
            self.apply_stream_snap();
        }
    }

    /// nice-plug's baked period-size default — only a stand-in for the
    /// case where the stream never ran (measure below is the truth).
    const DEFAULT_BUFFER: u32 = 512;

    /// The buffer default AS IT IS: while the live stream matches the
    /// settings page's current selection, that is the block size the
    /// driver actually granted (trust the driver — the requested
    /// default often is not the granted one). While the selection is
    /// still pending (not applied), the honest number is what
    /// following the default would REQUEST: nice-plug's baked 512.
    fn buffer_default(&self) -> u32 {
        let selection_is_live = self.settings.backend == self.applied.backend
            && self.settings.input_device == self.applied.input_device
            && self.settings.output_device == self.applied.output_device;
        match self.shared.buffer_frames.load() {
            n if selection_is_live && n > 0 => n,
            _ => Self::DEFAULT_BUFFER,
        }
    }

    /// Best-known driver rate. A fresh query wins; otherwise the value
    /// remembered at boot — the file's rate EQUALS what this process
    /// started with, so it was driver-resolved, not a draft the user
    /// is choosing right now. Queried once, remembered: a blind
    /// moment never erases the truth into a dash.
    fn driver_rate_known(&self) -> Option<f32> {
        self.driver_rate.or_else(|| {
            self.settings
                .sample_rate
                .filter(|r| self.applied.sample_rate == Some(*r))
        })
    }

    /// The concrete value the default hides — shown wherever the word
    /// "(default)" used to stand. None = genuinely unknowable (no system
    /// default device, silent driver) → rendered as an em dash.
    fn default_label(&self, dd: Dropdown) -> Option<String> {
        Some(match dd {
            Dropdown::Backend => self
                .backend_opts
                .iter()
                .find(|(id, _)| id == "auto")
                .map(|(_, l)| l.clone())
                .unwrap_or_else(|| "auto".to_owned()),
            Dropdown::Input => self.default_in.clone()?,
            Dropdown::Output => self.default_out.clone()?,
            Dropdown::Rate => fmt_rate(self.driver_rate_known()?),
            Dropdown::Buffer => self.buffer_default().to_string(),
            Dropdown::InputCh => "1".to_owned(),
            Dropdown::A4 => "440".to_owned(),
            Dropdown::Gate => "-60".to_owned(),
            Dropdown::InTune => format!("{IN_TUNE_CENTS:.0}"),
            Dropdown::GaugeCurve => format!("{GAUGE_KNEE_CENTS:.0}"),
        })
    }

    /// The ONE list a dropdown shows, is selected from, and is picked
    /// from: row 0 is the default — its concrete VALUE, starred — and
    /// the default never reappears among the plain options. Picking row
    /// 0 means "follow the default", which keeps tracking hardware
    /// (system device, driver rate) where a pinned option would freeze
    /// a snapshot. Pairs are (display label, raw value); second item of
    /// the tuple is the default row's raw value ("follow" sentinel).
    fn dropdown_rows(&self, dd: Dropdown) -> (Vec<(String, String)>, String) {
        // Plain options: (display label, raw value it selects).
        let opts: Vec<(String, String)> = match dd {
            Dropdown::Backend => self
                .backend_opts
                .iter()
                .map(|(id, l)| (l.clone(), id.clone()))
                .collect(),
            Dropdown::Input | Dropdown::Output => {
                let list = match dd {
                    Dropdown::Input => &self.input_opts,
                    _ => &self.output_opts,
                };
                list.iter().map(|n| (n.clone(), n.clone())).collect()
            }
            Dropdown::Rate => self
                .rate_opts
                .iter()
                .map(|l| (l.clone(), l.split(' ').next().unwrap_or("").to_owned()))
                .collect(),
            Dropdown::Buffer => self
                .buffer_opts
                .iter()
                .map(|l| (l.clone(), l.clone()))
                .collect(),
            Dropdown::InputCh => ["Auto (loudest)", "2", "3", "4", "5", "6", "7", "8"]
                .into_iter()
                .map(|l| (l.to_owned(), l.to_owned()))
                .collect(),
            Dropdown::A4 => A4_OPTIONS
                .iter()
                .map(|v| (format!("{v:.0} Hz"), format!("{v:.0}")))
                .collect(),
            Dropdown::Gate => GATE_OPTIONS
                .iter()
                .map(|v| (format!("{v:.0} dB"), format!("{v:.0}")))
                .collect(),
            Dropdown::InTune => IN_TUNE_OPTIONS
                .iter()
                .map(|v| (format!("{v:.0} \u{a2}"), format!("{v:.0}")))
                .collect(),
            Dropdown::GaugeCurve => GAUGE_KNEE_OPTIONS
                .iter()
                .map(|v| (format!("{v:.0} \u{a2} knee"), format!("{v:.0}")))
                .collect(),
        };
        // Raw value of the default row ("" when unknowable: the row
        // still exists — "\u{2014} *" — and still means follow).
        let def_raw = self.default_raw(dd);
        let head = match self.default_label(dd) {
            Some(d) => format!("{d} *"),
            None => "\u{2014} *".to_string(),
        };
        let mut rows = vec![(head, def_raw.clone())];
        rows.extend(
            opts.into_iter()
                .filter(|(_, raw)| raw.is_empty() || raw != &def_raw),
        );
        (rows, def_raw)
    }

    fn dropdown_items(&self, dd: Dropdown) -> Vec<String> {
        self.dropdown_rows(dd)
            .0
            .into_iter()
            .map(|(label, _)| label)
            .collect()
    }

    /// The raw value the default row selects ("" = unknowable default).
    fn default_raw(&self, dd: Dropdown) -> String {
        match dd {
            Dropdown::Backend => "auto".to_owned(),
            Dropdown::Input => self.default_in.clone().unwrap_or_default(),
            Dropdown::Output => self.default_out.clone().unwrap_or_default(),
            Dropdown::Rate => self
                .driver_rate_known()
                .map(|r| r.to_string())
                .unwrap_or_default(),
            Dropdown::Buffer => self.buffer_default().to_string(),
            Dropdown::InputCh => "1".to_owned(),
            Dropdown::A4 => "440".to_owned(),
            Dropdown::Gate => "-60".to_owned(),
            Dropdown::InTune => format!("{IN_TUNE_CENTS:.0}"),
            Dropdown::GaugeCurve => format!("{GAUGE_KNEE_CENTS:.0}"),
        }
    }

    /// The raw value the row shows right now (unset settings resolve to
    /// their default — that IS the current truth).
    fn current_raw(&self, dd: Dropdown) -> String {
        match dd {
            Dropdown::Backend => self
                .settings
                .backend
                .clone()
                .unwrap_or_else(|| "auto".into()),
            Dropdown::Input => self
                .settings
                .input_device
                .clone()
                .or_else(|| self.default_in.clone())
                .unwrap_or_default(),
            Dropdown::Output => self
                .settings
                .output_device
                .clone()
                .or_else(|| self.default_out.clone())
                .unwrap_or_default(),
            Dropdown::Rate => self
                .settings
                .sample_rate
                .map(|r| r.to_string())
                .or_else(|| self.driver_rate_known().map(|r| r.to_string()))
                .unwrap_or_default(),
            Dropdown::Buffer => self
                .settings
                .period_size
                .map(|p| p.to_string())
                .unwrap_or_else(|| self.buffer_default().to_string()),
            Dropdown::InputCh => match self.shared.input_channel.load() {
                0 => "Auto (loudest)".to_string(),
                n => n.to_string(),
            },
            Dropdown::A4 => format!("{:.0}", self.shared.a4_ref()),
            Dropdown::Gate => format!("{:.0}", self.shared.gate_db.load()),
            Dropdown::InTune => format!(
                "{:.0}",
                self.settings.in_tune_cents.unwrap_or(IN_TUNE_CENTS as f32)
            ),
            Dropdown::GaugeCurve => format!(
                "{:.0}",
                self.settings.gauge_knee_cents.unwrap_or(GAUGE_KNEE_CENTS)
            ),
        }
    }

    /// The row carrying the current truth: row 0 while the setting is
    /// unset (or set to exactly the default), else the matching option,
    /// else row 0 again (unknown value = shown as following default).
    fn selected_index(&self, dd: Dropdown) -> usize {
        let current = self.current_raw(dd);
        self.dropdown_rows(dd)
            .0
            .iter()
            .position(|(_, raw)| *raw == current)
            .unwrap_or(0)
    }

    /// True when the row's value IS the default — the star behind the
    /// value button, matching the starred default row in its dropdown.
    fn is_default(&self, dd: Dropdown) -> bool {
        let current = self.current_raw(dd);
        current.is_empty() || current == self.default_raw(dd)
    }

    fn pick(&mut self, dd: Dropdown, idx: usize) {
        // The raw value lives on the same row the user clicked — no
        // index arithmetic anywhere. Row 0 carries the default's raw
        // value; the setting is stored as None (follow) exactly then.
        let (rows, def_raw) = self.dropdown_rows(dd);
        let Some((_, raw)) = rows.get(idx).cloned() else {
            return;
        };
        let at_default = raw.is_empty() || raw == def_raw;
        match dd {
            Dropdown::Backend => {
                self.settings.backend = if raw == "auto" { None } else { Some(raw) };
                self.refresh_devices();
                #[cfg(feature = "standalone")]
                self.adopt_cold_backend();
            }
            Dropdown::Input => {
                self.settings.input_device = if at_default { None } else { Some(raw) };
                // Choosing an input picks the SAME interface as output:
                // one device owns both clock and monitoring path; a
                // mismatched output would silently be another driver.
                self.settings.output_device = self.settings.input_device.clone();
                self.refresh_stream_options();
                #[cfg(feature = "standalone")]
                self.adopt_cold_stream();
            }
            Dropdown::Output => {
                self.settings.output_device = if at_default { None } else { Some(raw) };
                // Duplex constraint: the output device narrows the buffer
                // range too — recompute.
                self.refresh_stream_options();
                #[cfg(feature = "standalone")]
                self.adopt_cold_stream();
            }
            Dropdown::Rate => {
                self.settings.sample_rate = if at_default { None } else { raw.parse().ok() };
            }
            Dropdown::Buffer => {
                self.settings.period_size = if at_default { None } else { raw.parse().ok() };
            }
            // Channel select is live for BOTH meter and detection: the RT
            // feed becomes exactly this channel (auto keeps the mono mix /
            // hottest-channel meter). No restart; APPLY persists.
            Dropdown::InputCh => {
                let ch = if raw == "Auto (loudest)" {
                    0
                } else {
                    raw.parse().unwrap_or(1)
                };
                self.settings.input_channel = if at_default { None } else { Some(ch) };
                self.shared.input_channel.store(ch);
            }
            // Params are live the moment they are picked (no restart needed);
            // APPLY only decides whether they persist to the next launch.
            Dropdown::A4 => {
                let v: f32 = raw.parse().unwrap_or(440.0);
                self.shared
                    .a4_ref_bits
                    .store((v as f64).to_bits(), Ordering::Relaxed);
                self.settings.a4_ref = if at_default { None } else { Some(v) };
            }
            Dropdown::Gate => {
                let v: f32 = raw.parse().unwrap_or(-60.0);
                self.shared.gate_db.store(v);
                self.settings.gate_db = if at_default { None } else { Some(v) };
            }
            // Editor look/judgement: applied the instant they are picked
            // (draw + verdict re-read settings every frame); APPLY persists.
            Dropdown::InTune => {
                let v: f32 = raw.parse().unwrap_or(IN_TUNE_CENTS as f32);
                self.settings.in_tune_cents = if at_default { None } else { Some(v) };
            }
            Dropdown::GaugeCurve => {
                let v: f32 = raw.parse().unwrap_or(GAUGE_KNEE_CENTS);
                self.settings.gauge_knee_cents = if at_default { None } else { Some(v) };
            }
        }
        self.open_dropdown = None;
        self.list_scroll = 0.0;
    }

    /// Draw `text` glyph-by-glyph on a fixed advance pitch (FiraCode is
    /// monospaced, so one digit's width is THE pitch). Each glyph is
    /// ink-centered inside its own slot, so the line's layout depends
    /// only on the character COUNT — signs, dots and the ¢ mark can no
    /// longer push neighbours around. Whole-string `ctx.text` cannot
    /// promise that: its centering follows the ink box, and glyph ink
    /// varies (+ vs −, and future fonts).
    #[allow(clippy::too_many_arguments)] // immediate-mode draw call, as in frost
    fn draw_tabular(
        &self,
        ctx: &mut Context,
        cx: f32,
        cy: f32,
        text: &str,
        size: f32,
        weight: f32,
        color: Color,
        z: f32,
    ) {
        let font = std::sync::Arc::clone(&self.font);
        let pitch = ctx.text_width(&font, "0", size, weight);
        if pitch <= 0.0 {
            // Font cannot be shaped: fall back to a plain draw rather
            // than silently vanishing.
            ctx.text(cx, cy, &font, text, size, weight, color, z);
            return;
        }
        let n = text.chars().count() as f32;
        for (i, ch) in text.chars().enumerate() {
            if ch == ' ' {
                continue; // blanks carry layout, not ink
            }
            let mut buf = [0u8; 4];
            ctx.text(
                cx + (i as f32 - (n - 1.0) / 2.0) * pitch,
                cy,
                &font,
                ch.encode_utf8(&mut buf),
                size,
                weight,
                color,
                z,
            );
        }
    }

    /// The closed dropdown's text: the current value, starred with `*`
    /// when it IS the default — so a glance says what would happen on
    /// RESET, and the star matches the dropdown's starred row.
    fn current_value_label(&self, dd: Dropdown) -> String {
        let mut s = self.value_text(dd);
        // A bare dash needs no star — the star decorates values, not
        // the absence of one.
        if self.is_default(dd) && !s.ends_with('*') && s != "\u{2014}" {
            s.push('*');
        }
        s
    }

    fn value_text(&self, dd: Dropdown) -> String {
        match dd {
            Dropdown::Backend => match self.settings.backend.as_deref() {
                // Nothing chosen yet: a bare dash — "auto" is what the
                // NEXT launch does, not a choice the user made.
                None => "\u{2014}".to_string(),
                Some(id) => self
                    .backend_opts
                    .iter()
                    .find(|(i, _)| i == id)
                    .map(|(_, l)| l.clone())
                    .unwrap_or_else(|| id.to_owned()),
            },
            Dropdown::Rate => match self.settings.sample_rate {
                Some(r) => fmt_rate(r),
                // Cold start: nothing chosen, nothing narrated. Warm
                // and unset: "follow the driver" IS the state — show
                // the number (queried or remembered), starred.
                None if self.cold => "\u{2014}".into(),
                None => self
                    .driver_rate_known()
                    .map(fmt_rate)
                    .unwrap_or_else(|| "\u{2014}".into()),
            },
            Dropdown::Buffer => match self.settings.period_size {
                // Same rule as Backend: unset is unset — the dropdown's
                // starred row names the value that follows (the live
                // driver-granted block size); the button stays a dash.
                None => "\u{2014}".to_string(),
                Some(p) => p.to_string(),
            },
            // Devices follow the same cold rule: on a fresh install the
            // panel does not announce the system default before the
            // user has shown any interest in devices at all. Warm and
            // unset, "follow the system default" is the state — name it.
            Dropdown::Input => self
                .settings
                .input_device
                .clone()
                .or_else(|| {
                    if self.cold {
                        None
                    } else {
                        self.default_in.clone()
                    }
                })
                .map(|d| clip(&d, 26))
                .unwrap_or_else(|| "\u{2014}".into()),
            Dropdown::Output => self
                .settings
                .output_device
                .clone()
                .or_else(|| {
                    if self.cold {
                        None
                    } else {
                        self.default_out.clone()
                    }
                })
                .map(|d| clip(&d, 26))
                .unwrap_or_else(|| "\u{2014}".into()),
            Dropdown::InputCh => match (self.cold, self.settings.input_channel) {
                (true, None) => "\u{2014}".to_string(),
                _ => match self.shared.input_channel.load() {
                    0 => "auto (loudest)".to_string(),
                    n => format!("channel {n}"),
                },
            },
            Dropdown::A4 => format!("{:.1} Hz", self.shared.a4_ref()),
            Dropdown::Gate => format!("{:.0} dB", self.shared.gate_db.load()),
            Dropdown::InTune => format!(
                "{:.0} \u{a2} window",
                self.settings.in_tune_cents.unwrap_or(IN_TUNE_CENTS as f32)
            ),
            Dropdown::GaugeCurve => format!(
                "{:.0} \u{a2} knee",
                self.settings.gauge_knee_cents.unwrap_or(GAUGE_KNEE_CENTS)
            ),
        }
    }

    /// True when the audio-class settings differ from what this process
    /// booted with — the ONLY condition that makes Apply restart.
    /// (Backend, devices, rate, buffer, input-latency: they configure the
    /// streams nice-plug opened at startup. Params — channel, A4, gate,
    /// window, curve — apply live and never set this.)
    fn audio_dirty(&self) -> bool {
        let (a, s) = (&self.applied, &self.settings);
        a.backend != s.backend
            || a.input_device != s.input_device
            || a.output_device != s.output_device
            || a.sample_rate != s.sample_rate
            || a.period_size != s.period_size
            || a.input_latency != s.input_latency
    }

    /// Anything to write at all: draft differs from the last save.
    /// Window GEOMETRY is excluded — it persists on its own (moved or
    /// resized, debounced; plus a flush at close), so counting it
    /// would keep APPLY perpetually "dirty" for merely using the app.
    fn apply_dirty(&self) -> bool {
        let (a, s) = (&self.saved, &self.settings);
        a.backend != s.backend
            || a.input_device != s.input_device
            || a.output_device != s.output_device
            || a.sample_rate != s.sample_rate
            || a.period_size != s.period_size
            || a.input_latency != s.input_latency
            || a.a4_ref != s.a4_ref
            || a.gate_db != s.gate_db
            || a.input_channel != s.input_channel
            || a.in_tune_cents != s.in_tune_cents
            || a.gauge_knee_cents != s.gauge_knee_cents
    }

    fn process_settings_page(&mut self, ctx: &mut Context, clicked: bool) {
        let (w, h) = ctx.size();
        let fit = (w / 720.0).min(h / 440.0).max(0.45);
        let mouse = ctx.mouse_position();
        let font = std::sync::Arc::clone(&self.font);
        let z = 0.4f32;

        // Escape changes nothing — it UN-changes: close an open dropdown,
        // or leave the page with drafts unsaved. Never a write, never a
        // restart.
        let esc = ctx.key_down(frost::KeyCode::Escape);
        if esc && !self.esc_held {
            if self.open_dropdown.is_some() {
                self.open_dropdown = None;
                self.list_scroll = 0.0;
            } else {
                self.page = Page::Main;
                self.save_note = None;
                return;
            }
        }
        self.esc_held = esc;

        // Audio block first, in signal order (what feeds me, which channel
        // of it, what I answer through, then the stream's shape), then a
        // gap, then the live param block.
        const ROWS: [(&str, Dropdown); 10] = [
            ("Backend", Dropdown::Backend),
            ("Input device", Dropdown::Input),
            ("Input channel", Dropdown::InputCh),
            ("Output device", Dropdown::Output),
            ("Buffer (frames)", Dropdown::Buffer),
            ("Sample rate", Dropdown::Rate),
            ("A4 reference", Dropdown::A4),
            ("Input gate", Dropdown::Gate),
            ("In-tune window", Dropdown::InTune),
            ("Gauge curve", Dropdown::GaugeCurve),
        ];
        // Rows 6+ are the param block: an extra gap drops them below the
        // audio block instead of running one uniform ladder. Spacing is
        // tuned so ten rows clear the APPLY button (bottom row ends at
        // −155·fit, APPLY's top edge is −175·fit).
        let row_y = |i: usize| {
            h / 2.0 - 40.0 * fit - i as f32 * 33.0 * fit - if i >= 6 { 14.0 } else { 0.0 } * fit
        };
        // Left column: genuinely LEFT-aligned. ctx.text centers INK, so
        // each label is centered at margin + half its measured width —
        // long names can no longer spill past the left screen edge.
        let lx = -w / 2.0 + 18.0 * fit;
        // Right column: value buttons, three characters wider, anchored by
        // their RIGHT edge at the screen margin — they cannot spill right
        // no matter how wide the dropdown text grows.
        let charw = ctx
            .text_width(&font, "0", 23.4 * fit, 550.0)
            .max(14.0 * fit);
        let bhw = 165.0 * fit + 3.0 * charw;
        let bx = w / 2.0 - 18.0 * fit - bhw;
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
            let lw = ctx.text_width(&font, label, 25.2 * fit, 550.0);
            ctx.text(lx + lw / 2.0, y, &font, *label, 25.2 * fit, 550.0, DIM, z);
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

        // Bottom actions: BACK · RESET · APPLY. The Apply label — and its
        // button's width — speak the truth: only an audio-class change
        // (the settings nice-plug baked at startup) requires a restart;
        // everything else goes live on save.
        let by = -h / 2.0 + 30.0 * fit;
        let needs_restart = self.audio_dirty();
        let apply_label = if needs_restart {
            "APPLY & RESTART"
        } else {
            "APPLY"
        };
        let aw = (ctx.text_width(&font, apply_label, 23.4 * fit, 700.0) + 26.0 * fit) / 2.0;
        let ax = w / 2.0 - 32.0 * fit - aw; // right-aligned with a margin
        // Nothing differs from the last save: the button stays, greyed,
        // and swallows nothing — there is no click when there is no do.
        let dirty = self.apply_dirty();
        let back = point_in(mouse, -w / 2.0 + 90.0 * fit, by, 65.0 * fit, 15.0 * fit);
        let reset = point_in(mouse, 0.0, by, 65.0 * fit, 15.0 * fit);
        let apply = dirty && point_in(mouse, ax, by, aw, 15.0 * fit);
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
            0.0,
            by,
            65.0 * fit,
            15.0 * fit,
            if reset { HOVER } else { TRACK },
            z,
        );
        ctx.text(
            0.0,
            by,
            &font,
            "RESET",
            23.4 * fit,
            650.0,
            if reset { AMBER } else { DIM },
            z + 0.01,
        );
        ctx.rectangle(
            ax,
            by,
            aw,
            15.0 * fit,
            if apply {
                HOVER
            } else if dirty {
                TRACK
            } else {
                BG
            },
            z,
        );
        ctx.text(
            ax,
            by,
            &font,
            apply_label,
            23.4 * fit,
            700.0,
            if apply {
                AMBER
            } else if dirty {
                GREEN
            } else {
                DIM
            },
            z + 0.01,
        );

        // Transient status only, on demand — no standing banner text.
        if let Some(note) = &self.save_note {
            ctx.text(
                0.0,
                by + 34.0 * fit,
                &font,
                clip(note, 90),
                21.6 * fit,
                550.0,
                AMBER,
                z,
            );
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
                if reset {
                    // Reset is a command, not a draft: defaults land in
                    // memory, in the live param atomics, and on disk at
                    // once. Audio streams keep running on the old config;
                    // the Apply button now truthfully says whether a
                    // restart is needed to catch up.
                    self.settings = Settings::default();
                    self.shared
                        .a4_ref_bits
                        .store(440f64.to_bits(), Ordering::Relaxed);
                    self.shared.gate_db.store(-60.0);
                    self.shared.input_channel.store(1);
                    self.refresh_devices();
                    self.save_note = match self.settings.save() {
                        Ok(()) => {
                            self.saved = self.settings.clone();
                            Some("settings reset to defaults".to_string())
                        }
                        Err(e) => Some(format!("save failed: {e}")),
                    };
                }
                if apply {
                    // Carry the window's current geometry along: "last
                    // screen position and size" is a setting like any
                    // other, saved when the user says Apply. On Windows
                    // the live rect feeds it (see gui.rs); elsewhere
                    // only the size is knowable, position is preserved.
                    #[cfg(feature = "standalone")]
                    {
                        let ls = self.shared.logical_size();
                        let pos = *self.shared.window_pos.lock().unwrap();
                        self.settings.window = Some(crate::settings::WindowGeom {
                            x: pos.map(|p| p.0).or(self.settings.window.and_then(|g| g.x)),
                            y: pos.map(|p| p.1).or(self.settings.window.and_then(|g| g.y)),
                            width: ls.width,
                            height: ls.height,
                        });
                    }
                    match self.settings.save() {
                        // The draft IS the file now — APPLY falls grey
                        // until something differs again.
                        Ok(()) if self.audio_dirty() => {
                            self.saved = self.settings.clone();
                            crate::request_restart();
                        }
                        Ok(()) => {
                            self.saved = self.settings.clone();
                            self.save_note = Some("saved — applied live".to_string());
                        }
                        Err(e) => self.save_note = Some(format!("save failed: {e}")),
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{GAUGE_KNEE_OPTIONS, cents_to_units, trim_zeros};

    #[test]
    fn trim_zeros_prunes_fill_keeps_anchor() {
        // Extra leading zeros become blanks…
        assert_eq!(trim_zeros("0055.0"), " 055.0");
        assert_eq!(trim_zeros("0005.0"), "  05.0");
        assert_eq!(trim_zeros("+00.7¢"), "+ 0.7¢");
        // …a single leading zero is the anchor and stays…
        assert_eq!(trim_zeros("0440.0"), "0440.0");
        assert_eq!(trim_zeros("+05.2¢"), "+05.2¢");
        assert_eq!(trim_zeros("0.7"), "0.7");
        // …no leading zeros: untouched; glyph counts are preserved.
        assert_eq!(trim_zeros("440.0"), "440.0");
        for s in ["0055.0", "0440.0", "440.0", "+00.7¢", "0.7"] {
            assert_eq!(trim_zeros(s).chars().count(), s.chars().count());
        }
    }

    #[test]
    fn gauge_scale_endpoints_symmetry_and_midroom() {
        // Endpoints land exactly on the gauge edges, at EVERY selectable
        // knee (the panel can pick any of them).
        for &knee in &GAUGE_KNEE_OPTIONS {
            assert!((cents_to_units(50.0, knee) - 240.0).abs() < 1e-2);
            assert!((cents_to_units(-50.0, knee) + 240.0).abs() < 1e-2);
            // Odd and monotone — the gauge never folds.
            let mut prev = f32::MIN;
            let mut c = -50.0f32;
            while c <= 50.0 {
                let x = cents_to_units(c, knee);
                assert!(x > prev, "monotone at {c} knee {knee}");
                assert!(
                    (cents_to_units(-c, knee) + x).abs() < 1e-2,
                    "odd at {c} knee {knee}"
                );
                prev = x;
                c += 0.5;
            }
            // The point of it all: the ±3 ¢ verdict zone is ≥ 3× the
            // linear span (14.4 units) at the DEFAULT knee…
            if (knee - 5.0).abs() < 0.01 {
                assert!(
                    cents_to_units(3.0, knee) >= 44.0,
                    "mid gains too little room"
                );
            }
            // …and smaller knees never shrink it.
            if knee <= 5.0 {
                assert!(cents_to_units(3.0, knee) >= cents_to_units(3.0, 5.0));
            }
            // Beyond-range input (EMA overshoot) cannot escape the gauge.
            assert!(cents_to_units(80.0, knee) <= 240.0 + 1e-2);
        }
        // The 10 ¢ mark sits before the gauge's midpoint at the default.
        assert!(
            cents_to_units(10.0, 5.0) < 120.0,
            "edges compress too weakly"
        );
    }
}
