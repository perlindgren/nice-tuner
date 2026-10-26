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
//!   ▂▃▅ level ▅▃▂        rms meter, y −95, halfwidth ≤ 160
//! ```

use crate::TunerSharedState;
use frost::{Color, Context, Process, Scene, SceneNode, Shape};
use pitch::freq_to_note;
use std::sync::{Arc, atomic::Ordering};

// Clarity gating lives in the detector (Config::min_clarity, audio thread);
// the UI only freshness-gates + level-gates (see process()).
/// cents → px on the gauge (±50 ¢ spans ±120 px)
const CENTS_PX: f32 = 2.4;
/// |cents| inside which we call it in tune, when latched
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

/// Static chrome: background, gauge tick marks, meter track. Rebuilt never;
/// everything that moves is drawn immediately in [`TunerUi::process`].
pub fn static_scene() -> Scene {
    let mut ticks = Vec::new();
    for cents in (-50..=50).step_by(10) {
        let x = cents as f32 * CENTS_PX;
        let long = cents % 50 == 0;
        let half = if long { 9.0 } else { 4.5 };
        ticks.push(Box::new(SceneNode {
            shape: Some(Shape::Rectangle {
                center: [x, 25.0],
                extent: [0.75, half],
                color: if long { DIM } else { TRACK },
            }),
            ..Default::default()
        }));
    }
    let meter_track = Box::new(SceneNode {
        shape: Some(Shape::Rectangle {
            center: [0.0, -95.0],
            extent: [160.0, 3.0],
            color: TRACK,
        }),
        ..Default::default()
    });
    Scene::new(SceneNode {
        shape: Some(Shape::Background { color: BG }),
        children: ticks
            .into_iter()
            .chain(std::iter::once(meter_track))
            .collect(),
        ..Default::default()
    })
}

/// The UI state machine + immediate-mode drawing, run inside frost's frame.
pub struct TunerUi {
    shared: Arc<TunerSharedState>,
    font: Arc<[u8]>,
    // smoothed display state
    needle_cents: f32, // EMA'd cents shown on the gauge
    latch_midi: i32,   // -1 = nothing latched
    latch_name: &'static str,
    latch_oct: i8,
    cents_shown: f64,
    freq_shown: f64,
    stable_ms: f32,
    green_t: f32,     // 0 = amber, 1 = green (lerped)
    meter_level: f32, // decayed meter fill, 0..1
}

impl TunerUi {
    pub fn new(shared: Arc<TunerSharedState>, font: Arc<[u8]>) -> Self {
        Self {
            shared,
            font,
            needle_cents: 0.0,
            latch_midi: -1,
            latch_name: "–",
            latch_oct: 0,
            cents_shown: 0.0,
            freq_shown: 0.0,
            stable_ms: 0.0,
            green_t: 0.0,
            meter_level: 0.0,
        }
    }
}

impl Process for TunerUi {
    fn process(&mut self, ctx: &mut Context, dt: f32) {
        let dt = dt.clamp(0.0, 0.25); // tolerate host pauses

        // ---- read shared state (written by the audio thread) --------------
        // The detector already applied its clarity gate on the audio thread
        // (push_samples only yields confident windows), so "signal present"
        // here = a *fresh* detection + level above the user gate.
        let freq = self.shared.freq_hz.load();
        let rms = self.shared.rms.load() as f64;
        let a4 = self.shared.a4_ref();
        let gate_db = self.shared.gate_db.load();
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
                self.latch_midi = note.midi as i32;
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
        let font = &self.font;
        let x = self.needle_cents * CENTS_PX * fit;

        // in-tune zone behind the needle
        ctx.rectangle(
            0.0,
            25.0 * fit,
            IN_TUNE_CENTS as f32 * CENTS_PX * fit,
            9.0 * fit,
            Color { a: 0.15, ..GREEN },
            0.05,
        );
        // needle
        ctx.line(
            x,
            (25.0 - 16.0) * fit,
            x,
            (25.0 + 16.0) * fit,
            main,
            2.5 * fit,
            0.2,
        );
        // note label
        let label = format!("{}{}", self.latch_name, self.latch_oct);
        ctx.text(0.0, 95.0 * fit, font, label, 96.0 * fit, 700.0, main, 0.2);
        // cents readout (one decimal, sign fixed for stable layout)
        let cents = if signal { self.cents_shown } else { 0.0 };
        let cents_s = format!(
            "{}{:.1}¢",
            if cents >= 0.0 { "+" } else { "−" },
            cents.abs()
        );
        ctx.text(0.0, -5.0 * fit, font, cents_s, 28.0 * fit, 400.0, main, 0.2);
        // frequency + reference
        let freq_s = if signal {
            format!("{:.1} Hz  ·  A4={:.0}", self.freq_shown, a4)
        } else {
            format!("———  ·  A4={:.0}", a4)
        };
        ctx.text(0.0, -45.0 * fit, font, freq_s, 22.0 * fit, 400.0, DIM, 0.2);
        // meter fill over the static track (freshness-gated, then decayed:
        // rms only updates on confident detections, so hold briefly and fade)
        let lvl_target = if fresh {
            ((rms_db - gate_db) / 60.0).clamp(0.0, 1.0) as f32
        } else {
            0.0
        };
        self.meter_level += (lvl_target - self.meter_level)
            * if lvl_target > self.meter_level {
                1.0
            } else {
                1.0 - (-dt / 0.3).exp()
            };
        let lvl = self.meter_level;
        ctx.rectangle(
            -160.0 * fit + 160.0 * fit * lvl,
            -95.0 * fit,
            160.0 * fit * lvl,
            3.0 * fit,
            main,
            0.1,
        );
    }
}

// ---- input mapping (baseview 0.3.3 / keyboard_types 0.8 → frost) ----------

pub fn map_key(key: &keyboard_types::Key) -> Option<frost::KeyCode> {
    use keyboard_types::Key;
    Some(match key {
        Key::Character(c) => match c.0.as_str() {
            "a" | "A" => frost::KeyCode::KeyA,
            "b" | "B" => frost::KeyCode::KeyB,
            "c" | "C" => frost::KeyCode::KeyC,
            "d" | "D" => frost::KeyCode::KeyD,
            "e" | "E" => frost::KeyCode::KeyE,
            "f" | "F" => frost::KeyCode::KeyF,
            "g" | "G" => frost::KeyCode::KeyG,
            "h" | "H" => frost::KeyCode::KeyH,
            "i" | "I" => frost::KeyCode::KeyI,
            "j" | "J" => frost::KeyCode::KeyJ,
            "k" | "K" => frost::KeyCode::KeyK,
            "l" | "L" => frost::KeyCode::KeyL,
            "m" | "M" => frost::KeyCode::KeyM,
            "n" | "N" => frost::KeyCode::KeyN,
            "o" | "O" => frost::KeyCode::KeyO,
            "p" | "P" => frost::KeyCode::KeyP,
            "q" | "Q" => frost::KeyCode::KeyQ,
            "r" | "R" => frost::KeyCode::KeyR,
            "s" | "S" => frost::KeyCode::KeyS,
            "t" | "T" => frost::KeyCode::KeyT,
            "u" | "U" => frost::KeyCode::KeyU,
            "v" | "V" => frost::KeyCode::KeyV,
            "w" | "W" => frost::KeyCode::KeyW,
            "x" | "X" => frost::KeyCode::KeyX,
            "y" | "Y" => frost::KeyCode::KeyY,
            "z" | "Z" => frost::KeyCode::KeyZ,
            "0" => frost::KeyCode::Digit0,
            "1" => frost::KeyCode::Digit1,
            "2" => frost::KeyCode::Digit2,
            "3" => frost::KeyCode::Digit3,
            "4" => frost::KeyCode::Digit4,
            "5" => frost::KeyCode::Digit5,
            "6" => frost::KeyCode::Digit6,
            "7" => frost::KeyCode::Digit7,
            "8" => frost::KeyCode::Digit8,
            "9" => frost::KeyCode::Digit9,
            _ => return None,
        },
        Key::Escape => frost::KeyCode::Escape,
        Key::Enter => frost::KeyCode::Enter,
        Key::Space => frost::KeyCode::Space,
        Key::Tab => frost::KeyCode::Tab,
        Key::Backspace => frost::KeyCode::Backspace,
        Key::Delete => frost::KeyCode::Delete,
        Key::ArrowLeft => frost::KeyCode::ArrowLeft,
        Key::ArrowRight => frost::KeyCode::ArrowRight,
        Key::ArrowUp => frost::KeyCode::ArrowUp,
        Key::ArrowDown => frost::KeyCode::ArrowDown,
        _ => return None,
    })
}

pub fn map_mouse(button: baseview::MouseButton) -> Option<frost::MouseButton> {
    Some(match button {
        baseview::MouseButton::Left => frost::MouseButton::Left,
        baseview::MouseButton::Right => frost::MouseButton::Right,
        baseview::MouseButton::Middle => frost::MouseButton::Middle,
        _ => return None,
    })
}
