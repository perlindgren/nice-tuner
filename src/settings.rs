//! Standalone audio-settings persistence and argv synthesis.
//!
//! nice-plug's standalone wrapper builds its audio streams from the parsed
//! CLI config before the plugin exists, and tears them down when the editor
//! closes — it cannot be reconfigured from inside the plugin. So the settings
//! panel persists changes here and the standalone entry respawns the process:
//! from the user's seat, a blink. Hosts never consult these values — in a
//! DAW the host owns the device — so the settings page does not exist there.

use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Read from and written to the process's working directory.
pub const FILE_NAME: &str = "nice-tuner-settings.json";

/// nice-plug's own `--input-latency` default (150 ms) is audible slap-back
/// for a live tuner; we stand in with this when neither the command line nor
/// the settings file has an opinion.
pub const OUR_INPUT_LATENCY_DEFAULT: f32 = 0.005;

/// Audio-system settings mirroring nice-plug's standalone CLI options.
/// `None` means "not set here" — the command line wins, then nice-plug's
/// own default (or our input-latency override above).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// A nice-plug `BackendType` value (clap spellings): `auto`, `jack`,
    /// `alsa`, `core-audio`, `wasapi`, `asio`, `dummy`.
    pub backend: Option<String>,
    /// Exact cpal device name — for ASIO, the ASIO *driver* name
    /// ("Komplete Input 1/2"), not a port/endpoint name.
    pub input_device: Option<String>,
    pub output_device: Option<String>,
    pub sample_rate: Option<f32>,
    pub period_size: Option<u32>,
    /// The alignment slack, in seconds, nice-plug inserts between its
    /// independent input and output streams.
    pub input_latency: Option<f32>,
    /// Plugin params, standalone-only. Inside a host these belong to the
    /// host (state save/load); standalone has no host, so the settings page
    /// plays that role. Not CLI flags — never fed to `effective_args`.
    pub a4_ref: Option<f32>,
    pub gate_db: Option<f32>,
    /// Input channel, 1-based; absent = channel 1 (the tuner's default
    /// single-instrument cable). An explicit 0 selects auto (hottest
    /// channel). A noisy sibling makes auto wrong — the right input
    /// should be picked explicitly.
    pub input_channel: Option<u32>,
    /// Tuning-verdict half-width and gauge curvature, cents (None = 3.0 /
    /// 5.0). Look-and-judgement values read by the editor itself, also in
    /// a host — unlike the audio settings, nothing RT consults them.
    pub in_tune_cents: Option<f32>,
    pub gauge_knee_cents: Option<f32>,
    /// Standalone window geometry (top-left in physical screen px; size
    /// in logical units). None = never saved → the window centers itself
    /// on first launch. Hosts own their windows; this is standalone-only.
    pub window: Option<WindowGeom>,
}

/// Serialized window placement for the standalone binary. `x`/`y` are
/// physical screen px (top-left) and known only on Windows — elsewhere
/// baseview has no position concept at all; size is logical px. Both
/// always round-trip: unknown position simply means "center again".
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct WindowGeom {
    pub x: Option<i32>,
    pub y: Option<i32>,
    pub width: f32,
    pub height: f32,
}

impl Settings {
    /// Repair known bad values from older writers. Panel builds before the
    /// clap-key fix wrote the cpal spelling "coreaudio", which nice-plug's
    /// argument parser rejects — the app then died at startup argv parsing,
    /// indistinguishable from "device refuses to open".
    pub fn normalize(&mut self) {
        if self.backend.as_deref() == Some("coreaudio") {
            self.backend = Some("core-audio".to_string());
        }
    }

    pub fn path() -> PathBuf {
        std::env::current_dir()
            .unwrap_or_else(|_| PathBuf::from("."))
            .join(FILE_NAME)
    }

    /// Whether a settings FILE exists at all — distinct from "settings
    /// have values". The panel reads this at startup: a cold start (no
    /// file) shows unset audio rows as dashes, not narrated resolutions.
    pub fn exists() -> bool {
        Self::path().exists()
    }

    /// A missing file is the default; a corrupt file is the default plus a
    /// warning to show. Startup must never depend on a file that can rot.
    pub fn load() -> (Settings, Option<String>) {
        match std::fs::read_to_string(Self::path()) {
            Err(_) => (Settings::default(), None),
            Ok(text) if text.trim().is_empty() => (Settings::default(), None),
            Ok(text) => match serde_json::from_str::<Settings>(&text) {
                Ok(mut s) => {
                    s.normalize();
                    (s, None)
                }
                Err(e) => (
                    Settings::default(),
                    Some(format!(
                        "ignoring unreadable {}: {e}",
                        Self::path().display()
                    )),
                ),
            },
        }
    }

    pub fn save(&self) -> Result<(), String> {
        let text = serde_json::to_string_pretty(self).map_err(|e| e.to_string())?;
        std::fs::write(Self::path(), text).map_err(|e| format!("{}: {e}", Self::path().display()))
    }
}

/// Does `raw` already carry this option (long flag or any short form)?
fn cli_has(raw: &[String], flag: &str, shorts: &[&str]) -> bool {
    raw.iter()
        .any(|a| a == flag || shorts.iter().any(|s| a == s))
}

fn format_rate(r: f32) -> String {
    if r.fract() == 0.0 {
        format!("{}", r as u32)
    } else {
        r.to_string()
    }
}

/// Synthesize the argv handed to nice-plug's clap parser. Precedence:
/// explicit command-line flags, then the settings file, then our defaults.
/// `raw` includes argv[0] (the parser expects it).
pub fn effective_args(settings: &Settings, raw: &[String]) -> Vec<String> {
    let mut args: Vec<String> = raw.to_vec();
    let mut push = |flag: &str, value: String| {
        args.push(flag.to_owned());
        args.push(value);
    };

    if let Some(b) = settings
        .backend
        .as_deref()
        .filter(|_| !cli_has(raw, "--backend", &["-b"]))
    {
        push("--backend", b.to_owned());
    }
    if let Some(d) = settings
        .input_device
        .as_deref()
        .filter(|_| !cli_has(raw, "--input-device", &[]))
    {
        push("--input-device", d.to_owned());
    }
    if let Some(d) = settings
        .output_device
        .as_deref()
        .filter(|_| !cli_has(raw, "--output-device", &[]))
    {
        push("--output-device", d.to_owned());
    }
    if let Some(r) = settings
        .sample_rate
        .filter(|_| !cli_has(raw, "--sample-rate", &["-r"]))
    {
        push("--sample-rate", format_rate(r));
    }
    if let Some(p) = settings
        .period_size
        .filter(|_| !cli_has(raw, "--period-size", &["-p"]))
    {
        push("--period-size", p.to_string());
    }
    if settings
        .input_latency
        .filter(|_| !cli_has(raw, "--input-latency", &[]))
        .is_some()
    {
        let l = settings.input_latency.unwrap();
        push("--input-latency", l.to_string());
    } else if !cli_has(raw, "--input-latency", &[]) {
        push("--input-latency", OUR_INPUT_LATENCY_DEFAULT.to_string());
    }
    args
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_geometry_round_trips_and_defaults_absent() {
        let s = Settings {
            window: Some(WindowGeom {
                x: Some(-1200),
                y: Some(80),
                width: 900.0,
                height: 560.0,
            }),
            ..Default::default()
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: Settings = serde_json::from_str(&json).unwrap();
        assert_eq!(back, s);
        // Old files (no window key) load as "never saved" → the window
        // centres itself on the next start.
        let old: Settings = serde_json::from_str("{}").unwrap();
        assert_eq!(old.window, None);
        // Size-only entries (platforms without a position API omit x/y).
        let size_only: Settings =
            serde_json::from_str(r#"{"window":{"width":800.0,"height":500.0}}"#).unwrap();
        assert_eq!(
            size_only.window,
            Some(WindowGeom {
                x: None,
                y: None,
                width: 800.0,
                height: 500.0
            })
        );
    }

    #[test]
    fn command_line_beats_settings_file() {
        let s = Settings {
            period_size: Some(64),
            ..Default::default()
        };
        let raw = vec!["nice-tuner".to_owned(), "-p".to_owned(), "128".to_owned()];
        let args = effective_args(&s, &raw);
        // The file's 64 must not appear; the user's 128 must.
        assert!(!args.iter().any(|a| a == "64"));
        let p = args.iter().position(|a| a == "-p").unwrap();
        assert_eq!(args[p + 1], "128");
    }

    #[test]
    fn settings_file_beats_our_input_latency_default() {
        let s = Settings {
            input_latency: Some(0.02),
            ..Default::default()
        };
        let args = effective_args(&s, &["x".to_owned()]);
        let i = args.iter().position(|a| a == "--input-latency").unwrap();
        assert_eq!(args[i + 1], "0.02");
        assert!(!args.iter().any(|a| a == "0.005"));
    }

    #[test]
    fn no_opinions_yet_yields_our_default_latency_only() {
        let args = effective_args(&Settings::default(), &["x".to_owned()]);
        let i = args.iter().position(|a| a == "--input-latency").unwrap();
        assert_eq!(args[i + 1], "0.005");
        // No other flags invented.
        assert_eq!(args.len(), 3);
    }

    #[test]
    fn json_round_trip_and_lenient_reading() {
        let s = Settings {
            backend: Some("asio".into()),
            input_device: Some("Komplete Input 1/2".into()),
            period_size: Some(64),
            input_channel: Some(1),
            a4_ref: Some(432.0),
            in_tune_cents: Some(2.0),
            gauge_knee_cents: Some(7.0),
            ..Default::default()
        };
        let text = serde_json::to_string(&s).unwrap();
        let back: Settings = serde_json::from_str(&text).unwrap();
        assert_eq!(back, s);
        // Unknown fields ignored, missing fields None — forward compatible.
        let loose: Settings =
            serde_json::from_str(r#"{"backend":"wasapi","future_thing":42}"#).unwrap();
        assert_eq!(loose.backend.as_deref(), Some("wasapi"));
        assert_eq!(loose.period_size, None);
    }

    #[test]
    fn normalize_repairs_legacy_coreaudio_key() {
        let mut s = Settings {
            backend: Some("coreaudio".into()),
            ..Default::default()
        };
        s.normalize();
        assert_eq!(s.backend.as_deref(), Some("core-audio"));
        // Correct values pass through untouched.
        let mut ok = Settings {
            backend: Some("wasapi".into()),
            ..Default::default()
        };
        ok.normalize();
        assert_eq!(ok.backend.as_deref(), Some("wasapi"));
    }
}
