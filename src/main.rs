//! Standalone target for testing the tuner without a DAW.

use nice_plug::prelude::*;
use nice_tuner::TunerPlugin;
use nice_tuner::settings::{self, Settings};
use std::sync::atomic::Ordering;

fn main() {
    nice_tuner::STANDALONE_MODE.store(true, Ordering::Relaxed);

    let raw: Vec<String> = std::env::args().collect();
    let (settings, warning) = Settings::load();
    if let Some(warning) = warning {
        eprintln!("nice-tuner: {warning}");
    }
    // Precedence: explicit command-line flags, then the settings file,
    // then our defaults (nice-plug's 150 ms input-latency included).
    let args = settings::effective_args(&settings, &raw);

    nice_export_standalone_with_args::<TunerPlugin, _>(args);

    // The settings page's "Apply & restart" unwinds the editor through a
    // deliberate window close; honor it with a fresh process — new settings
    // are read from the file at its startup, which is the point.
    if nice_tuner::restart_requested() {
        respawn(&raw);
    }
}

fn respawn(raw: &[String]) -> ! {
    let exe = std::env::current_exe().expect("locate own executable for restart");
    // The dying parent may still hold the audio driver: ASIO drivers are
    // per-process exclusive DLLs whose teardown completes asynchronously
    // after the stream drop. Spawning immediately races the new process
    // against the release — the classic "device refuses to open" on Apply
    // & restart. A short pause makes the child the sole claimant.
    std::thread::sleep(std::time::Duration::from_millis(500));
    match std::process::Command::new(exe)
        .args(raw.iter().skip(1))
        .spawn()
    {
        Ok(_child) => std::process::exit(0),
        Err(e) => {
            eprintln!("nice-tuner: restart failed ({e}) — start the tuner again by hand");
            std::process::exit(1);
        }
    }
}
