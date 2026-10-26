//! Standalone target for testing the tuner without a DAW.

use nice_plug::prelude::*;
use nice_tuner::TunerPlugin;
use nice_tuner::settings::{self, Settings};
use std::sync::atomic::Ordering;

fn main() {
    // Helper mode for the settings panel (see devices::snapshot_via_child):
    // enumerate one backend, print JSON on stdout, exit — WITHOUT opening
    // a window or a stream. The whole point is a process that holds no
    // ASIO driver, so it can see drivers the live app cannot.
    {
        let argv: Vec<String> = std::env::args().collect();
        if argv.get(1).map(String::as_str) == Some("--list-devices") {
            let backend = argv
                .get(2)
                .filter(|s| !s.is_empty())
                .cloned()
                .unwrap_or_else(|| "auto".to_owned());
            let inp = argv.get(3).filter(|s| !s.is_empty()).map(String::as_str);
            let outp = argv.get(4).filter(|s| !s.is_empty()).map(String::as_str);
            let snap = nice_tuner::devices::snapshot(&backend, inp, outp);
            print!("{}", serde_json::to_string(&snap).unwrap_or_default());
            return;
        }
    }

    nice_tuner::STANDALONE_MODE.store(true, Ordering::Relaxed);

    let raw: Vec<String> = std::env::args().collect();
    let (settings, warning) = Settings::load();
    if let Some(warning) = warning {
        eprintln!("nice-tuner: {warning}");
    }

    // "(default)" sample rate means the rate the DRIVER is set to (the NI
    // control panel's number), not nice-plug's hard-coded 48 kHz. Resolve
    // it once here; an explicit file value or -r/--sample-rate still wins
    // (effective_args applies CLI-over-file precedence as usual).
    // The answer is WRITTEN BACK: this boot's rate is then known to the
    // settings page even when a later ASIO query lands blind — queried
    // once, remembered, not re-guessed.
    let mut settings = settings;
    if settings.sample_rate.is_none()
        && let Some(rate) = nice_tuner::devices::driver_default_rate(
            settings.backend.as_deref().unwrap_or("auto"),
            settings.input_device.as_deref(),
            settings.output_device.as_deref(),
        )
    {
        settings.sample_rate = Some(rate);
        if let Err(e) = settings.save() {
            eprintln!("nice-tuner: could not remember driver rate: {e}");
        }
    }

    // Open-attempt loop: ASIO is exclusive per driver, so "device refuses
    // to open" is usually transient — the previous instance (Apply &
    // restart) is still releasing it, or a host like Reaper owns it right
    // now. A run that dies within its first seconds is a failed startup,
    // so retry briefly; a run that lasted is a normal window close.
    const ATTEMPTS: u32 = 4;
    for attempt in 1..=ATTEMPTS {
        let args = settings::effective_args(&settings, &raw);
        let started = std::time::Instant::now();
        nice_export_standalone_with_args::<TunerPlugin, _>(args);

        // The settings page's "Apply & restart" unwinds the editor through
        // a deliberate window close; honor it with a fresh process — new
        // settings are read from the file at its startup, which is the point.
        if nice_tuner::restart_requested() {
            respawn(&raw);
        }
        if started.elapsed() >= std::time::Duration::from_secs(3) {
            return; // user closed a working window
        }
        if attempt < ATTEMPTS {
            eprintln!(
                "nice-tuner: audio device would not open (attempt {attempt}/{ATTEMPTS}); \
                 it may still be releasing from a restart or held by another app (Reaper?). \
                 Retrying in 700 ms — Ctrl+C to stop."
            );
            std::thread::sleep(std::time::Duration::from_millis(700));
        }
    }
    eprintln!(
        "nice-tuner: could not open the audio device after {ATTEMPTS} attempts. \
         Close any app using it (Reaper's audio system, another tuner window) \
         and start again."
    );
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
