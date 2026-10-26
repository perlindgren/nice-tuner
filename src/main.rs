//! Standalone target for testing the tuner without a DAW.

use nice_plug::prelude::*;
use nice_tuner::TunerPlugin;

/// nice-plug's default `--input-latency` (0.15 s) buffers input against output
/// as slack between its two independent device streams — a value chosen for
/// general-purpose plugins on unreliable hardware, which on a live tuner is
/// audible slap-back. Default the margin to 5 ms instead; an explicit
/// `--input-latency` on the command line still wins.
fn args_with_low_latency_default() -> Vec<String> {
    let mut args: Vec<String> = std::env::args().collect();
    let user_set_it = args
        .iter()
        .any(|a| a == "--input-latency" || a.starts_with("--input-latency="));
    if !user_set_it {
        args.insert(1, "--input-latency".to_owned());
        args.insert(2, "0.005".to_owned());
    }
    args
}

fn main() {
    nice_export_standalone_with_args::<TunerPlugin, _>(args_with_low_latency_default());
}
