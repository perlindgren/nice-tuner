//! Callback-budget probe: how long does ONE 64-frame `push_samples` take?
//!
//! Background: detection is bursty — most callbacks just buffer samples, one
//! in a while runs a full window search. With a 64-frame period at 48 kHz the
//! RT budget per callback is 1.33 ms; an unoptimized (opt-level 0) search
//! burst blows it, which is why debug builds needed large buffers.
//! `Cargo.toml` pins `[profile.dev.package.pitch] opt-level = 3` for exactly
//! this reason; this test guards that budget in the dev profile.

use pitch::{Config, PitchDetector};
use std::time::Instant;

const SR: f64 = 48000.0;
const PERIOD: usize = 64;

#[test]
fn callback_budget_64_frames() {
    let cfg = Config {
        sample_rate: SR,
        ..Config::default()
    };
    let mut det = PitchDetector::new(cfg);

    let total = (12.0 * SR) as usize;
    let mut times: Vec<f64> = Vec::with_capacity(total / PERIOD);
    let start = Instant::now();
    let mut buf = [0.0f32; PERIOD];
    let mut detections = 0usize;
    for n in 0..total / PERIOD {
        let t0 = Instant::now();
        for (i, s) in buf.iter_mut().enumerate() {
            let ph = 2.0 * std::f64::consts::PI * 110.0 * ((n * PERIOD + i) as f64) / SR;
            *s = ph.sin() as f32 * 0.5;
        }
        if det.push_samples(&buf).is_some() {
            detections += 1;
        }
        times.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let wall = start.elapsed().as_secs_f64();

    times.sort_by(|a, b| a.total_cmp(b));
    let max = *times.last().unwrap();
    let p99 = times[times.len() * 99 / 100];

    println!(
        "budget: wall {wall:.2}s for 12 s audio ({:.1}x realtime), \
         p99 {:.3} ms, max {:.3} ms, detections {detections}",
        12.0 / wall,
        p99,
        max
    );

    // The detector must keep up: faster than 4× realtime amortized.
    assert!(wall < 3.0, "cannot keep up: {wall:.2}s for 12s audio");
    // Worst single callback must stay inside 4× the 1.33 ms budget. Tighter
    // would flake on a loaded CI box; unoptimized pitch code burst ~10 ms+,
    // so this still catches the regression it guards.
    assert!(
        max < 4.0,
        "worst callback {max:.2} ms blows the 64-frame budget"
    );
    assert!(detections > 50, "detector silent: {detections} detections");
}
