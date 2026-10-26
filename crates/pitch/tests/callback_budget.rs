//! Detection-cost probe: how long does ONE `push_samples` call take?
//!
//! Detection is bursty — most calls just buffer samples; one in a while runs
//! a full window search. That cost lands on the editor's frost-frame pump
//! (nice-plug's main/UI thread; analysis deliberately does NOT run on the
//! audio thread), where the frame budget is ~16.7 ms. In the dev profile an
//! opt-level-0 search burst measures ~2.3 ms and visibly stutters the frame
//! loop, which is why `Cargo.toml` pins
//! `[profile.dev.package.pitch] opt-level = 3` (burst → ~0.26 ms). The
//! assertions below are what enforces that pin: at opt-level 0 this fails.

use pitch::{Config, PitchDetector};
use std::time::Instant;

const SR: f64 = 48000.0;
const PERIOD: usize = 64;

#[test]
fn search_burst_stays_inside_the_frame_budget() {
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

    // The UI pump must keep up comfortably: faster than 4× realtime.
    assert!(wall < 3.0, "cannot keep up: {wall:.2}s for 12s audio");
    // Discriminator is p99 (stable under uniform CPU load; the absolute max
    // is not): pinned dev profile measures ~0.2 ms, unpinned ~2.2 ms — an
    // order of magnitude apart, so 1.0 ms separates them without flaking on
    // a loaded machine. The max bound just catches pathological regressions.
    assert!(
        p99 < 1.0,
        "p99 {p99:.2} ms: bursts approaching the unpinned (~2.2 ms) regime"
    );
    assert!(
        max < 4.0,
        "pathological burst {max:.2} ms — over a quarter frame"
    );
    assert!(detections > 50, "detector silent: {detections} detections");
}
