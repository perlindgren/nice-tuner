//! Demo: render a 440 Hz sine, stream it through the detector, print hits.
//!
//! Run with `cargo run --example wave`.

use pitch::{freq_to_note, Config, PitchDetector};

fn main() {
    let sample_rate = 48000.0f64;
    let seconds = 2.0;
    let n = (sample_rate * seconds) as usize;

    // 440 Hz test tone, amplitude 0.5.
    let samples: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f64 / sample_rate;
            (0.5 * (2.0 * std::f64::consts::PI * 440.0 * t).sin()) as f32
        })
        .collect();

    let mut det = PitchDetector::new(Config::default());

    println!("feeding {} s of 440 Hz sine (chunk = 512 frames)", seconds);
    let mut hits = 0usize;
    for chunk in samples.chunks(512) {
        if let Some(d) = det.push_samples(chunk) {
            let note = freq_to_note(d.freq_hz, 440.0);
            hits += 1;
            println!(
                "hit {:3}: {:8.2} Hz  clarity {:.4}  rms {:.4}  ->  {}{}  cents {:+.1}",
                hits, d.freq_hz, d.clarity, d.rms, note.name, note.octave, note.cents
            );
        }
    }
    if hits == 0 {
        println!("(no detections — should not happen for this signal)");
    }
}
