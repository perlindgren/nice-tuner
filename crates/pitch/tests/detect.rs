//! YIN detector tests: sines, sawtooth (octave robustness), silence/noise,
//! note mapping and irregular-chunk streaming.
//!
//! All float assertions use tolerances, never exact equality.

use pitch::{freq_to_note, Config, Detection, PitchDetector};

const SR: f64 = 48000.0;

/// Sine helper: amplitude `amp`, frequency `f`, `dur` seconds at SR.
fn sine(f: f64, amp: f32, dur: f64) -> Vec<f32> {
    let n = (SR * dur) as usize;
    let mut v = Vec::with_capacity(n);
    // f64 phase accumulation, then to f32 — mirrors how a host would render.
    for i in 0..n {
        let t = i as f64 / SR;
        v.push((amp as f64 * (2.0 * std::f64::consts::PI * f * t).sin()) as f32);
    }
    v
}

/// Sawtooth-ish tone: 5 harmonics with 1/k amplitudes (hand-summed sines).
fn saw5(f: f64, amp: f32, dur: f64) -> Vec<f32> {
    let n = (SR * dur) as usize;
    let mut v = Vec::with_capacity(n);
    // Normalize by the sum of harmonic amplitudes so peak stays ~amp.
    let norm: f64 = (1..=5).map(|k| 1.0 / k as f64).sum();
    for i in 0..n {
        let t = i as f64 / SR;
        let mut s = 0.0f64;
        for k in 1..=5 {
            s += (2.0 * std::f64::consts::PI * f * k as f64 * t).sin() / k as f64;
        }
        v.push(((amp as f64) * s / norm) as f32);
    }
    v
}

/// Run samples in `chunk`-sized pieces, return every detection emitted.
fn run_chunks(det: &mut PitchDetector, samples: &[f32], chunk: usize) -> Vec<Detection> {
    let mut out = Vec::new();
    for c in samples.chunks(chunk) {
        if let Some(d) = det.push_samples(c) {
            out.push(d);
        }
    }
    out
}

/// Feed a tone and return the best (highest-clarity) detection.
fn best_detection(samples: &[f32], cfg: Config) -> Detection {
    let mut det = PitchDetector::new(cfg);
    let dets = run_chunks(&mut det, samples, 256);
    assert!(!dets.is_empty(), "expected at least one detection");
    *dets
        .iter()
        .max_by(|a, b| a.clarity.partial_cmp(&b.clarity).unwrap())
        .unwrap()
}

// ---- spec test 1: A4 440 Hz sine ------------------------------------------

#[test]
fn a4_sine_freq_note_and_clarity() {
    let d = best_detection(&sine(440.0, 0.5, 2.0), Config::default());
    assert!(
        (d.freq_hz - 440.0).abs() < 1.0,
        "freq {} far from 440",
        d.freq_hz
    );
    assert!(d.clarity > 0.9, "clarity {} too low", d.clarity);

    let n = freq_to_note(d.freq_hz, 440.0);
    assert_eq!(n.name, "A");
    assert_eq!(n.octave, 4);
    assert!(n.cents.abs() < 5.0, "cents {} too far from 0", n.cents);
}

// ---- spec test 2: low E2 and guitar E4 -------------------------------------

#[test]
fn low_e2_sine() {
    let d = best_detection(&sine(82.41, 0.6, 2.5), Config::default());
    assert!(
        (d.freq_hz - 82.41).abs() < 0.5,
        "freq {} far from 82.41",
        d.freq_hz
    );
    assert!(d.clarity > 0.9, "clarity {} too low", d.clarity);
}

#[test]
fn guitar_e4_sine() {
    let d = best_detection(&sine(329.63, 0.5, 2.0), Config::default());
    assert!(
        (d.freq_hz - 329.63).abs() < 1.0,
        "freq {} far from 329.63",
        d.freq_hz
    );
    assert!(d.clarity > 0.9, "clarity {} too low", d.clarity);
}

// ---- spec test 3: sawtooth octave robustness --------------------------------

#[test]
fn sawtooth_110hz_no_octave_error() {
    let d = best_detection(&saw5(110.0, 0.5, 2.0), Config::default());
    assert!(
        (d.freq_hz - 110.0).abs() < 1.0,
        "freq {} far from 110 (octave error?)",
        d.freq_hz
    );
}

// ---- spec test 4: silence and noise ------------------------------------------

/// Small deterministic LCG (Knuth MMIX constants) producing f32 in -1..1.
struct Lcg(u64);
impl Lcg {
    fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        // Use the high bits; low LCG bits are poorly random.
        let u = (self.0 >> 40) as f32; // 24 bits
        u / (1u32 << 23) as f32 * 2.0 - 1.0
    }
}

#[test]
fn silence_produces_no_confident_detection() {
    let mut cfg = Config::default();
    cfg.min_clarity = 0.0; // observe raw results, ungated
    let mut det = PitchDetector::new(cfg);
    let silence = vec![0.0f32; (SR * 2.0) as usize];
    let dets = run_chunks(&mut det, &silence, 512);
    // Either nothing at all, or only low-clarity results below the gate.
    for d in &dets {
        assert!(
            d.clarity < cfg_default_min_clarity(),
            "silence produced clarity {}",
            d.clarity
        );
    }
}

fn cfg_default_min_clarity() -> f64 {
    Config::default().min_clarity
}

#[test]
fn white_noise_never_confident() {
    // Ungated so we can inspect raw clarity values.
    let mut cfg = Config::default();
    cfg.min_clarity = 0.0;
    let mut det = PitchDetector::new(cfg);

    let mut rng = Lcg::new(0x1234_5678_9abc_def0);
    let noise: Vec<f32> = (0..(SR * 3.0) as usize).map(|_| 0.5 * rng.next_f32()).collect();
    let dets = run_chunks(&mut det, &noise, 333);

    // Detections may exist (window completed) but must never be confident.
    for d in &dets {
        assert!(
            d.clarity <= 0.5,
            "noise produced clarity {} (> 0.5)",
            d.clarity
        );
    }

    // Also check the fully-gated path emits nothing.
    let mut gated = PitchDetector::new(Config::default());
    let mut rng2 = Lcg::new(0xdead_beef_cafe_1234);
    let noise2: Vec<f32> = (0..(SR * 2.0) as usize).map(|_| 0.5 * rng2.next_f32()).collect();
    let dets2 = run_chunks(&mut gated, &noise2, 512);
    assert!(
        dets2.is_empty(),
        "gated detector emitted {} detections on noise",
        dets2.len()
    );
}

// ---- spec test 5: A#4 note mapping + semitone boundary -----------------------

#[test]
fn note_a_sharp4_and_boundary() {
    let n = freq_to_note(466.16, 440.0);
    assert_eq!(n.name, "A#");
    assert_eq!(n.octave, 4);

    // Exactly halfway between A4 and A#4: cents must sit at +/-50.
    let boundary = 440.0 * 2f64.powf(0.5 / 12.0);
    let b = freq_to_note(boundary, 440.0);
    assert!(
        (b.cents.abs() - 50.0).abs() < 0.6,
        "boundary cents {} not near +/-50",
        b.cents
    );
    assert!(b.cents >= -50.0 && b.cents <= 50.0);

    // Nearest semitone frequency consistency at 440 tuning.
    let a = freq_to_note(443.0, 440.0);
    assert_eq!(a.name, "A");
    assert!((a.nearest_freq - 440.0).abs() < 1e-9);
    assert!(a.cents > 0.0); // sharp of A
}

// ---- spec test 6: streaming in irregular chunks ------------------------------

#[test]
fn streaming_irregular_chunks_matches_bulk() {
    let sig = sine(440.0, 0.5, 3.0);

    // Chunked: cycle of awkward sizes.
    let sizes = [7usize, 128, 511, 1024];
    let mut det = PitchDetector::new(Config::default());
    let mut dets = Vec::new();
    let mut i = 0usize;
    let mut idx = 0usize;
    while idx < sig.len() {
        let n = sizes[i % sizes.len()].min(sig.len() - idx);
        if let Some(d) = det.push_samples(&sig[idx..idx + n]) {
            dets.push(d);
        }
        idx += n;
        i += 1;
    }
    assert!(
        !dets.is_empty(),
        "no detection within the first 3 s of chunked audio"
    );

    // Bulk: one giant push of the same samples.
    let mut bulk = PitchDetector::new(Config::default());
    let bulk_det = bulk
        .push_samples(&sig)
        .expect("bulk push should complete windows");

    let last_chunked = dets.last().unwrap();
    assert!(
        (last_chunked.freq_hz - bulk_det.freq_hz).abs() < 0.2,
        "chunked {} vs bulk {}",
        last_chunked.freq_hz,
        bulk_det.freq_hz
    );
}

// ---- spec test 7: allocation-free steady state --------------------------------
// Implemented as a unit test in src/lib.rs (`alloc_tests` module) with a
// counting GlobalAlloc armed only on the measuring thread, because
// #[global_allocator] must live in the crate root and would otherwise make
// every other test in the process count allocations against a shared counter.

// ---- extra sanity: rms tracks level ---------------------------------------------

#[test]
fn rms_tracks_input_level() {
    let d_loud = best_detection(&sine(440.0, 0.8, 1.5), Config::default());
    let d_quiet = best_detection(&sine(440.0, 0.05, 1.5), Config::default());
    // Sine RMS ~ amp / sqrt(2); allow slack for the filter + f32 rounding.
    assert!(
        d_loud.rms > d_quiet.rms * 5.0,
        "loud {} quiet {}",
        d_loud.rms,
        d_quiet.rms
    );
    assert!((d_loud.rms as f64 - 0.8 / 2f64.sqrt()).abs() < 0.15);
}
