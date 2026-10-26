//! Hard gate: the *detection* path of the pitch detector performs zero heap
//! allocations once the detector is warm.
//!
//! The buffers (`window`, `d`, `cmnd`) are allocated in `PitchDetector::new`,
//! which runs from the plugin's `activate()` — the real-audio-callback
//! contract then requires silence from the allocator on every block. This
//! test enforces that contract at the DSP level with a counting global
//! allocator, independent of nice-plug's `assert_process_allocs` (which
//! guards the full `process()` callback only in debug builds at runtime).
//!
//! Kept to one test in its own binary so the counting window has no other
//! allocator traffic: the libtest harness is parked while this runs.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

static COUNTING: AtomicBool = AtomicBool::new(false);
static ALLOCS: AtomicUsize = AtomicUsize::new(0);

struct Counting;

unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if COUNTING.load(Ordering::Relaxed) {
            ALLOCS.fetch_add(1, Ordering::Relaxed);
        }
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Feed `sample_rate`-Hz audio to the detector and assert the measured
/// stretch allocated nothing.
#[test]
fn detection_path_never_allocates_when_warm() {
    use pitch::{Config, PitchDetector};

    const SR: f32 = 44100.0;
    let cfg = Config {
        sample_rate: SR as f64,
        min_freq: 30.0,
        max_freq: 2100.0,
        ..Config::default()
    };
    let mut detector = PitchDetector::new(cfg);

    // 12 s of a 110 Hz sine with a touch of harmonic content and a slow
    // amplitude ramp — plenty of completed windows and varying RMS.
    let n = (SR * 12.0) as usize;
    let signal: Vec<f32> = (0..n)
        .map(|i| {
            let t = i as f32 / SR;
            let env = 0.4 + 0.3 * (std::f32::consts::TAU * 0.5 * t).sin();
            env * ((std::f32::consts::TAU * 110.0 * t).sin()
                + 0.4 * (std::f32::consts::TAU * 220.0 * t).sin())
        })
        .collect();

    // Warm-up stretch: first passes may touch lazily initialized state.
    // (Detector buffers themselves were allocated in `new`.)
    let warm = (SR * 2.0) as usize;
    for chunk in signal[..warm].chunks(512) {
        let _ = detector.push_samples(chunk);
    }

    // Measured stretch: ten full seconds of detections must stay dark.
    ALLOCS.store(0, Ordering::Relaxed);
    COUNTING.store(true, Ordering::Relaxed);
    let mut detections = 0usize;
    for chunk in signal[warm..].chunks(512) {
        if detector.push_samples(chunk).is_some() {
            detections += 1;
        }
    }
    COUNTING.store(false, Ordering::Relaxed);

    let allocs = ALLOCS.load(Ordering::Relaxed);
    // Sanity: the signal really did drive detections, otherwise the zero
    // below would be vacuous.
    assert!(
        detections > 50,
        "expected detections to exercise the hot path, got {detections}"
    );
    assert_eq!(
        allocs, 0,
        "detection path performed {allocs} heap allocations"
    );
}
