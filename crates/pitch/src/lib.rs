//! # pitch
//!
//! Real-time monophonic pitch detection for a guitar / bass / violin style
//! hardware-instrument tuner inside an audio plugin.
//!
//! The crate is `std`-only (no external dependencies); all math uses the
//! built-in `f32`/`f64` intrinsics.
//!
//! ## Algorithm
//!
//! [`PitchDetector`] implements **YIN** (de Cheveigné & Kawahara, *J. Acoust.
//! Soc. Am.* 111(2), 2002) on an integer-decimated copy of the input signal:
//!
//! 1. A 2-pole Butterworth low-pass (Q = 1/sqrt(2), cutoff = 0.45 * decimated
//!    Nyquist) runs at the host rate to prevent aliasing, then the signal is
//!    decimated by [`Config::downsample_factor`] (internal analysis rate =
//!    `sample_rate / downsample_factor`).
//! 2. Decimated samples fill a 2048-sample analysis window; after each pass
//!    the last half is kept (hop = window / 2, i.e. 50 % overlap).
//! 3. The difference function `d(tau) = sum_j (x[j] - x[j+tau])^2` is computed
//!    in `f32` for every lag, and the **cumulative mean normalized
//!    difference** (CMND) is formed with a running cumulative sum:
//!    `cmnd(tau) = d(tau) / ((1/tau) * sum_{k=1..=tau} d(k))`.
//! 4. The search range is forced to `lag_min = ceil(sr' / max_freq)` ..
//!    `lag_max = floor(sr' / min_freq)` at the analysis rate `sr'`.
//! 5. Absolute threshold: the first local minimum of the CMND below `0.10`
//!    within the range is chosen (falling back to the CMND global minimum
//!    when nothing dips below the threshold). A parabolic interpolation of
//!    the three CMND samples around the chosen lag yields a fractional lag.
//! 6. `freq_hz = sr' / lag`, `rms` is the RMS of the analysis window, and
//!    `clarity = clamp(1.0 - cmnd_min, 0.0, 1.0)` where `cmnd_min` is the
//!    vertex value of the parabola fitted at the chosen lag. Strongly
//!    periodic windows have a CMND dip far below the 0.10 threshold, giving
//!    clarity close to 1.0; noise or silence produce CMND values around 1.0
//!    or above, giving clarity near 0.0.
//!
//! ## Clarity gate
//!
//! [`PitchDetector::push_samples`] returns `Some(Detection)` only on a call
//! that completed an analysis window *and* whose result passed
//! [`Config::min_clarity`]; otherwise `None`. Consumers wanting raw
//! (ungated) results can set `min_clarity = 0.0`.
//!
//! ## Allocation behaviour
//!
//! All buffers (analysis window, difference function, CMND) are allocated in
//! [`PitchDetector::new`] and reused for the lifetime of the detector;
//! `push_samples` never grows any heap in steady state (enforced by
//! construction — indexing into pre-sized vectors, no iterators that
//! allocate, no formatting — and checked by a counting-allocator unit test).
//!
//! ## Graceful degradation
//!
//! If [`Config::min_freq`] demands a longer lag than the 2048-sample window
//! can support, the lag range is clamped so that at least 64 samples are
//! summed, which silently raises the effective minimum detectable frequency
//! (never a panic or a garbage result).

mod note;

pub use note::{freq_to_note, Note};

/// Tuner / detector configuration.
#[derive(Debug, Clone, Copy)]
pub struct Config {
    /// Host sample rate in Hz (e.g. 48000.0).
    pub sample_rate: f64,
    /// Lowest fundamental to search for, in Hz (e.g. 60.0, low open E of a bass).
    pub min_freq: f64,
    /// Highest fundamental to search for, in Hz (e.g. 2000.0).
    pub max_freq: f64,
    /// Integer decimation factor applied before analysis. The detector runs
    /// at `sample_rate / downsample_factor`. Values < 1 are clamped to 1.
    pub downsample_factor: usize,
    /// Confidence gate in `0.0..=1.0`: windows whose clarity falls below
    /// this value produce `None` from [`PitchDetector::push_samples`].
    pub min_clarity: f64,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            sample_rate: 48000.0,
            min_freq: 60.0,
            max_freq: 2000.0,
            downsample_factor: 4,
            min_clarity: 0.85,
        }
    }
}

/// Result of one completed analysis window.
#[derive(Debug, Clone, Copy)]
pub struct Detection {
    /// Detected fundamental frequency in Hz (fractional via parabolic
    /// interpolation of the CMND).
    pub freq_hz: f64,
    /// Confidence in `0.0..=1.0`; `1.0 - cmnd_min` at the chosen lag
    /// (see the crate-level docs for the exact mapping).
    pub clarity: f64,
    /// RMS level of the analysed (decimated) window, for gating / level metering.
    pub rms: f32,
}

/// Analysis window length in *decimated* samples (2048 @ 12 kHz = ~170 ms,
/// comfortably > 2 periods for 80 Hz fundamentals and their CMND needs).
const WINDOW_SIZE: usize = 2048;
/// Hop between consecutive analysis passes (50 % overlap: the last half of
/// each window is kept and refilled).
const HOP_SIZE: usize = WINDOW_SIZE / 2;
/// Minimum number of samples that must remain for the difference-function
/// sum; bounds the maximum usable lag to `WINDOW_SIZE - MIN_SUM_LEN`.
const MIN_SUM_LEN: usize = 64;
/// Maximum lag the difference buffers are sized for.
const MAX_LAG: usize = WINDOW_SIZE - MIN_SUM_LEN;
/// YIN absolute threshold (paper: 0.10).
const CMND_THRESHOLD: f32 = 0.10;

/// 2-pole Butterworth low-pass (RBJ cookbook LPF, Q = 1/sqrt(2)), used to
/// band-limit the signal before decimation.
#[derive(Debug, Clone, Copy)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    x1: f64,
    x2: f64,
    y1: f64,
    y2: f64,
}

impl Biquad {
    /// Design a low-pass biquad at sample rate `fs` with cutoff `fc`.
    fn lowpass(fs: f64, fc: f64) -> Self {
        let fc = fc.min(fs * 0.49).max(fs * 1e-9);
        let w0 = 2.0 * std::f64::consts::PI * fc / fs;
        let cos_w = w0.cos();
        let sin_w = w0.sin();
        // Q = 1/sqrt(2) (Butterworth)  =>  alpha = sin(w0) / (2Q) = sin(w0)/sqrt(2).
        let alpha = sin_w * std::f64::consts::FRAC_1_SQRT_2;
        let a0 = 1.0 + alpha;
        Biquad {
            b0: (1.0 - cos_w) * 0.5 / a0,
            b1: (1.0 - cos_w) / a0,
            b2: (1.0 - cos_w) * 0.5 / a0,
            a1: -2.0 * cos_w / a0,
            a2: (1.0 - alpha) / a0,
            x1: 0.0,
            x2: 0.0,
            y1: 0.0,
            y2: 0.0,
        }
    }

    #[inline]
    fn process(&mut self, x: f64) -> f64 {
        let y = self.b0 * x + self.b1 * self.x1 + self.b2 * self.x2
            - self.a1 * self.y1
            - self.a2 * self.y2;
        self.x2 = self.x1;
        self.x1 = x;
        self.y2 = self.y1;
        self.y1 = y;
        y
    }

    fn reset(&mut self) {
        self.x1 = 0.0;
        self.x2 = 0.0;
        self.y1 = 0.0;
        self.y2 = 0.0;
    }
}

/// Real-time monophonic pitch detector (YIN). See the crate-level docs for
/// the algorithm and allocation guarantees.
///
/// ```
/// use pitch::{Config, PitchDetector};
/// let det = PitchDetector::new(Config::default());
/// # let _ = det;
/// ```
pub struct PitchDetector {
    /// Sanitized copy of the user config (kept live so `set_*` can update
    /// the search range without reallocation).
    cfg: Config,
    /// Anti-aliasing filter, run at the host rate before decimation.
    lowpass: Biquad,
    /// Decimation phase counter (keeps every `downsample_factor`-th sample).
    decimate_phase: usize,
    /// Analysis window of decimated samples (`WINDOW_SIZE`, preallocated).
    window: Vec<f32>,
    /// Number of valid samples currently in `window`.
    filled: usize,
    /// Difference function scratch (preallocated, index = lag).
    d: Vec<f32>,
    /// CMND scratch (preallocated, index = lag).
    cmnd: Vec<f32>,
}

impl PitchDetector {
    /// Create a detector, allocating all internal buffers up front.
    ///
    /// Invalid config fields are clamped to sane ranges rather than rejected
    /// (see the module docs on graceful degradation).
    pub fn new(cfg: Config) -> Self {
        let cfg = sanitize(cfg);
        let analysis_rate = cfg.sample_rate / cfg.downsample_factor as f64;
        let lowpass = Biquad::lowpass(cfg.sample_rate, analysis_rate * 0.45);
        PitchDetector {
            cfg,
            lowpass,
            decimate_phase: 0,
            window: vec![0.0f32; WINDOW_SIZE],
            filled: 0,
            d: vec![0.0f32; MAX_LAG + 1],
            cmnd: vec![0.0f32; MAX_LAG + 1],
        }
    }

    /// Feed host-rate audio (any chunk size).
    ///
    /// Internally low-passes, decimates, and runs one YIN pass per completed
    /// analysis window (window = 2048 decimated samples, hop = window / 2).
    /// Returns the detection of the **last** window completed by this call,
    /// or `None` if no window completed or the result was gated by
    /// [`Config::min_clarity`].
    ///
    /// Allocation-free in steady state: every buffer is sized in `new()` and
    /// only written through indexing here. (Checked by the counting
    /// `GlobalAlloc` unit test in this crate.)
    pub fn push_samples(&mut self, samples: &[f32]) -> Option<Detection> {
        let factor = self.cfg.downsample_factor;
        let mut last = None;
        for &s in samples {
            let filtered = self.lowpass.process(f64::from(s));
            self.decimate_phase += 1;
            if self.decimate_phase < factor {
                continue;
            }
            self.decimate_phase = 0;
            self.window[self.filled] = filtered as f32;
            self.filled += 1;
            if self.filled == WINDOW_SIZE {
                let det = self.analyse();
                // Overlap by keeping the last half of the window.
                self.window.copy_within(HOP_SIZE..WINDOW_SIZE, 0);
                self.filled = HOP_SIZE;
                last = det;
            }
        }
        last
    }

    /// Update the lowest searchable frequency at runtime (allocation-free;
    /// out-of-range values are clamped).
    pub fn set_min_freq(&mut self, f: f64) {
        self.cfg.min_freq = f;
        sanitize_freq_bounds(&mut self.cfg);
    }

    /// Update the highest searchable frequency at runtime (allocation-free;
    /// out-of-range values are clamped).
    pub fn set_max_freq(&mut self, f: f64) {
        self.cfg.max_freq = f;
        sanitize_freq_bounds(&mut self.cfg);
    }

    /// Clear all streaming state (filter memory, decimation phase, partial
    /// window) as if the detector had just been created.
    pub fn reset(&mut self) {
        self.lowpass.reset();
        self.decimate_phase = 0;
        self.filled = 0;
        for v in self.window.iter_mut() {
            *v = 0.0;
        }
    }

    /// One YIN pass over the filled analysis window. Borrow-free of the
    /// allocator; returns `None` on an invalid lag range or when the clarity
    /// gate rejects the estimate.
    fn analyse(&mut self) -> Option<Detection> {
        // Copy out scalar config first so `self` fields can be destructured
        // into disjoint borrows below.
        let cfg = self.cfg;
        let sr = cfg.sample_rate / cfg.downsample_factor as f64;

        // Forced search range from min/max freq at the analysis rate.
        let mut lag_max = (sr / cfg.min_freq).floor().max(1.0) as usize;
        if lag_max > MAX_LAG {
            // Window too short for min_freq's lag range: clamp gracefully
            // (effective minimum detectable frequency becomes sr / MAX_LAG).
            lag_max = MAX_LAG;
        }
        let lag_min = (sr / cfg.max_freq).ceil().max(1.0) as usize;
        if lag_min > lag_max {
            return None;
        }
        // Fixed integration length for every lag, so the CMND normalization
        // is comparable across lags.
        let n = WINDOW_SIZE - lag_max;

        let PitchDetector {
            ref window,
            ref mut d,
            ref mut cmnd,
            ..
        } = *self;
        let x: &[f32] = window;

        // RMS of the analysed window (level meter / external gating).
        let mut ssq = 0f32;
        for &v in x.iter() {
            ssq += v * v;
        }
        let rms = (ssq / WINDOW_SIZE as f32).sqrt();

        // Difference function + CMND in one pass with a running cumulative
        // sum. f32 differencing is plenty: values are O(n * amp^2) <= ~1e4.
        cmnd[0] = 1.0;
        let mut cum = 0f32;
        for tau in 1..=lag_max {
            let mut sum = 0f32;
            for j in 0..n {
                let diff = x[j] - x[j + tau];
                sum += diff * diff;
            }
            d[tau] = sum;
            cum += sum;
            cmnd[tau] = if cum > f32::MIN_POSITIVE {
                (sum * tau as f32) / cum
            } else {
                1.0
            };
        }

        // Absolute threshold: first local CMND minimum below 0.10 inside the
        // forced range; otherwise fall back to the CMND global minimum.
        let mut tau_best = usize::MAX;
        let mut tau = lag_min;
        while tau <= lag_max {
            if cmnd[tau] < CMND_THRESHOLD {
                while tau + 1 <= lag_max && cmnd[tau + 1] < cmnd[tau] {
                    tau += 1;
                }
                tau_best = tau;
                break;
            }
            tau += 1;
        }
        if tau_best == usize::MAX {
            let mut best = lag_min;
            for t in lag_min..=lag_max {
                if cmnd[t] < cmnd[best] {
                    best = t;
                }
            }
            tau_best = best;
        }

        // Parabolic interpolation on the CMND around the chosen lag.
        let (t_frac, cmnd_min) = if tau_best > 0 && tau_best < lag_max {
            let y0 = cmnd[tau_best - 1];
            let y1 = cmnd[tau_best];
            let y2 = cmnd[tau_best + 1];
            let denom = y0 - 2.0 * y1 + y2;
            if denom > 1e-30 {
                let delta = (0.5 * (y0 - y2) / denom).clamp(-1.0, 1.0);
                let vertex = y1 - 0.25 * (y0 - y2) * delta;
                (
                    tau_best as f64 + f64::from(delta),
                    f64::from(vertex.max(0.0)),
                )
            } else {
                (tau_best as f64, f64::from(y1))
            }
        } else {
            (tau_best as f64, f64::from(cmnd[tau_best]))
        };

        if !(t_frac > 0.5) {
            return None;
        }
        let freq_hz = sr / t_frac;
        // Clarity mapping: 1.0 minus the CMND minimum at the chosen lag,
        // clamped into 0..=1. See crate docs.
        let clarity = (1.0 - cmnd_min).clamp(0.0, 1.0);
        if freq_hz.is_finite() && clarity >= cfg.min_clarity {
            Some(Detection {
                freq_hz,
                clarity,
                rms,
            })
        } else {
            None
        }
    }
}

/// Clamp the whole config into a usable range.
fn sanitize(mut cfg: Config) -> Config {
    if !(cfg.sample_rate.is_finite() && cfg.sample_rate > 0.0) {
        cfg.sample_rate = 48000.0;
    }
    cfg.downsample_factor = cfg.downsample_factor.max(1);
    cfg.min_clarity = if cfg.min_clarity.is_finite() {
        cfg.min_clarity.clamp(0.0, 1.0)
    } else {
        0.85
    };
    sanitize_freq_bounds(&mut cfg);
    cfg
}

/// Clamp `min_freq`/`max_freq` into `(0, analysis_nyquist]` with
/// `min_freq < max_freq`, keeping the detector able to run for any input.
fn sanitize_freq_bounds(cfg: &mut Config) {
    let analysis_rate = cfg.sample_rate / cfg.downsample_factor as f64;
    if !(cfg.min_freq.is_finite() && cfg.min_freq > 0.0) {
        cfg.min_freq = 60.0;
    }
    if !(cfg.max_freq.is_finite() && cfg.max_freq > 0.0) {
        cfg.max_freq = 2000.0;
    }
    cfg.min_freq = cfg.min_freq.clamp(1.0, analysis_rate * 0.45);
    cfg.max_freq = cfg
        .max_freq
        .clamp(cfg.min_freq * 1.01, analysis_rate * 0.5);
}

// ---------------------------------------------------------------------------
// Test-only allocation counting.
//
// A `GlobalAlloc` wrapper affects the whole process, but the counter only
// ticks while the *current thread's* `ARMED` flag is set, so unrelated test
// threads running in parallel cannot pollute the measurement. The flag is a
// const-initialized `Cell<bool>` thread-local: no lazy-init heap allocation
// happens inside the allocator itself.
// ---------------------------------------------------------------------------
#[cfg(test)]
mod alloc_tests {
    use super::*;
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    use std::sync::atomic::{AtomicUsize, Ordering};

    thread_local! {
        /// Only count allocations made by the armed thread.
        static ARMED: Cell<bool> = const { Cell::new(false) };
    }

    /// Allocation count (alloc calls) attributed to armed threads.
    static COUNT: AtomicUsize = AtomicUsize::new(0);

    struct CountingAlloc;

    unsafe impl GlobalAlloc for CountingAlloc {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            if ARMED.with(|a| a.get()) {
                COUNT.fetch_add(1, Ordering::Relaxed);
            }
            System.alloc(layout)
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            System.dealloc(ptr, layout)
        }
    }

    #[global_allocator]
    static ALLOC: CountingAlloc = CountingAlloc;

    /// Spec test 7: after warm-up, `push_samples` must not touch the heap.
    #[test]
    fn push_samples_does_not_allocate_in_steady_state() {
        // Sanity: the counting allocator must actually count on this thread,
        // otherwise the zero-assertion below would be vacuous.
        COUNT.store(0, Ordering::Relaxed);
        ARMED.with(|a| a.set(true));
        let probe: Vec<u8> = (0..1024usize).map(|i| i as u8).collect();
        let probe_count = COUNT.load(Ordering::Relaxed);
        drop(probe);
        assert!(
            probe_count > 0,
            "counting allocator not observing allocations (test harness broken)"
        );
        COUNT.store(0, Ordering::Relaxed);

        let sr = 48000.0f64;
        let sig: Vec<f32> = (0..sr as usize)
            .map(|i| {
                (0.5 * (2.0 * std::f64::consts::PI * 440.0 * i as f64 / sr).sin()) as f32
            })
            .collect();

        let mut det = PitchDetector::new(Config::default());
        det.push_samples(&sig); // warm-up: reach steady state

        COUNT.store(0, Ordering::Relaxed);
        // (still armed from the probe above)
        let mut n_det = 0usize;
        for chunk in sig.chunks(512) {
            if det.push_samples(chunk).is_some() {
                n_det += 1;
            }
        }
        let count = COUNT.load(Ordering::Relaxed);
        ARMED.with(|a| a.set(false));

        assert!(n_det > 0, "sanity: expected detections for a 440 Hz sine");
        assert_eq!(
            count, 0,
            "push_samples allocated {count} times in steady state"
        );
    }
}
