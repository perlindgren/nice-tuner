//! Input-meter ballistics, as pure state over (rms, frame-peak, clock).
//!
//! Kept free of GPU/threading so every rule the user specified is nailed
//! down by unit tests instead of by eye:
//!
//! * the bar chases RMS with a fast attack and a moderate release;
//! * the peak marker jumps to any input exceeding the CURRENTLY SHOWN
//!   marker and re-arms the 1 s stick — that is the only arming condition;
//! * after 1 s without an exceedance the marker falls at 24 dB/s (real
//!   wall-clock dt, frame-rate proof), floored at the bar so it can never
//!   display below the level it is measuring;
//! * silence drains both layers to -120 dBFS.

/// Peak-hold level meter. All inputs are dBFS.
#[derive(Clone, Copy, Debug)]
pub struct PeakMeter {
    level_db: f32,
    peak_db: f32,
    stick_until_ms: u64,
    last_ms: u64,
}

/// Hold time after the marker was last beaten by the input.
pub const STICK_MS: u64 = 1000;
/// Marker fall rate, dB per second.
pub const FALL_DB_PER_S: f32 = 24.0;
/// Value treated as digital silence.
pub const SILENCE_DB: f32 = -120.0;

impl PeakMeter {
    pub fn new(now_ms: u64) -> Self {
        Self {
            level_db: SILENCE_DB,
            peak_db: SILENCE_DB,
            stick_until_ms: 0,
            last_ms: now_ms,
        }
    }

    /// Advance one frame. `rms_db` is this frame's RMS level (SILENCE_DB if
    /// no audio), `frame_peak_db` this frame's sample peak (SILENCE_DB if
    /// none), `now_ms` the wall clock.
    pub fn tick(&mut self, rms_db: f32, frame_peak_db: f32, now_ms: u64) {
        let dt = now_ms.saturating_sub(self.last_ms).min(200) as f32 / 1000.0;
        self.last_ms = now_ms;

        // Bar: RMS with fast attack / moderate release.
        let k = if rms_db > self.level_db { 0.5 } else { 0.08 };
        self.level_db += (rms_db - self.level_db) * k;

        // Marker: exceedance of the SHOWN value sticks it; else, after the
        // stick lapses, a constant-rate fall floored at the bar.
        if frame_peak_db > self.peak_db {
            self.peak_db = frame_peak_db;
            self.stick_until_ms = now_ms + STICK_MS;
        } else if now_ms >= self.stick_until_ms {
            self.peak_db = (self.peak_db - FALL_DB_PER_S * dt).max(self.level_db);
        }
    }

    pub fn level_db(&self) -> f32 {
        self.level_db
    }

    pub fn peak_db(&self) -> f32 {
        self.peak_db
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Drive `n` frames of `step_ms` apart at constant levels.
    fn hold(m: &mut PeakMeter, rms: f32, peak: f32, n: u32, step_ms: u64, t: &mut u64) {
        for _ in 0..n {
            *t += step_ms;
            m.tick(rms, peak, *t);
        }
    }

    #[test]
    fn first_exceedance_jumps_and_sticks() {
        let mut t = 0u64;
        let mut m = PeakMeter::new(t);
        hold(&mut m, -30.0, -12.0, 1, 16, &mut t);
        assert_eq!(m.peak_db(), -12.0, "instant rise");
        hold(&mut m, -30.0, -40.0, 30, 16, &mut t); // ~480 ms of quieter input
        assert_eq!(m.peak_db(), -12.0, "sticks through the whole second");
    }

    #[test]
    fn falls_after_exactly_one_second_of_no_exceedance() {
        let mut m = PeakMeter::new(0);
        let t = 16u64;
        m.tick(-30.0, -12.0, t);
        // Still within the stick: frozen.
        for dt_ms in [300, 600, 900] {
            m.tick(-30.0, -40.0, 16 + dt_ms);
            assert_eq!(m.peak_db(), -12.0, "frozen at {dt_ms} ms");
        }
        // Past the stick: falling.
        m.tick(-30.0, -40.0, 16 + 1100);
        assert!(m.peak_db() < -12.0, "must have started falling");
    }

    #[test]
    fn fall_rate_is_24_db_per_second_and_floors_at_bar() {
        let mut t = 0u64;
        let mut m = PeakMeter::new(t);
        // Loud strum (peak -12) over a quiet sustain (RMS -50, so the bar
        // floor cannot truncate the one-second fall measurement).
        m.tick(-50.0, -12.0, 16);
        assert_eq!(m.peak_db(), -12.0);
        // Let the bar settle to the sustain level while the stick holds.
        hold(&mut m, -50.0, -80.0, 40, 16, &mut t); // 640 ms < stick
        assert_eq!(m.peak_db(), -12.0, "still stuck during settle");
        // Past the stick, fall for ~1 s in small steps: 24 dB. The stick
        // expires at t = 1016 ms; frames 63..=125 (t 1024..2016) fall for
        // 1008 ms ⇒ 24.2 dB. (Settle ran 40 frames, so 85 more.)
        hold(&mut m, -50.0, -80.0, 85, 16, &mut t);
        let drop = -12.0 - m.peak_db();
        assert!(
            (drop - FALL_DB_PER_S).abs() < 0.6,
            "one second should drop ~24 dB, dropped {drop}"
        );
        // Keep falling: pinned at the bar, never below.
        hold(&mut m, -50.0, -80.0, 400, 16, &mut t);
        assert!(
            (m.peak_db() - m.level_db()).abs() < 0.5,
            "marker floors at the bar: peak {}, level {}",
            m.peak_db(),
            m.level_db()
        );
    }

    #[test]
    fn exceedance_mid_fall_jumps_and_rearms() {
        let mut t = 0u64;
        let mut m = PeakMeter::new(t);
        m.tick(-40.0, -10.0, 16);
        hold(&mut m, -40.0, -70.0, 130, 16, &mut t); // past stick: falling
        let shown = m.peak_db();
        assert!(shown < -10.0, "should be mid-fall, got {shown}");
        // A strum BELOW the original peak but ABOVE the shown marker.
        t += 16;
        m.tick(-30.0, -18.0, t);
        assert_eq!(m.peak_db(), -18.0, "jumps to the new, lower max");
        // Fresh stick: frozen a full second...
        m.tick(-30.0, -70.0, t + 900);
        assert_eq!(m.peak_db(), -18.0, "re-armed: still frozen at +900 ms");
        // ...then falling again.
        m.tick(-30.0, -70.0, t + 1100);
        assert!(m.peak_db() < -18.0, "falling again after the fresh second");
    }

    #[test]
    fn silence_drains_to_the_floor() {
        let mut t = 0u64;
        let mut m = PeakMeter::new(t);
        m.tick(-20.0, -6.0, 16);
        hold(&mut m, SILENCE_DB, SILENCE_DB, 400, 16, &mut t);
        assert!(m.level_db() < -100.0, "bar drains, got {}", m.level_db());
        assert!(m.peak_db() < -100.0, "marker drains, got {}", m.peak_db());
    }
}
