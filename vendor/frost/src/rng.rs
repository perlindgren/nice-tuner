//! A small, dependency-free random number generator.
//!
//! [`Rng`] is a splitmix64 generator: pure 64-bit integer arithmetic, so it
//! runs identically on every platform, with no seeding system calls and no
//! `rand` crate in the dependency tree. It is meant
//! for game-time randomness — spawn points, velocities, lifetimes — where a
//! full-featured distribution library is overkill.
//!
//! Seed from the clock with [`Rng::new`] for a different stream on each
//! run, or fix the stream with [`Rng::with_seed`] for reproducible
//! simulations and tests.

use std::time::{SystemTime, UNIX_EPOCH};

/// A small splitmix64 pseudo-random number generator.
///
/// The state advances by the golden-ratio constant on every draw and is
/// mixed with two multiply-xor rounds, so consecutive values are well
/// distributed even though the generator is only a few lines long.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rng(u64);

impl Rng {
    /// Seed from the clock: the nanoseconds since the Unix epoch, mixed
    /// with the golden-ratio constant so a degenerate clock still gives a
    /// healthy seed. Each run gets a different stream.
    pub fn new() -> Self {
        let t = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_nanos() as u64)
            .unwrap_or(0);
        Self(t ^ 0x9E37_79B9_7F4A_7C15)
    }

    /// Seed from an exact value: the same seed always gives the same
    /// stream, for reproducible simulations and tests.
    pub fn with_seed(seed: u64) -> Self {
        Self(seed)
    }

    /// The stream's current state — the value the next draw advances from.
    /// Storing it, and restoring it with [`Self::set_state`], resumes the stream
    /// exactly where it left off, which is what a save file needs.
    pub fn state(&self) -> u64 {
        self.0
    }

    /// Restores the stream to `state`: the next draw continues exactly
    /// where a stream that reached `state` would have.
    pub fn set_state(&mut self, state: u64) {
        self.0 = state;
    }

    /// The next pseudo-random `u64`.
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        z
    }

    /// A uniform `f32` in `[0, 1)`: the top 24 bits of the next `u64`.
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u32 << 24) as f32
    }

    /// A uniform `f32` in `[lo, hi)`.
    pub fn in_range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + self.next_f32() * (hi - lo)
    }
}

impl Default for Rng {
    /// A clock-seeded generator: the same as [`Rng::new`].
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The same seed gives the same stream: the generator is
    /// deterministic, which is what reproducible simulations and tests
    /// need.
    #[test]
    fn the_same_seed_gives_the_same_stream() {
        let mut a = Rng::with_seed(42);
        let mut b = Rng::with_seed(42);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
            assert_eq!(a.next_f32(), b.next_f32());
        }
    }

    /// Different seeds give different streams.
    #[test]
    fn different_seeds_give_different_streams() {
        let mut a = Rng::with_seed(0);
        let mut b = Rng::with_seed(1);
        assert_ne!(a.next_u64(), b.next_u64());
    }

    /// `next_f32` stays in [0, 1) and `in_range` stays in its bounds.
    #[test]
    fn the_uniforms_stay_in_their_bounds() {
        let mut rng = Rng::with_seed(1234567);
        for _ in 0..10_000 {
            let x = rng.next_f32();
            assert!((0.0..1.0).contains(&x), "next_f32 out of range: {x}");
            let y = rng.in_range(-2.5, 7.5);
            assert!((-2.5..7.5).contains(&y), "in_range out of bounds: {y}");
        }
    }
}
