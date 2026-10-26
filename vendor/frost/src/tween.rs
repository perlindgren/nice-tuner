//! Time-based value animation, independent of the renderer.
//!
//! A [`Tween`] linearly interpolates a [`Tweenable`] value from `from` to
//! `to` over a duration; each [`Tween::tick`] advances it by the frame's
//! delta time and returns the current value, so a [`crate::Process`] can
//! drive it once per frame and apply the result to the scene.

/// A value that a [`Tween`] can interpolate.
///
/// Implement it for your own types to tween them; it is implemented for
/// `f32` and `[f32; 2]`.
pub trait Tweenable: Copy {
    /// Linearly interpolates from `self` toward `other`: `t = 0.0` yields
    /// `self`, `t = 1.0` yields `other`.
    fn tween(self, other: Self, t: f32) -> Self;
}

impl Tweenable for f32 {
    fn tween(self, other: Self, t: f32) -> Self {
        self + (other - self) * t
    }
}

impl Tweenable for [f32; 2] {
    fn tween(self, other: Self, t: f32) -> Self {
        [self[0].tween(other[0], t), self[1].tween(other[1], t)]
    }
}

/// How a [`Tween`] behaves once its duration has elapsed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Repeat {
    /// Travel to the target and stay there.
    Once,
    /// Jump back to the start and travel again.
    Loop,
    /// Travel to the target, then back to the start, forever.
    PingPong,
}

/// A linear tween of a value from `from` to `to` over `duration` seconds.
///
/// Each [`Tween::tick`] advances the tween by a time step and returns the
/// current interpolated value. With the default [`Repeat::PingPong`] the
/// value travels `from -> to -> from -> ...` at constant speed; see
/// [`Tween::repeat`] for the other modes.
#[derive(Clone, Copy, Debug)]
pub struct Tween<T: Tweenable> {
    from: T,
    to: T,
    /// Elapsed time, in seconds, since the tween was created.
    time: f32,
    /// Seconds for one leg (the `from -> to` travel).
    duration: f32,
    repeat: Repeat,
}

impl<T: Tweenable> Tween<T> {
    /// Creates a [`Repeat::PingPong`] tween whose one-way travel from `from`
    /// to `to` takes `duration` seconds.
    pub fn new(from: T, to: T, duration: f32) -> Self {
        Self {
            from,
            to,
            time: 0.0,
            duration: duration.max(1e-6),
            repeat: Repeat::PingPong,
        }
    }

    /// Sets the repeat mode.
    pub fn repeat(mut self, repeat: Repeat) -> Self {
        self.repeat = repeat;
        self
    }

    /// The value the tween starts from.
    pub fn from(&self) -> T {
        self.from
    }

    /// The value the tween travels to.
    pub fn to(&self) -> T {
        self.to
    }

    /// The elapsed time, in seconds, since the tween was created.
    pub fn time(&self) -> f32 {
        self.time
    }

    /// Seconds for one leg (the `from -> to` travel).
    pub fn duration(&self) -> f32 {
        self.duration
    }

    /// The repeat mode.
    pub fn repeat_mode(&self) -> Repeat {
        self.repeat
    }

    /// Restores the elapsed time to `time` seconds: the next [`tick`]
    /// continues from the value the tween had at that moment, so an
    /// in-flight tween can be saved and resumed mid-travel.
    pub fn set_time(&mut self, time: f32) {
        self.time = time;
    }

    /// Advances by `dt` seconds and returns the current interpolated value.
    pub fn tick(&mut self, dt: f32) -> T {
        self.time += dt;
        match self.repeat {
            Repeat::Once => {
                let t = (self.time / self.duration).min(1.0);
                self.from.tween(self.to, t)
            }
            Repeat::Loop => {
                let t = (self.time % self.duration) / self.duration;
                self.from.tween(self.to, t)
            }
            Repeat::PingPong => {
                let cycle = self.time % (2.0 * self.duration);
                let t = if cycle < self.duration {
                    cycle / self.duration
                } else {
                    1.0 - (cycle - self.duration) / self.duration
                };
                self.from.tween(self.to, t)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ping_pong_travels_to_and_fro() {
        let mut tw = Tween::new(0.0, 100.0, 1.0); // 1s per leg, ping-pong
        assert_eq!(tw.tick(0.5), 50.0); // halfway to `to`
        assert_eq!(tw.tick(0.5), 100.0); // at `to`
        assert_eq!(tw.tick(0.5), 50.0); // halfway back
        assert_eq!(tw.tick(0.5), 0.0); // back at `from`, cycle restarts
    }

    #[test]
    fn once_stays_at_the_end() {
        let mut tw = Tween::new(0.0, 10.0, 1.0).repeat(Repeat::Once);
        assert_eq!(tw.tick(0.5), 5.0);
        assert_eq!(tw.tick(0.5), 10.0);
        assert_eq!(tw.tick(5.0), 10.0); // stays at the target
    }

    #[test]
    fn loop_wraps_at_the_end() {
        let mut tw = Tween::new(0.0, 10.0, 1.0).repeat(Repeat::Loop);
        assert_eq!(tw.tick(0.5), 5.0);
        assert_eq!(tw.tick(0.4), 9.0);
        assert_eq!(tw.tick(0.1), 0.0); // wrapped back to `from`
        assert_eq!(tw.tick(0.5), 5.0);
    }

    #[test]
    fn interpolates_vectors() {
        let mut tw = Tween::new([0.0, 10.0], [20.0, 0.0], 1.0);
        assert_eq!(tw.tick(0.5), [10.0, 5.0]);
    }
}
