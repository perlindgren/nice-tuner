//! Simple particle simulation, independent of the renderer.
//!
//! A [`ParticleSystem`] is a plain collection of [`Particle`]s: each holds
//! a position, a velocity, a remaining lifetime, a size, a rotation, and a
//! color. A [`ParticleSystem::update`] advances every particle by the
//! frame's delta time under a constant gravity and removes the dead ones,
//! so a [`crate::Process`] can spawn, update, and draw the particles once
//! per frame. The drawing stays with the caller: put the system in the
//! node's [`crate::Shape::Particles`] shape to draw it in the node's local
//! space (transformed, scaled, and tinted by the node), or, for batches
//! that live directly in the window's user space, with the deprecated
//! [`crate::Canvas::particles`] draw.

use crate::objects::Color;

/// A single particle: a position, a velocity, a lifetime, a size, a
/// rotation, and a color.
///
/// Build one as a struct literal and [`ParticleSystem::spawn`] it, and read
/// the system's [`ParticleSystem::particles`] back to draw each one.
///
/// The particle's `size`, `angle`, and `color` are pure render properties:
/// the simulation in [`ParticleSystem::update`] never touches them, so the
/// caller can animate them (spin the `angle`, fade the `color`) between
/// frames.
#[derive(Clone, Debug, PartialEq)]
pub struct Particle {
    /// Position: the node's local space when the system is held in the
    /// node's [`crate::Shape::Particles`] shape, or the window's user space
    /// (window-centered, y up) for the deprecated
    /// [`crate::Canvas::particles`] draw.
    pub pos: [f32; 2],
    /// Velocity, in pixels per second.
    pub vel: [f32; 2],
    /// Remaining lifetime, in seconds; the particle dies when it runs out.
    pub life: f32,
    /// The lifetime the particle was spawned with, so a caller can compute
    /// its life fraction (`life / max_life`) for a fade.
    pub max_life: f32,
    /// Draw size, in pixels: the radius when the particle is drawn as a
    /// circle, the half-width when it is drawn as a rectangle or a sprite
    /// (see [`crate::ParticleShape`]).
    pub size: f32,
    /// The particle's own rotation, in radians, on top of the node's: the
    /// shape it is drawn with is rotated by this much in the node's local
    /// space (positive counterclockwise, like the node's transform).
    /// `0.0` leaves the batch's shape unrotated.
    pub angle: f32,
    /// The particle's own color, multiplied with the batch's base color
    /// (and, for a node-attached batch, the node's composed modulate):
    /// per-particle modulation. White (`1.0, 1.0, 1.0`) leaves the batch's
    /// color unchanged.
    pub color: Color,
}

/// A plain collection of [`Particle`]s, advanced by [`ParticleSystem::update`].
///
/// The simulation is pure math: nothing here knows about the renderer. A
/// [`crate::Process`] spawns particles, updates them each frame, and draws
/// them.
#[derive(Clone, Debug, Default)]
pub struct ParticleSystem {
    /// The live particles, in spawn order.
    pub particles: Vec<Particle>,
}

impl ParticleSystem {
    /// Creates an empty system.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a particle. The system has no capacity, so the caller controls
    /// the spawn rate.
    pub fn spawn(&mut self, particle: Particle) {
        self.particles.push(particle);
    }

    /// Advances every particle by `dt` seconds under a constant `gravity`
    /// (in pixels per second, per second) and removes the dead ones.
    ///
    /// Each step is a semi-implicit Euler: the velocity takes the gravity
    /// first, then the position takes the new velocity, which stays stable
    /// under gravity; then the lifetime is decremented and the particles
    /// whose life has run out are removed.
    pub fn update(&mut self, dt: f32, gravity: [f32; 2]) {
        for p in &mut self.particles {
            p.vel[0] += gravity[0] * dt;
            p.vel[1] += gravity[1] * dt;
            p.pos[0] += p.vel[0] * dt;
            p.pos[1] += p.vel[1] * dt;
            p.life -= dt;
        }
        self.particles.retain(|p| p.life > 0.0);
    }

    /// The number of live particles.
    pub fn len(&self) -> usize {
        self.particles.len()
    }

    /// Whether the system has no live particles.
    pub fn is_empty(&self) -> bool {
        self.particles.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A particle at `pos` with `vel`, a lifetime of `life` seconds, and
    /// the given `size`; its `max_life` is its `life`, and it is unrotated
    /// and white.
    fn particle(pos: [f32; 2], vel: [f32; 2], life: f32, size: f32) -> Particle {
        Particle {
            pos,
            vel,
            life,
            max_life: life,
            size,
            angle: 0.0,
            color: Color {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
        }
    }

    #[test]
    fn update_integrates_position_and_velocity() {
        // A particle moving right at 10 px/s under gravity pulling it down
        // at 2 px/s/s, for a 1s step: the velocity takes the gravity first
        // (semi-implicit Euler), then the position takes the new velocity.
        let mut sys = ParticleSystem::new();
        sys.spawn(particle([0.0, 0.0], [10.0, 0.0], 10.0, 1.0));
        sys.update(1.0, [0.0, -2.0]);
        let p = &sys.particles[0];
        assert_eq!(p.vel, [10.0, -2.0]);
        assert_eq!(p.pos, [10.0, -2.0]);
        assert_eq!(p.life, 9.0);
    }

    #[test]
    fn dead_particles_are_removed() {
        let mut sys = ParticleSystem::new();
        sys.spawn(particle([0.0, 0.0], [0.0, 0.0], 1.0, 1.0));
        sys.spawn(particle([5.0, 0.0], [0.0, 0.0], 2.0, 1.0));
        sys.update(1.0, [0.0, 0.0]); // the first one's life runs out exactly
        assert_eq!(sys.len(), 1);
        assert_eq!(sys.particles[0].pos, [5.0, 0.0]);
        sys.update(1.0, [0.0, 0.0]); // the second one runs out too
        assert!(sys.is_empty());
    }

    #[test]
    fn spawn_preserves_order() {
        let mut sys = ParticleSystem::new();
        sys.spawn(particle([1.0, 0.0], [0.0, 0.0], 1.0, 1.0));
        sys.spawn(particle([2.0, 0.0], [0.0, 0.0], 1.0, 1.0));
        sys.spawn(particle([3.0, 0.0], [0.0, 0.0], 1.0, 1.0));
        assert_eq!(
            sys.particles.iter().map(|p| p.pos[0]).collect::<Vec<_>>(),
            [1.0, 2.0, 3.0]
        );
    }

    #[test]
    fn steps_are_deterministic() {
        // The same spawn and step, twice, give the same particles.
        let step = |sys: &mut ParticleSystem| {
            sys.spawn(particle([0.0, 0.0], [0.0, 10.0], 5.0, 2.0));
            sys.update(0.1, [0.0, -1.0]);
        };
        let mut a = ParticleSystem::new();
        let mut b = ParticleSystem::new();
        step(&mut a);
        step(&mut b);
        assert_eq!(a.particles, b.particles);
    }
}
