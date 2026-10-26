//! Bodies in motion: a [`Collider`] given a velocity and a weight, and the
//! collisions that trade their momentum without spending any energy.
//!
//! The [`crate::collision`] module owns static geometry — where a shape is,
//! whether two overlap, and the vector that separates them — and no state.
//! This module puts that geometry in motion. A [`Body`] is a shape that moves:
//! the shape is a `Collider` and owns its own center, the velocity carries
//! that center forward, and the weight decides how the body answers a push.
//! [`Body::advance`] integrates position by the velocity for a frame; the two
//! resolve functions settle a frame's contacts.
//!
//! **Every collision here is perfectly elastic.** The normal component of the
//! approach is reversed whole, not damped, and a body meeting a wall reflects
//! with restitution `1.0` (see [`crate::collision::reflect`]). So the total
//! kinetic energy of the bodies a frame settles — `Σ ½ · weight · |v|²` — is
//! the same before and after the contacts; [`kinetic_energy`] reads it, and a
//! test watches it stay flat across a three-body round. That is a property of
//! the impulse arithmetic below, not a correction applied after the fact: the
//! impulse is equal and opposite on the two bodies (so momentum is conserved),
//! acts only along the contact normal (so the sideways slide is untouched),
//! and reverses the normal closing speed exactly (so energy is conserved).
//!
//! **What this is not.** Integration is discrete — a body teleports by
//! `velocity * dt` each frame — so a body moving farther in one frame than a
//! wall is thick steps clean through it: `Collider::cast` is the tool for the
//! day a demo needs to prevent that, and no demo here needs it. There is no
//! angular momentum, friction, or gravity: weight is the whole material, and
//! it only ever resists a push along a normal.

use crate::collision::{Collider, dot, reflect};

/// A shape in motion: a [`Collider`] (which owns the position, as it owns the
/// position for every other shape in the crate), the velocity its center moves
/// at in pixels per second, and its weight — the mass a collision trades
/// momentum by.
///
/// The weight is in arbitrary units and must be positive; only its ratios
/// matter. A body twice another's weight takes half the velocity change from
/// the same collision, and a heavy body meeting a light one barely notices.
pub struct Body {
    /// The body's shape and position: its center is the body's position.
    pub collider: Collider,
    /// The velocity of the center, in pixels per second.
    pub velocity: [f32; 2],
    /// The mass, in arbitrary units. Positive.
    pub weight: f32,
}

impl Body {
    /// A body: a shape, the velocity it starts with, and its weight.
    pub fn new(collider: Collider, velocity: [f32; 2], weight: f32) -> Self {
        Body {
            collider,
            velocity,
            weight,
        }
    }

    /// The body's center — its position, read out of the shape that owns it.
    pub fn center(&self) -> [f32; 2] {
        match &self.collider {
            Collider::Box(b) => b.center,
            Collider::Circle(c) => c.center,
        }
    }

    /// The reciprocal of the weight: the number the collision arithmetic
    /// actually divides by, zero-when-infinite for a wall.
    fn inverse_weight(&self) -> f32 {
        1.0 / self.weight
    }

    /// Move the body along its velocity for `dt` seconds. Collisions are not
    /// checked here: a frame advances every body, then settles the contacts.
    pub fn advance(&mut self, dt: f32) {
        self.collider
            .translate([self.velocity[0] * dt, self.velocity[1] * dt]);
    }
}

/// Settle one contact between two bodies, moving and turning both.
///
/// On overlap the bodies are first pushed apart along the contact normal,
/// each giving way in proportion to how light it is — the lighter does most
/// of the moving — and then the elastic impulse reverses their approach:
/// equal and opposite impulses along the normal, sized so the speed at which
/// they close is the speed at which they part. A pair that overlaps but is
/// already separating only gets the positional fix, never a second kick.
///
/// `true` when the bodies overlapped and were dealt with, `false` when they
/// never touched and nothing changed.
pub fn resolve_pair(a: &mut Body, b: &mut Body) -> bool {
    // The push-out normal points the way `a` must move to leave `b`, so it is
    // the contact normal from `b` to `a`; both bodies answer to it with
    // matching signs below.
    let Some(out) = a.collider.push_out(&b.collider) else {
        return false;
    };
    let (inv_a, inv_b) = (a.inverse_weight(), b.inverse_weight());
    let inv_sum = inv_a + inv_b;

    // Stand them apart first, splitting the penetration by inverse weight so
    // neither sinks into the other and the heavier holds its ground.
    let (share_a, share_b) = (inv_a / inv_sum, inv_b / inv_sum);
    a.collider.translate([
        out.dir[0] * out.depth * share_a,
        out.dir[1] * out.depth * share_a,
    ]);
    b.collider.translate([
        -out.dir[0] * out.depth * share_b,
        -out.dir[1] * out.depth * share_b,
    ]);

    // The velocity's share of the answer, but only for a closing pair: a
    // relative velocity along the separating normal means they are already
    // parting, and the impulse must not pull them back.
    let closing = dot(
        [a.velocity[0] - b.velocity[0], a.velocity[1] - b.velocity[1]],
        out.dir,
    );
    if closing >= 0.0 {
        return true;
    }

    // `e = 1` folded into the `-2`: the impulse magnitude per unit inverse
    // mass. Divided by the total inverse weight so a light body's velocity
    // swings further than a heavy one's for the same momentum.
    let j = -2.0 * closing / inv_sum;
    a.velocity = [
        a.velocity[0] + j * inv_a * out.dir[0],
        a.velocity[1] + j * inv_a * out.dir[1],
    ];
    b.velocity = [
        b.velocity[0] - j * inv_b * out.dir[0],
        b.velocity[1] - j * inv_b * out.dir[1],
    ];
    true
}

/// Settle one contact between a body and a shape that never moves — a wall,
/// a post, the edge of the world.
///
/// The static shape is immovable, so the whole positional fix and the whole
/// velocity reversal fall on the body: it is pushed clear of the shape and
/// reflected off it with a restitution of `1.0`, keeping every joule. The
/// body's speed is exactly what it was; only its direction changes.
///
/// `true` when the body was inside the shape and is now out and turned,
/// `false` when they never touched.
pub fn resolve_static(body: &mut Body, wall: &Collider) -> bool {
    let Some(out) = body.collider.push_out(wall) else {
        return false;
    };
    body.collider
        .translate([out.dir[0] * out.depth, out.dir[1] * out.depth]);
    body.velocity = reflect(body.velocity, out.dir, 1.0);
    true
}

/// The living total: the kinetic energy of the bodies, `Σ ½ · weight · |v|²`.
/// It carries no position — only motion — so it is the number a frame's
/// contacts must not change.
pub fn kinetic_energy(bodies: &[Body]) -> f32 {
    bodies
        .iter()
        .map(|b| 0.5 * b.weight * dot(b.velocity, b.velocity))
        .sum()
}

/// The total momentum of the bodies, `Σ weight · v`. The vector sum, so the
/// directions count: two equal bodies in a dead tie have none between them.
pub fn momentum(bodies: &[Body]) -> [f32; 2] {
    bodies.iter().fold([0.0; 2], |sum, b| {
        [
            sum[0] + b.weight * b.velocity[0],
            sum[1] + b.weight * b.velocity[1],
        ]
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::collision::{Circle, OrientedBox};

    fn box_body(center: [f32; 2], half: [f32; 2], vel: [f32; 2], weight: f32) -> Body {
        Body::new(Collider::Box(OrientedBox::new(center, half)), vel, weight)
    }

    fn circle_body(center: [f32; 2], radius: f32, vel: [f32; 2], weight: f32) -> Body {
        Body::new(Collider::Circle(Circle { center, radius }), vel, weight)
    }

    fn close(a: f32, b: f32) -> bool {
        (a - b).abs() <= 1e-4 * a.abs().max(b.abs()).max(1.0)
    }

    #[test]
    fn two_equal_masses_trade_velocities_head_on() {
        // The textbook tie: two equal masses meeting dead-on exchange
        // velocities whole — the classic Newton's cradle pair.
        let mut a = box_body([-4.0, 0.0], [5.0, 5.0], [100.0, 0.0], 2.0);
        let mut b = box_body([4.0, 0.0], [5.0, 5.0], [-100.0, 0.0], 2.0);
        assert!(resolve_pair(&mut a, &mut b));
        assert!(close(a.velocity[0], -100.0) && close(a.velocity[1], 0.0));
        assert!(close(b.velocity[0], 100.0) && close(b.velocity[1], 0.0));
    }

    #[test]
    fn the_heavier_body_barely_notices_the_lighter() {
        // A heavy body driven into a stationary light one loses almost
        // nothing, and the light one is flung off at close to twice speed.
        let mut heavy = box_body([-4.0, 0.0], [5.0, 5.0], [100.0, 0.0], 100.0);
        let mut light = box_body([4.0, 0.0], [5.0, 5.0], [0.0, 0.0], 1.0);
        assert!(resolve_pair(&mut heavy, &mut light));
        assert!(heavy.velocity[0] > 95.0, "the heavy one slows a hair");
        assert!(light.velocity[0] > 195.0, "the light one is flung clear");
    }

    #[test]
    fn energy_and_momentum_survive_a_glancing_pair() {
        // A glancing blow: the normal is at an angle, so the impulse turns
        // the bodies rather than stopping them. The two ledger lines that a
        // frictionless elastic impulse cannot move are energy and momentum.
        let mut bodies = [
            circle_body([-6.0, 4.0], 5.0, [120.0, -40.0], 3.0),
            circle_body([0.0, 0.0], 5.0, [-30.0, 60.0], 5.0),
        ];
        let energy_before = kinetic_energy(&bodies);
        let momentum_before = momentum(&bodies);

        let (a, b) = split_two(&mut bodies, 0, 1);
        assert!(resolve_pair(a, b));
        assert!(
            close(kinetic_energy(&bodies), energy_before),
            "energy is spent by nothing here"
        );
        let after = momentum(&bodies);
        assert!(
            close(after[0], momentum_before[0]) && close(after[1], momentum_before[1]),
            "momentum is traded, never created"
        );
    }

    #[test]
    fn a_separating_pair_is_pushed_apart_but_never_kicked() {
        // Overlapping yet already parting: the positional fix stands them
        // clear, and the velocities — which already point apart — are left
        // exactly as they were. A body must not be sped up by a body behind
        // it merely because their boxes still overlap for one more frame.
        let mut a = box_body([-4.0, 0.0], [5.0, 5.0], [-100.0, 0.0], 2.0);
        let mut b = box_body([4.0, 0.0], [5.0, 5.0], [100.0, 0.0], 2.0);
        assert!(resolve_pair(&mut a, &mut b));
        assert!(close(a.velocity[0], -100.0));
        assert!(close(b.velocity[0], 100.0));
        // And they come out not overlapping.
        assert!(!a.collider.intersects(&b.collider));
    }

    #[test]
    fn the_wall_turns_a_body_without_spending_it() {
        // A bounce off a static wall keeps the speed to the joule; only the
        // direction changes. This is the pair impulse with an immovable
        // partner, so it must agree with a plain elastic reflection.
        let mut ball = circle_body([95.0, 0.0], 10.0, [80.0, 30.0], 1.0);
        let wall = Collider::Box(OrientedBox::new([100.0, 0.0], [5.0, 100.0]));
        let speed_before = dot(ball.velocity, ball.velocity).sqrt();
        assert!(resolve_static(&mut ball, &wall));
        let speed_after = dot(ball.velocity, ball.velocity).sqrt();
        assert!(close(speed_before, speed_after), "a wall takes nothing");
        assert!(ball.velocity[0] < 0.0, "turned back the way it came");
        assert!(!ball.collider.intersects(&wall), "and out of the wall");
    }

    #[test]
    fn penetration_is_split_by_inverse_weight() {
        // Deep overlap, unequal masses: the light body covers almost all the
        // separation and the heavy one barely shifts, in exact proportion to
        // their weights.
        let mut heavy = box_body([1.0, 0.0], [5.0, 5.0], [0.0, 0.0], 99.0);
        let mut light = box_body([-1.0, 0.0], [5.0, 5.0], [0.0, 0.0], 1.0);
        let light_before = light.center()[0];
        let heavy_before = heavy.center()[0];
        assert!(resolve_pair(&mut heavy, &mut light));
        let light_moved = light.center()[0] - light_before;
        let heavy_moved = heavy.center()[0] - heavy_before;
        assert!(
            close(light_moved.abs(), 99.0 * heavy_moved.abs()),
            "the light one does ninety-nine times the giving way"
        );
        assert!(
            light_moved.signum() != heavy_moved.signum(),
            "they stand apart, not on each other"
        );
        assert!(!heavy.collider.intersects(&light.collider));
    }

    #[test]
    fn bodies_that_never_touch_are_left_alone() {
        let mut a = box_body([-50.0, 0.0], [5.0, 5.0], [10.0, 0.0], 1.0);
        let mut b = box_body([50.0, 0.0], [5.0, 5.0], [-10.0, 0.0], 1.0);
        assert!(!resolve_pair(&mut a, &mut b));
        assert!(close(a.velocity[0], 10.0) && close(b.velocity[0], -10.0));
        let mut ball = circle_body([0.0, 0.0], 5.0, [0.0, 0.0], 1.0);
        assert!(!resolve_static(
            &mut ball,
            &Collider::Box(OrientedBox::new([90.0, 0.0], [5.0, 5.0]))
        ));
    }

    #[test]
    fn advance_walks_the_center_and_keeps_the_shape() {
        let mut boxy = box_body([0.0, 0.0], [4.0, 6.0], [30.0, -20.0], 1.0);
        boxy.advance(0.5);
        assert!(close(boxy.center()[0], 15.0) && close(boxy.center()[1], -10.0));
        let Collider::Box(b) = &boxy.collider else {
            panic!("a box stays a box");
        };
        assert_eq!(b.half, [4.0, 6.0]);
        assert!(close(b.angle, 0.0));
    }

    #[test]
    fn a_three_body_round_keeps_the_ledger_balanced() {
        // Fire a body into a still row of two, then let the shaken row hit
        // the walls: after every contact of a busy frame, the energy of the
        // whole set is what it started as. Momentum can move between bodies
        // and vanish into a wall (a wall is an outside force), but energy has
        // nowhere to leak.
        let mut bodies = [
            circle_body([-30.0, 0.0], 8.0, [150.0, 0.0], 2.0),
            circle_body([0.0, 0.0], 8.0, [0.0, 0.0], 3.0),
            circle_body([40.0, 12.0], 8.0, [-40.0, 25.0], 1.5),
        ];
        let walls = [
            Collider::Box(OrientedBox::new([0.0, 200.0], [400.0, 50.0])),
            Collider::Box(OrientedBox::new([0.0, -200.0], [400.0, 50.0])),
            Collider::Box(OrientedBox::new([200.0, 0.0], [50.0, 400.0])),
            Collider::Box(OrientedBox::new([-200.0, 0.0], [50.0, 400.0])),
        ];
        let energy_before = kinetic_energy(&bodies);
        for _ in 0..60 {
            for b in bodies.iter_mut() {
                b.advance(1.0 / 60.0);
            }
            for i in 0..bodies.len() {
                for j in (i + 1)..bodies.len() {
                    let (a, b) = split_two(&mut bodies, i, j);
                    resolve_pair(a, b);
                }
            }
            for body in bodies.iter_mut() {
                for wall in &walls {
                    resolve_static(body, wall);
                }
            }
        }
        let after = kinetic_energy(&bodies);
        assert!(
            (after - energy_before).abs() <= 1e-3 * energy_before,
            "the ledger balances across every wall and body alike:              {after} against {energy_before}"
        );
    }

    /// Borrow the `i`th and `j`th bodies (`i < j`) at once.
    fn split_two(bodies: &mut [Body], i: usize, j: usize) -> (&mut Body, &mut Body) {
        let (head, tail) = bodies.split_at_mut(j);
        (&mut head[i], &mut tail[0])
    }
}
