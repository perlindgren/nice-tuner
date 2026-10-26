//! 2D collision math: shapes, overlap tests, and push-out resolution.
//!
//! Pure math — no winit, no wgpu, no state — so it is unit-tested without a
//! GPU on every platform. Coordinates
//! are the same window-centered, y-up space the rest of frost draws in: x to
//! the right, y up, angles counter-clockwise in radians.
//!
//! This module answers questions; it owns no world and does no integration.
//! The game's [`Process`](crate::Process) keeps the positions and
//! velocities, builds a [`Collider`] from that state each frame, applies
//! [`Collider::push_out`] together with [`reflect`], and writes the new
//! position back into the scene node's transform.
//!
//! Segments are asked the same way, with no more state: [`Collider::cast`]
//! says where a segment first enters a shape, and the two policies built on
//! it differ only in what they do when an endpoint is inside a shape —
//! [`occluded`] is the CPU twin of the shaders' `occluded`, where a receiver
//! inside an occluder is lit rather than blocked, while [`visible`] answers
//! line of sight, where a shape containing either endpoint blocks.
//!
//! When a game outgrows this — stacking, joints, ragdolls, dozens of
//! mutually dynamic bodies that need a broadphase, or continuous collision
//! for very fast projectiles — the replacement is `rapier2d`: move the state
//! into its rigid bodies and swap these two call sites; the scene bridge and
//! the rendering stay the same.

/// The f32 epsilon: shapes whose gap is this or larger are touching, not
/// overlapping.
const EPS: f32 = 1e-6;

/// The slack a convexity check allows, in pixels, plus a term relative to the
/// edge's distance from the origin: a corner lies *on* the two edges that meet
/// it, so the check is testing equality, and f32 rounding at a coordinate of a
/// few hundred pixels is a few ten-thousandths wide. The segment epsilon would
/// reject a well-formed polygon for it. A dent a player could see is worth a
/// hundred times this.
const CONVEX_EPS: f32 = 1e-4;

/// The epsilon a segment's direction must exceed along an axis for that
/// axis's slab to clip the interval at all — the shaders' `1e-6` kept
/// verbatim, so the CPU and GPU segment tests take the same branch for the
/// same ray.
const SLAB_EPS: f32 = 1e-6;

/// The dot product of two 2D vectors.
pub(crate) fn dot(a: [f32; 2], b: [f32; 2]) -> f32 {
    a[0] * b[0] + a[1] * b[1]
}

/// The length of a 2D vector.
fn len(v: [f32; 2]) -> f32 {
    (v[0] * v[0] + v[1] * v[1]).sqrt()
}

/// A box centered at `center` with half-extents `half`, rotated
/// counter-clockwise by `angle` radians about its center.
///
/// `half` uses the same convention as `Canvas::rectangle` (dx/dy): the
/// full size of the box is `2 * half`. An `angle` of `0.0` is axis-aligned.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OrientedBox {
    /// The box's center in window-centered pixels.
    pub center: [f32; 2],
    /// Half the box's extent along each of its own axes.
    pub half: [f32; 2],
    /// Rotation in radians, counter-clockwise from the axis-aligned pose.
    pub angle: f32,
}

impl OrientedBox {
    /// An axis-aligned box.
    pub fn new(center: [f32; 2], half: [f32; 2]) -> Self {
        Self {
            center,
            half,
            angle: 0.0,
        }
    }

    /// A box rotated counter-clockwise by `angle` radians.
    pub fn rotated(center: [f32; 2], half: [f32; 2], angle: f32) -> Self {
        Self {
            center,
            half,
            angle,
        }
    }

    /// The box's unit axes in world coordinates: `axis(0)` along the box's
    /// own x (the `half[0]` extent) and `axis(1)` along its own y.
    pub fn axis(&self, i: usize) -> [f32; 2] {
        let (sin, cos) = self.angle.sin_cos();
        if i == 0 { [cos, sin] } else { [-sin, cos] }
    }

    /// The box's four corners in world coordinates, counter-clockwise from
    /// the box's own lower-left corner.
    pub fn corners(&self) -> [[f32; 2]; 4] {
        let (sin, cos) = self.angle.sin_cos();
        let [hw, hh] = self.half;
        [[-hw, -hh], [-hw, hh], [hw, hh], [hw, -hh]].map(|[x, y]| {
            [
                self.center[0] + x * cos - y * sin,
                self.center[1] + x * sin + y * cos,
            ]
        })
    }
}

/// A circle.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Circle {
    /// The circle's center in window-centered pixels.
    pub center: [f32; 2],
    /// The circle's radius in pixels.
    pub radius: f32,
}

/// A collision shape.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Collider {
    /// A (possibly rotated) box.
    Box(OrientedBox),
    /// A circle.
    Circle(Circle),
}

/// The minimum translation that separates two overlapping shapes: move the
/// shape by `dir * depth` and it no longer overlaps the other.
#[derive(Clone, Copy, Debug)]
pub struct PushOut {
    /// A unit vector, pointing the way the shape should be moved — away
    /// from the other shape.
    pub dir: [f32; 2],
    /// The distance to move along `dir`.
    pub depth: f32,
}

impl Collider {
    /// Whether the two shapes overlap: `true` when one intrudes into the
    /// other's interior by more than the f32 epsilon, `false` when they are
    /// apart or merely touching.
    pub fn intersects(&self, other: &Self) -> bool {
        self.push_out(other).is_some()
    }

    /// The minimum translation vector that pushes `self` out of `other`, or
    /// `None` when they do not overlap.
    pub fn push_out(&self, other: &Self) -> Option<PushOut> {
        match (self, other) {
            (Collider::Box(a), Collider::Box(b)) => box_push_out(a, b),
            (Collider::Circle(a), Collider::Circle(b)) => circle_push_out(a, b),
            (Collider::Circle(a), Collider::Box(b)) => circle_box_push_out(a, b),
            (Collider::Box(a), Collider::Circle(b)) => {
                // Push the circle out of the box, then flip: the relative
                // separation is symmetric, so the box moves the other way by
                // the same amount.
                circle_box_push_out(b, a).map(|po| PushOut {
                    dir: [-po.dir[0], -po.dir[1]],
                    depth: po.depth,
                })
            }
        }
    }

    /// Slide the shape: every center coordinate moves by `delta`, and
    /// nothing else changes — a box keeps its size and its angle.
    pub fn translate(&mut self, delta: [f32; 2]) {
        let center = match self {
            Collider::Box(b) => &mut b.center,
            Collider::Circle(c) => &mut c.center,
        };
        center[0] += delta[0];
        center[1] += delta[1];
    }

    /// Where the segment from `from` to `to` first enters this shape: the `t`
    /// in `0.0 <= t < 1.0` of `from + t * (to - from)`, or `None` when the
    /// segment misses the shape or merely grazes it.
    ///
    /// The interior counts and the boundary does not, so a segment ending
    /// exactly on the boundary is `None` — the same convention as
    /// [`Collider::push_out`], where touching is not overlapping. A `from`
    /// already inside is `Some(0.0)`, and a zero-length segment hits exactly
    /// when its point is inside.
    ///
    /// The one rough edge, inherited from the shaders' `occluded` that this
    /// mirrors: a segment lying *exactly* in the plane of a box face counts
    /// as inside that slab rather than on its boundary, so a ray skimming a
    /// face can hit. It is deliberate — a directional light's rays share one
    /// direction exactly, so that case is reached routinely rather than by
    /// accident — and pinned by a test.
    pub fn cast(&self, from: [f32; 2], to: [f32; 2]) -> Option<f32> {
        match self {
            Collider::Box(b) => box_segment_t(from, to, b),
            Collider::Circle(c) => circle_segment_t(from, to, c),
        }
    }

    /// Whether `p` is inside the shape or exactly on its boundary — the
    /// inclusive test, which is what both the shaders' occluder skip and
    /// [`Collider::occludes`] want.
    pub fn contains(&self, p: [f32; 2]) -> bool {
        match self {
            Collider::Box(b) => {
                let (sin, cos) = b.angle.sin_cos();
                let rel = [p[0] - b.center[0], p[1] - b.center[1]];
                let lx = rel[0] * cos + rel[1] * sin;
                let ly = -rel[0] * sin + rel[1] * cos;
                lx.abs() <= b.half[0] && ly.abs() <= b.half[1]
            }
            Collider::Circle(c) => len([p[0] - c.center[0], p[1] - c.center[1]]) <= c.radius,
        }
    }

    /// Whether this shape blocks the light at `from` from reaching `to`: the
    /// segment enters it, and `to` is not itself inside it.
    ///
    /// The second half is the shaders' rule — an occluder never shades its
    /// own surface or interior, so a receiver inside a box is lit and the box
    /// shadows only what lies behind it. Note the asymmetry it leaves: a
    /// *light* inside an occluder still casts from its far side, because the
    /// skip tests the receiver alone.
    pub fn occludes(&self, from: [f32; 2], to: [f32; 2]) -> bool {
        !self.contains(to) && self.cast(from, to).is_some()
    }
}

/// The `t` where the segment from `from` to `to` enters `b`, clipped slab by
/// slab. Both endpoints go into the box's own frame first, where it is
/// axis-aligned about the origin — the same transform [`circle_box_push_out`]
/// uses — and each of the two slabs then narrows the interval of `t` spent
/// inside, exactly as the shaders' `occluded` does.
fn box_segment_t(from: [f32; 2], to: [f32; 2], b: &OrientedBox) -> Option<f32> {
    let (sin, cos) = b.angle.sin_cos();
    let to_local = |p: [f32; 2]| -> [f32; 2] {
        let rel = [p[0] - b.center[0], p[1] - b.center[1]];
        [rel[0] * cos + rel[1] * sin, -rel[0] * sin + rel[1] * cos]
    };
    let ll = to_local(from);
    let lp = to_local(to);
    let d = [lp[0] - ll[0], lp[1] - ll[1]];

    let mut tmin: f32 = 0.0;
    let mut tmax: f32 = 1.0;
    for i in 0..2 {
        let h = b.half[i];
        if d[i].abs() > SLAB_EPS {
            let t1 = (-h - ll[i]) / d[i];
            let t2 = (h - ll[i]) / d[i];
            tmin = tmin.max(t1.min(t2));
            tmax = tmax.min(t1.max(t2));
        } else if ll[i] < -h || ll[i] > h {
            // Parallel to this slab and outside it: there is no crossing to
            // find, whatever the other axis says.
            return None;
        }
    }

    // The interval must have positive length and reach into the open segment
    // `(0, 1)`: an exit at exactly `t = 1` is a graze of the far face, and an
    // entry at exactly `t = 1` a touch of the near one.
    (tmax > tmin && tmax > 0.0 && tmin < 1.0).then_some(tmin)
}

/// The `t` where the segment from `from` to `to` enters `c`, by the
/// quadratic the circle's equation and the segment's line give together.
fn circle_segment_t(from: [f32; 2], to: [f32; 2], c: &Circle) -> Option<f32> {
    let d = [to[0] - from[0], to[1] - from[1]];
    let f = [from[0] - c.center[0], from[1] - c.center[1]];
    let a = dot(d, d);
    if a <= EPS * EPS {
        // A zero-length segment has no direction to solve for: it hits only
        // if the point itself is inside.
        return (len(f) <= c.radius).then_some(0.0);
    }
    let b = 2.0 * dot(f, d);
    let disc = b * b - 4.0 * a * (dot(f, f) - c.radius * c.radius);
    if disc <= 0.0 {
        // A miss, or a tangency: the boundary is not the interior.
        return None;
    }
    let s = disc.sqrt();
    let (t0, t1) = ((-b - s) / (2.0 * a), (-b + s) / (2.0 * a));
    (t1 > 0.0 && t0 < 1.0).then_some(t0.max(0.0))
}

/// The nearest shape the segment from `from` to `to` enters, as its index in
/// `shapes`, or `None` when it enters none. The caller keeps the identity of
/// its own shapes — this module reports only which of the ones it was handed,
/// so it stays as stateless as the tests it answers. Ties go to the earlier
/// index.
pub fn first_hit(from: [f32; 2], to: [f32; 2], shapes: &[Collider]) -> Option<usize> {
    let mut best: Option<(f32, usize)> = None;
    for (i, s) in shapes.iter().enumerate() {
        if let Some(t) = s.cast(from, to)
            && best.is_none_or(|(bt, _)| t < bt)
        {
            best = Some((t, i));
        }
    }
    best.map(|(_, i)| i)
}

/// Whether any occluder blocks the light at `from` from reaching `to`: the
/// CPU's twin of the shaders' `occluded`, answering the same question the
/// pixels are answered with — see [`Collider::occludes`] for the rules it
/// applies, and [`visible`] for the line-of-sight reading of the same field.
pub fn occluded(from: [f32; 2], to: [f32; 2], occluders: &[Collider]) -> bool {
    occluders.iter().any(|o| o.occludes(from, to))
}

/// Whether a viewer at `from` sees a target at `to`: `false` as soon as one
/// shape's interior meets the segment, at any `t` including `0.0`. So a shape
/// containing either endpoint blocks the view, which is the opposite answer
/// [`occluded`] gives for a receiver inside an occluder — a lamp inside a
/// crate lights the crate, a crate-mate inside it is not *seen* through it.
pub fn visible(from: [f32; 2], to: [f32; 2], blockers: &[Collider]) -> bool {
    first_hit(from, to, blockers).is_none()
}

/// The box–box minimum translation vector, by the separating-axis theorem:
/// two boxes overlap only when their projections overlap on every one of
/// the four candidate axes (two per box), and the axis of least penetration
/// is the minimum translation that separates them.
fn box_push_out(a: &OrientedBox, b: &OrientedBox) -> Option<PushOut> {
    let axes = [a.axis(0), a.axis(1), b.axis(0), b.axis(1)];
    let mut best: Option<([f32; 2], f32)> = None; // (axis, penetration)

    for &axis in &axes {
        let ra = a.half[0] * dot(a.axis(0), axis).abs() + a.half[1] * dot(a.axis(1), axis).abs();
        let rb = b.half[0] * dot(b.axis(0), axis).abs() + b.half[1] * dot(b.axis(1), axis).abs();
        let d = dot([b.center[0] - a.center[0], b.center[1] - a.center[1]], axis);
        let pen = ra + rb - d.abs();
        if pen <= EPS {
            return None; // A separating axis: no overlap.
        }
        match best {
            Some((_, p)) if p <= pen => {}
            _ => best = Some((axis, pen)),
        }
    }

    let (axis, pen) = best?; // Every axis overlapped.
    // Orient the axis so it points from `b` toward `a`: when `b` lies on
    // the + side of `a` along the axis, `a` must be pushed along −.
    let s = dot([b.center[0] - a.center[0], b.center[1] - a.center[1]], axis);
    let dir = if s > 0.0 { [-axis[0], -axis[1]] } else { axis };
    Some(PushOut { dir, depth: pen })
}

/// The circle–circle minimum translation vector.
fn circle_push_out(a: &Circle, b: &Circle) -> Option<PushOut> {
    let d = [a.center[0] - b.center[0], a.center[1] - b.center[1]];
    let dist = len(d);
    let pen = a.radius + b.radius - dist;
    if pen <= EPS {
        return None;
    }
    let dir = if dist > EPS {
        [d[0] / dist, d[1] / dist]
    } else {
        [1.0, 0.0] // Concentric: a deterministic, arbitrary axis.
    };
    Some(PushOut { dir, depth: pen })
}

/// The circle–box minimum translation vector, pushing the circle out of the
/// box. The circle's center is transformed into the box's own frame, where
/// the box is an axis-aligned square about the origin.
fn circle_box_push_out(c: &Circle, b: &OrientedBox) -> Option<PushOut> {
    let (sin, cos) = b.angle.sin_cos();
    let rel = [c.center[0] - b.center[0], c.center[1] - b.center[1]];
    // Rotate `rel` by `-angle` into the box's frame.
    let lx = rel[0] * cos + rel[1] * sin;
    let ly = -rel[0] * sin + rel[1] * cos;

    let (dir_local, depth) = if lx.abs() <= b.half[0] && ly.abs() <= b.half[1] {
        // The center is inside the box: the circle is clear only once it has
        // moved entirely past the nearest face.
        let dx = b.half[0] - lx.abs();
        let dy = b.half[1] - ly.abs();
        let sx = if lx < 0.0 { -1.0 } else { 1.0 };
        let sy = if ly < 0.0 { -1.0 } else { 1.0 };
        if dx <= dy {
            ([sx, 0.0], c.radius + dx)
        } else {
            ([0.0, sy], c.radius + dy)
        }
    } else {
        // The center is outside: push away from the closest point on the box.
        let cx = lx.clamp(-b.half[0], b.half[0]);
        let cy = ly.clamp(-b.half[1], b.half[1]);
        let d = [lx - cx, ly - cy];
        let dist = len(d);
        let pen = c.radius - dist;
        if pen <= EPS {
            return None;
        }
        let dir = if dist > EPS {
            [d[0] / dist, d[1] / dist]
        } else {
            [1.0, 0.0]
        };
        (dir, pen)
    };

    // Rotate the local direction by `+angle` back to world.
    Some(PushOut {
        dir: [
            dir_local[0] * cos - dir_local[1] * sin,
            dir_local[0] * sin + dir_local[1] * cos,
        ],
        depth,
    })
}

/// Reflect a velocity about a collision normal, with restitution.
///
/// `n` is the unit collision normal, pointing the way the body was pushed —
/// away from the surface it hit (see [`PushOut::dir`]). `e` is the
/// restitution in `[0, 1]`: `0.0` kills the velocity's normal component (the
/// body slides along the surface), `1.0` is a perfectly elastic bounce. A
/// velocity already moving away from the surface is returned unchanged.
pub fn reflect(vel: [f32; 2], n: [f32; 2], e: f32) -> [f32; 2] {
    let vn = dot(vel, n);
    if vn >= 0.0 {
        return vel;
    }
    let k = (1.0 + e) * vn;
    [vel[0] - k * n[0], vel[1] - k * n[1]]
}

/// The most edges a [`Convex`] may have, and the number of sides a
/// [`Convex::disc`] is cut into.
///
/// One number serves both because the GPU cost of a polygon is per edge: every
/// shadow ray clips against every edge of every occluder, and a light with a
/// penumbra fires nine rays per pixel. Sixteen is enough to be round — the
/// sixteen-gon inscribed in a disc stays within two percent of its radius,
/// which at any size a player can see is under a pixel or two of edge
/// travel — and small enough that a hundred lights still cost a fragment
/// nothing. Shapes needing more are the author's to simplify, not the
/// renderer's to approximate.
pub const MAX_PLANES: usize = 16;

/// The number of sides [`Convex::disc`] uses: the widest polygon the type
/// allows, so the tessellated circle is as round as the cap permits.
pub const DISC_SIDES: usize = MAX_PLANES;

/// A convex polygon, held as the half-plane of each of its edges.
///
/// The half-planes are the representation rather than a derived one: a convex
/// shape *is* the intersection of the half-planes its edges bound, which is
/// what makes the segment test the box's slab test with more slabs — no
/// new geometry, no new case. Normals are normalised and their offsets
/// computed once, here, so a shader clipping a ray against sixteen edges does
/// sixteen dot products instead of sixteen square roots.
///
/// This is the occluder's shape, not a physics body: [`Collider`] gains no
/// variant for it, because pushing two polygons apart needs the separating
/// axis of a curved boundary and a contact normal worth its name, and that is
/// a step of its own rather than a free rider on this one.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Convex {
    /// Each edge as `(n.x, n.y, c, _)` with `n` the outward unit normal: a
    /// point is inside the edge when `n.x * p.x + n.y * p.y <= c`.
    planes: [[f32; 4]; MAX_PLANES],
    /// How many of `planes` are in use. Always at least three.
    count: u8,
    /// The polygon's own axis-aligned bounds, so a caller can reject a ray
    /// that misses it without touching a single edge.
    bounds: ([f32; 2], [f32; 2]),
}

impl Convex {
    /// The polygon bounded by `points`, given in order around it either way.
    ///
    /// The winding is inferred from the signed area, so the same shape drawn
    /// backwards is the same polygon — the y-flip between image space and
    /// pixel space reverses a winding on its way to the GPU, and a shape that
    /// silently turned inside out there would have its normals pointing
    /// inwards and shadow everything.
    ///
    /// `None` is the answer for anything that cannot be a convex occluder:
    /// fewer than three corners, more than [`MAX_PLANES`], a repeated or
    /// nearly-coincident corner, a coordinate that is not a number, or a
    /// corner that bulges outward. Nothing is decomposed and nothing is
    /// repaired: a concave outline is a shape the author meant differently
    /// than the renderer would guess, so it is reported rather than drawn
    /// wrong.
    pub fn new(points: &[[f32; 2]]) -> Option<Self> {
        if points.len() < 3 || points.len() > MAX_PLANES {
            return None;
        }
        if !points.iter().all(|p| p[0].is_finite() && p[1].is_finite()) {
            return None;
        }
        let doubled_area: f32 = points
            .iter()
            .zip(points.iter().cycle().skip(1))
            .map(|([ax, ay], [bx, by])| ax * by - bx * ay)
            .sum();
        if doubled_area.abs() < EPS {
            return None; // collinear, or a shape with no area to occlude with
        }
        // Order the corners counter-clockwise in the signed-area sense, so one
        // normal convention serves every input winding.
        let ordered: Vec<[f32; 2]> = if doubled_area > 0.0 {
            points.to_vec()
        } else {
            points.iter().rev().copied().collect()
        };
        let center = [
            ordered.iter().map(|p| p[0]).sum::<f32>() / ordered.len() as f32,
            ordered.iter().map(|p| p[1]).sum::<f32>() / ordered.len() as f32,
        ];
        let mut planes = [[0.0f32; 4]; MAX_PLANES];
        for i in 0..ordered.len() {
            let a = ordered[i];
            let b = ordered[(i + 1) % ordered.len()];
            let d = [b[0] - a[0], b[1] - a[1]];
            let length = len(d);
            if length < EPS {
                return None; // a repeated corner: the edge has no direction
            }
            let mut n = [d[1] / length, -d[0] / length];
            let mut c = n[0] * a[0] + n[1] * a[1];
            if n[0] * center[0] + n[1] * center[1] > c {
                // The interior landed outside this edge: the winding was
                // already the other way round, or the corner bulges out.
                n = [-n[0], -n[1]];
                c = -c;
            }
            if n[0] * center[0] + n[1] * center[1] > c {
                return None;
            }
            planes[i] = [n[0], n[1], c, 0.0];
        }
        // Every corner inside every edge is what convexity means, and it is
        // cheap to check here rather than to discover as a wrong shadow.
        for p in &ordered {
            for plane in planes.iter().take(ordered.len()) {
                let slack = CONVEX_EPS + plane[2].abs() * 1e-6;
                if plane[0] * p[0] + plane[1] * p[1] > plane[2] + slack {
                    return None;
                }
            }
        }
        let lo = [
            ordered.iter().map(|p| p[0]).fold(f32::INFINITY, f32::min),
            ordered.iter().map(|p| p[1]).fold(f32::INFINITY, f32::min),
        ];
        let hi = [
            ordered
                .iter()
                .map(|p| p[0])
                .fold(f32::NEG_INFINITY, f32::max),
            ordered
                .iter()
                .map(|p| p[1])
                .fold(f32::NEG_INFINITY, f32::max),
        ];
        Some(Self {
            planes,
            count: ordered.len() as u8,
            bounds: (lo, hi),
        })
    }

    /// A regular `DISC_SIDES`-sided polygon with its corners on the circle of
    /// `radius` about `center`: the shape a disc becomes on the way to the
    /// occluder field, where a curved edge has to be told to the GPU as flat
    /// ones.
    ///
    /// The disc's own exact form stays available as [`Circle`] — physics wants
    /// the round normal, and the shadow is the only consumer that cannot have
    /// it. The gap between the two is bounded by the polygon's inradius,
    /// `radius * cos(pi / DISC_SIDES)`, and pinned by a test, because it is
    /// the tolerance every GPU-versus-CPU shadow comparison over a circle has
    /// to be measured against.
    pub fn disc(center: [f32; 2], radius: f32) -> Self {
        let step = std::f32::consts::TAU / DISC_SIDES as f32;
        let points: Vec<[f32; 2]> = (0..DISC_SIDES)
            .map(|k| {
                let (sin, cos) = (k as f32 * step).sin_cos();
                [center[0] + radius * cos, center[1] + radius * sin]
            })
            .collect();
        // Regular, convex and within the cap by construction, so there is no
        // error to return; a radius that is not a number is the author's.
        Self::new(&points).expect("a regular polygon is convex")
    }

    /// The edges as `(n.x, n.y, c, _)`, in the order they run around the
    /// shape: exactly the ones it has, which are the bytes a packer writes
    /// into the occluder field. The length *is* the edge count, deliberately —
    /// the backing array is sized for the widest polygon a shape may be, so
    /// handing it out whole would let a caller ship a hexagon's record wearing
    /// sixteen edges, ten of them zeros that clip nothing and cost a shadow
    /// ray each.
    pub fn planes(&self) -> &[[f32; 4]] {
        &self.planes[..self.count as usize]
    }

    /// The polygon's axis-aligned bounds in its own space, as the low and high
    /// corners. A ray missing these misses the shape, which is what makes the
    /// bounds worth shipping to the GPU: four comparisons reject a shadow ray
    /// before sixteen clip it.
    pub fn bounds(&self) -> ([f32; 2], [f32; 2]) {
        self.bounds
    }

    /// The center and half-extents of those bounds, in the [`OrientedBox`]
    /// convention the occluder record already carries: the full size is twice
    /// the half.
    pub fn bounds_as_box(&self) -> OrientedBox {
        let (lo, hi) = self.bounds;
        OrientedBox::new(
            [(lo[0] + hi[0]) / 2.0, (lo[1] + hi[1]) / 2.0],
            [(hi[0] - lo[0]) / 2.0, (hi[1] - lo[1]) / 2.0],
        )
    }

    /// Whether `p` is inside the polygon or exactly on its edge — the
    /// inclusive test, matching [`Collider::contains`] so that the receiver
    /// inside an occluder is lit whether the occluder is a box, a disc or a
    /// rock.
    pub fn contains(&self, p: [f32; 2]) -> bool {
        self.planes[..self.count as usize]
            .iter()
            .all(|plane| plane[0] * p[0] + plane[1] * p[1] <= plane[2])
    }

    /// Where the segment `from` to `to` enters the polygon, as `t` in `from +
    /// t * (to - from)`, or `None` when it misses: the same interval clip the
    /// box performs, one edge at a time, and the same acceptance rule — an
    /// exit at exactly `t = 1` is a graze of the far edge, an entry at exactly
    /// `t = 1` a touch of the near one.
    pub fn cast(&self, from: [f32; 2], to: [f32; 2]) -> Option<f32> {
        let planes = self.planes();
        let mut tmin = 0.0f32;
        let mut tmax = 1.0f32;
        for plane in planes {
            // Which side of the edge each endpoint falls on. The segment lies
            // inside the edge where `dot(n, p) - c` is at or below zero, and
            // that quantity is linear along it, so one crossing bounds the
            // interval — and a segment already through the edge on both ends
            // needs no division at all, which is where the box's slab epsilon
            // would otherwise be needed.
            let da = plane[0] * from[0] + plane[1] * from[1] - plane[2];
            let db = plane[0] * to[0] + plane[1] * to[1] - plane[2];
            if da > 0.0 {
                if db > 0.0 {
                    return None; // both ends outside this edge: no crossing
                }
                tmin = tmin.max(da / (da - db));
            } else if db > 0.0 {
                tmax = tmax.min(da / (da - db));
            }
        }
        (tmax > tmin && tmax > 0.0 && tmin < 1.0).then_some(tmin)
    }

    /// Whether this polygon blocks the light at `from` from reaching `to`:
    /// the segment enters it, and `to` is not itself inside it. The receiver
    /// rule is [`Collider::occludes`]'s, unchanged — a surface inside an
    /// occluder is lit, and the occluder shadows only what lies behind it.
    pub fn occludes(&self, from: [f32; 2], to: [f32; 2]) -> bool {
        !self.contains(to) && self.cast(from, to).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, FRAC_PI_4, SQRT_2};

    /// Whether two 2D vectors are within `eps` component-wise.
    fn close(a: [f32; 2], b: [f32; 2], eps: f32) -> bool {
        (a[0] - b[0]).abs() < eps && (a[1] - b[1]).abs() < eps
    }

    // -- OrientedBox basics --

    #[test]
    fn unrotated_box_axes_and_corners() {
        let b = OrientedBox::new([10.0, 20.0], [30.0, 40.0]);
        assert_eq!(b.axis(0), [1.0, 0.0]);
        assert_eq!(b.axis(1), [0.0, 1.0]);
        assert_eq!(
            b.corners(),
            [[-20.0, -20.0], [-20.0, 60.0], [40.0, 60.0], [40.0, -20.0]]
        );
    }

    #[test]
    fn quarter_turn_rotates_box() {
        let b = OrientedBox::rotated([0.0, 0.0], [10.0, 20.0], FRAC_PI_2);
        assert!(close(b.axis(0), [0.0, 1.0], 1e-5));
        assert!(close(b.axis(1), [-1.0, 0.0], 1e-5));
        let expected = [[20.0, -10.0], [-20.0, -10.0], [-20.0, 10.0], [20.0, 10.0]];
        for (got, want) in b.corners().iter().zip(expected.iter()) {
            assert!(close(*got, *want, 1e-4));
        }
    }

    // -- Box–box --

    #[test]
    fn aabb_overlap_and_miss() {
        let a = Collider::Box(OrientedBox::new([0.0, 0.0], [50.0, 50.0]));
        let near = Collider::Box(OrientedBox::new([80.0, 0.0], [50.0, 50.0]));
        assert!(a.intersects(&near)); // a spans ±50, near spans 30..130.
        let far = Collider::Box(OrientedBox::new([100.1, 0.0], [50.0, 50.0]));
        assert!(!a.intersects(&far)); // A 0.1 gap.
    }

    #[test]
    fn exact_touching_is_not_overlap() {
        let a = Collider::Box(OrientedBox::new([0.0, 0.0], [50.0, 50.0]));
        let b = Collider::Box(OrientedBox::new([100.0, 0.0], [50.0, 50.0]));
        assert!(!a.intersects(&b));
    }

    #[test]
    fn subpixel_gap_is_not_overlap() {
        let a = Collider::Box(OrientedBox::new([0.0, 0.0], [1.0, 1.0]));
        let b = Collider::Box(OrientedBox::new([1.999999, 0.0], [1.0, 1.0]));
        assert!(!a.intersects(&b)); // A ~1e-6 gap: within the epsilon.
    }

    #[test]
    fn aabb_push_out_from_four_sides() {
        let a = Collider::Box(OrientedBox::new([0.0, 0.0], [10.0, 10.0]));
        let from_right = Collider::Box(OrientedBox::new([15.0, 0.0], [10.0, 10.0]));
        let po = a.push_out(&from_right).unwrap();
        assert!(close(po.dir, [-1.0, 0.0], 1e-5));
        assert!((po.depth - 5.0).abs() < 1e-5);

        let from_left = Collider::Box(OrientedBox::new([-15.0, 0.0], [10.0, 10.0]));
        let po = a.push_out(&from_left).unwrap();
        assert!(close(po.dir, [1.0, 0.0], 1e-5));
        assert!((po.depth - 5.0).abs() < 1e-5);

        let from_above = Collider::Box(OrientedBox::new([0.0, 15.0], [10.0, 10.0]));
        let po = a.push_out(&from_above).unwrap();
        assert!(close(po.dir, [0.0, -1.0], 1e-5));
        assert!((po.depth - 5.0).abs() < 1e-5);

        let from_below = Collider::Box(OrientedBox::new([0.0, -15.0], [10.0, 10.0]));
        let po = a.push_out(&from_below).unwrap();
        assert!(close(po.dir, [0.0, 1.0], 1e-5));
        assert!((po.depth - 5.0).abs() < 1e-5);
    }

    #[test]
    fn rotated_bar_overlaps_horizontal_bar() {
        let a = Collider::Box(OrientedBox::rotated([0.0, 0.0], [40.0, 10.0], 0.0));
        let b = Collider::Box(OrientedBox::rotated([20.0, 0.0], [10.0, 40.0], FRAC_PI_4));
        assert!(a.intersects(&b));
    }

    #[test]
    fn rotated_box_half_pixel_clear_is_not_overlap() {
        // A diamond whose left vertex is a half pixel past the square's
        // right edge: apart.
        let a = Collider::Box(OrientedBox::new([0.0, 0.0], [10.0, 10.0]));
        let b = Collider::Box(OrientedBox::rotated(
            [10.0 + 10.0 * SQRT_2 + 0.5, 0.0],
            [10.0, 10.0],
            FRAC_PI_4,
        ));
        assert!(!a.intersects(&b));
    }

    #[test]
    fn rotated_box_half_pixel_in_overlaps_and_pushes_along_x() {
        let a = Collider::Box(OrientedBox::new([0.0, 0.0], [10.0, 10.0]));
        let b = Collider::Box(OrientedBox::rotated(
            [10.0 + 10.0 * SQRT_2 - 0.5, 0.0],
            [10.0, 10.0],
            FRAC_PI_4,
        ));
        let po = a.push_out(&b).unwrap();
        assert!(close(po.dir, [-1.0, 0.0], 1e-4));
        assert!((po.depth - 0.5).abs() < 1e-4);
    }

    #[test]
    fn rotated_push_out_picks_least_penetration_axis() {
        // A diamond nudged half a pixel above a horizontal bar: the bar's
        // own y axis is the least-penetrated one, so that is the MTV.
        let a = Collider::Box(OrientedBox::rotated([0.0, 0.0], [40.0, 10.0], 0.0));
        let b = Collider::Box(OrientedBox::rotated([30.0, 1.0], [10.0, 10.0], FRAC_PI_4));
        let po = a.push_out(&b).unwrap();
        assert!(close(po.dir, [0.0, -1.0], 1e-4));
        assert!((po.depth - (9.0 + 10.0 * SQRT_2)).abs() < 1e-4);
    }

    #[test]
    fn push_out_actually_separates() {
        let ab = OrientedBox::rotated([0.0, 0.0], [30.0, 10.0], 0.3);
        let bb = OrientedBox::rotated([25.0, 5.0], [40.0, 8.0], -0.7);
        let a = Collider::Box(ab);
        let b = Collider::Box(bb);
        assert!(a.intersects(&b));

        let po = a.push_out(&b).unwrap();
        let moved = OrientedBox::rotated(
            [
                ab.center[0] + po.dir[0] * po.depth,
                ab.center[1] + po.dir[1] * po.depth,
            ],
            ab.half,
            ab.angle,
        );
        assert!(!Collider::Box(moved).intersects(&b));
    }

    // -- Circle–circle --

    #[test]
    fn circle_overlap_touch_miss() {
        let a = Collider::Circle(Circle {
            center: [0.0, 0.0],
            radius: 30.0,
        });
        let b = Collider::Circle(Circle {
            center: [40.0, 0.0],
            radius: 30.0,
        });
        assert!(a.intersects(&b));
        let po = a.push_out(&b).unwrap();
        assert!(close(po.dir, [-1.0, 0.0], 1e-5));
        assert!((po.depth - 20.0).abs() < 1e-4);

        let c = Collider::Circle(Circle {
            center: [60.0, 0.0],
            radius: 30.0,
        });
        assert!(!a.intersects(&c)); // Exactly touching.
    }

    #[test]
    fn concentric_circles_push_out_is_deterministic() {
        let a = Collider::Circle(Circle {
            center: [0.0, 0.0],
            radius: 10.0,
        });
        let b = Collider::Circle(Circle {
            center: [0.0, 0.0],
            radius: 10.0,
        });
        let po = a.push_out(&b).unwrap();
        assert!(close(po.dir, [1.0, 0.0], 1e-6));
        assert!((po.depth - 20.0).abs() < 1e-5);
    }

    // -- Circle–box --

    #[test]
    fn circle_aabb_edge_push_out() {
        let sq = Collider::Box(OrientedBox::new([0.0, 0.0], [50.0, 50.0]));
        let c = Collider::Circle(Circle {
            center: [55.0, 0.0],
            radius: 20.0,
        });
        assert!(c.intersects(&sq));
        let po = c.push_out(&sq).unwrap();
        assert!(close(po.dir, [1.0, 0.0], 1e-5));
        assert!((po.depth - 15.0).abs() < 1e-4); // 50 + 20 - 55.
    }

    #[test]
    fn circle_aabb_corner_push_out() {
        let sq = Collider::Box(OrientedBox::new([0.0, 0.0], [50.0, 50.0]));
        let c = Collider::Circle(Circle {
            center: [60.0, 60.0],
            radius: 15.0,
        });
        let po = c.push_out(&sq).unwrap();
        assert!(close(po.dir, [SQRT_2 / 2.0, SQRT_2 / 2.0], 1e-4));
        let depth = 15.0 - (200.0_f32).sqrt();
        assert!((po.depth - depth).abs() < 1e-4);
    }

    #[test]
    fn circle_aabb_touching_is_not_overlap() {
        let sq = Collider::Box(OrientedBox::new([0.0, 0.0], [50.0, 50.0]));
        let c = Collider::Circle(Circle {
            center: [70.0, 0.0],
            radius: 20.0,
        });
        assert!(!c.intersects(&sq)); // Closest point at distance 20 = radius.
    }

    #[test]
    fn circle_center_inside_box_pushes_out_past_nearest_face() {
        let sq = Collider::Box(OrientedBox::new([0.0, 0.0], [50.0, 30.0]));
        let c = Collider::Circle(Circle {
            center: [10.0, 0.0],
            radius: 20.0,
        });
        let po = c.push_out(&sq).unwrap();
        // The nearest face is the top one (30 away, not 40): the circle
        // clears once it has moved radius + face distance along +y.
        assert!(close(po.dir, [0.0, 1.0], 1e-6));
        assert!((po.depth - 50.0).abs() < 1e-4);
    }

    #[test]
    fn circle_fully_inside_box_pushes_out_full_clearance() {
        let sq = Collider::Box(OrientedBox::new([0.0, 0.0], [50.0, 50.0]));
        let c = Collider::Circle(Circle {
            center: [0.0, 0.0],
            radius: 10.0,
        });
        let po = c.push_out(&sq).unwrap();
        assert!(close(po.dir, [1.0, 0.0], 1e-6));
        assert!((po.depth - 60.0).abs() < 1e-4); // radius + face distance.
    }

    #[test]
    fn circle_rotated_box_push_out() {
        // A horizontal bar turned a quarter turn is a vertical bar with the
        // half-extents swapped.
        let bar = Collider::Box(OrientedBox::rotated([0.0, 0.0], [40.0, 10.0], FRAC_PI_2));
        let c = Collider::Circle(Circle {
            center: [0.0, 45.0],
            radius: 20.0,
        });
        let po = c.push_out(&bar).unwrap();
        assert!(close(po.dir, [0.0, 1.0], 1e-4));
        assert!((po.depth - 15.0).abs() < 1e-4);
    }

    // -- Box–circle symmetry --

    #[test]
    fn box_push_out_of_circle_is_inverse_of_circle() {
        let a = Collider::Box(OrientedBox::new([0.0, 0.0], [30.0, 20.0]));
        let c = Collider::Circle(Circle {
            center: [40.0, 0.0],
            radius: 25.0,
        });
        let ba = a.push_out(&c).unwrap();
        let cb = c.push_out(&a).unwrap();
        assert!(close(ba.dir, [-cb.dir[0], -cb.dir[1]], 1e-6));
        assert!((ba.depth - cb.depth).abs() < 1e-5);
        assert!((ba.depth - 15.0).abs() < 1e-4); // 30 + 25 - 40.
    }

    // -- reflect --

    #[test]
    fn reflect_zero_restitution_slides() {
        // Hitting a wall whose normal points +x while moving toward it:
        // the normal component is killed, the tangential one survives.
        let v = reflect([-100.0, 50.0], [1.0, 0.0], 0.0);
        assert!(close(v, [0.0, 50.0], 1e-4));
    }

    #[test]
    fn reflect_full_restitution_bounces() {
        let v = reflect([-100.0, 50.0], [1.0, 0.0], 1.0);
        assert!(close(v, [100.0, 50.0], 1e-4));
    }

    #[test]
    fn reflect_ignores_velocity_away_from_surface() {
        let v = reflect([100.0, 50.0], [1.0, 0.0], 0.5);
        assert!(close(v, [100.0, 50.0], 1e-6));
    }

    // -- Segment queries --

    /// The box the segment tests share: centered at the origin, spanning ±50.
    fn seg_box() -> Collider {
        Collider::Box(OrientedBox::new([0.0, 0.0], [50.0, 50.0]))
    }

    /// The circle the segment tests share: centered at the origin, radius 50.
    fn seg_circle() -> Collider {
        Collider::Circle(Circle {
            center: [0.0, 0.0],
            radius: 50.0,
        })
    }

    #[test]
    fn segment_enters_a_box_at_the_near_face() {
        let t = seg_box().cast([-100.0, 0.0], [100.0, 0.0]).unwrap();
        assert!((t - 0.25).abs() < 1e-5); // x = -50 of a 200 px segment.
    }

    #[test]
    fn segment_touching_a_box_face_is_not_a_hit() {
        assert_eq!(seg_box().cast([-100.0, 0.0], [-50.0, 0.0]), None);
        assert_eq!(seg_box().cast([-100.0, 0.0], [-60.0, 0.0]), None);
    }

    #[test]
    fn segment_ending_inside_a_box_hits_at_the_face() {
        let t = seg_box().cast([-100.0, 0.0], [0.0, 0.0]).unwrap();
        assert!((t - 0.5).abs() < 1e-5);
    }

    #[test]
    fn segment_starting_inside_a_box_hits_at_zero() {
        let t = seg_box().cast([0.0, 0.0], [100.0, 0.0]).unwrap();
        assert!(t.abs() < 1e-6);
    }

    #[test]
    fn segment_parallel_and_outside_a_slab_misses() {
        // Ten pixels above the box, running along it: the degenerate slab has
        // nothing to clip, so the parallel-outside branch is what rejects.
        assert_eq!(seg_box().cast([-100.0, 60.0], [100.0, 60.0]), None);
    }

    #[test]
    fn segment_skimming_a_box_face_counts_as_inside_that_slab() {
        // The shaders' convention, mirrored on purpose: `ll.y > half.y` is an
        // exclusive test, so a ray lying exactly in the face plane is inside
        // the slab rather than on its boundary. A directional light's rays
        // share one direction exactly, so this branch is reached routinely.
        let t = seg_box().cast([-100.0, 50.0], [100.0, 50.0]).unwrap();
        assert!((t - 0.25).abs() < 1e-5);
    }

    #[test]
    fn segment_hits_a_rotated_box_through_its_own_axes() {
        // A quarter turn turns the 80 x 20 bar into a 20 x 80 one: entered
        // along y at -40 of a 200 px segment.
        let bar = Collider::Box(OrientedBox::rotated([0.0, 0.0], [40.0, 10.0], FRAC_PI_2));
        let t = bar.cast([0.0, -100.0], [0.0, 100.0]).unwrap();
        assert!((t - 0.3).abs() < 1e-5);
    }

    #[test]
    fn degenerate_segment_hits_only_when_its_point_is_inside() {
        assert_eq!(seg_box().cast([0.0, 0.0], [0.0, 0.0]), Some(0.0));
        assert_eq!(seg_box().cast([100.0, 0.0], [100.0, 0.0]), None);
        assert_eq!(seg_circle().cast([0.0, 0.0], [0.0, 0.0]), Some(0.0));
        assert_eq!(seg_circle().cast([100.0, 0.0], [100.0, 0.0]), None);
    }

    #[test]
    fn segment_enters_a_circle_at_the_near_crossing() {
        let t = seg_circle().cast([-100.0, 0.0], [100.0, 0.0]).unwrap();
        assert!((t - 0.25).abs() < 1e-5);
        assert_eq!(seg_circle().cast([-100.0, 0.0], [-50.0, 0.0]), None);
    }

    #[test]
    fn segment_tangent_to_a_circle_is_not_a_hit() {
        assert_eq!(seg_circle().cast([-100.0, 50.0], [100.0, 50.0]), None);
        assert_eq!(seg_circle().cast([-100.0, 60.0], [100.0, 60.0]), None);
    }

    #[test]
    fn contains_is_inclusive_of_the_boundary() {
        assert!(seg_box().contains([50.0, 0.0]));
        assert!(seg_box().contains([0.0, 0.0]));
        assert!(!seg_box().contains([50.1, 0.0]));
        assert!(seg_circle().contains([0.0, 50.0]));
        assert!(!seg_circle().contains([0.0, 50.1]));
    }

    // -- Occlusion and line of sight --

    #[test]
    fn occluded_ignores_the_occluder_holding_the_receiver() {
        // The shaders' skip: a receiver inside the box is lit by the lamp,
        // even though the segment runs through the box to reach it.
        let b = seg_box();
        assert!(b.cast([-100.0, 0.0], [0.0, 0.0]).is_some());
        assert!(!b.occludes([-100.0, 0.0], [0.0, 0.0]));
        assert!(!occluded([-100.0, 0.0], [0.0, 0.0], &[b]));
    }

    #[test]
    fn occluded_reports_a_light_inside_the_occluder() {
        // The other half of the same rule: the skip tests the receiver alone,
        // so a lamp inside a crate still shades what lies beyond it.
        let b = seg_box();
        assert!(b.occludes([0.0, 0.0], [100.0, 0.0]));
        assert!(occluded([0.0, 0.0], [100.0, 0.0], &[b]));
    }

    #[test]
    fn occluded_uses_the_open_segment() {
        // The light is 10 px clear of the box and the ray stops short of it.
        assert!(!occluded([-60.0, 0.0], [-55.0, 0.0], &[seg_box()]));
        assert!(occluded([-60.0, 0.0], [60.0, 0.0], &[seg_box()]));
    }

    #[test]
    fn visible_blocks_both_endpoints_inside() {
        // Where `occluded` says lit, `visible` says hidden: the same field,
        // the opposite policy for an endpoint inside a shape.
        assert!(!visible([-100.0, 0.0], [0.0, 0.0], &[seg_box()]));
        assert!(!visible([0.0, 0.0], [100.0, 0.0], &[seg_box()]));
        assert!(visible([-100.0, 80.0], [100.0, 80.0], &[seg_box()]));
    }

    #[test]
    fn an_empty_field_neither_shades_hides_nor_hits() {
        assert!(!occluded([-100.0, 0.0], [100.0, 0.0], &[]));
        assert!(visible([-100.0, 0.0], [100.0, 0.0], &[]));
        assert_eq!(first_hit([-100.0, 0.0], [100.0, 0.0], &[]), None);
    }

    #[test]
    fn first_hit_reports_the_nearest_shape_by_index() {
        let near = Collider::Box(OrientedBox::new([100.0, 0.0], [20.0, 20.0]));
        let far = Collider::Box(OrientedBox::new([300.0, 0.0], [20.0, 20.0]));
        let (from, to) = ([-100.0, 0.0], [400.0, 0.0]);
        assert_eq!(first_hit(from, to, &[near, far]), Some(0));
        assert_eq!(first_hit(from, to, &[far, near]), Some(1));
        assert_eq!(first_hit(from, to, &[far]), Some(0));
        assert_eq!(first_hit([200.0, 0.0], [250.0, 0.0], &[near, far]), None);
    }

    #[test]
    fn first_hit_mixes_shape_kinds_by_distance() {
        let circle = seg_circle();
        let beyond = Collider::Box(OrientedBox::new([300.0, 0.0], [20.0, 20.0]));
        let (from, to) = ([-100.0, 0.0], [400.0, 0.0]);
        assert_eq!(first_hit(from, to, &[beyond, circle]), Some(1));
        assert_eq!(
            first_hit([200.0, 0.0], [400.0, 0.0], &[beyond, circle]),
            Some(0)
        );
    }

    #[test]
    fn a_convex_polygon_blocks_the_light_between_its_edges() {
        // An octagon standing between lamp and wall: the wall behind it is in
        // shadow, the wall beside it is not, and the difference is the
        // polygon's silhouette rather than its bounding box's — the corner
        // points are the ones a box would get wrong.
        let points: Vec<[f32; 2]> = (0..8)
            .map(|k| {
                let a = std::f32::consts::TAU * k as f32 / 8.0;
                [40.0 * a.cos(), 40.0 * a.sin()]
            })
            .collect();
        let octagon = Convex::new(&points).expect("a regular octagon is convex");
        let lamp = [-100.0, 0.0];
        assert!(
            octagon.occludes(lamp, [100.0, 0.0]),
            "a ray straight through the octagon should be blocked"
        );
        // Past the silhouette, where the light reaches and the polygon's
        // bounds would have lied. The octagon's top corner at (0, 40) is the
        // tangent from this lamp, throwing shadow to y = 80 at x = 100; its
        // bounds' near corner at (-40, 40) is a steeper tangent still, and
        // would have thrown shadow to y = 133. A point between the two is lit
        // by the shape and darkened by its box.
        let beyond = [100.0, 100.0];
        assert!(
            !octagon.occludes(lamp, beyond),
            "past the silhouette the light should reach"
        );
        assert!(
            occluded(lamp, beyond, &[Collider::Box(octagon.bounds_as_box())]),
            "the same ray against the bounds should be blocked: the edges are \
             what changed the answer, which is the whole point of the shape"
        );
        assert!(
            octagon.contains([0.0, 0.0]),
            "the octagon's own middle is inside it"
        );
        // And the bounds agree with the shape they wrap.
        let (lo, hi) = octagon.bounds();
        assert!(lo[0] < -39.0 && hi[1] > 39.0, "the bounds wrap the corners");
    }

    #[test]
    fn a_receiver_inside_a_convex_polygon_is_lit() {
        // The rule the shaders implement and `occluded` promises: an occluder
        // never shades its own interior, so the same segment that is blocked
        // to a point beyond the shape is not blocked to a point within it.
        let square = Convex::new(&[[-20.0, -20.0], [20.0, -20.0], [20.0, 20.0], [-20.0, 20.0]])
            .expect("a square is convex");
        assert!(square.occludes([-100.0, 0.0], [100.0, 0.0]));
        assert!(
            !square.occludes([-100.0, 0.0], [0.0, 0.0]),
            "a receiver inside the polygon is lit, not blocked"
        );
        // A light inside still casts: the skip tests the receiver alone, so
        // the far side of the shape is shadowed from within.
        assert!(
            square.occludes([0.0, 0.0], [100.0, 0.0]),
            "a lamp inside an occluder still shades what is behind it"
        );
    }

    #[test]
    fn the_same_polygon_wound_backwards_is_the_same_polygon() {
        // Image space flips y on the way to a framebuffer, which reverses a
        // winding. Normals derived from the wrong winding point inward and
        // turn the shape into its own complement, so the winding is inferred,
        // not assumed — and both orders must answer alike.
        let rock = [[0.0, -30.0], [25.0, -10.0], [15.0, 25.0], [-20.0, 20.0]];
        let forward = Convex::new(&rock).expect("this rock is convex");
        let reversed: Vec<[f32; 2]> = rock.iter().rev().copied().collect();
        let backward = Convex::new(&reversed).expect("wound the other way, still convex");
        assert_eq!(forward, backward, "the winding should not reach the answer");
        for probe in [[0.0, 0.0], [60.0, 0.0], [-60.0, 40.0], [10.0, 100.0]] {
            assert_eq!(
                forward.contains(probe),
                backward.contains(probe),
                "{probe:?} got two answers"
            );
        }
    }

    #[test]
    fn a_shape_that_cannot_be_a_convex_occluder_is_refused() {
        // Concave, degenerate, tiny, huge or numb: each of these would be a
        // shadow the renderer invented rather than the one that was drawn.
        let l_shape = [
            [0.0, 0.0],
            [40.0, 0.0],
            [40.0, 20.0],
            [20.0, 20.0],
            [20.0, 40.0],
            [0.0, 40.0],
        ];
        assert!(Convex::new(&l_shape).is_none(), "an L is not convex");
        assert!(
            Convex::new(&[[0.0, 0.0], [10.0, 0.0], [20.0, 0.0]]).is_none(),
            "three points on a line enclose nothing"
        );
        assert!(
            Convex::new(&[[0.0, 0.0], [10.0, 0.0], [10.0, 0.0], [0.0, 10.0]]).is_none(),
            "a repeated corner leaves an edge without a direction"
        );
        assert!(
            Convex::new(&[[0.0, 0.0], [10.0, 0.0]]).is_none(),
            "two corners make no shape"
        );
        let too_many: Vec<[f32; 2]> = (0..MAX_PLANES + 1)
            .map(|k| {
                let a = std::f32::consts::TAU * k as f32 / (MAX_PLANES + 1) as f32;
                [a.cos(), a.sin()]
            })
            .collect();
        assert!(
            Convex::new(&too_many).is_none(),
            "past the cap the GPU cost is the author's to choose"
        );
        assert!(
            Convex::new(&[[0.0, 0.0], [f32::NAN, 5.0], [0.0, 10.0]]).is_none(),
            "a coordinate that is not a number is not a shape"
        );
    }

    #[test]
    fn a_tessellated_disc_stays_within_two_percent_of_the_round_one() {
        // The tolerance every GPU shadow over a circle is measured against:
        // the sixteen-gon's corners are on the circle and its edges cut
        // inside, so the two shapes differ only in the thin band between the
        // inradius and the radius — never outside the circle, never inside
        // the inradius.
        let radius = 40.0;
        let polygon = Convex::disc([0.0, 0.0], radius);
        let circle = Collider::Circle(Circle {
            center: [0.0, 0.0],
            radius,
        });
        let inradius = radius * (std::f32::consts::PI / DISC_SIDES as f32).cos();
        assert!(
            inradius > radius * 0.98,
            "a {DISC_SIDES}-gon should sit within two percent of its circle"
        );
        for k in 0..720 {
            let a = std::f32::consts::TAU * k as f32 / 720.0;
            let direction = [a.cos(), a.sin()];
            let at = |d: f32| [direction[0] * d, direction[1] * d];
            assert!(
                polygon.contains(at(inradius - 0.5)) && circle.contains(at(inradius - 0.5)),
                "every direction agrees deep inside, at {a}"
            );
            assert!(
                !polygon.contains(at(radius + 0.5)) && !circle.contains(at(radius + 0.5)),
                "neither shape reaches beyond its radius, at {a}"
            );
            // Where they disagree at all, it is inside the band and nowhere
            // wider: the polygon is contained in the circle by construction.
            for probe in [radius - 1.0, radius, radius * 0.99] {
                if polygon.contains(at(probe)) != circle.contains(at(probe)) {
                    assert!(
                        probe > inradius,
                        "a disagreement at {probe} is outside the tessellation band"
                    );
                }
            }
        }
    }
}
