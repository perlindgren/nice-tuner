//! Authored geometry, baked into what the engine can actually use.
//!
//! A sprite's body, a tile's solid top, a character's hurt box are authored
//! facts about artwork: they live beside the sheet they describe, in the
//! sprite's or tile's own space, and they are written once and read by
//! thousands of instances. This module is where authored data turns into
//! engine shapes. [`Baker`] takes [`Feature`]s, each placed somewhere in a
//! node's space, and [`finish`](Baker::finish)es them into [`Baked`]: the flat
//! lists a frame and a movement step want.
//!
//! Everything stays in **the node's own space**, in and out. A baked set
//! travels with the node's [`Transform`](crate::Transform), which is what lets
//! one bake serve every instance, and it is why a stretched, turned sprite
//! still throws a shadow with square corners: nothing is ever baked into world
//! coordinates, so nothing has to approximate a sheared box.
//!
//! Three roles, three policies, because three consumers want different things
//! from the same authored shape:
//!
//! * [`Role::Occlude`] feeds the occluder field. The GPU reads one record per
//!   shape and tests every lit pixel against every record, so a floor of four
//!   thousand solid tiles is four thousand shadow tests per pixel per light.
//!   Boxes that span the same distance across and touch or overlap along the
//!   other axis are therefore fused into strips, and strips into blocks: the
//!   same covered points, far fewer records. Fusing also retires the shared
//!   edge, which is the one seam a light ray can slip along.
//! * [`Role::Solid`] feeds movement, where fusing is worth having for the same
//!   reason — but only within one [`Material`], because a crate and a
//!   trampoline that happen to touch must not become one wall that bounces.
//! * [`Role::Hit`] feeds the game's own logic: which hurt box was struck,
//!   which tile was clicked. Fusing them would answer the question with the
//!   wrong box, so hits come through exactly as authored.
//!
//! Only rectangles fuse: a fusion is taken when the result is still one
//! rectangle covering exactly the points the parts covered, which an L or a
//! circle never is. Nothing is dropped quietly either. A shape the bake cannot
//! honour — a box with no width, or a corner that is not a number — comes back
//! from [`Baker::add`] as a [`Rejection`] the loader can name, instead of
//! vanishing into a list that looks fine. A painted sprite that silently cannot
//! cast a shadow is the failure this path exists to end.
//!
//! An occluding *circle* is no longer among the refusals, and it is not
//! flattened on the way in either: the bake keeps it round and the
//! tessellation happens at the border, in [`Baked::declare_occluders`], where
//! the circle becomes the sixteen-gon the light field speaks. The authored
//! description outlives the bake, so the CPU is asked about the circle that was
//! drawn while the GPU is given the edges it can hold — the same arrangement
//! that lets a fused box keep its own turn until the transform carries it.
//!
//! An occluding *convex polygon* is honoured as well, and it is the one
//! shape that lives in the bake rather than being fused or cut: it arrives
//! as [`Footprint::poly`], joins [`Baked`]'s polygon list whole, and is
//! declared through [`Canvas::occluder_polygon`] exactly as authored — the
//! outline is already what the light field speaks. Asked to block movement
//! or to be struck, the same polygon is a refusal:
//! [`collision`](crate::collision) has no polygon body, and deriving one
//! is a step of its own.
//!
//! Not here yet: capsules, and the convex polygon for the roles that push
//! bodies apart. The reader that builds these features from the `.ron`
//! sidecar that already rides each sprite has arrived as
//! [`crate::shapes`] (`docs/authored-shapes.md`). A tile-grid broadphase
//! for line queries over very large maps is still missing — fused strips
//! keep a linear scan cheap enough until a real map says otherwise.

use crate::canvas::Canvas;
use crate::collision::{Circle, Collider, Convex, OrientedBox};
use crate::objects::Transform;

/// How far apart two authored edges may be and still count as the same edge,
/// in local pixels. Loose enough to forgive a sidecar that wrote `15.999998`
/// where its neighbour wrote `16.0`, tight enough that a deliberate gap stays
/// a gap.
const MERGE_EPS: f32 = 1e-3;

/// How close to a quarter turn an angle may be and be called that quarter
/// turn. Fusing boxes whose turns differ by a hair would move their edges by
/// more than [`MERGE_EPS`] once flattened, so near-quarter angles are snapped
/// to it — which is also what lets a tile placed through a quarter turn fuse
/// with its upright neighbours.
const ANGLE_SNAP: f32 = 1e-6;

/// What an authored shape is for.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    /// Blocks light. Baked into the rectangles a frame declares, so it
    /// decides where a light's shadow falls.
    Occlude,
    /// Blocks movement. Baked into [`collision`](crate::collision) shapes a
    /// movement step pushes a body out of, carrying its [`Material`].
    Solid,
    /// Marks a target. Baked into shapes the game tests to ask *what did I
    /// hit*, and never fused, because the answer has to name the box that was
    /// authored.
    Hit,
}

/// Why a convex polygon is refused wherever two shapes must be pushed
/// apart, as the stable text a [`Rejection`] carries and a loader prints.
///
/// Occlusion only ever asks whether a segment crossed an edge, which the
/// half-planes answer directly; a body needs the separating axis of a
/// curved boundary and a contact normal worth the name, which is a step of
/// its own rather than a free rider on this one. The reader quotes the
/// same sentence when a sidecar asks a polygon to be solid or hit, so the
/// author hears one reason from whichever end they come to.
pub(crate) const POLY_NOT_PUSHED: &str = "a convex polygon occludes; pushing two polygons apart needs the separating axis of a curved boundary and a contact normal worth the name, which is a step of its own";

/// How a shape that blocks movement behaves when something meets it.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Material {
    /// The restitution of a bounce off this shape: the `e` of
    /// [`reflect`](crate::reflect). `0.0` is dead, `1.0` is perfectly elastic.
    /// Shapes of different restitution are never fused, whatever their
    /// geometry.
    pub bounce: f32,
}

impl Material {
    /// Impassable and dead: nothing bounces off it.
    pub const DEAD: Material = Material { bounce: 0.0 };
}

/// One authored shape, in the space of the thing it was authored for: a
/// sprite's own frame, or one atlas cell's.
#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(clippy::large_enum_variant)] // the polygon rides by value, like `Convex` everywhere else in the crate; a bake runs once per sprite load, not per frame, and nothing here wants a `Footprint` to stop being `Copy`
pub enum Footprint {
    /// A rectangle, possibly turned about its own center.
    Box(OrientedBox),
    /// A circle.
    Circle(Circle),
    /// A convex polygon, held as the half-planes of its own edges.
    ///
    /// It occludes, and nothing else: the roles that push bodies apart ask
    /// for a separating axis and a contact normal, which a polygon cannot
    /// yet give (`POLY_NOT_PUSHED`).
    Poly(Convex),
}

impl Footprint {
    /// An axis-aligned rectangle at `center` with half-extents `half`, the
    /// same convention as [`Canvas::rectangle`]: the full size is `2 * half`.
    pub fn rect(center: [f32; 2], half: [f32; 2]) -> Self {
        Self::Box(OrientedBox::new(center, half))
    }

    /// A rectangle turned counter-clockwise by `angle` radians about its
    /// center.
    pub fn tilted(center: [f32; 2], half: [f32; 2], angle: f32) -> Self {
        Self::Box(OrientedBox::rotated(center, half, angle))
    }

    /// A circle of `radius` at `center`.
    pub fn disc(center: [f32; 2], radius: f32) -> Self {
        Self::Circle(Circle { center, radius })
    }

    /// The convex polygon `shape`, as [`Convex::new`] accepted it — which
    /// is the only way one exists. A shape that was concave or crossed
    /// never becomes a footprint; the refusal is the author's to read.
    pub fn poly(shape: Convex) -> Self {
        Self::Poly(shape)
    }

    /// The shape as a [`Collider`], for the roles that answer questions,
    /// or `None` for the one shape that has no collider: the convex
    /// polygon occludes, but `Collider` gained no polygon variant, and a
    /// role asked to push a polygon apart takes the refusal rather than a
    /// hull nobody drew.
    pub fn collider(self) -> Option<Collider> {
        match self {
            Self::Box(b) => Some(Collider::Box(b)),
            Self::Circle(c) => Some(Collider::Circle(c)),
            Self::Poly(_) => None,
        }
    }

    /// The same shape in another space: turned counter-clockwise by
    /// `place.angle` about its own origin, then shifted to `place.at` — the
    /// composition a node's transform gives its children.
    pub fn placed(self, place: Placement) -> Self {
        let (sin, cos) = place.angle.sin_cos();
        let turn = |p: [f32; 2]| {
            [
                p[0] * cos - p[1] * sin + place.at[0],
                p[0] * sin + p[1] * cos + place.at[1],
            ]
        };
        match self {
            Self::Box(b) => Self::Box(OrientedBox::rotated(
                turn(b.center),
                b.half,
                b.angle + place.angle,
            )),
            Self::Circle(c) => Self::Circle(Circle {
                center: turn(c.center),
                radius: c.radius,
            }),
            Self::Poly(p) => {
                if place == Placement::IDENTITY {
                    return Self::Poly(p);
                }
                // A `Convex` keeps its edges, not its corners, so the
                // corners come back out of them as each pair of
                // neighbouring half-planes' crossing, travel through the
                // placement like any other point, and the outline is
                // rebuilt from the moved corners. The move is rigid and
                // convexity survives it, so `Convex::new` — which was
                // asked to accept this shape once already — has nothing
                // new to refuse; were it ever to disagree, the authored
                // polygon stands rather than any stand-in, because a
                // shape in the wrong place is still the shape somebody
                // drew and a bounding box would be a quiet hull.
                let planes = p.planes();
                let mut moved: Vec<[f32; 2]> = Vec::with_capacity(planes.len());
                for i in 0..planes.len() {
                    let (a, b) = (planes[i], planes[(i + 1) % planes.len()]);
                    let det = a[0] * b[1] - a[1] * b[0];
                    moved.push(turn([
                        (a[2] * b[1] - a[1] * b[2]) / det,
                        (a[0] * b[2] - a[2] * b[0]) / det,
                    ]));
                }
                Self::Poly(Convex::new(&moved).unwrap_or(p))
            }
        }
    }

    /// Whether every number the shape is made of is a real number.
    fn finite(self) -> bool {
        match self {
            Self::Box(b) => {
                b.center[0].is_finite()
                    && b.center[1].is_finite()
                    && b.half[0].is_finite()
                    && b.half[1].is_finite()
                    && b.angle.is_finite()
            }
            Self::Circle(c) => {
                c.center[0].is_finite() && c.center[1].is_finite() && c.radius.is_finite()
            }
            Self::Poly(p) => {
                let (lo, hi) = p.bounds();
                lo[0].is_finite() && lo[1].is_finite() && hi[0].is_finite() && hi[1].is_finite()
            }
        }
    }

    /// Whether the shape covers no points at all: a box with a side of zero or
    /// less, a circle without a radius, or a polygon that was never allowed
    /// to enclose nothing in the first place.
    fn empty(self) -> bool {
        match self {
            Self::Box(b) => b.half[0] <= 0.0 || b.half[1] <= 0.0,
            Self::Circle(c) => c.radius <= 0.0,
            // `Convex::new` refuses an outline with no area, so a polygon
            // that exists at all has something to occlude with.
            Self::Poly(_) => false,
        }
    }
}

/// One authored fact about a sprite frame or an atlas cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Feature {
    /// What the shape is for.
    pub role: Role,
    /// The shape, in the authoring space.
    pub shape: Footprint,
    /// How it behaves, for the roles that care.
    pub material: Material,
}

impl Feature {
    /// A shape that blocks light.
    pub fn occluding(shape: Footprint) -> Self {
        Self {
            role: Role::Occlude,
            shape,
            material: Material::DEAD,
        }
    }

    /// A shape that blocks movement, with the restitution to bounce off it
    /// with.
    pub fn solid(shape: Footprint, bounce: f32) -> Self {
        Self {
            role: Role::Solid,
            shape,
            material: Material { bounce },
        }
    }

    /// A shape that only ever gets asked what was struck.
    pub fn hit(shape: Footprint) -> Self {
        Self {
            role: Role::Hit,
            shape,
            material: Material::DEAD,
        }
    }
}

/// Where one authored set sits inside the node's space.
///
/// A sprite frame is placed at the origin and unturned — its features are
/// already in the sprite's space — while one atlas cell of a tile map is placed
/// at its tile's center, turned by however that tile is oriented.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Placement {
    /// Where the authored set's origin lands.
    pub at: [f32; 2],
    /// How far it is turned counter-clockwise on the way, in radians.
    pub angle: f32,
}

impl Placement {
    /// Left where it was authored: the sprite's own frame.
    pub const IDENTITY: Placement = Placement {
        at: [0.0, 0.0],
        angle: 0.0,
    };

    /// Moved to `at`, unturned.
    pub const fn at(at: [f32; 2]) -> Self {
        Placement { at, angle: 0.0 }
    }

    /// Moved to `at` and turned by `angle` radians counter-clockwise.
    pub const fn turned(at: [f32; 2], angle: f32) -> Self {
        Placement { at, angle }
    }
}

/// Authored geometry the bake could not honour, and why it was left out.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Rejection {
    /// Where in the slice handed to [`Baker::add`] the shape came from.
    pub index: usize,
    /// What it was authored to do.
    pub role: Role,
    /// Why it was left out, as stable text a loader can print and a test can
    /// name.
    pub why: &'static str,
}

/// A solid that survived the bake: a shape, and the restitution of a bounce off
/// it, which [`crate::reflect`] wants at the moment of the bounce.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Solid {
    /// The shape to test against.
    pub shape: Collider,
    /// The restitution to reflect with.
    pub bounce: f32,
}

/// What a bake produced: flat lists, all in the node's own space, each
/// ready for the consumer it belongs to.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Baked {
    /// The fused rectangles a frame declares as shadows.
    pub occluders: Vec<OrientedBox>,
    /// The circles authored to cast, kept circles: a fusion of two circles is
    /// no circle at all, so these are counted rather than merged, and they are
    /// cut into the polygon the light field speaks only on the way out, in
    /// [`Baked::declare_occluders`] — which is also why the CPU's shadow
    /// question goes to the round shape and not to its approximation.
    pub disc_occluders: Vec<Circle>,
    /// The convex polygons authored to cast, each kept as the outline it was
    /// drawn with: a fusion of two polygons is a shape nobody drew, and no
    /// record here needs a tessellation — the outline is already the edges
    /// the light field speaks. These are declared through
    /// [`Canvas::occluder_polygon`], in their own space like every other
    /// record.
    pub poly_occluders: Vec<Convex>,
    /// The shapes that block movement, fused within each material.
    pub solids: Vec<Solid>,
    /// The shapes that only answer *what was struck*, exactly as authored.
    pub hits: Vec<Collider>,
}

impl Baked {
    /// Declare every baked occluder to `canvas` under `world`, the node's
    /// transform into user space: the one call that turns baked geometry into
    /// a shadow.
    ///
    /// Each record keeps its shape local — a box's own turn rides along in the
    /// transform it is declared with — so a fused strip of tiles authored at
    /// any angle enters the light field exactly as a hand-written
    /// [`Canvas::occluder`] call would.
    ///
    /// A baked circle is tessellated here and nowhere earlier:
    /// [`Convex::disc`](crate::Convex::disc) cuts it into the polygon the field
    /// carries, the one place a round shape has to become flat ones. The bake
    /// had every right to keep it a circle, so the approximation is made once,
    /// at the last moment, by the only function that needs it. An authored
    /// polygon arrives at that moment already flat — it goes to
    /// [`Canvas::occluder_polygon`] as the outline it was drawn with.
    pub fn declare_occluders(&self, canvas: &mut Canvas, world: Transform) {
        for b in &self.occluders {
            let to_node = Transform::rotate(b.angle).compose(&Transform::translate(b.center));
            canvas.occluder(to_node.compose(&world), [0.0, 0.0], b.half);
        }
        for c in &self.disc_occluders {
            canvas.occluder_polygon(world, &Convex::disc(c.center, c.radius));
        }
        // The authored polygons need no last-moment tessellation: the
        // outline the author drew is already the edges the field carries.
        for p in &self.poly_occluders {
            canvas.occluder_polygon(world, p);
        }
    }

    /// The baked occluders as [`Collider`]s, for asking the same question on
    /// the CPU: [`occluded`](crate::occluded) and [`visible`](crate::visible)
    /// over this list answer it the way the shader will, in the node's space.
    ///
    /// Round occluders come back round, deliberately, even though the shader
    /// will be handed their sixteen-gon. The two differ by under two percent of
    /// a radius, which is how far a shadow's edge is allowed to move; asking
    /// the CPU about the polygon too would fold the answer into the question,
    /// and a shadow that disagreed would no longer say which side was wrong.
    ///
    /// The authored polygons are not in this list: [`Collider`] has no
    /// polygon variant, and the honest CPU twin of their shadow question
    /// would have to cross an edge exactly as the field does, which is the
    /// same missing step as the polygon collider that would push bodies.
    /// Declaring them is unaffected — the field itself carries their edges.
    pub fn occluder_colliders(&self) -> Vec<Collider> {
        self.occluders
            .iter()
            .copied()
            .map(Collider::Box)
            .chain(self.disc_occluders.iter().copied().map(Collider::Circle))
            .collect()
    }

    /// Whether the bake left nothing behind.
    pub fn is_empty(&self) -> bool {
        self.occluders.is_empty()
            && self.disc_occluders.is_empty()
            && self.poly_occluders.is_empty()
            && self.solids.is_empty()
            && self.hits.is_empty()
    }
}

/// Where one accepted shape is routed.
#[allow(clippy::large_enum_variant)] // the same by-value polygon as `Footprint` above; a route is a hand-off inside one `add` call, not a thing that travels
enum Route {
    Occluder(OrientedBox),
    /// An occluding circle, bound for the round list rather than the fused
    /// one: nothing about a circle survives being merged with anything else.
    Disc(Circle),
    /// An occluding convex polygon, bound for the list it is declared from:
    /// it fuses with nothing, and it is already the edges the field speaks.
    Poly(Convex),
    Shape(Collider),
}

/// Judge one placed shape: where it goes, or why it cannot go anywhere.
fn classify(shape: Footprint, role: Role) -> Result<Route, &'static str> {
    if !shape.finite() {
        return Err("its coordinates are not finite numbers");
    }
    if shape.empty() {
        return Err("it has no area, so it occludes and blocks nothing");
    }
    match (role, shape) {
        (Role::Occlude, Footprint::Box(b)) => Ok(Route::Occluder(b)),
        (Role::Occlude, Footprint::Circle(c)) => Ok(Route::Disc(c)),
        (Role::Occlude, Footprint::Poly(p)) => Ok(Route::Poly(p)),
        (Role::Solid | Role::Hit, s) => match s.collider() {
            Some(c) => Ok(Route::Shape(c)),
            // The one footprint without a collider: a convex polygon asked
            // to push a body, or to be named as what was struck.
            None => Err(POLY_NOT_PUSHED),
        },
    }
}

/// Collects placed features and bakes them.
///
/// Collecting before fusing is what lets a tile map be baked at all: a cell's
/// features are authored once, placed at every tile that uses the cell, and
/// only then fused — two tiles of the same wall fuse because they end up
/// adjacent in the map's space, which nothing knows while the cell is still
/// one cell.
///
/// ```
/// use frost::{Baker, Feature, Footprint, Placement};
///
/// // One atlas cell: a floor tile solid across its whole 16 by 16 face.
/// let cell = [Feature::solid(Footprint::rect([0.0, 0.0], [8.0, 8.0]), 0.0)];
/// let mut baker = Baker::new();
/// for col in 0..4 {
///     // A rejection is authored geometry that will do nothing, so a loader
///     // prints what comes back rather than dropping it on the floor.
///     let placed = baker.add(Placement::at([col as f32 * 16.0, 0.0]), &cell);
///     assert!(placed.is_empty());
/// }
/// let baked = baker.finish();
/// // Four tiles of one material, edge to edge: one solid, not four.
/// assert_eq!(baked.solids.len(), 1);
/// ```
#[derive(Clone, Debug, Default)]
pub struct Baker {
    occluders: Vec<OrientedBox>,
    disc_occluders: Vec<Circle>,
    poly_occluders: Vec<Convex>,
    solids: Vec<(u32, Collider)>,
    hits: Vec<Collider>,
}

impl Baker {
    /// An empty bake.
    pub fn new() -> Self {
        Self::default()
    }

    /// Place one authored set — a sprite frame's features, or one atlas cell's
    /// — into the node's space, and keep what the bake can use.
    ///
    /// Returns what was left out. The result is `#[must_use]`: a rejection is
    /// authored geometry that will quietly do nothing, which is the one thing
    /// an author cannot see by playing the game.
    #[must_use = "a rejected feature is authored geometry that does nothing; name it"]
    pub fn add(&mut self, place: Placement, features: &[Feature]) -> Vec<Rejection> {
        let mut rejected = Vec::new();
        for (index, feature) in features.iter().copied().enumerate() {
            let shape = feature.shape.placed(place);
            match classify(shape, feature.role) {
                Ok(Route::Occluder(b)) => self.occluders.push(canonical(&b)),
                Ok(Route::Disc(c)) => self.disc_occluders.push(c),
                Ok(Route::Poly(p)) => self.poly_occluders.push(p),
                Ok(Route::Shape(s)) => match feature.role {
                    // Solids are grouped by turn before they fuse, so their
                    // turn is folded into a quarter turn first. A hit is never
                    // grouped with anything, so it keeps the exact description
                    // the author wrote — which is the whole reason it is kept
                    // apart from the others.
                    Role::Solid => self.solids.push((
                        feature.material.bounce.to_bits(),
                        match s {
                            Collider::Box(b) => Collider::Box(canonical(&b)),
                            circle => circle,
                        },
                    )),
                    _ => self.hits.push(s),
                },
                Err(why) => rejected.push(Rejection {
                    index,
                    role: feature.role,
                    why,
                }),
            }
        }
        rejected
    }

    /// Fuse what can be fused and hand over the lists.
    pub fn finish(self) -> Baked {
        use std::collections::BTreeMap;

        let mut groups: BTreeMap<u32, Vec<OrientedBox>> = BTreeMap::new();
        for b in self.occluders {
            groups.entry(b.angle.to_bits()).or_default().push(b);
        }
        let mut occluders = Vec::new();
        for boxes in groups.values() {
            occluders.extend(fuse(boxes));
        }

        let mut groups: BTreeMap<(u32, u32), Vec<OrientedBox>> = BTreeMap::new();
        let mut circles: Vec<Solid> = Vec::new();
        for (bounce, shape) in self.solids {
            match shape {
                Collider::Box(b) => groups
                    .entry((b.angle.to_bits(), bounce))
                    .or_default()
                    .push(b),
                Collider::Circle(c) => circles.push(Solid {
                    shape: Collider::Circle(c),
                    bounce: f32::from_bits(bounce),
                }),
            }
        }
        let mut solids: Vec<Solid> = Vec::new();
        for ((_, bounce), boxes) in &groups {
            let bounce = f32::from_bits(*bounce);
            solids.extend(fuse(boxes).into_iter().map(|b| Solid {
                shape: Collider::Box(b),
                bounce,
            }));
        }
        solids.extend(circles);

        Baked {
            occluders,
            // Authored order, unfused: two circles side by side are two
            // circles, and the only way to make one of them another shape would
            // be to invent a shape nobody drew.
            disc_occluders: self.disc_occluders,
            // The same reason, in edges: an outline is declared as drawn.
            poly_occluders: self.poly_occluders,
            solids,
            hits: self.hits,
        }
    }
}

/// The same box, with its turn folded into `0 .. π/2`.
///
/// A box turned by a quarter turn with its extents swapped covers exactly the
/// points the unswapped box does, so this rewrites the description without
/// moving a single point. Two boxes then share a fusion group when they cover
/// the same kind of ground: an upright tile and one laid down sideways come out
/// of here alike, and a box's angle is either a quarter turn or genuinely its
/// own, never the float residue of a composition that meant a quarter turn.
///
/// Away from a quarter turn nothing is folded, so two boxes share a group only
/// when their turns are the same number. A turn reached by two different
/// calculations therefore costs one extra record — and only that, since a
/// group's frame is one of its own members, which is what lets the flattened
/// fusion be exact rather than nearly so. Buckets wider than a rounding error
/// would buy the records back and charge the difference to the edges.
fn canonical(b: &OrientedBox) -> OrientedBox {
    let mut out = *b;
    while out.angle >= std::f32::consts::FRAC_PI_2 {
        out.angle -= std::f32::consts::FRAC_PI_2;
        out.half = [out.half[1], out.half[0]];
    }
    while out.angle < 0.0 {
        out.angle += std::f32::consts::FRAC_PI_2;
        out.half = [out.half[1], out.half[0]];
    }
    if out.angle < ANGLE_SNAP {
        out.angle = 0.0;
    } else if std::f32::consts::FRAC_PI_2 - out.angle < ANGLE_SNAP {
        // A hair under a quarter turn is a quarter turn, which is no turn at
        // all on a box whose extents are swapped.
        out.angle = 0.0;
        out.half = [out.half[1], out.half[0]];
    }
    out
}

/// Fuse a set of boxes that share a turn into as few boxes as still cover
/// exactly the same points.
///
/// A fusion is taken only when the result is still one rectangle: two boxes may
/// merge when they agree across one axis and touch or overlap along the other.
/// An L therefore stays two boxes, and a row of tiles becomes one strip, which
/// then joins the strip under it into a block. The loop runs until a pass
/// changes nothing; a pass can only shorten the list, so it ends.
fn fuse(boxes: &[OrientedBox]) -> Vec<OrientedBox> {
    if boxes.len() < 2 {
        return boxes.to_vec();
    }
    // Every box here carries the same turn, so the whole set flattens into
    // that frame, fuses axis-aligned, and turns back.
    let frame = boxes[0].angle;
    let (sin, cos) = frame.sin_cos();
    let flat = |p: [f32; 2]| [p[0] * cos + p[1] * sin, -p[0] * sin + p[1] * cos];
    let up = |p: [f32; 2]| [p[0] * cos - p[1] * sin, p[0] * sin + p[1] * cos];

    let mut rects: Vec<Rect> = boxes
        .iter()
        .map(|b| {
            let c = flat(b.center);
            Rect {
                lo: [c[0] - b.half[0], c[1] - b.half[1]],
                hi: [c[0] + b.half[0], c[1] + b.half[1]],
            }
        })
        .collect();

    loop {
        let before = rects.len();
        rects = flatten_axis(&rects, 0);
        rects = flatten_axis(&rects, 1);
        if rects.len() == before {
            break;
        }
    }

    rects
        .into_iter()
        .map(|r| {
            let c = up([(r.lo[0] + r.hi[0]) / 2.0, (r.lo[1] + r.hi[1]) / 2.0]);
            OrientedBox::rotated(
                c,
                [(r.hi[0] - r.lo[0]) / 2.0, (r.hi[1] - r.lo[1]) / 2.0],
                frame,
            )
        })
        .collect()
}

/// A rectangle inside one fusion group, in the group's own frame.
#[derive(Clone, Copy, Debug)]
struct Rect {
    lo: [f32; 2],
    hi: [f32; 2],
}

impl Rect {
    /// The rectangle grown to cover `other` as well.
    fn joined(&self, other: &Rect) -> Rect {
        Rect {
            lo: [self.lo[0].min(other.lo[0]), self.lo[1].min(other.lo[1])],
            hi: [self.hi[0].max(other.hi[0]), self.hi[1].max(other.hi[1])],
        }
    }
}

/// Merge every rectangle whose span across `across` is the same and which touch
/// or overlap along `axis`, into rectangles that run the whole way.
fn flatten_axis(rects: &[Rect], axis: usize) -> Vec<Rect> {
    let across = 1 - axis;
    let mut order: Vec<usize> = (0..rects.len()).collect();
    order.sort_by(|&a, &b| {
        rects[a].lo[across]
            .total_cmp(&rects[b].lo[across])
            .then(rects[a].hi[across].total_cmp(&rects[b].hi[across]))
            .then(rects[a].lo[axis].total_cmp(&rects[b].lo[axis]))
    });

    let mut out: Vec<Rect> = Vec::with_capacity(rects.len());
    let mut at = 0;
    while at < order.len() {
        // Everything spanning the same distance across, within the merge
        // tolerance, makes one row to run along the axis.
        let span = rects[order[at]];
        let mut row = vec![order[at]];
        let mut next = at + 1;
        while next < order.len()
            && (rects[order[next]].lo[across] - span.lo[across]).abs() <= MERGE_EPS
            && (rects[order[next]].hi[across] - span.hi[across]).abs() <= MERGE_EPS
        {
            row.push(order[next]);
            next += 1;
        }
        row.sort_by(|&a, &b| rects[a].lo[axis].total_cmp(&rects[b].lo[axis]));

        let mut run = rects[row[0]];
        for &i in &row[1..] {
            let r = rects[i];
            if r.lo[axis] <= run.hi[axis] + MERGE_EPS {
                // Touching, overlapping, or swallowed whole: either way the
                // run grows to cover it, and covering it is the union, since
                // the two already share their span across.
                run = run.joined(&r);
            } else {
                out.push(run);
                run = r;
            }
        }
        out.push(run);
        at = next;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Place a set and insist the bake took the whole of it: a test that
    /// quietly baked nothing would be asserting nothing.
    fn accept(baker: &mut Baker, place: Placement, features: &[Feature]) {
        let rejected = baker.add(place, features);
        assert!(rejected.is_empty(), "the bake refused {rejected:?}");
    }

    /// One floor tile: opaque across its whole 16 × 16 face.
    fn tile() -> Feature {
        Feature::occluding(Footprint::rect([0.0, 0.0], [8.0, 8.0]))
    }

    #[test]
    fn a_rectangle_reaches_every_role_it_was_authored_for() {
        let shape = Footprint::rect([4.0, -2.0], [10.0, 5.0]);
        let mut baker = Baker::new();
        let rejected = baker.add(
            Placement::IDENTITY,
            &[
                Feature::occluding(shape),
                Feature::solid(shape, 0.6),
                Feature::hit(shape),
            ],
        );
        assert!(rejected.is_empty());
        let baked = baker.finish();
        assert_eq!(baked.occluders.len(), 1);
        assert_eq!(baked.solids.len(), 1);
        assert_eq!(baked.hits.len(), 1);
        assert_eq!(baked.occluders[0].center, [4.0, -2.0]);
        assert_eq!(baked.occluders[0].half, [10.0, 5.0]);
        assert_eq!(baked.solids[0].bounce, 0.6);
        assert!(baked.hits[0].contains([4.0, -2.0]));
    }

    #[test]
    fn an_occluding_circle_stays_round_and_reaches_the_field_as_a_polygon() {
        // The round case, end to end: honoured rather than refused, kept round
        // by the bake, put to the CPU's shadow question as a circle, and cut
        // into edges only at the last step — the declared record.
        let mut baker = Baker::new();
        let rejected = baker.add(
            Placement::IDENTITY,
            &[
                Feature::occluding(Footprint::disc([4.0, -3.0], 12.0)),
                Feature::solid(Footprint::disc([4.0, -3.0], 12.0), 0.0),
            ],
        );
        assert!(
            rejected.is_empty(),
            "an occluding circle is honoured now, got {rejected:?}"
        );
        let baked = baker.finish();
        assert!(
            baked.occluders.is_empty(),
            "nothing about a circle turns into a box"
        );
        assert_eq!(baked.disc_occluders.len(), 1);
        assert_eq!(baked.disc_occluders[0].center, [4.0, -3.0]);
        assert_eq!(baked.disc_occluders[0].radius, 12.0);
        assert_eq!(baked.solids.len(), 1);
        assert!(matches!(baked.solids[0].shape, Collider::Circle(_)));
        assert_eq!(baked.occluder_colliders().len(), 1);
        assert!(
            matches!(baked.occluder_colliders()[0], Collider::Circle(_)),
            "the CPU is asked about the circle the author drew, not its \
             sixteen-gon"
        );
        assert!(!baked.is_empty());

        // And the border: one declared occluder, carrying the authored center,
        // the edges it was cut into, and the node's transform.
        let mut canvas = Canvas::new((64, 64), 1.0);
        baked.declare_occluders(&mut canvas, Transform::translate([10.0, 20.0]));
        assert_eq!(canvas.occluders.len(), 1);
        let declared = canvas.occluders[0]
            .polygon
            .expect("a disc declares itself as a polygon");
        assert_eq!(
            declared.planes().len(),
            crate::collision::DISC_SIDES,
            "the declared disc should carry every edge the tessellation has"
        );
        let bounds = declared.bounds_as_box();
        assert_eq!(
            bounds.center,
            [4.0, -3.0],
            "the polygon is the authored disc, not a new shape centred elsewhere"
        );
        assert!(
            (bounds.half[0] - 12.0).abs() < 1e-4 && (bounds.half[1] - 12.0).abs() < 1e-4,
            "the bounds should wrap the disc, got {:?}",
            bounds.half
        );
    }

    #[test]
    fn two_occluding_circles_are_two_circles() {
        // Fusion is the bake's whole economy, and it stops at the circle: two
        // discs side by side do not become a strip, a capsule or one bigger
        // disc, because no shape covers their points and is still what was
        // authored. They are counted, and each is cut on its own.
        let mut baker = Baker::new();
        accept(
            &mut baker,
            Placement::IDENTITY,
            &[
                Feature::occluding(Footprint::disc([-6.0, 0.0], 6.0)),
                Feature::occluding(Footprint::disc([6.0, 0.0], 6.0)),
            ],
        );
        let baked = baker.finish();
        assert!(baked.occluders.is_empty());
        assert_eq!(
            baked.disc_occluders.len(),
            2,
            "touching discs stay two discs"
        );
        assert_eq!(baked.occluder_colliders().len(), 2);
    }

    #[test]
    fn a_convex_polygon_reaches_the_field_as_its_own_outline() {
        // The round case again, one step further: the shape is honoured,
        // unfused and untessellated, and the declared record carries the
        // authored half-planes themselves — the edges, not a box around
        // them and not a new polygon.
        let corners = [[-8.0, -4.0], [10.0, -2.0], [2.0, 9.0]];
        let rock = Convex::new(&corners).expect("this triangle is convex");
        let mut baker = Baker::new();
        accept(
            &mut baker,
            Placement::IDENTITY,
            &[Feature::occluding(Footprint::poly(rock))],
        );
        let baked = baker.finish();
        assert!(
            baked.occluders.is_empty() && baked.disc_occluders.is_empty(),
            "nothing about a polygon turns into a box or a circle"
        );
        assert_eq!(baked.poly_occluders, [rock]);
        assert!(!baked.is_empty());
        assert_eq!(
            baked.occluder_colliders().len(),
            0,
            "the polygon is not in the CPU's list either: Collider has no \
             polygon variant, and a bounding box would be a hull"
        );

        let mut canvas = Canvas::new((64, 64), 1.0);
        baked.declare_occluders(&mut canvas, Transform::translate([10.0, 20.0]));
        assert_eq!(canvas.occluders.len(), 1);
        let declared = canvas.occluders[0]
            .polygon
            .expect("a polygon declares itself with its edges");
        assert_eq!(
            declared.planes(),
            rock.planes(),
            "the field is handed the authored outline, verbatim"
        );
    }

    #[test]
    fn a_placed_polygon_travels_with_the_cell_it_was_placed_into() {
        // The one reconstruction `placed` does: a cell's polygon shifted
        // two tiles over arrives shifted — its corners are rebuilt from
        // the edges, moved and refitted, and must land where the cell did
        // rather than where the cell's picture sits by default.
        let corners = [[-8.0, -4.0], [10.0, -2.0], [2.0, 9.0]];
        let rock = Convex::new(&corners).expect("this triangle is convex");
        let want = rock.bounds_as_box().center;
        let mut baker = Baker::new();
        accept(
            &mut baker,
            Placement::at([32.0, 16.0]),
            &[Feature::occluding(Footprint::poly(rock))],
        );
        let moved = baker.finish().poly_occluders.swap_remove(0);
        assert_eq!(
            moved.planes().len(),
            rock.planes().len(),
            "the same edges travel, no more and none added"
        );
        let got = moved.bounds_as_box().center;
        assert!(
            (got[0] - (want[0] + 32.0)).abs() < 1e-3 && (got[1] - (want[1] + 16.0)).abs() < 1e-3,
            "the bounds moved to {got:?}, expected [{}, {}]",
            want[0] + 32.0,
            want[1] + 16.0
        );
    }

    #[test]
    fn a_convex_polygon_is_refused_where_two_shapes_must_be_pushed_apart() {
        // Occlusion asks whether a segment crossed an edge; movement and
        // hurt boxes ask for a separating axis and a contact normal, and
        // the bake says so instead of answering with a hull.
        let corners = [[-8.0, -4.0], [10.0, -2.0], [2.0, 9.0]];
        let rock = Convex::new(&corners).expect("this triangle is convex");
        let mut baker = Baker::new();
        let rejected = baker.add(
            Placement::IDENTITY,
            &[
                Feature::solid(Footprint::poly(rock), 0.0),
                Feature::hit(Footprint::poly(rock)),
            ],
        );
        assert_eq!(rejected.len(), 2);
        for why in [rejected[0].why, rejected[1].why] {
            assert!(
                why.contains("separating axis"),
                "the refusal must name what is missing, got {why:?}"
            );
        }
        assert!(baker.finish().is_empty());
    }

    #[test]
    fn a_shape_with_no_area_or_no_numbers_is_rejected() {
        let mut baker = Baker::new();
        let rejected = baker.add(
            Placement::IDENTITY,
            &[
                Feature::solid(Footprint::rect([0.0, 0.0], [8.0, 0.0]), 0.0),
                Feature::hit(Footprint::rect([0.0, 0.0], [8.0, -1.0])),
                Feature::solid(Footprint::disc([0.0, 0.0], 0.0), 0.0),
                Feature::hit(Footprint::disc([f32::NAN, 0.0], 4.0)),
            ],
        );
        assert_eq!(rejected.len(), 4);
        assert_eq!(rejected[2].index, 2);
        assert!(baker.finish().is_empty());
    }

    #[test]
    fn a_hit_box_is_never_fused_or_rewritten() {
        // Two hurt boxes side by side: the game has to be able to say which one
        // was struck, so the bake hands both back. And a third, authored
        // through a quarter turn, comes back turned — nothing rewrites the box
        // the author will be asked to name.
        let mut baker = Baker::new();
        accept(
            &mut baker,
            Placement::IDENTITY,
            &[
                Feature::hit(Footprint::rect([0.0, 0.0], [8.0, 8.0])),
                Feature::hit(Footprint::rect([16.0, 0.0], [8.0, 8.0])),
                Feature::hit(Footprint::tilted(
                    [32.0, 0.0],
                    [8.0, 12.0],
                    std::f32::consts::FRAC_PI_2,
                )),
            ],
        );
        let hits = baker.finish().hits;
        assert_eq!(hits.len(), 3);
        let Collider::Box(turned) = hits[2] else {
            panic!("a box was authored, a box comes back");
        };
        assert_eq!(turned.angle, std::f32::consts::FRAC_PI_2);
        assert_eq!(turned.half, [8.0, 12.0]);
    }

    #[test]
    fn solids_fuse_only_within_one_material() {
        let mut baker = Baker::new();
        accept(
            &mut baker,
            Placement::IDENTITY,
            &[
                Feature::solid(Footprint::rect([0.0, 0.0], [8.0, 8.0]), 0.2),
                Feature::solid(Footprint::rect([16.0, 0.0], [8.0, 8.0]), 0.2),
                Feature::solid(Footprint::rect([32.0, 0.0], [8.0, 8.0]), 0.9),
            ],
        );
        let baked = baker.finish();
        assert_eq!(baked.solids.len(), 2);
        let slab = baked.solids.iter().find(|s| s.bounce == 0.2).unwrap();
        let trampoline = baked.solids.iter().find(|s| s.bounce == 0.9).unwrap();
        // The two slabs of one material join into a box 32 wide: the union's
        // own half-extent, not the pair's.
        assert_eq!(
            slab.shape,
            Collider::Box(OrientedBox::new([8.0, 0.0], [16.0, 8.0]))
        );
        assert_eq!(
            trampoline.shape,
            Collider::Box(OrientedBox::new([32.0, 0.0], [8.0, 8.0]))
        );
    }

    #[test]
    fn occluders_fuse_whatever_their_material_says() {
        // A shadow does not care what casts it: a crate and a trampoline under
        // one lamp throw one shadow once their boxes join.
        let mut baker = Baker::new();
        accept(
            &mut baker,
            Placement::IDENTITY,
            &[
                Feature::solid(Footprint::rect([0.0, 0.0], [8.0, 8.0]), 0.2),
                Feature::solid(Footprint::rect([16.0, 0.0], [8.0, 8.0]), 0.9),
                Feature::occluding(Footprint::rect([0.0, 0.0], [8.0, 8.0])),
                Feature::occluding(Footprint::rect([16.0, 0.0], [8.0, 8.0])),
            ],
        );
        let baked = baker.finish();
        assert_eq!(baked.occluders.len(), 1);
        assert_eq!(baked.solids.len(), 2);
    }

    #[test]
    fn a_row_of_tiles_becomes_one_strip_and_a_block_becomes_one_box() {
        let mut baker = Baker::new();
        for col in 0..5 {
            accept(
                &mut baker,
                Placement::at([col as f32 * 16.0, 0.0]),
                &[tile()],
            );
        }
        let baked = baker.finish();
        assert_eq!(baked.occluders.len(), 1);
        assert_eq!(baked.occluders[0].center, [32.0, 0.0]);
        assert_eq!(baked.occluders[0].half, [40.0, 8.0]);

        let mut baker = Baker::new();
        for row in 0..4 {
            for col in 0..5 {
                accept(
                    &mut baker,
                    Placement::at([col as f32 * 16.0, row as f32 * 16.0]),
                    &[tile()],
                );
            }
        }
        let baked = baker.finish();
        assert_eq!(baked.occluders.len(), 1);
        assert_eq!(baked.occluders[0].center, [32.0, 24.0]);
        assert_eq!(baked.occluders[0].half, [40.0, 32.0]);
    }

    #[test]
    fn an_l_keeps_its_two_boxes() {
        // The union is not a rectangle, so no fusion is taken: a rectangle
        // fitted to the L would shadow points nothing stands in front of.
        let mut baker = Baker::new();
        accept(
            &mut baker,
            Placement::IDENTITY,
            &[
                Feature::occluding(Footprint::rect([0.0, 0.0], [8.0, 8.0])),
                Feature::occluding(Footprint::rect([16.0, 0.0], [8.0, 8.0])),
                Feature::occluding(Footprint::rect([0.0, 16.0], [8.0, 8.0])),
            ],
        );
        assert_eq!(baker.finish().occluders.len(), 2);
    }

    #[test]
    fn a_gap_stays_a_gap() {
        let mut baker = Baker::new();
        accept(
            &mut baker,
            Placement::IDENTITY,
            &[
                Feature::occluding(Footprint::rect([0.0, 0.0], [8.0, 8.0])),
                Feature::occluding(Footprint::rect([17.0, 0.0], [8.0, 8.0])),
            ],
        );
        assert_eq!(baker.finish().occluders.len(), 2);
    }

    #[test]
    fn a_tile_laid_sideways_fuses_with_its_upright_neighbours() {
        // Authored 16 wide and 32 tall, placed through a quarter turn, the
        // middle tile covers 32 wide and 16 tall — the same ground as its
        // upright twins, so the three fuse into one strip.
        let mut baker = Baker::new();
        accept(
            &mut baker,
            Placement::IDENTITY,
            &[Feature::occluding(Footprint::rect(
                [-32.0, 0.0],
                [16.0, 8.0],
            ))],
        );
        accept(
            &mut baker,
            Placement::turned([0.0, 0.0], std::f32::consts::FRAC_PI_2),
            &[Feature::occluding(Footprint::rect([0.0, 0.0], [8.0, 16.0]))],
        );
        accept(
            &mut baker,
            Placement::IDENTITY,
            &[Feature::occluding(Footprint::rect(
                [32.0, 0.0],
                [16.0, 8.0],
            ))],
        );
        let baked = baker.finish();
        assert_eq!(baked.occluders.len(), 1);
        assert_eq!(baked.occluders[0].center, [0.0, 0.0]);
        assert_eq!(baked.occluders[0].half, [48.0, 8.0]);
    }

    #[test]
    fn every_quarter_turn_lands_in_the_same_fusion_group() {
        // The residue of a composed quarter turn is the arithmetic this module
        // has to survive: a tile placed through three turns, and one placed
        // backwards through one, are both axis-aligned ground, and they must
        // join the upright tiles either side of them.
        for turn in [
            std::f32::consts::FRAC_PI_2,
            std::f32::consts::PI,
            3.0 * std::f32::consts::FRAC_PI_2,
            -std::f32::consts::FRAC_PI_2,
        ] {
            let mut baker = Baker::new();
            accept(
                &mut baker,
                Placement::IDENTITY,
                &[Feature::occluding(Footprint::rect(
                    [-32.0, 0.0],
                    [16.0, 8.0],
                ))],
            );
            // Authored standing up as 16 wide and 32 tall; turned by `turn`,
            // it covers the 32 wide and 16 tall ground between its neighbours
            // - except at a half turn, which leaves it standing.
            let middle = if turn.abs() == std::f32::consts::PI {
                Footprint::rect([0.0, 0.0], [16.0, 8.0])
            } else {
                Footprint::rect([0.0, 0.0], [8.0, 16.0])
            };
            accept(
                &mut baker,
                Placement::turned([0.0, 0.0], turn),
                &[Feature::occluding(middle)],
            );
            accept(
                &mut baker,
                Placement::IDENTITY,
                &[Feature::occluding(Footprint::rect(
                    [32.0, 0.0],
                    [16.0, 8.0],
                ))],
            );
            let baked = baker.finish();
            assert_eq!(
                baked.occluders.len(),
                1,
                "a turn of {turn} fused into {} boxes",
                baked.occluders.len()
            );
            assert_eq!(baked.occluders[0].half, [48.0, 8.0]);
        }
    }

    #[test]
    fn a_shared_odd_angle_fuses_on_agreement_and_misses_on_drift() {
        // Two boxes of one orientation, edge to edge along their own axis, at
        // an angle nobody would call a quarter turn. `first` and `second` are
        // the turns each box carries: the same number reaches both, or the
        // second arrives there through arithmetic of its own.
        let (sin, cos) = 0.4f32.sin_cos();
        let along_own_x = [32.0 * cos, 32.0 * sin];
        let pair = |first: f32, second: f32| {
            let mut baker = Baker::new();
            accept(
                &mut baker,
                Placement::IDENTITY,
                &[Feature::occluding(Footprint::tilted(
                    [0.0, 0.0],
                    [16.0, 8.0],
                    first,
                ))],
            );
            accept(
                &mut baker,
                Placement::IDENTITY,
                &[Feature::occluding(Footprint::tilted(
                    along_own_x,
                    [16.0, 8.0],
                    second,
                ))],
            );
            baker.finish().occluders
        };
        assert_eq!(
            pair(0.4, 0.4).len(),
            1,
            "two boxes of one turn, edge to edge, should be one"
        );

        // One ULP of drift between them and they stay two records, however
        // surely they touch: an angle is folded near a quarter turn, not near
        // an arbitrary one, and folding exactly is what buys the flattened
        // fusion its exactness. The cost of the conservatism is records per
        // pixel; the cost of an approximate frame would be edges.
        let drifted = (0.4f32 + 10.0) - 10.0;
        assert_ne!(drifted, 0.4, "the drift the test is about vanished");
        assert_eq!(pair(0.4, drifted).len(), 2);
        // And the same turn reached two ways still covers the same ground, so
        // the two records are correct, just not as few as they could be.
        let two = pair(0.4, drifted);
        for (want_angle, want_center) in [(0.4, [0.0, 0.0]), (drifted, along_own_x)] {
            let found = two.iter().find(|b| b.angle == want_angle).unwrap();
            assert_eq!(found.center, want_center);
            assert_eq!(found.half, [16.0, 8.0]);
        }
    }

    #[test]
    fn a_box_that_is_turned_waits_for_a_neighbour_turned_the_same_way() {
        // An eighth turn is nobody's quarter turn: the box keeps it, fuses with
        // nothing, and still casts at its angle.
        let mut baker = Baker::new();
        accept(
            &mut baker,
            Placement::IDENTITY,
            &[Feature::occluding(Footprint::tilted(
                [0.0, 0.0],
                [8.0, 8.0],
                0.4,
            ))],
        );
        accept(
            &mut baker,
            Placement::IDENTITY,
            &[Feature::occluding(Footprint::rect([16.0, 0.0], [8.0, 8.0]))],
        );
        let baked = baker.finish();
        assert_eq!(baked.occluders.len(), 2);
        assert_eq!(baked.occluders[1].angle, 0.4);
    }

    #[test]
    fn baking_stays_in_the_nodes_space() {
        let mut baker = Baker::new();
        accept(&mut baker, Placement::at([-40.0, 12.0]), &[tile()]);
        let baked = baker.finish();
        // Placed, but never baked into world coordinates: the node's transform
        // still goes on the front of the whole set.
        assert_eq!(baked.occluders[0].center, [-40.0, 12.0]);
        assert_eq!(baked.occluders[0].angle, 0.0);
    }

    /// Assert that a fused set covers the same points as the set it was fused
    /// from, sampled off the integer lattice the boxes were built on so no
    /// sample has to guess which side of an edge it is on.
    fn assert_same_coverage(authored: &[Collider], fused: &[Collider], case: usize, across: f32) {
        let mut x = -10.25;
        while x < across {
            let mut y = -10.25;
            while y < across {
                let p = [x, y];
                let was = authored.iter().any(|c| c.contains(p));
                let now = fused.iter().any(|c| c.contains(p));
                assert_eq!(now, was, "case {case} point {p:?} changed hands");
                y += 0.5;
            }
            x += 0.5;
        }
    }

    #[test]
    fn fusion_covers_exactly_the_points_the_tiles_covered() {
        // The safety property of the whole module: a point is inside some fused
        // box exactly when it was inside one of the authored ones. A fusion
        // that overshoots throws a shadow nothing stands in; one that falls
        // short lights a point a wall covers.
        let mut seed: u32 = 0x2545_f491;
        let mut rand = move || {
            seed ^= seed << 13;
            seed ^= seed >> 17;
            seed ^= seed << 5;
            (seed >> 8) as f32 / (1u32 << 24) as f32
        };
        let trunc = |v: f32| v as u32 as f32;

        // Messy piles: every width and position its own, so most boxes find no
        // partner and the few that do must agree exactly.
        for case in 0..12 {
            let mut baker = Baker::new();
            let mut authored: Vec<Collider> = Vec::new();
            for _ in 0..40 {
                let w = 1.0 + trunc(rand() * 4.0);
                let h = 1.0 + trunc(rand() * 4.0);
                let x = trunc(rand() * 24.0);
                let y = trunc(rand() * 24.0);
                let shape = Footprint::rect([x + w / 2.0, y + h / 2.0], [w / 2.0, h / 2.0]);
                authored.push(shape.collider().expect("a rect is baked to a collider"));
                accept(
                    &mut baker,
                    Placement::IDENTITY,
                    &[Feature::occluding(shape)],
                );
            }
            let fused = baker.finish().occluder_colliders();
            assert!(fused.len() <= authored.len(), "case {case} grew");
            assert_same_coverage(&authored, &fused, case, 30.0);
        }

        // The shape a tile map actually makes: one cell size on a lattice,
        // cells randomly missing, which is where rows fuse into strips and
        // strips into blocks over several passes.
        for case in 0..8 {
            let mut baker = Baker::new();
            let mut authored: Vec<Collider> = Vec::new();
            for row in 0..8 {
                for col in 0..8 {
                    if trunc(rand() * 2.0) == 0.0 {
                        continue;
                    }
                    let shape = Footprint::rect([col as f32 * 16.0, row as f32 * 16.0], [8.0, 8.0]);
                    authored.push(shape.collider().expect("a rect is baked to a collider"));
                    accept(
                        &mut baker,
                        Placement::IDENTITY,
                        &[Feature::occluding(shape)],
                    );
                }
            }
            let fused = baker.finish().occluder_colliders();
            // A lattice this regular always gives up something: two adjacent
            // cells are never far apart.
            assert!(
                fused.len() < authored.len(),
                "case {case} fused nothing: {} boxes from {}",
                fused.len(),
                authored.len()
            );
            assert_same_coverage(&authored, &fused, case, 130.0);
        }
    }

    #[test]
    fn a_declared_occluder_carries_its_turn_in_the_transform() {
        let mut baker = Baker::new();
        accept(&mut baker, Placement::turned([10.0, 20.0], 0.4), &[tile()]);
        let baked = baker.finish();
        let mut canvas = Canvas::new((100, 100), 1.0);
        baked.declare_occluders(&mut canvas, Transform::translate([100.0, 0.0]));
        assert_eq!(canvas.occluders.len(), 1);
        let o = &canvas.occluders[0];
        // The shape stays a plain box at its own origin; its turn rides along
        // in the transform, so the record keeps the form the shader reads.
        assert_eq!(o.center, [0.0, 0.0]);
        assert_eq!(o.half, [8.0, 8.0]);
        // And the transform really does carry the turn, the placement, and the
        // node's own move in that order, with the canvas fold last of all: the
        // box's own corner, turned by 0.4, set down at [10, 20], moved 100 to
        // the right by the node, then turned into panel pixels.
        let got = o.world.apply([8.0, 8.0]);
        let (sin, cos) = 0.4f32.sin_cos();
        let want = [
            110.0 + 8.0 * cos - 8.0 * sin + 50.0,
            50.0 - (20.0 + 8.0 * sin + 8.0 * cos),
        ];
        assert!(
            (got[0] - want[0]).abs() < 1e-3 && (got[1] - want[1]).abs() < 1e-3,
            "declared corner {got:?}, expected {want:?}"
        );
    }

    #[test]
    fn the_cpu_and_the_frame_ask_a_shadow_question_in_the_same_space() {
        // A fused strip is one record for the GPU and one collider for the CPU,
        // and the CPU question is asked in the node's space.
        let mut baker = Baker::new();
        for col in 0..6 {
            accept(
                &mut baker,
                Placement::at([col as f32 * 16.0, 0.0]),
                &[tile()],
            );
        }
        accept(
            &mut baker,
            Placement::at([80.0, 40.0]),
            &[Feature::occluding(Footprint::disc([0.0, 0.0], 10.0))],
        );
        let baked = baker.finish();
        let occluders = baked.occluder_colliders();
        assert_eq!(
            occluders.len(),
            2,
            "one fused strip and one round occluder, both in the CPU's list"
        );
        // Straight through the strip: blocked.
        assert!(crate::occluded([-60.0, 0.0], [100.0, 0.0], &occluders));
        // And forty pixels up — clear of the strip by every one of a tile's
        // eight half-heights — the ray now runs into the disc. The round
        // occluder reached the question the CPU asks, in the node's space,
        // without a rectangle being asked to stand in for it.
        assert!(
            crate::occluded([-60.0, 40.0], [100.0, 40.0], &occluders),
            "the disc authored at [80, 40] should block the ray above the strip"
        );
        assert!(
            !crate::occluded([-60.0, 80.0], [100.0, 80.0], &occluders),
            "and the ray above the disc should still be clear"
        );
    }
}
