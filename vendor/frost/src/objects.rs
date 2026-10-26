//! The scene objects: the [`Transform`], [`Color`], [`Light`], [`Shape`],
//! [`SceneNode`], and [`Scene`] types that make up the scene tree drawn by
//! frost.

use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::particles::ParticleSystem;

/// An RGBA color with channels in `0.0..=1.0`, used for the background and
/// for the color of each drawn object.
///
/// The alpha channel is the color's opacity: shapes emit
/// `coverage * a`, and sprites and text emit `texture_alpha * opacity * a`
/// (their separate `alpha` field and the color's alpha multiply). `1.0`
/// (opaque) is the default for the colors created by [`Shape::sprite`] and
/// [`Shape::text`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Color {
    /// Red channel.
    pub r: f32,
    /// Green channel.
    pub g: f32,
    /// Blue channel.
    pub b: f32,
    /// Alpha channel: the color's opacity. `1.0` is opaque.
    pub a: f32,
}

impl Color {
    /// The channels as `[r, g, b, a]`.
    pub(crate) fn channels(&self) -> [f32; 4] {
        [self.r, self.g, self.b, self.a]
    }

    /// Multiplies this color by `other` channel by channel.
    pub(crate) fn mul(&self, other: Color) -> Color {
        Color {
            r: self.r * other.r,
            g: self.g * other.g,
            b: self.b * other.b,
            a: self.a * other.a,
        }
    }

    /// The linear interpolation from `self` toward `other` at fraction
    /// `t`, channel by channel: `self` at `t = 0.0` and `other` at
    /// `t = 1.0`.
    pub fn lerp(&self, other: Color, t: f32) -> Color {
        Color {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }
}

/// White at full opacity: the identity for channel-wise multiplication, the
/// [`SceneNode::modulate`] that leaves colors unchanged.
pub(crate) const WHITE: Color = Color {
    r: 1.0,
    g: 1.0,
    b: 1.0,
    a: 1.0,
};

/// The default ambient color: a dim neutral gray, the floor of a scene's
/// light field where no light reaches. [`Scene::new`] and
/// [`Scene::default`] start from it, and a scene's
/// [`ambient`](Scene::ambient) field can be set to it (or to any other
/// color) to change the floor of every lit receiver.
pub const AMBIENT: Color = Color {
    r: 0.3,
    g: 0.3,
    b: 0.3,
    a: 1.0,
};

/// A 2D affine transform: `p' = m * p + t`.
///
/// The rows of `m` are `m[0]` and `m[1]`, so a point `(x, y)` maps to
/// `(m[0][0]*x + m[0][1]*y + t[0], m[1][0]*x + m[1][1]*y + t[1])`. Rotation
/// angles are measured from the +x axis toward +y, i.e. counter-clockwise in
/// the window-centered y-up coordinates.
#[derive(Clone, Copy, Debug)]
pub struct Transform {
    pub(crate) m: [[f32; 2]; 2],
    pub(crate) t: [f32; 2],
}

impl Transform {
    /// The identity transform (no change).
    pub const fn identity() -> Self {
        Self {
            m: [[1.0, 0.0], [0.0, 1.0]],
            t: [0.0, 0.0],
        }
    }

    /// A translation by `offset`, `[x, y]` in y-up user units.
    pub const fn translate(offset: [f32; 2]) -> Self {
        Self {
            m: [[1.0, 0.0], [0.0, 1.0]],
            t: offset,
        }
    }

    /// A rotation by `angle` radians, counter-clockwise in y-up coordinates.
    pub fn rotate(angle: f32) -> Self {
        let (s, c) = angle.sin_cos();
        Self {
            m: [[c, -s], [s, c]],
            t: [0.0, 0.0],
        }
    }

    /// A scale by `scale[0]` horizontally and `scale[1]` vertically.
    pub const fn scale(scale: [f32; 2]) -> Self {
        Self {
            m: [[scale[0], 0.0], [0.0, scale[1]]],
            t: [0.0, 0.0],
        }
    }

    /// A uniform scale by `s`.
    pub const fn scale_uniform(s: f32) -> Self {
        Self::scale([s, s])
    }

    /// The node-local coordinates of an image anchor: the offset from a
    /// sprite's center to the anchor pixel, for a sprite that is centered
    /// on its node's origin.
    ///
    /// Image pixels are numbered from `(0, 0)` at the upper-left, `y`
    /// down; node-local coordinates have the image's center on the node's
    /// origin, `y` up. So the anchor at image pixels `(ax, ay)` of an
    /// image of `size` pixels is the offset `(ax - size[0] / 2,
    /// size[1] / 2 - ay)`, the y flip included.
    pub const fn anchor(anchor: [f32; 2], size: [f32; 2]) -> [f32; 2] {
        [anchor[0] - size[0] / 2.0, size[1] / 2.0 - anchor[1]]
    }

    /// The transform that applies `self` first, then `other`:
    /// `self.compose(other)` maps a point `p` to `other.apply(self.apply(p))`.
    ///
    /// In matrix form, with `p' = m * p + t`, the composition multiplies in
    /// the opposite order of the application order: `other.m * self.m`.
    pub fn compose(&self, other: &Transform) -> Transform {
        let s = self.m;
        let o = other.m;
        Transform {
            m: [
                [
                    o[0][0] * s[0][0] + o[0][1] * s[1][0],
                    o[0][0] * s[0][1] + o[0][1] * s[1][1],
                ],
                [
                    o[1][0] * s[0][0] + o[1][1] * s[1][0],
                    o[1][0] * s[0][1] + o[1][1] * s[1][1],
                ],
            ],
            t: [
                o[0][0] * self.t[0] + o[0][1] * self.t[1] + other.t[0],
                o[1][0] * self.t[0] + o[1][1] * self.t[1] + other.t[1],
            ],
        }
    }

    /// Applies the transform to a point.
    pub fn apply(&self, p: [f32; 2]) -> [f32; 2] {
        [
            self.m[0][0] * p[0] + self.m[0][1] * p[1] + self.t[0],
            self.m[1][0] * p[0] + self.m[1][1] * p[1] + self.t[1],
        ]
    }

    /// This transform with its translation scaled by `f`: the same linear
    /// part, and the translation `(f * t[0], f * t[1])`.
    pub(crate) fn scaled_translation(&self, f: f32) -> Transform {
        Transform {
            m: self.m,
            t: [self.t[0] * f, self.t[1] * f],
        }
    }

    /// The inverse transform, or `None` if the transform is degenerate (its
    /// linear part collapses points onto a line or a point).
    pub(crate) fn invert(&self) -> Option<Transform> {
        let a = self.m[0][0];
        let b = self.m[0][1];
        let c = self.m[1][0];
        let d = self.m[1][1];
        let det = a * d - b * c;
        if det.abs() < 1e-12 {
            return None;
        }
        let inv = 1.0 / det;
        Some(Transform {
            m: [[d * inv, -b * inv], [-c * inv, a * inv]],
            t: [
                -(self.t[0] * d - self.t[1] * b) * inv,
                (self.t[0] * c - self.t[1] * a) * inv,
            ],
        })
    }

    /// The scale factors along the x and y axes (the norms of the matrix
    /// columns), so the anti-alias band can be kept a constant size in
    /// screen pixels under scaling.
    pub(crate) fn scales(&self) -> [f32; 2] {
        [
            (self.m[0][0] * self.m[0][0] + self.m[1][0] * self.m[1][0]).sqrt(),
            (self.m[0][1] * self.m[0][1] + self.m[1][1] * self.m[1][1]).sqrt(),
        ]
    }
}

/// The geometry of each particle in a [`Shape::Particles`] batch: one shape
/// for the whole batch, shared by every particle.
///
/// The shape's extent is scaled by each particle's own
/// [`Particle::size`](crate::Particle::size) (its half-width), its
/// orientation by each particle's own
/// [`Particle::angle`](crate::Particle::angle), and its color by the
/// particle's own [`Particle::color`](crate::Particle::color) — the batch
/// sets *what* each particle looks like,
/// while the particles keep their individual position, scale, rotation, and
/// tint.
#[derive(Clone, Debug, Default)]
pub enum ParticleShape {
    /// A filled circle with the particle's `size` as its radius.
    #[default]
    Circle,
    /// A filled rectangle, `size * 2` wide and `size * aspect * 2` tall,
    /// centered on the particle.
    Rectangle {
        /// The rectangle's height-to-width ratio; `1.0` is a square.
        aspect: f32,
    },
    /// A sprite image: each particle draws the whole image, scaled so its
    /// width is `2 * size`, and centered on the particle — the texture's
    /// own center `(1/2, 1/2)` sits on the particle's position.
    ///
    /// Created by [`ParticleShape::sprite`] and [`ParticleShape::sprite_bytes`];
    /// the pixels are decoded up front and shared behind an [`Arc`], like a
    /// [`Shape::Sprite`].
    Sprite {
        /// The RGBA8 pixel data, row by row, top row first.
        data: Arc<[u8]>,
        /// The texture width in pixels.
        width: u32,
        /// The texture height in pixels.
        height: u32,
        /// The buffer's generation, identifying `data` to the backend's
        /// texture cache.
        generation: u64,
    },
}

impl ParticleShape {
    /// Creates a sprite particle shape from the PNG file at `path`.
    ///
    /// The file is read and decoded to RGBA8 immediately, so a missing file
    /// or a non-PNG file fails here, not at render time. The pixels live
    /// behind an [`Arc`], so cloning a scene containing the shape is cheap:
    /// all the clones share the same buffer.
    pub fn sprite(path: impl AsRef<Path>) -> Result<Self, SpriteError> {
        let bytes = std::fs::read(path.as_ref()).map_err(SpriteError::Io)?;
        Self::sprite_bytes(&bytes)
    }

    /// Creates a sprite particle shape from PNG data already in memory, for
    /// example bytes embedded into the binary with `include_bytes!`.
    ///
    /// Like [`ParticleShape::sprite`], the bytes are decoded to RGBA8 up
    /// front, so a non-PNG buffer fails here, not at render time, and the
    /// pixels live behind an [`Arc`], but no file is read — this is how
    /// sprite shapes are created in environments without a file system,
    /// such as a web browser.
    pub fn sprite_bytes(bytes: impl AsRef<[u8]>) -> Result<Self, SpriteError> {
        let image = image::load_from_memory(bytes.as_ref()).map_err(SpriteError::Decode)?;
        let rgba = image.to_rgba8();
        let (width, height) = rgba.dimensions();
        let data = Arc::from(rgba.into_raw().into_boxed_slice());
        Ok(Self::Sprite {
            data,
            width,
            height,
            generation: NEXT_SPRITE_GENERATION.fetch_add(1, Ordering::Relaxed),
        })
    }

    /// The shape's kind as the render pipeline reads it: `0.0` for a
    /// circle, `1.0` for a rectangle, `2.0` for a sprite.
    pub(crate) fn kind(&self) -> f32 {
        match self {
            Self::Circle => 0.0,
            Self::Rectangle { .. } => 1.0,
            Self::Sprite { .. } => 2.0,
        }
    }

    /// The shape's height-to-width factor: `1.0` for a circle, the given
    /// `aspect` for a rectangle, and the texture's height over its width for
    /// a sprite.
    pub(crate) fn aspect(&self) -> f32 {
        match self {
            Self::Circle => 1.0,
            Self::Rectangle { aspect } => *aspect,
            Self::Sprite { width, height, .. } => {
                (*height as f32).max(1.0) / (*width as f32).max(1.0)
            }
        }
    }
}

/// One tile placed on a tile map: a copy of an atlas sub-rectangle, laid
/// at `center` in the map node's local space, `size` wide and tall, and
/// tinted by `color`.
///
/// A tile is placed like a rectangle and sampled like a sprite: the node's
/// transform, scale, and modulate move, stretch, and tint the placement
/// exactly as they would a rectangle's, while [`Tile::uv`] picks *what*
/// the tile shows out of the map's atlas. `uv` is
/// `[min_u, min_v, max_u, max_v]` in `[0, 1]` texture coordinates with
/// the top-left origin a PNG carries — so the cell at `(col, row)` of an
/// even `cols`-by-`rows` atlas grid spans
/// `(col / cols, row / rows, (col + 1) / cols, (row + 1) / rows)`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Tile {
    /// The tile's center, in the map node's local space.
    pub center: [f32; 2],
    /// The tile's full width and height in local units: `32.0` by `32.0`
    /// lays a 32-pixel atlas cell at one texture pixel per scene pixel.
    pub size: [f32; 2],
    /// The atlas sub-rectangle to show, `[min_u, min_v, max_u, max_v]`.
    pub uv: [f32; 4],
    /// The tile's own tint, multiplied with the batch's `color`, the
    /// node's composed modulate, and the sampled texel's own alpha.
    /// White shows the atlas uncolored.
    pub color: Color,
    /// The atlas cell's own orientation: clockwise quarter-turns,
    /// 0..=3, applied in screen space before the flips. An odd turn
    /// lays the cell sideways, so the painted quad swaps the tile's
    /// width and height — `size` keeps describing the cell as
    /// authored, the renderer turns the extent.
    pub rot: u8,
    /// The cell mirrored left-to-right, in screen space, after `rot`:
    /// an x-flip always means sideways, whatever the orientation.
    pub flip_x: bool,
    /// The cell mirrored top-to-bottom, in screen space, after `rot`.
    pub flip_y: bool,
}

impl Tile {
    /// A tile at `center` of `size`, showing the atlas sub-rectangle
    /// `uv`, in the atlas's own colors.
    pub fn new(center: [f32; 2], size: [f32; 2], uv: [f32; 4]) -> Self {
        Self {
            center,
            size,
            uv,
            color: Color {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
            rot: 0,
            flip_x: false,
            flip_y: false,
        }
    }

    /// The tile with its cell turned `rot` clockwise quarter-turns
    /// (counts past three wrap) and mirrored on the `flip_x` and
    /// `flip_y` screen axes — the orientation of a painted cell.
    #[must_use]
    pub const fn transformed(mut self, rot: u8, flip_x: bool, flip_y: bool) -> Self {
        self.rot = rot % 4;
        self.flip_x = flip_x;
        self.flip_y = flip_y;
        self
    }
}

/// A set of light layers: the 32-layer bitmask a [`Light`] emits on, a
/// [`SceneNode::lit_layers`] receiver receives from, and a
/// [`SceneNode::occludes_layers`] occluder shadows on.
///
/// Light layers are a filter, not a depth: they decide *which* lights reach
/// *which* surfaces, never the draw order — that belongs to [`Layer`] and
/// [`SceneNode::order`]. A light reaches a receiver when their sets share a
/// layer; a shadow lands on a receiver when the occluder's set and the
/// receiver's `lit_layers` share a layer. Layers are the cheap answer to
/// lighting scenes that should not light each other: a tilemap whose own
/// tiles must not shadow each other, a sky nothing occludes, a HUD the
/// scene's lamps never touch.
///
/// The default is the set holding only **layer 0** — the set every light,
/// receiver, and occluder starts on — so a scene that names no layers is
/// one big layer-0 world and behaves exactly as it did before layers
/// existed. Anything written `..Default::default()` (or left out of a
/// [`SceneNode`] literal) is on layer 0.
///
/// This is the struct-literal pattern: set a layer only where the scene
/// asks for the filter, and let every other node take the default.
///
/// [`Layer`]: crate::objects::Layer
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct LightLayers(u32);

impl Default for LightLayers {
    /// The default set: layer 0 alone.
    fn default() -> Self {
        Self::layer(0)
    }
}

impl LightLayers {
    /// The empty set: light that reaches nothing, shadows that land
    /// nowhere. Useful to mute one end of the filter without dropping the
    /// node's `lit`/`occludes` flag.
    pub const NONE: Self = Self(0);
    /// Every one of the 32 layers: a light shining on all receivers, an
    /// occluder shadowing all of them.
    pub const ALL: Self = Self(u32::MAX);

    /// The set holding exactly one layer. The layer number wraps every 32
    /// (`layer(0) == layer(32)`), so it never panics; numbers past 31 are
    /// still reachable but share a bit with their lower twins.
    #[must_use]
    pub const fn layer(n: u32) -> Self {
        Self(1u32 << (n % 32))
    }

    /// This set's raw bits: bit `n` is layer `n`. The bit layout is the
    /// wire format the backend packs; apps rarely need it, but tests and
    /// serializing scene formats like to name bits directly.
    #[must_use]
    pub const fn bits(self) -> u32 {
        self.0
    }

    /// A set from raw bits, the inverse of [`bits`](Self::bits).
    #[must_use]
    pub const fn from_bits(bits: u32) -> Self {
        Self(bits)
    }

    /// The union of two sets: every layer either holds.
    #[must_use]
    pub const fn union(self, other: Self) -> Self {
        Self(self.0 | other.0)
    }

    /// Whether the two sets share at least one layer — the question the
    /// renderer asks of a light and a receiver, and of an occluder and the
    /// receiver its shadow would land on.
    #[must_use]
    pub const fn intersects(self, other: Self) -> bool {
        self.0 & other.0 != 0
    }

    /// Whether `n` names a layer this set holds.
    #[must_use]
    pub const fn contains(self, n: u32) -> bool {
        self.0 & (1u32 << (n % 32)) != 0
    }
}

impl std::ops::BitOr for LightLayers {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        self.union(rhs)
    }
}

impl std::ops::BitOrAssign for LightLayers {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

/// A light source: the color, strength, falloff extent, and cone of a
/// light, held by a [`Shape::Light`] node.
///
/// A light is never drawn: it contributes to the frame's light field, and
/// every lit receiver ([`SceneNode::lit`]) evaluates it per pixel at render
/// time. The node carrying the light provides the position: the light sits
/// at the node's local origin, so the node's transform and its ancestors'
/// place it in the scene, and the node's composed modulate tints its color,
/// exactly as they would tint a shape's.
///
/// By default a light is omni-directional — it lights every pixel within
/// `radius`. Setting a narrower [`spread`](Self::spread) turns it into a
/// cone: only pixels inside the cone receive its light, feathered out at
/// the edges by [`softness`](Self::softness).
///
/// [`Light::directional`] is the third kind: light from a source so far
/// away — a sun, a distant star — that its rays are parallel. It has no
/// position and its light never fades with distance, so the node carrying
/// it only aims it (and tints it); a scene may hold any number, like a
/// planet lit by two stars at once.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Light {
    /// The light's color, multiplied with its intensity and with the
    /// falloff at each lit pixel.
    pub color: Color,
    /// The light's strength, multiplied with its color: `1.0` is full
    /// strength, `0.0` is off.
    pub intensity: f32,
    /// The falloff extent, in user units (pixels): the light reaches
    /// exactly `radius` pixels from its position, fading to zero at the
    /// edge.
    ///
    /// A negative radius is the sentinel that marks a directional light:
    /// [`Light::directional`] sets it, and the light then ignores its
    /// position and this value entirely.
    pub radius: f32,
    /// The direction the light points, in radians, counterclockwise from
    /// the local +x axis, measured in the node's local (y-up) space: like
    /// every other direction on a node, the node's transform and its
    /// ancestors' rotate it along with the position, so rotating the node
    /// turns the beam. Ignored when the light is omni-directional
    /// ([`spread`](Self::spread) at or beyond [`TAU`](std::f32::consts::TAU)).
    pub direction: f32,
    /// The cone's full opening angle, in radians: the angle between the
    /// cone's two edges, measured around [`direction`](Self::direction).
    /// `TAU` (or more) is omni-directional — the light reaches every
    /// direction, as if there were no cone at all; [`PI`](std::f32::consts::PI)
    /// is a half-plane; smaller angles narrow the beam; `0.0` degenerates
    /// to the axis alone.
    pub spread: f32,
    /// How wide the cone's edge feather is, in radians: the band just
    /// inside each cone edge across which the light ramps smoothly from
    /// full strength to none, instead of stopping dead at the edge — the
    /// cheap stand-in for a real penumbra. `0.0` keeps the hard edge;
    /// more than [`spread`](Self::spread) / 2 feathers the whole cone
    /// (clamped, so the axis stays fully lit). Ignored when the light is
    /// omni-directional.
    pub softness: f32,
    /// The light's penumbra — the soft edge of the shadows it casts — as
    /// the radius, in user units (pixels), of the light's own disk.
    ///
    /// The shadow rays sample that disk (nine taps, rotated per pixel)
    /// and the light passes in proportion to how much of it an occluder
    /// leaves uncovered, so a shadow fades across an edge as wide as the
    /// disk instead of dropping from full light to full dark in one
    /// pixel. `0.0` keeps the hard shadow of a mathematical point light.
    ///
    /// This applies to every light, cone or omni, and only to shadows:
    /// an occluder-free light looks exactly as it did. The light's own
    /// [`radius`](Self::radius) falloff is unrelated.
    ///
    /// A [`directional`](Light::directional) light's shadow softness is
    /// geometric: its source is distant, so a disk of `penumbra` pixels
    /// at that distance fades shadows by proportionally less the closer
    /// the occluder sits to the shadow — values around 20 and up give
    /// suns and stars shadows worth calling soft.
    pub penumbra: f32,
    /// How many shadow rays a penumbra'd **omni** light sweeps through
    /// its disk per lit pixel: 1..=9, taken from a grid ordered so the
    /// first four are a balanced cross and all nine fill the 3×3.
    /// Weak hardware trades diffusion for frame rate one ray at a
    /// time; the default, 9, is the full grid. Ignored when
    /// [`penumbra`](Self::penumbra) is zero — that is the single hard
    /// ray by definition — and by cone and directional lights, whose
    /// third uniform slot carries the cone's edge feather instead.
    pub shadow_taps: u32,
    /// The light layers this light shines on: it reaches every receiver
    /// whose [`SceneNode::lit_layers`] shares a layer with this set.
    ///
    /// The default — [`LightLayers::default()`], layer 0 alone — is the
    /// one world every pre-layer scene shares. A lamp meant to light both
    /// the scenery and a surface the scenery must not shadow (a tilemap
    /// lighting the tiles *and* the far wall behind them) joins the two
    /// sets with [`union`](LightLayers::union) or `|`; see the lighting
    /// amendment in `ARCHITECTURE.md`.
    pub layers: LightLayers,
}

impl Light {
    /// An omni-directional point light: full-circle coverage, lighting
    /// every direction equally.
    pub fn point(color: Color, intensity: f32, radius: f32) -> Self {
        Self {
            color,
            intensity,
            // Clamped here, not at draw: a point light must never carry
            // the negative radius that marks a directional one, so a
            // stray negative degrades to a degenerate zero-radius light
            // rather than accidentally becoming a sun.
            radius: radius.max(0.0),
            // The direction is irrelevant for a full circle; +x keeps the
            // recorded cone axis defined.
            direction: 0.0,
            spread: std::f32::consts::TAU,
            softness: 0.0,
            penumbra: 0.0,
            shadow_taps: 9,
            layers: LightLayers::default(),
        }
    }

    /// A cone (spot) light: a beam of total opening angle `spread`
    /// radians, pointing `direction` radians counterclockwise from the
    /// local +x axis in the node's local space. See the
    /// [`direction`](Self::direction) and [`spread`](Self::spread) fields.
    ///
    /// `softness` feathers the cone's two edges: the light fades out over
    /// this many radians of angle just inside each edge instead of
    /// stopping dead there — `0.0` is a hard edge, and values beyond
    /// `spread / 2` clamp to feathering the whole cone. See
    /// [`Light::softness`].
    pub fn cone(
        color: Color,
        intensity: f32,
        radius: f32,
        direction: f32,
        spread: f32,
        softness: f32,
    ) -> Self {
        Self {
            color,
            intensity,
            radius: radius.max(0.0),
            direction,
            spread,
            softness,
            penumbra: 0.0,
            shadow_taps: 9,
            layers: LightLayers::default(),
        }
    }

    /// A directional light: light from a source so far away that its
    /// rays are parallel — a sun, a distant star. It has no position and
    /// its strength does not fall off with distance: every pixel the
    /// occluders leave uncovered receives it at full `intensity`.
    ///
    /// `direction` is the direction the light *travels*, in radians
    /// counterclockwise from the local +x axis: a dawn sun rising in the
    /// east (screen right) travels toward `-x` and downward over the
    /// ground, so its direction points left and down, near `PI`. The
    /// node carrying the light only aims it — rotating the node turns
    /// the daylight along with it — and its position and scale do not
    /// matter at all.
    ///
    /// A scene may hold any number: two [`Shape::Light`] nodes, each a
    /// star, light a planet from both sides at once, each casting its
    /// own parallel shadows. Directional lights still cast shadows and
    /// take a [`penumbra`](Self::penumbra); see [`Light::with_penumbra`]
    /// for how its radius reads on a distant source.
    pub fn directional(color: Color, intensity: f32, direction: f32) -> Self {
        Self {
            color,
            intensity,
            // The negative radius is the sentinel: a point or cone light
            // with no radius is degenerate anyway, so no finite value
            // here could mean anything else, and the shader swaps the
            // falloff and shadow rays for the parallel treatment.
            radius: -1.0,
            direction,
            // The cone gate has no meaning for parallel rays: full
            // circle, so every uncovered pixel receives the light.
            spread: std::f32::consts::TAU,
            softness: 0.0,
            penumbra: 0.0,
            shadow_taps: 9,
            layers: LightLayers::default(),
        }
    }

    /// The same light with a shadow penumbra: the light is treated as a
    /// disk of `radius` user units (pixels) when occluders cut its
    /// light, so its shadows fade across edges as wide as the disk
    /// instead of ending on a hard silhouette. See
    /// [`Light::penumbra`]; a `radius` of `0` — the constructor default —
    /// is the hard point-light shadow.
    pub fn with_penumbra(mut self, radius: f32) -> Self {
        self.penumbra = radius.max(0.0);
        self
    }

    /// Set the omni penumbra's tap count: see
    /// [`shadow_taps`](Self::shadow_taps). Values clamp to 1..=9 as
    /// they pack, so a request outside the grid simply buys its
    /// nearest end.
    pub fn with_shadow_taps(mut self, taps: u32) -> Self {
        self.shadow_taps = taps;
        self
    }

    /// The same light shining on a set of [`LightLayers`]: it reaches every
    /// receiver whose [`SceneNode::lit_layers`] shares a layer with
    /// `layers`. See [`Light::layers`].
    #[must_use]
    pub fn with_layers(mut self, layers: LightLayers) -> Self {
        self.layers = layers;
        self
    }
}

/// A shape that a [`SceneNode`] can hold.
///
/// A shape is a filled geometric shape drawn in the node's local space — a
/// circle, a rectangle, a sprite, text, or a particle batch — or a
/// [`Shape::Light`], which is never drawn and only lights the scene.
///
/// Coordinates and sizes are in the node's local space, in pixels; the
/// composed transforms of every ancestor apply to the shape, except for
/// [`Shape::Background`], which fills the whole window regardless of
/// transform.
#[derive(Clone, Debug)]
pub enum Shape {
    /// A filled circle centered at `center` with `radius`.
    Circle {
        center: [f32; 2],
        radius: f32,
        color: Color,
    },
    /// A filled rectangle centered at `center` with half-widths `extent`.
    Rectangle {
        center: [f32; 2],
        extent: [f32; 2],
        color: Color,
    },
    /// A polyline: `points` connected in order by straight segments, stroked
    /// with `width`, drawn in **one draw call** no matter how many points it
    /// has — the batched alternative to a run of thin rectangles or a
    /// chain of lines.
    ///
    /// The points and the width are in the node's local space, in pixels,
    /// and the node's transform and scale apply to the stroke exactly as
    /// they do to the other shapes (the width scales with the geometric
    /// mean of the scale's two axes, like a particle's size, so a uniform
    /// scale is exact). At most 128 points are drawn — the shader packs a
    /// fixed 128-point array — and a polyline with fewer than two points
    /// draws nothing.
    ///
    /// Replacing a node's `points` in place is the cheap way to animate a
    /// stroke: the shape is cloned once per node per frame like any other,
    /// and the draw list still records one draw for the whole chain.
    Polyline {
        /// The vertices in the node's local space, connected in order.
        points: Vec<[f32; 2]>,
        /// The stroke's width, in pixels.
        width: f32,
        /// The stroke's color.
        color: Color,
    },
    /// Fills the whole window with `color`.
    ///
    /// The node's transform (and its ancestors') is ignored, and the
    /// background is always at the very back of the frame: at render time it
    /// becomes the window's clear color, so it costs no draw call. Hang it on
    /// the scene root or anywhere else in the tree — position does not matter.
    /// When several nodes hold backgrounds, the last one in depth-first call
    /// order is the one that shows.
    Background { color: Color },
    /// A sprite: the RGBA pixels of a PNG image, created by
    /// [`Shape::sprite`] or [`Shape::sprite_nearest`].
    ///
    /// The sprite is centered on the node's origin, one texture pixel per
    /// scene pixel, and the node's transform and scale apply to it exactly as
    /// they do to the other shapes. `color` is a per-pixel tint multiplied
    /// with every sampled pixel; white (`r: 1.0, g: 1.0, b: 1.0, a: 1.0`)
    /// leaves the texture unchanged. The texture's own alpha channel is
    /// scaled by `alpha` and by `color.a`, so the sprite's overall opacity
    /// is the product of the two.
    Sprite {
        /// The RGBA8 pixel data, row by row, top row first.
        data: Arc<[u8]>,
        /// The texture width in pixels.
        width: u32,
        /// The texture height in pixels.
        height: u32,
        /// The per-pixel tint, multiplied with every sampled pixel.
        color: Color,
        /// The sprite's opacity, multiplied with the texture's own alpha
        /// channel. 1.0 (the default) leaves the texture's transparency
        /// unchanged; 0.0 makes the sprite fully transparent.
        alpha: f32,
        /// The texture's sampling filter, applied whenever the quad is not
        /// the texture's own size: [`SpriteFilter::Linear`] by default.
        filter: SpriteFilter,
        /// The buffer's generation, identifying `data` to the backend's
        /// texture cache.
        generation: u64,
        /// A game-authored WGSL program that paints this sprite in place
        /// of the engine's sprite pipeline, attached by
        /// [`Shape::with_program`]. `None` — every sprite the engine and
        /// every scene without a custom effect carries — is the engine's
        /// own shader, unchanged.
        program: Option<Arc<str>>,
    },
    /// A text: `text` laid out with the font loaded by [`Shape::text`] at
    /// `size` pixels per em.
    ///
    /// The text block is centered on the node's origin, one font pixel per
    /// scene pixel, and the node's transform and scale apply to it exactly
    /// as they do with the other shapes. Each glyph is drawn as a tinted
    /// texture quad sampling a shared glyph atlas: `color` is the glyph
    /// color (white gives white text), and the text's overall opacity is the
    /// product of `alpha` and `color.a`.
    ///
    /// `weight` instantiates the font's `wght` variation axis: `400.0` is
    /// the standard Regular, `700.0` a full bold — on a variable font (like
    /// `assets/fonts/FiraCode-VariableFont_wght.ttf`, whose axis runs 300
    /// to 700) any value in between renders genuinely thicker or thinner.
    /// A static font has no axis to instantiate and draws at its own
    /// weight, whatever this says. Set it through [`Shape::with_weight`].
    Text {
        /// The string to lay out.
        text: String,
        /// The font file's bytes, shared by every text shape that loaded
        /// the same file.
        font: Arc<[u8]>,
        /// The font size in pixels per em.
        size: f32,
        /// The `wght` variation axis value; 400.0 (the default from
        /// [`Shape::text`] and [`Shape::text_bytes`]) is Regular.
        weight: f32,
        /// The glyph color, multiplied with every mask pixel.
        color: Color,
        /// The text's opacity, multiplied with the mask's own alpha
        /// channel.
        alpha: f32,
    },
    /// A batch of live particles from a [`ParticleSystem`], simulated in the
    /// node's local space and drawn as one instanced draw, each particle as
    /// the batch's `shape` at its local position, fading by its remaining
    /// life fraction.
    ///
    /// Every particle shares the batch's [`ParticleShape`], but keeps its
    /// own position, scale, rotation, and color: each particle is drawn as
    /// `shape`, scaled by its `size`, rotated by its `angle` (on top of the
    /// node's rotation), and tinted by its `color` — so a spinning
    /// rectangle or a tumbling sprite needs no engine support, just
    /// `particle.angle += spin * dt;` in the update.
    ///
    /// The batch rides the node's world transform and scale exactly like the
    /// other shapes: a particle at local `(x, y)` lands wherever the node's
    /// world transform maps it, and each size is multiplied by the geometric
    /// mean of the world's two axis scales — exact under a uniform scale, and
    /// area-preserving under a non-uniform one (the particle's shape cannot
    /// be stretched independently along the two axes, so it keeps its
    /// proportions). The batch is tinted by `color` multiplied with the
    /// node's composed modulate and with each particle's own `color`, and it
    /// draws at the node's composed order.
    ///
    /// The engine never advances the simulation: the node's owner spawns into
    /// the system and calls [`ParticleSystem::update`] each frame, in the same
    /// local space the particles are drawn in — for a system held on `node`,
    /// `if let Some(Shape::Particles { system, .. }) = node.shape.as_mut() {
    /// system.update(dt, gravity); }`. A system with no live particles draws
    /// nothing.
    Particles {
        /// The live system whose particles make up the batch.
        system: ParticleSystem,
        /// The batch's base tint, multiplied with every particle (and its
        /// own color) and then with the node's composed modulate; white
        /// leaves the particles uncolored by the batch itself.
        color: Color,
        /// The geometry each particle is drawn as; the whole batch shares
        /// it. [`ParticleShape::Circle`] (the default) draws circles as
        /// before.
        shape: ParticleShape,
    },
    /// A tile map: [`Tile`] placements sharing one atlas texture, drawn
    /// in **one instanced draw call** however many tiles the map holds —
    /// the batching spine of the particle batch, aimed at static art.
    ///
    /// Created by [`Shape::tilemap`] and [`Shape::tilemap_bytes`], which
    /// decode the atlas PNG up front and start with no tiles; place with
    /// [`Shape::push_tile`] or [`Shape::set_tiles`]. Like
    /// [`Shape::Particles`], the map lives in the node's local space:
    /// every placement rides the node's world transform and scale (a tile
    /// is a box, so its two axes scale independently — the exact answer
    /// a rectangle gives, not the particle batch's geometric mean), is
    /// tinted by `color` times the node's composed modulate times each
    /// tile's own `color`, and draws at the node's composed order. A map
    /// with no tiles draws nothing.
    ///
    /// `clip` cuts the map to a view rectangle, in the node's local
    /// space, as `[min_x, min_y, max_x, max_y]`: the tile quads are
    /// scissored to the rectangle the node's transform maps it to, so a
    /// map can be enormous while only its visible window ever reaches
    /// the screen. `None` clips to nothing.
    ///
    /// The map lights like every other surface when the node says
    /// `lit`: the light field multiplies each tile's pixel and occluders
    /// cast across the floor — a shadowed map is the point. An unlit
    /// node draws it exactly as it always was. The atlas is sampled
    /// with
    /// `filter` — [`SpriteFilter::Nearest`] by construction, since tile
    /// atlases are pixel art; [`Shape::set_filter`] changes it.
    TileMap {
        /// The atlas's RGBA8 pixel data, row by row, top row first — the
        /// same shared-buffer arrangement, texture-cache identity, and
        /// upload cost as a [`Shape::Sprite`]'s: a tileset drawn on the
        /// map and shown as a sprite shares one GPU texture.
        data: Arc<[u8]>,
        /// The atlas texture's width in pixels.
        width: u32,
        /// The atlas texture's height in pixels.
        height: u32,
        /// The buffer's generation, identifying `data` to the backend's
        /// texture cache.
        generation: u64,
        /// How the atlas is sampled when a tile is scaled away from its
        /// texture size.
        filter: SpriteFilter,
        /// The placements. Order among them is draw order within the
        /// single instanced call: tiles later in the list win where two
        /// placements overlap.
        tiles: Vec<Tile>,
        /// The view rectangle the map is scissored to, in the node's
        /// local space, as `[min_x, min_y, max_x, max_y]`; `None` draws
        /// unclipped.
        clip: Option<[f32; 4]>,
        /// The batch's base tint, multiplied with the node's composed
        /// modulate and every tile's own `color`. White leaves the
        /// atlas as authored.
        color: Color,
    },
    /// A light: contributes the node's [`Light`] to the frame's light
    /// field.
    ///
    /// The light is never drawn — it only lights the [`SceneNode::lit`]
    /// receivers, evaluated per pixel at render time. It sits at the
    /// node's local origin: the node's transform (and its ancestors')
    /// position it in the scene, rotates its cone (if it has one), and
    /// tints its color through the composed modulate, exactly as they
    /// would a shape's. The radius is in user units, mapped one to one to
    /// pixels.
    Light {
        /// The light's color, strength, falloff extent, and cone.
        light: Light,
    },
}

/// How a sprite's texture is sampled: the filter the GPU sampler applies
/// whenever the quad that samples it is not the texture's own size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum SpriteFilter {
    /// Bilinear filtering: each output pixel is a weighted average of the
    /// texture's four nearest texels, which smooths scaled edges. The right
    /// choice for photographic images, and the default.
    #[default]
    Linear,
    /// Nearest-neighbor filtering: each output pixel copies exactly one
    /// texel, so scaled edges stay hard instead of blurring — the filter
    /// for pixel-art-style output.
    Nearest,
}

/// The next sprite buffer generation. It starts at 1 (never 0) so a buffer
/// generation can never be confused with the constant `0` that static,
/// never-repacked buffers carry. Every sprite buffer created by the
/// `Shape::sprite*` and `ParticleShape::sprite*` constructors gets its own
/// generation, so the backend's `(pointer, generation)` texture cache can
/// never hit a stale texture when a freed buffer's address is recycled by a
/// new, same-sized one.
static NEXT_SPRITE_GENERATION: AtomicU64 = AtomicU64::new(1);

impl Shape {
    /// Creates a sprite shape from the PNG file at `path`.
    ///
    /// The file is read and decoded to RGBA8 immediately, so a missing file
    /// or a non-PNG file fails here, not at render time. The pixels live
    /// behind an [`Arc`], so cloning a sprite shape — or any scene node or
    /// scene containing one — is cheap: all the clones share the same
    /// buffer.
    pub fn sprite(path: impl AsRef<Path>) -> Result<Self, SpriteError> {
        let bytes = std::fs::read(path.as_ref()).map_err(SpriteError::Io)?;
        Self::sprite_bytes(&bytes)
    }

    /// Creates a sprite shape from PNG data already in memory, for example
    /// bytes embedded into the binary with `include_bytes!`.
    ///
    /// Like [`Shape::sprite`], the bytes are decoded to RGBA8 up front, so a
    /// non-PNG buffer fails here, not at render time, and the pixels live
    /// behind an [`Arc`], but no file is read — this is how sprite shapes
    /// are created in environments without a file system, such as a web
    /// browser.
    pub fn sprite_bytes(bytes: impl AsRef<[u8]>) -> Result<Self, SpriteError> {
        Self::sprite_bytes_filtered(bytes, SpriteFilter::Linear)
    }

    /// Creates a sprite shape from the PNG file at `path`, sampling it with
    /// nearest-neighbor filtering: each output pixel copies exactly one
    /// texel, so the image keeps its hard, unblurred edges when it is
    /// scaled — the filter for pixel-art-style output.
    ///
    /// Like [`Shape::sprite`], the file is read and decoded to RGBA8
    /// immediately and the pixels live behind an [`Arc`]; the only
    /// difference is the sampling filter.
    pub fn sprite_nearest(path: impl AsRef<Path>) -> Result<Self, SpriteError> {
        let bytes = std::fs::read(path.as_ref()).map_err(SpriteError::Io)?;
        Self::sprite_bytes_filtered(&bytes, SpriteFilter::Nearest)
    }

    /// Creates a nearest-neighbor sprite shape from PNG data already in
    /// memory, for example bytes embedded into the binary with
    /// `include_bytes!`: the in-memory counterpart of
    /// [`Shape::sprite_nearest`].
    pub fn sprite_bytes_nearest(bytes: impl AsRef<[u8]>) -> Result<Self, SpriteError> {
        Self::sprite_bytes_filtered(bytes, SpriteFilter::Nearest)
    }

    /// The shared implementation behind the sprite constructors: decodes
    /// `bytes` to RGBA8 and wraps the pixels with the given sampling
    /// `filter`.
    fn sprite_bytes_filtered(
        bytes: impl AsRef<[u8]>,
        filter: SpriteFilter,
    ) -> Result<Self, SpriteError> {
        let image = image::load_from_memory(bytes.as_ref()).map_err(SpriteError::Decode)?;
        let rgba = image.to_rgba8();
        let (width, height) = rgba.dimensions();
        let data = Arc::from(rgba.into_raw().into_boxed_slice());
        Ok(Self::Sprite {
            data,
            width,
            height,
            color: Color {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
            alpha: 1.0,
            filter,
            generation: NEXT_SPRITE_GENERATION.fetch_add(1, Ordering::Relaxed),
            program: None,
        })
    }

    /// Creates a text shape from the font file at `path`.
    ///
    /// The file is read immediately and checked to be a TrueType/OpenType
    /// font, so a missing file or a non-font file fails here, not at render
    /// time. The bytes live behind an [`Arc`], so every text shape that
    /// loaded the same file shares one buffer, and the render pipeline can
    /// cache one glyph atlas per (font, size) pair.
    pub fn text(
        path: impl AsRef<Path>,
        text: impl Into<String>,
        size: f32,
    ) -> Result<Self, TextError> {
        let bytes = std::fs::read(path.as_ref()).map_err(TextError::Io)?;
        Self::text_bytes(&bytes, text, size)
    }

    /// Creates a text shape from font data already in memory, for example
    /// bytes embedded into the binary with `include_bytes!`.
    ///
    /// Like [`Shape::text`], the bytes are checked to be a
    /// TrueType/OpenType font up front and live behind an [`Arc`], but no
    /// file is read — this is how text shapes are created in environments
    /// without a file system, such as a web browser.
    pub fn text_bytes(font: &[u8], text: impl Into<String>, size: f32) -> Result<Self, TextError> {
        swash::FontRef::from_index(font, 0).ok_or(TextError::InvalidFont)?;
        Ok(Self::Text {
            text: text.into(),
            font: Arc::from(font),
            size,
            weight: 400.0,
            color: Color {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
            alpha: 1.0,
        })
    }

    /// Creates a text shape from a font buffer already shared behind an
    /// [`Arc`].
    ///
    /// This is [`Shape::text_bytes`] without the copy: the shape reuses
    /// the caller's buffer. Since the glyph atlas cache is keyed on that
    /// buffer's pointer, every shape built from one `Arc` shares one
    /// atlas — while two `text_bytes` calls on the same bytes mint two
    /// buffers, and two atlases. Per-frame text (an animated splash)
    /// should share the `Arc`: the pointer stays stable, so the atlas
    /// is built once instead of rebuilt and re-uploaded every frame.
    pub fn text_shared(
        font: Arc<[u8]>,
        text: impl Into<String>,
        size: f32,
    ) -> Result<Self, TextError> {
        swash::FontRef::from_index(&font, 0).ok_or(TextError::InvalidFont)?;
        Ok(Self::Text {
            text: text.into(),
            font,
            size,
            weight: 400.0,
            color: Color {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
            alpha: 1.0,
        })
    }

    /// Sets the text's `wght` variation-axis value: `400.0` is the Regular
    /// [`Shape::text`] and [`Shape::text_bytes`] default, `700.0` a full
    /// bold, and anything in between renders proportionally thicker —
    /// e.g. `520.0` reads as a Medium for a variable font like FiraCode
    /// Variable. A font without a weight axis ignores this; the value only
    /// changes how the glyphs are instantiated, not the layout size.
    pub fn with_weight(mut self, weight: f32) -> Self {
        if let Self::Text { weight: w, .. } = &mut self {
            *w = weight;
        }
        self
    }

    /// The sprite's texture size in pixels, `[width, height]`, if the
    /// shape is a sprite: `None` for circles, rectangles, backgrounds and
    /// texts.
    pub fn sprite_size(&self) -> Option<[f32; 2]> {
        match self {
            Self::Sprite { width, height, .. } => Some([*width as f32, *height as f32]),
            Self::TileMap { width, height, .. } => Some([*width as f32, *height as f32]),
            _ => None,
        }
    }

    /// Creates a tile-map shape from the PNG atlas at `path`, empty of
    /// tiles and sampled nearest-neighbor.
    ///
    /// The file is read and decoded to RGBA8 immediately, so a missing
    /// file or a non-PNG file fails here, not at render time; the pixels
    /// live behind an [`Arc`] and share the backend's texture cache with
    /// a [`Shape::Sprite`] of the same image. Place tiles with
    /// [`Shape::push_tile`] and cut the map to its window with
    /// [`Shape::set_clip`].
    pub fn tilemap(path: impl AsRef<Path>) -> Result<Self, SpriteError> {
        let bytes = std::fs::read(path.as_ref()).map_err(SpriteError::Io)?;
        Self::tilemap_bytes(&bytes)
    }

    /// Creates a tile-map shape from the PNG atlas in `bytes`: the
    /// in-memory counterpart of [`Shape::tilemap`], for atlases embedded
    /// with `include_bytes!` or held by an editor.
    pub fn tilemap_bytes(bytes: impl AsRef<[u8]>) -> Result<Self, SpriteError> {
        let image = image::load_from_memory(bytes.as_ref()).map_err(SpriteError::Decode)?;
        let rgba = image.to_rgba8();
        let (width, height) = rgba.dimensions();
        let data = Arc::from(rgba.into_raw().into_boxed_slice());
        Ok(Self::TileMap {
            data,
            width,
            height,
            generation: NEXT_SPRITE_GENERATION.fetch_add(1, Ordering::Relaxed),
            filter: SpriteFilter::Nearest,
            tiles: Vec::new(),
            clip: None,
            color: Color {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
        })
    }

    /// Adds one [`Tile`] placement to the map, on top of every tile
    /// already there. Does nothing to a shape that is not a tile map.
    pub fn push_tile(&mut self, tile: Tile) {
        if let Self::TileMap { tiles, .. } = self {
            tiles.push(tile);
        }
    }

    /// Replaces the map's placements. Does nothing to a shape that is
    /// not a tile map.
    pub fn set_tiles(&mut self, tiles: impl IntoIterator<Item = Tile>) {
        if let Self::TileMap { tiles: old, .. } = self {
            *old = tiles.into_iter().collect();
        }
    }

    /// Cuts the map to the view rectangle `[min_x, min_y, max_x, max_y]`
    /// in its local space (`None` for no clip). Does nothing to a shape
    /// that is not a tile map.
    pub fn set_clip(&mut self, clip: Option<[f32; 4]>) {
        if let Self::TileMap { clip: c, .. } = self {
            *c = clip;
        }
    }

    /// Sets the atlas's sampling filter. Does nothing to a shape that is
    /// not a tile map.
    pub fn set_filter(&mut self, filter: SpriteFilter) {
        if let Self::TileMap { filter: f, .. } = self {
            *f = filter;
        }
    }

    /// Attaches a game-authored WGSL program to a sprite: the source
    /// takes the sprite pipeline's place wherever this shape draws, and
    /// paints the sprite with whatever the program's fragment stage
    /// decides. Other shape kinds carry no program — the engine's own
    /// pipelines own them — and are returned unchanged.
    ///
    /// The contract is the sprite pipeline's, restated for a new author:
    ///
    /// - The source is compiled behind [`crate::LIGHTING_WGSL`], the same
    ///   prelude the engine's lit kinds are assembled from, so the
    ///   program may call `light_mix`, walk `occluded`, and bind the
    ///   frame's light and occluder fields by declaring them itself.
    /// - The entry points are the engine's: `vs_main` (the full-screen
    ///   triangle — every scene pipeline takes the scissor, not a vertex
    ///   buffer), and BOTH `fs_main` and `fs_emissive`, so the pipeline
    ///   fits the scene pass whether or not the machine carries the
    ///   emissive target.
    /// - The per-draw uniform is [`crate::Shape`]'s sprite block, 112
    ///   bytes, packed by the engine from the node's fields:
    ///   `to_local`/`translation`/`size` (the sprite's inverse world
    ///   transform and extent), `tint` (color × modulate), `alpha`,
    ///   `lit`, `lit_layers`, `uv_rect`, `emission` (node emission ×
    ///   modulate), and `glow`. The program declares its own mirror
    ///   struct at `@group(0) @binding(0)` and reads it as it likes —
    ///   `@binding(1)` is the sprite's texture, `@binding(2)` its
    ///   sampler, `@binding(3)`/`@binding(4)` the light and occluder
    ///   storage fields, exactly as `sprite.wgsl` binds them. Nothing
    ///   checks the mirror's layout at runtime: the byte layout lives in
    ///   `shaders/sprite.wgsl`'s `SpriteUniforms`, and a program whose
    ///   struct drifts from it reads garbage where the engine wrote
    ///   numbers.
    /// - Blend is the alpha blend, always: a program that wants to add
    ///   its light can do the arithmetic itself and write the result.
    ///
    /// A program that fails to compile is the driver's complaint to
    /// hear, mid-frame, from the device's error sink — which is why
    /// [`crate::validate_wgsl`] exists: a game's test suite can fail on
    /// a bad shader the way `cargo test` fails on the engine's own,
    /// against the same prelude the renderer will prepend.
    ///
    /// ```no_run
    /// # fn demo(shape: frost::Shape, source: String) {
    /// let dissolved = shape.with_program(source);
    /// # }
    /// ```
    #[must_use]
    pub fn with_program(mut self, source: impl Into<Arc<str>>) -> Self {
        if let Self::Sprite { program, .. } = &mut self {
            *program = Some(source.into());
        }
        self
    }

    /// Reads like the wrapper it anticipates: `target` painted by
    /// `program`. Today the two are the same mechanism — the program
    /// takes the sprite pipeline's place for this shape, in place — but
    /// the call shape stands where a screen-space `Shape::Effect` would
    /// one day render its target to an offscreen image first and let the
    /// program sample that finished picture. Such a wrapper needs
    /// per-target render targets and pass splits the one-pass scene
    /// carries no room for yet; when it arrives it will compile its
    /// programs exactly as these are compiled, over the same prelude and
    /// the same validation, which is why the seam lives here.
    #[must_use]
    pub fn painted_by(target: Self, program: impl Into<Arc<str>>) -> Self {
        target.with_program(program)
    }
}

/// An error while creating a [`Shape::Text`] from a font file.
#[derive(Debug)]
pub enum TextError {
    /// The file could not be read.
    Io(std::io::Error),
    /// The file was read but is not a TrueType/OpenType font.
    InvalidFont,
}

impl std::fmt::Display for TextError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "failed to read the font file: {err}"),
            Self::InvalidFont => write!(f, "the font file is not a TrueType/OpenType font"),
        }
    }
}

impl std::error::Error for TextError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::InvalidFont => None,
        }
    }
}

/// An error while creating a [`Shape::Sprite`] from a PNG file.
#[derive(Debug)]
pub enum SpriteError {
    /// The file could not be read.
    Io(std::io::Error),
    /// The file was read but is not a decodable PNG image.
    Decode(image::ImageError),
}

impl std::fmt::Display for SpriteError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Io(err) => write!(f, "failed to read the sprite file: {err}"),
            Self::Decode(err) => {
                write!(f, "failed to decode the sprite as a PNG image: {err}")
            }
        }
    }
}

impl std::error::Error for SpriteError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(err) => Some(err),
            Self::Decode(err) => Some(err),
        }
    }
}

/// A node in a scene tree.
///
/// Every node knows how to update itself and walk its own children, so a
/// whole tree updates with a single call to [`Node::visit`]: the visit runs
/// on every node in post-order (children before their parent), and each
/// node's [`Node::process`] runs once per visit.
pub trait Node {
    /// Per-frame update for this node, called once per [`Node::visit`],
    /// after the node's children have been visited. The default does
    /// nothing.
    fn process(&mut self) {}

    /// Visits this node and all of its descendants in post-order: each
    /// child's [`Node::visit`] runs before [`Node::process`] on this node,
    /// so a node's `process` sees its children already updated.
    fn visit(&mut self);
}

/// A node in a [`Scene`] tree.
///
/// A node is a [`Transform`], a `scale`, a `modulate`, and an `order`, plus
/// its own optional [`Shape`] and its children. The transform and the scale
/// are relative to the node's parent and apply to the node's own shape as
/// well as composing onto all descendants; the modulate multiplies into the
/// node's own shape color as well as composing onto all descendants'; the
/// order does the same for the subtree's draw order. A node without a shape
/// (`shape: None`) is a pure group or pivot node, and a node without
/// children is a leaf. A node literal can leave any fields out by writing
/// `..SceneNode::default()`: the omitted fields take the identity transform,
/// no scale, a white modulate, order `0.0`, no lighting, no occlusion,
/// no shape, and no children.
///
/// A [`SceneNode`] is a [`Node`]: its [`Node::visit`] walks its children and
/// then runs its (default, no-op) [`Node::process`], so a whole scene tree
/// updates with a single [`Scene::visit`].
#[derive(Clone, Debug)]
pub struct SceneNode {
    /// The transform from the parent's coordinate space to this node's,
    /// applied to the node's own shape and composed onto its children.
    pub transform: Transform,
    /// The node's non-uniform scale, applied in its own coordinate space
    /// before its transform, so it scales the node's shape and its whole
    /// subtree. `[1.0, 1.0]` (the default) is no scaling.
    pub scale: [f32; 2],
    /// The node's color modulation, multiplied channel by channel into the
    /// node's own shape color and composed (multiplied) onto its children's,
    /// so it tints the whole subtree, just like the scale and the transform:
    /// the node's `modulate` multiplies the modulation inherited from its
    /// ancestors, and that product is what its children inherit. White
    /// (all channels `1.0`, the default) leaves the colors unchanged.
    pub modulate: Color,
    /// The node's draw-order offset, added to the order inherited from its
    /// ancestors: the node's own shape draws at that total, and the same
    /// total is passed on to the children, so the order propagates down the
    /// whole subtree, just like the scale and the transform. `0.0` (the
    /// default) keeps the subtree at its inherited draw order.
    pub order: f32,
    /// Whether the node's own shape is lit by the frame's light field: when
    /// true, the shape's color is multiplied, per pixel, by the scene's
    /// ambient color plus the contribution of every light at that pixel,
    /// plus the node's own [`emission`](Self::emission).
    ///
    /// Unlike the transform, scale, modulate, and order, the flag does not
    /// propagate to the children: it applies to this node's own shape only,
    /// and each child keeps its own. `false` (the default) draws the shape
    /// unlit, exactly as before.
    pub lit: bool,
    /// The light layers this node's shape receives from when [`lit`](Self::lit)
    /// is set: a [`Light`] reaches this surface when the light's
    /// [`Light::layers`] and this set share at least one layer.
    ///
    /// The default — [`LightLayers::default()`], layer 0 — holds every
    /// light that names no layers, which is every light of every scene
    /// written before layers existed: leaving this field out of a node
    /// literal keeps the old behavior exactly. It rides the same
    /// node-only, non-propagating rule as `lit`.
    pub lit_layers: LightLayers,
    /// Whether the node's own shape occludes the frame's lights: when true
    /// the shape's silhouette blocks the light's path to every lit pixel
    /// behind it, casting a shadow — a rectangle as itself, a circle as the
    /// polygon cut from it. How
    /// sharp that shadow's edge is belongs to the light, not the occluder:
    /// a point-sized light ([`Light::penumbra`] `0.0`) is cut off on a hard
    /// per-pixel silhouette, while a light with a penumbra fades across an
    /// edge as wide as its disk. The shadow is evaluated in the frame's
    /// pixel space from the shape's full composed transform, so a
    /// rotated, scaled, or translated node casts the shadow of wherever it
    /// actually sits.
    ///
    /// Like [`SceneNode::lit`], the flag does not propagate to the
    /// children: it applies to this node's own shape only, and each child
    /// keeps its own. `false` (the default) casts no shadow. A node can
    /// carry both flags: a `lit: true, occludes: true` wall is lit on its
    /// own surface and still blocks the light from what lies behind it.
    pub occludes: bool,
    /// The light layers this node's shadow lands on: the occluder cuts a
    /// lit pixel's light only when this set and the *receiver*'s
    /// [`lit_layers`](Self::lit_layers) share at least one layer. The
    /// filter answers "whose surfaces do I shadow", not "which lights do I
    /// block" — the same wall can shadow the far wall behind it while the
    /// floor it stands on, lit from a different layer, sees no shadow
    /// from it at all.
    ///
    /// The default — [`LightLayers::default()`], layer 0 — shadows every
    /// receiver that names no layers, which is every receiver of every
    /// scene written before layers existed. It matters only when
    /// [`occludes`](Self::occludes) is set; declared occluders take their
    /// set through
    /// [`Canvas::occluder_layers`](crate::Canvas::occluder_layers).
    pub occludes_layers: LightLayers,
    /// The node's own emission: light the surface puts out, multiplied into
    /// its own color, so a surface stays visible in the dark and lights add
    /// on top of it rather than being its only source of brightness:
    /// `pixel = color * (ambient + lights + emission)` when [`SceneNode::lit`],
    /// `pixel = color * (1 + emission)` when it is not.
    ///
    /// Unlike [`SceneNode::lit`], this does not care whether any light
    /// reaches the surface — an emissive node glows in a scene with no
    /// lights at all, and glows the same amount inside a beam. Like `lit`,
    /// it applies to this node's own shape only, does not propagate to
    /// children, and is multiplied by the composed modulate. It lights the surface
    /// and nothing else: emission never spills onto neighbors, and nothing
    /// shadows it — to light the surroundings as well, attach a
    /// [`Shape::Light`] child node. Spill is **bloom**, an effect of the
    /// renderer over bright pixels rather than a property of a node: where
    /// the app asks for it ([`Config::bloom`](crate::Config::bloom)) it is
    /// this field's brightness, blurred at four widths and added back over
    /// the frame.
    ///
    /// The color's `a` scales the emission (rgb times `a`), so fading it in
    /// and out needs only an alpha ramp. Its channels are not clamped: over
    /// 1.0 asks for more light than a display can show, and the frame buffer
    /// keeps the overshoot rather than clipping it. That overshoot is exactly
    /// what bloom is built to find — its threshold is where "bright" starts
    /// handing some of its light to the pixels around it — so a lamp meant to
    /// light a room is written past 1.0 and one meant merely to be seen is
    /// not. With bloom off, a value past 1.0 still reaches the screen as
    /// plain white; [`tone_map`](crate::Config::tone_map) is what would say
    /// otherwise.
    ///
    /// The default — black — is off.
    pub emission: Color,
    /// The color of the light this node hands to the glow, when its halo should
    /// be a different color from the surface that throws it.
    ///
    /// [`emission`](Self::emission) does two jobs at once: it is the surface's
    /// own color in the frame, and it is what the bloom chain reads, so a lamp's
    /// core and its halo always arrive in the same hue. This field takes the
    /// second job — set, the emissive channel carries this color instead while
    /// the surface keeps its own, which is what lets a filament read white
    /// while the room it glows into goes amber.
    ///
    /// It is a multiplier on the same pixels emission multiplies: the channel
    /// holds `texture × glow`, so like emission it can take a channel away and
    /// cannot supply one the artwork does not have. And like emission it lights
    /// no neighbour; that takes a [`Shape::Light`](crate::Shape::Light) child.
    ///
    /// The default, `None`, is "the glow is the emission" — how every scene in
    /// the tree looks, and the only behaviour there was before this field. It
    /// rides the composed modulate as emission does, so fading a node fades its
    /// halo with it.
    pub glow: Option<Color>,
    /// Whether this node adds its light to the frame instead of covering what
    /// is behind it.
    ///
    /// The house rule is that rgb is radiance, alpha is coverage, and the blend
    /// does the rest — which is why an opaque wall drawn over a lamp takes the
    /// lamp's glow away along with its color. This field opts out of the second
    /// half: the alpha stops meaning how much of the world the sprite hides and
    /// starts meaning how much light it deposits, so a soft radial ramp lands as
    /// a lamp's glow instead of as a dark bell over the wall behind it. Both
    /// attachments take it that way — the frame's color and the emissive channel
    /// — which is what lets one authored sprite be a lamp's visible core and the
    /// thing the bloom chain blurs, from one asset and one tint.
    ///
    /// It belongs to artwork rather than to lighting, because a glow's falloff
    /// *is* its content: a blob the engine synthesizes has to invent one, and
    /// every engine that has shipped such a blob has heard about it. What this
    /// still does not do is light anything. A glow that illuminates the room
    /// needs a [`Shape::Light`](crate::Shape::Light) beside it and the two
    /// agreeing by hand, as they do everywhere else in the tree.
    ///
    /// Coverage adds as well, so a frame's alpha can pass 1.0 under a glow. The
    /// frame's alpha goes on to the window, where it clamps, and nothing inside
    /// the engine reads it.
    ///
    /// The default, `false`, is the ordinary covering draw. Sprites are the only
    /// kind that can ask: a shape is hard-edged, and an additive hard edge is a
    /// bright disc, which the emission channel already draws without needing a
    /// second rule.
    ///
    /// Nothing in the tree asks for it yet; the pipeline it selects is exercised
    /// by the `blend` tests and by nothing else so far. It went into the eyes hall
    /// as an authored glow and came back cleaner without one — an iris quad is
    /// already a drawn falloff, so a sprite underneath it is a second and wider
    /// light on the same source. What still wants it is a scene whose light is
    /// *not* the shape of whatever throws it: a lantern whose bulb is a dark
    /// fixture, say, or a window the room is lit through.
    pub additive: bool,
    /// The shape this node draws, in its own local space, if any.
    pub shape: Option<Shape>,
    /// The child nodes, positioned in this node's coordinate space.
    pub children: Vec<Box<SceneNode>>,
    /// Engine-managed owner marker: the [`crate::Diagnostics`] overlay sets
    /// this to `true` on the nodes it appends to the scene, and the marker
    /// rides onto every draw those nodes produce — glyph quads included —
    /// so the render loop can count the overlay's share of the frame's draw
    /// calls behind [`crate::Context::frame_diagnostic_draw_calls`].
    ///
    /// The field is public so that node literals keep working with
    /// `..Default::default()` from outside the crate; demo code should
    /// leave it at the default `false` and never set it itself.
    pub diagnostic: bool,
}

impl Default for SceneNode {
    /// An identity node: identity transform, no scale, a white modulate,
    /// order `0.0`, no lighting, no occlusion, no emission, no shape, and no
    /// children.
    fn default() -> Self {
        Self {
            transform: Transform::identity(),
            scale: [1.0, 1.0],
            modulate: WHITE,
            order: 0.0,
            lit: false,
            lit_layers: LightLayers::default(),
            occludes: false,
            occludes_layers: LightLayers::default(),
            emission: Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0,
            },
            glow: None,
            additive: false,
            shape: None,
            children: Vec::new(),
            diagnostic: false,
        }
    }
}

impl Node for SceneNode {
    /// Visits every child first, then updates this node itself (post-order),
    /// so [`Node::process`] sees the children already updated.
    fn visit(&mut self) {
        for child in &mut self.children {
            child.visit();
        }
        self.process();
    }
}

/// A rendering layer of a [`Scene`]: an independent draw group with its own
/// layer order, its own parallax speed, and its own root node.
///
/// Layers are hard draw partitions: every node of a layer is drawn after
/// (on top of) every node of a layer with a lower `order`, and before every
/// node of a layer with a higher `order`, no matter the nodes' local
/// `order` values. Within a layer the local ordering applies: lower local
/// order first, ties resolved in tree order.
///
/// The scene's root subtree is itself a group, at the implicit layer order
/// `0.0`, declared before every explicit layer: it is drawn before any
/// layer with the same or a higher order, and after any layer with a lower
/// order. A [`Shape::Background`] inside a layer keeps the background
/// behavior of every group: it never draws, it only contributes to the
/// frame's clear color.
#[derive(Clone, Debug, Default)]
pub struct Layer {
    /// The layer's order: higher order is closer to the camera (drawn
    /// later, on top of layers with a lower order).
    pub order: f32,
    /// The layer's parallax speed, as a multiple of the scene's camera
    /// motion: `1.0` (the default) follows the camera exactly, a higher
    /// speed makes the layer's contents move faster than the camera (the
    /// layer reads as closer), and a lower speed slower (farther away).
    /// Ignored when the scene has no [`Scene::camera`].
    pub speed: f32,
    /// The layer's repetition: `(repeat_x, repeat_y)`. `(0.0, 0.0)` (the
    /// default) draws the layer once. A non-zero component tiles the layer
    /// in both directions along that axis with a period of `abs(component)`
    /// pixels, so a moving camera can scroll through the layer infinitely:
    /// the layer is drawn once per copy of its content that can overlap
    /// the window — with a pure-translation camera, only the few copies
    /// the window's box and the content's span around it reach, because
    /// the minimum period is the window's width/height — and an object
    /// crossing a tile boundary is split into wrapping slices, the part
    /// past the boundary appearing on the opposite side of the window
    /// (each copy is re-used, with just a displacement, and clipped by the
    /// per-object scissor test, so no object is ever drawn twice). A
    /// non-zero period below the window size on that axis, or an object
    /// whose extent in a repeating axis exceeds its period (the same object
    /// would be drawn twice), aborts the program with an error. The
    /// component's sign is ignored.
    ///
    /// The repetition is a periodic extension of the layer's *current*
    /// content: each copy carries the content as it is now, displaced by
    /// its offset, so content that moves through the layer's space is
    /// re-tiled as it moves. An object a camera follows exactly stays at
    /// the window's center wherever it wanders: its wrapped copies sit one
    /// full period away, off the window's edge, because the period is at
    /// least the window size. A unique object (such as a player) is
    /// usually given its own non-repeating layer, because it is not part
    /// of the pattern that should tile.
    pub repeat: [f32; 2],
    /// The root node of the layer's tree, walked like a scene's root.
    pub root: SceneNode,
}

impl Layer {
    /// Creates a layer at the default order `0.0` and the default speed
    /// `1.0`, from its root node.
    pub fn new(root: SceneNode) -> Self {
        Self {
            order: 0.0,
            speed: 1.0,
            repeat: [0.0, 0.0],
            root,
        }
    }
}

/// The position of a node within a [`Scene`]: the group that contains the
/// node — the base group (the scene's root subtree) or one of the scene's
/// explicit [`Layer`]s — and the child-index path from that group's root
/// down to the node.
#[derive(Clone, Debug)]
pub struct NodePath {
    /// The group the node lives in: `None` for the base group (the scene's
    /// root subtree), `Some(i)` for the `i`th layer of the scene.
    pub group: Option<usize>,
    /// Child indices from the group's root down to the node: the first is
    /// the root's child, the second that child's child, and so on. An empty
    /// path names the group's root itself.
    pub children: Vec<usize>,
}

/// A tree of [`SceneNode`]s rooted at a single node, plus zero or more
/// rendering [`Layer`]s.
///
/// Pass a scene to [`run`](crate::run): every frame the user's
/// [`Process`](crate::Process) runs, then the tree is updated with
/// [`Scene::visit`] (every node's [`Node::process`], children before their
/// parent, on the root tree and on every layer's tree), and only then is
/// the scene drawn — so both can mutate it in place (via
/// [`Context::scene`](crate::Context::scene)) to animate it. Use
/// [`Canvas::draw_scene`](crate::Canvas::draw_scene) to draw additional
/// scenes.
///
/// The root subtree and the layers are separate draw groups: the root
/// subtree is the group at the implicit layer order `0.0`, and each layer
/// is its own group at its [`Layer::order`]. Groups are painted by ascending
/// order — higher order closer to the camera, drawn later, on top — and
/// within a group the local ordering applies.
///
/// If the scene has a [`Scene::camera`], the scene is drawn in that node's
/// coordinate space instead of the fixed window-centered user space: the
/// camera node stays at the window origin and the rest of the scene moves
/// and rotates around it, so a camera hung under a moving node follows it.
/// Each group renders from the camera scaled by its speed — the base group
/// at the full speed `1.0`, and each layer at its [`Layer::speed`].
#[derive(Clone, Debug)]
pub struct Scene {
    /// The root node; the scene is walked depth-first from here.
    pub root: SceneNode,
    /// The scene's rendering layers, in declaration order.
    pub layers: Vec<Layer>,
    /// The scene's camera node, if any: the scene is drawn in that node's
    /// coordinate space, so the node stays at the window origin while the
    /// rest of the scene moves around it. `None` (the default) draws the
    /// scene in the fixed window-centered user space.
    pub camera: Option<NodePath>,
    /// The ambient color of the scene's light field: the floor every
    /// [`SceneNode::lit`] receiver's color is multiplied by even where no
    /// light reaches. A lit shape's pixel color becomes
    /// `color * (ambient + light contributions)` — with a black ambient,
    /// pixels no light reaches are fully dark; with the default dim gray,
    /// they read as a darkened version of the shape's color.
    pub ambient: Color,
}

impl Scene {
    /// Creates a scene from its root node, with no layers, no camera, and
    /// the default dim-gray ambient.
    pub fn new(root: SceneNode) -> Self {
        Self {
            root,
            layers: Vec::new(),
            camera: None,
            ambient: AMBIENT,
        }
    }

    /// Updates the whole tree in one call: visits the root, which visits
    /// every child, and so on down the tree, so each node's
    /// [`Node::process`] runs exactly once, children before their parent —
    /// on the root tree and on every layer's tree.
    pub fn visit(&mut self) {
        self.root.visit();
        for layer in &mut self.layers {
            layer.root.visit();
        }
    }

    /// The world transform of the node at `path` — the node's own scale and
    /// transform composed over its ancestors', up to its group's root — or
    /// `None` if the path names no node (a missing layer or a missing
    /// child).
    pub(crate) fn world_at(&self, path: &NodePath) -> Option<Transform> {
        let root = match path.group {
            None => &self.root,
            Some(index) => &self.layers.get(index)?.root,
        };
        let mut node = root;
        let mut parent = Transform::identity();
        for &index in &path.children {
            let local = Transform::scale(node.scale).compose(&node.transform);
            parent = local.compose(&parent);
            node = node.children.get(index)?;
        }
        let local = Transform::scale(node.scale).compose(&node.transform);
        Some(local.compose(&parent))
    }

    /// The world transform of the scene's camera node, if the scene has a
    /// camera and its path names a node.
    pub(crate) fn camera_world(&self) -> Option<Transform> {
        self.camera.as_ref().and_then(|path| self.world_at(path))
    }
}

impl Default for Scene {
    /// An empty scene: an identity root node with no shape and no children,
    /// no layers, no camera, and the default dim-gray ambient, for apps that
    /// only use the immediate draw methods.
    fn default() -> Self {
        Self {
            root: SceneNode::default(),
            layers: Vec::new(),
            camera: None,
            ambient: AMBIENT,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_is_unlit_by_default() {
        assert!(!SceneNode::default().lit);
    }

    #[test]
    fn default_node_carries_no_light() {
        // The rest state a scene that recycles nodes between lives
        // returns them to is exactly this default: no emission to
        // lift the light mix, no glow to bloom, no additive blend to
        // brighten, and a white modulate that tints nothing. Games
        // define "washed" by these values — changing one silently
        // rewrites every recycled node's rest.
        let node = SceneNode::default();
        assert_eq!(
            node.emission,
            Color {
                r: 0.0,
                g: 0.0,
                b: 0.0,
                a: 0.0
            }
        );
        assert_eq!(node.glow, None);
        assert!(!node.additive);
        assert_eq!(node.modulate, WHITE);
    }

    #[test]
    fn scene_ambient_is_a_dim_gray_by_default() {
        let ambient = Scene::default().ambient;
        assert_eq!(
            ambient,
            Color {
                r: 0.3,
                g: 0.3,
                b: 0.3,
                a: 1.0
            }
        );
        // `Scene::new` carries the same default.
        assert_eq!(Scene::new(SceneNode::default()).ambient, ambient);
    }

    #[test]
    fn light_layers_default_to_layer_zero() {
        // The default is the one world every pre-layer scene shares —
        // and a layer number past 31 wraps back onto the same set, so
        // counting never escapes it.
        assert_eq!(LightLayers::default(), LightLayers::layer(0));
        assert_eq!(LightLayers::layer(32), LightLayers::layer(0));
        assert_eq!(LightLayers::default().bits(), 1);
        assert!(LightLayers::default().contains(0));
        assert!(!LightLayers::default().contains(1));
        // A node that names no layers receives and shadows on layer 0,
        // which is what keeps a pre-layer scene behaving exactly as it
        // did before layers existed.
        assert_eq!(SceneNode::default().lit_layers, LightLayers::default());
        assert_eq!(SceneNode::default().occludes_layers, LightLayers::default());
    }

    #[test]
    fn light_layers_union_intersects_and_contains() {
        let two_and_seven = LightLayers::layer(2) | LightLayers::layer(7);
        assert!(two_and_seven.intersects(LightLayers::layer(7)));
        assert!(!two_and_seven.intersects(LightLayers::layer(0)));
        assert!(two_and_seven.contains(2) && two_and_seven.contains(7));
        assert!(!two_and_seven.contains(3));
        assert_eq!(
            two_and_seven.union(LightLayers::layer(0)),
            LightLayers::from_bits(0b1000_0101)
        );
        // The raw bits round-trip through `from_bits`, and the two ends
        // of the filter behave as the renderer expects: the empty set
        // shares a layer with nothing, `ALL` with everything.
        assert_eq!(LightLayers::from_bits(two_and_seven.bits()), two_and_seven);
        assert!(!LightLayers::NONE.intersects(LightLayers::ALL));
        assert!(LightLayers::ALL.intersects(two_and_seven));
    }
}
