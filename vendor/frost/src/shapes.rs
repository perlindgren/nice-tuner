//! The shapes an author drew in the picture, read out of a sprite's `.ron`
//! sidecar: the `shapes` key's entries, in one pass, as both views of them.
//!
//! The vocabulary is one flat list, one entry per shape, with the kind as
//! a field and the three consumers' roles as flags on the same entry — a
//! wall is solid *and* occluding, and one role per entry would make the
//! author write its geometry twice:
//!
//! ```text
//! shapes: [
//!     (kind: "rect", at: [16.0, 8.0], size: [24.0, 16.0], solid: true),
//!     (kind: "circle", at: [16.0, 2.0], radius: 6.0, occludes: true),
//!     (kind: "poly", at: [0.0, 0.0], points: [[0.0, 0.0], [12.0, 4.0], [2.0, 14.0]], occludes: true),
//!     // one wall, both jobs: bouncy, and it throws a shadow
//!     (kind: "rect", at: [16.0, 30.0], size: [32.0, 8.0], solid: (bounce: 0.2), occludes: true),
//! ]
//! ```
//!
//! `kind` is `rect` (needs `size: [w, h]`, full pixels), `circle` (needs
//! `radius`) or `poly` (needs `points`, an array of `[x, y]` corners);
//! `capsule` is written into the grammar but refused for now, by name,
//! because there is no capsule shape anywhere to make of it. `solid` is
//! `true` or `(bounce: <number>)`, and a polygon is honoured as an
//! occluder only — pushing two polygons apart needs a separating axis and
//! a contact normal worth the name, which is a step of its own.
//!
//! The read serves two consumers, so it has two arms:
//! [`read_shapes`] hands back what the author drew, in the texture's own
//! pixels, plus every error — the arm a drawing tool uses, drawing every
//! entry whose geometry it could read while the errors speak beside it
//! (`sprite_util` overlays the shapes over the sprite exactly this way).
//! [`authored_shapes`] is the arm a [`Baker`](crate::Baker) feeds: the
//! same single pass, but all-or-nothing — one defect is the whole
//! document refused — and its entries stand in the node's space, every
//! role of an entry a [`Feature`] of its own.
//!
//! [`bake_sidecar`] is the load-time seam that puts the two together:
//! it finds `<name>.ron` beside the `<name>.png` a game is about to
//! draw, parses it, reads it and bakes it — the sprite's authored set
//! once, in the sprite's space, for the game to declare and collide per
//! frame through [`Baked`]'s own exits. A mis-authored sidecar fails the
//! load loudly, because a game quietly missing its collision is the bug
//! this whole module exists to make impossible.
//!
//! This is a vocabulary for a document designed for additions. The
//! sidecar round-trips through `sprite_util` with its comments and its
//! unknown keys intact — the tool interprets `atlas` and two-number
//! structs and preserves everything else — so `shapes:` joins `flower_anchors`
//! and friends as just another key the tool will not touch, and the next
//! key can arrive the same way. Two rules get rediscovered expensively
//! otherwise, so they are stated here where the reader can also enforce
//! them:
//!
//! * **Positions are arrays.** `at: [16.0, 8.0]`, never `at: (16.0, 8.0)`:
//!   `sprite_util` reads any struct with exactly two numeric fields as a
//!   draggable position spot, so a tuple is a handle the tool lets you
//!   grab and a shape whose numbers move under the author's hands. The
//!   reader refuses a tuple with exactly this explanation.
//! * **Errors are returned, never logged.** The sidecar's standing policy
//!   — an unparseable file: the sprite simply loads — is right for an
//!   atlas grid and wrong for hit boxes: a sprite that silently loads with
//!   no collision is a bug nobody can see by playing. Both arms hand the
//!   caller every error it found, all of them in one pass, and the caller
//!   decides whether the sprite loads; the reader never logs-and-continues.
//!
//! Authored numbers are in the texture's pixels: x right, y *down*, origin
//! at the image's top-left. The engine's footprints stand centred on the
//! node in user space, which is y-*up*, one unit per pixel — so the flip
//! happens here, in [`pixel_to_node`] and nowhere else, the way
//! `docs/authored-shapes.md` records it must. An authored shape that
//! reaches a baker takes the flip once, on its way out of this module;
//! an authored shape that reaches the author's eyes never takes it at
//! all. A polygon is convex or refused: [`Convex::new`] rejects a concave
//! or self-crossing outline, and this reader passes that refusal straight
//! through as its own message — nothing is decomposed, and nothing is
//! quietly hulled. Convexity survives a reflection, so it is decided in
//! the pixels it was authored in, before the flip — a drawing tool and a
//! baker must agree on which outlines are the same outline.

use crate::bake::{Baked, Baker, Feature, Footprint, POLY_NOT_PUSHED, Placement, Rejection};
use crate::collision::{Convex, MAX_PLANES};
use crate::ron::{Kind, Val};
use std::path::Path;

/// Every error one reading of a sidecar produced, at once.
///
/// An author fixing a sidecar wants the whole list, not the first
/// complaint and a rebuild per round trip, so one reading of the
/// document reports every defect it finds in a single pass. The crate's
/// standing policy for sidecars is that a bad file still loads the
/// sprite; that is the right call for an atlas grid and the wrong one for
/// a hit box, so this travels to the caller, who owns the decision — a
/// missing hit box is invisible in play, and this must never be decided
/// by a quiet log line.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
#[must_use = "an unreadable shape is authored geometry that does nothing; name the errors rather than dropping them"]
pub struct ShapeErrors {
    /// Every complaint, in document order: whole errors first, then each
    /// entry's own.
    pub errors: Vec<ShapeError>,
}

impl ShapeErrors {
    /// How many errors the reading found.
    pub fn len(&self) -> usize {
        self.errors.len()
    }

    /// Whether the reading found nothing to complain about.
    pub fn is_empty(&self) -> bool {
        self.errors.is_empty()
    }
}

impl std::fmt::Display for ShapeErrors {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for (i, err) in self.errors.iter().enumerate() {
            if i > 0 {
                f.write_str("\n")?;
            }
            write!(f, "{err}")?;
        }
        Ok(())
    }
}

impl std::error::Error for ShapeErrors {}

/// One thing about the sidecar's `shapes` the reader could not make a
/// feature from, in the voice of [`crate::bake::Rejection`]: a sentence
/// that names where it came from, which an author can act on and a test
/// can name.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShapeError {
    /// Which entry said it: `None` is a defect of the `shapes` key
    /// itself, `Some(i)` the `i`-th entry of the list. The RON tree is a
    /// preservation tree and carries no source lines, so the entry's
    /// position is the finest address there is to give.
    pub entry: Option<usize>,
    /// The complaint as a sentence, prefixed with where it came from.
    pub message: String,
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.message)
    }
}

/// One entry of the `shapes` list, as its author drew it: the geometry
/// in the texture's own pixels — x right, y down from the image's
/// top-left — with the roles the entry asked for.
///
/// This is what the author sees and what a tool draws; [`authored_shapes`]
/// turns the same entries into the engine's [`Feature`]s, which is where
/// the pixel-to-node flip happens. An entry stands here whenever its
/// *geometry* read — kind, place and shape all understood — even when
/// something else about it failed: a tool showing the author their own
/// sidecar wants the outlines it could read on screen *while* it reports
/// what it refused. An entry whose geometry it could not read is absent,
/// and its errors say why.
#[derive(Clone, Debug, PartialEq)]
pub struct Authored {
    /// The entry's seat in the `shapes` list — the same address the
    /// errors of [`ShapeError::entry`] give.
    pub entry: usize,
    /// The centre, in the image's pixels.
    pub at: [f32; 2],
    /// What the entry is, in the image's pixels.
    pub shape: AuthoredShape,
    /// The three questions, exactly as answered; a bad answer is off
    /// *and* in the error list. `bounce` belongs to `solid`.
    pub solid: bool,
    pub bounce: f32,
    pub occludes: bool,
    pub hit: bool,
}

/// The geometry of one authored entry, in the image's pixels.
#[derive(Clone, Debug, PartialEq)]
pub enum AuthoredShape {
    /// Centred on the entry's `at`; full width and height, not
    /// half-extents — half-extents are the engine's convention, not the
    /// author's.
    Rect { size: [f32; 2] },
    /// Centred on the entry's `at`.
    Circle { radius: f32 },
    /// Corners *relative to* the entry's `at`, closed; convex (the read
    /// refused anything else) and no more than [`MAX_PLANES`] of them.
    Poly { points: Vec<[f32; 2]> },
}

/// Read the `shapes` key of a parsed sidecar as authored: every entry
/// whose geometry read, in the texture's pixels, and every error the
/// document holds — at once, in one pass over `root`, the sidecar's
/// parsed tree.
///
/// The two halves answer different questions and both travel back
/// together: a drawing tool paints the list it got and reads the errors
/// aloud (which is what `sprite_util` does — shapes on the sprite, the
/// first refusal on the status line), while a baker wants
/// [`authored_shapes`], which treats any error as a refusal of the whole
/// document. A geometry that read but whose *roles* failed — a polygon
/// asked to push, a flag written as a number — draws all the same: the
/// picture shows what was drawn, the errors say what was refused.
///
/// No `shapes` key, or an empty list, is not an error and returns
/// nothing — which is where a caller's derive-from-bounds default takes
/// over: a sprite or tile with no authored shapes derives them from the
/// bounds it already has. The key is authoritative only once it is
/// present, because the moment an author draws a shape the picture has
/// stopped describing it.
pub fn read_shapes(root: &Val) -> (Vec<Authored>, ShapeErrors) {
    let mut errors = Vec::new();
    let Some(shapes) = top_level_key(root, "shapes") else {
        return (Vec::new(), ShapeErrors::default());
    };
    let Val::Seq { items, .. } = shapes else {
        errors.push(problem(
            None,
            "the value of `shapes` should be a list of entries like (kind: \"rect\", ..)",
        ));
        return (Vec::new(), ShapeErrors { errors });
    };
    let mut authored = Vec::new();
    for (index, item) in items.iter().enumerate() {
        authored.extend(read_entry(index, &item.val, &mut errors));
    }
    (authored, ShapeErrors { errors })
}

/// Read the `shapes` key of a parsed sidecar into the features a
/// [`Baker`](crate::Baker) takes, with the authored pixels turned into the node's
/// space.
///
/// This is [`read_shapes`] and the flip: one pass reads every entry, and
/// any error at all refuses the whole document — a missing hit box is
/// invisible in play, so the caller gets `Err` with *every* defect in it,
/// never a partial bake that looks fine. `size_px` is the image the
/// shapes were drawn on, `[width, height]` in pixels: authored positions
/// are pixels of that image with y down from its top-left, and the
/// conversion lands each shape centred on the node, y up. Each role an
/// entry asked for becomes its own [`Feature`], all of them over the one
/// footprint.
pub fn authored_shapes(root: &Val, size_px: [f32; 2]) -> Result<Vec<Feature>, ShapeErrors> {
    let (authored, errs) = read_shapes(root);
    if !errs.is_empty() {
        return Err(errs);
    }
    let mut errors = Vec::new();
    let mut features = Vec::new();
    for entry in &authored {
        if let Some(shape) = footprint(entry, size_px, &mut errors) {
            if entry.solid {
                features.push(Feature::solid(shape, entry.bounce));
            }
            if entry.occludes {
                features.push(Feature::occluding(shape));
            }
            if entry.hit {
                features.push(Feature::hit(shape));
            }
        }
    }
    if errors.is_empty() {
        Ok(features)
    } else {
        Err(ShapeErrors { errors })
    }
}

/// Why a sprite's authored set could not be loaded and baked.
///
/// Every variant is a voice, not a log line: the loader hands the whole
/// complaint to its caller, who decides whether the sprite still loads.
/// A game that starts with silently missing collision is the bug this
/// exists to prevent; a missing sidecar is not a complaint at all —
/// [`bake_sidecar`] answers it with an empty [`Baked`].
#[derive(Debug)]
pub enum SidecarError {
    /// The sidecar exists but could not be read off the disk.
    Io(std::io::Error),
    /// The PNG's own size could not be read — the sidecar's numbers mean
    /// nothing without the image they were authored against.
    Png(image::ImageError),
    /// The sidecar is not the RON subset the parser accepts. The tool's
    /// own policy is *unparseable: the sprite simply loads* — right for
    /// a tree the author is mid-edit on — but at the load seam the tree
    /// carries hit boxes, so the failure travels on.
    Ron(String),
    /// The `shapes` key was readable and refused: the whole list of
    /// [`ShapeErrors`], in one pass, exactly as [`authored_shapes`]
    /// reports them.
    Shapes(ShapeErrors),
    /// The reader handed the baker a footprint the baker itself refused.
    /// It should be unreachable — the reader's geometry is finite and
    /// non-empty by construction, which is the baker's whole refusal
    /// list — but a refusal that *did* happen is reported, never
    /// dropped on the floor: a quietly missing hit box is the crime.
    Bake(Vec<Rejection>),
}

impl std::fmt::Display for SidecarError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SidecarError::Io(err) => write!(f, "the sidecar could not be read: {err}"),
            SidecarError::Png(err) => write!(f, "the sprite's size could not be read: {err}"),
            SidecarError::Ron(err) => write!(f, "the sidecar does not parse: {err}"),
            SidecarError::Shapes(errs) => write!(f, "{errs}"),
            SidecarError::Bake(rejected) => {
                for (i, r) in rejected.iter().enumerate() {
                    if i > 0 {
                        f.write_str("\n")?;
                    }
                    write!(f, "feature {}: {:?}", r.index, r.why)?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for SidecarError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            SidecarError::Io(err) => Some(err),
            SidecarError::Png(err) => Some(err),
            _ => None,
        }
    }
}

/// The load-time seam: a sprite's authored set, loaded from its files —
/// the `<name>.ron` found beside the `<name>.png`, parsed, read, and
/// baked once into the sprite's own space.
///
/// The [`Baked`] answer rides every instance: declare its occluders and
/// collide against its solids wherever a node or a tile stands, and one
/// brick sidecar walls a whole hall (see the `authored_hall` example).
/// A sprite *without* a sidecar is the seam's most common answer and not
/// an error: an empty [`Baked`], where a caller that wants bounds-derived
/// shapes derives them — which role those bounds play is the caller's
/// business, which is why the seam does not guess.
pub fn bake_sidecar(png: impl AsRef<Path>) -> Result<Baked, SidecarError> {
    let png = png.as_ref();
    let (w, h) = image::image_dimensions(png).map_err(SidecarError::Png)?;
    let sidecar = png.with_extension("ron");
    let text = match std::fs::read_to_string(&sidecar) {
        Ok(text) => text,
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => return Ok(Baked::default()),
        Err(err) => return Err(SidecarError::Io(err)),
    };
    bake_sidecar_text([w as f32, h as f32], &text)
}

/// [`bake_sidecar`] with its inputs already in hand: the image's size,
/// which the sidecar's pixels are authored against, and the sidecar's
/// text. The wasm games reach the seam this way — both files come out of
/// the binary through `include_bytes!` and `include_str!`, where there is
/// no directory beside a PNG to look in.
pub fn bake_sidecar_text(size_px: [f32; 2], sidecar: &str) -> Result<Baked, SidecarError> {
    let doc = crate::ron::parse_doc(sidecar).map_err(SidecarError::Ron)?;
    let features = authored_shapes(&doc.root, size_px).map_err(SidecarError::Shapes)?;
    if features.is_empty() {
        return Ok(Baked::default());
    }
    let mut baker = Baker::new();
    let rejected = baker.add(Placement::IDENTITY, &features);
    if rejected.is_empty() {
        Ok(baker.finish())
    } else {
        Err(SidecarError::Bake(rejected))
    }
}

/// One authored entry's footprint in the node's space: the geometry
/// flipped through the one conversion, full sizes become half-extents,
/// and the polygon's corners travel through the same single conversion
/// as every other authored position — the convexity that made the read
/// accept the outline was decided in the pixels and survives the flip
/// (a reflection preserves straight lines and betweenness, so a convex
/// outline stays convex), which is what lets a refusal-free read reach
/// here; if it somehow did not, the conversion says so rather than
/// baking a shape nobody authored.
fn footprint(
    entry: &Authored,
    size_px: [f32; 2],
    errors: &mut Vec<ShapeError>,
) -> Option<Footprint> {
    let center = pixel_to_node(entry.at, size_px);
    match &entry.shape {
        AuthoredShape::Rect { size } => {
            Some(Footprint::rect(center, [size[0] / 2.0, size[1] / 2.0]))
        }
        AuthoredShape::Circle { radius } => Some(Footprint::disc(center, *radius)),
        AuthoredShape::Poly { points } => {
            let moved: Vec<[f32; 2]> = points
                .iter()
                .map(|p| pixel_to_node([entry.at[0] + p[0], entry.at[1] + p[1]], size_px))
                .collect();
            match Convex::new(&moved) {
                Some(outline) => Some(Footprint::poly(outline)),
                None => {
                    errors.push(problem(
                        Some(entry.entry),
                        "that outline is concave or self-crossing: a polygon is convex or refused — it is not decomposed, and not hulled",
                    ));
                    None
                }
            }
        }
    }
}

/// Where one point authored in the texture's pixels stands in the node's
/// space: pixel `(x, y)` of a `w × h` image becomes
/// `(x - w / 2, h / 2 - y)` — centred on the node, one unit per pixel,
/// and y flipped from the image's down to the engine's up.
///
/// This is the whole conversion and the only place the flip lives, so it
/// cannot be applied for one shape and skipped for another; polygon
/// corners go through it like every other position. The flip reverses a
/// winding — a corner order that runs counter-clockwise in pixels runs
/// clockwise in node space — and nothing here fixes that, because
/// [`Convex::new`] infers the winding from the signed area instead of
/// assuming it.
fn pixel_to_node(px: [f32; 2], size_px: [f32; 2]) -> [f32; 2] {
    [px[0] - size_px[0] / 2.0, size_px[1] / 2.0 - px[1]]
}

/// The value of `name` under the document's root, whether the root is a
/// struct (the shape a sidecar has) or a map, whose keys the parser
/// renders to display strings — a quoted key arrives quoted.
fn top_level_key<'v>(root: &'v Val, name: &str) -> Option<&'v Val> {
    match root {
        Val::Struct { fields, .. } => fields
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, it)| &it.val),
        Val::Map { entries, .. } => entries
            .iter()
            .find(|(k, _)| k == name || k == &format!("\"{name}\""))
            .map(|(_, it)| &it.val),
        _ => None,
    }
}

/// Read one entry of the list, collecting every complaint it earns into
/// `errors` and returning it whenever its *geometry* read. The roles
/// travel with it as answered — a bad answer is off and in the error
/// list — because the picture and the bake ask different questions of a
/// broken entry: the picture shows the outline it can still show, the
/// bake receives no features from a document holding any error at all.
fn read_entry(index: usize, entry: &Val, errors: &mut Vec<ShapeError>) -> Option<Authored> {
    let Val::Struct { fields, .. } = entry else {
        errors.push(problem(
            Some(index),
            "an entry is a struct like (kind: \"rect\", at: [16.0, 8.0], size: [24.0, 16.0], solid: true)",
        ));
        return None;
    };
    let field = |key: &str| fields.iter().find(|(k, _)| k == key).map(|(_, it)| &it.val);

    // A closed vocabulary: what the tool preserves, this reader must at
    // least notice it was not asked for.
    for (name, _) in fields {
        if name.is_empty() {
            errors.push(problem(
                Some(index),
                "every field of an entry is named, `key: value`",
            ));
        } else if !KNOWN_KEYS.contains(&name.as_str()) {
            errors.push(problem(
                Some(index),
                format!(
                    "unknown key `{name}`: an entry knows kind, at, size, radius, points, solid, occludes and hit, and nothing else"
                ),
            ));
        }
    }

    let kind = match field("kind") {
        Some(v) => match word(v) {
            Some(k @ ("rect" | "circle" | "poly" | "capsule")) => Some(k),
            Some(other) => {
                errors.push(problem(
                    Some(index),
                    format!("kind `{other}` is not a shape: the vocabulary is rect, circle, poly (capsule is not implemented yet)"),
                ));
                None
            }
            None => {
                errors.push(problem(
                    Some(index),
                    "kind is a quoted word: \"rect\", \"circle\" or \"poly\"",
                ));
                None
            }
        },
        None => {
            errors.push(problem(
                Some(index),
                "no kind: an entry says what it is, \"rect\", \"circle\" or \"poly\"",
            ));
            None
        }
    };
    if kind == Some("capsule") {
        errors.push(problem(
            Some(index),
            "kind `capsule` is refused: capsules are in the vocabulary but implemented nowhere yet, so there is no shape to make of the entry",
        ));
        return None;
    }

    let mut at_px = None;
    match field("at") {
        None => errors.push(problem(
            Some(index),
            "no `at`: a shape stands at its centre, [x, y] in the image's pixels",
        )),
        Some(v) => {
            if let Some(p) = pair(v) {
                at_px = Some(p);
            } else if is_two_number_struct(v) {
                errors.push(problem(
                    Some(index),
                    "`at` is a struct with two numbers, which sprite_util reads as a draggable position spot and lets you move: a shape that slides under the author's hands is the bug, so authored positions are arrays, [x, y], and stay where they were written",
                ));
            } else {
                errors.push(problem(
                    Some(index),
                    "`at` is an array of two numbers, [x, y], in the image's pixels",
                ));
            }
        }
    }

    // The geometry, per kind. Each kind's own key or nothing else: the
    // hints name what was found when the document confused one for the
    // other, because "a circle needs a radius, found `size`" is the
    // error that teaches the vocabulary.
    let mut shape = None;
    match (kind, at_px) {
        (Some("rect"), Some(at)) => match field("size").and_then(pair) {
            Some(size) => shape = Some(ShapeAt::Rect { at, size }),
            None if field("size").is_some() => errors.push(problem(
                Some(index),
                "`size` is an array of two numbers, [width, height] in full pixels; half-extents are the engine's, not the author's",
            )),
            None => errors.push(problem(
                Some(index),
                "a rect needs `size`: [width, height] in full pixels",
            )),
        },
        (Some("circle"), Some(at)) => match field("radius").and_then(number) {
            Some(radius) => shape = Some(ShapeAt::Circle { at, radius }),
            None if field("radius").is_some() => errors.push(problem(
                Some(index),
                "`radius` is a single number, in pixels",
            )),
            None if field("size").is_some() => errors.push(problem(
                Some(index),
                "a circle needs `radius`: found `size` — the round shape wants one number",
            )),
            None => errors.push(problem(
                Some(index),
                "a circle needs `radius`: one number, in pixels",
            )),
        },
        (Some("poly"), Some(at)) => shape = read_poly(index, at, field("points"), errors),
        // `kind` and `at` each reported their own problem already.
        (Some(_), None) | (None, _) => {}
        // `capsule` returned above; nothing else is a known kind.
        (Some(_), Some(_)) => {}
    }

    // The three questions, on the same entry: they are what a consumer
    // asks of a shape, not properties of the drawing.
    let mut bounce = 0.0;
    let mut solid = false;
    match field("solid") {
        None => {}
        Some(Val::Struct { fields: sf, .. }) => {
            let bounces = sf.len() == 1 && sf[0].0 == "bounce";
            match bounces.then(|| number(&sf[0].1.val)).flatten() {
                Some(b) => {
                    bounce = b;
                    solid = true;
                }
                None => errors.push(problem(
                    Some(index),
                    "`solid` is `true`, `false`, or `(bounce: <number>)` — a material with nothing in it is no material",
                )),
            }
        }
        Some(v) => match boolean(v) {
            Some(b) => solid = b,
            None => errors.push(problem(
                Some(index),
                "`solid` is `true`, `false`, or `(bounce: <number>)`",
            )),
        },
    }
    let occludes = flag(field("occludes"), "occludes", index, errors);
    let mut hit = flag(field("hit"), "hit", index, errors);

    // A polygon occludes; the roles that push bodies apart get the bake's
    // own reason, quoted from the same constant so both ends say it
    // alike. The entry's other roles still stand: a wall authored solid
    // *and* occluding keeps the shadow while the author learns the rest.
    let mut refused = false;
    if kind == Some("poly") {
        if solid {
            errors.push(problem(Some(index), POLY_NOT_PUSHED));
            solid = false;
            refused = true;
        }
        if hit {
            errors.push(problem(Some(index), POLY_NOT_PUSHED));
            hit = false;
            refused = true;
        }
    }

    let wanted_any = solid || occludes || hit;
    if !wanted_any && !refused {
        errors.push(problem(
            Some(index),
            "`solid`, `occludes` and `hit` are all off: an authored shape nobody wants is a mistake, not a comment",
        ));
    }

    // The geometry draws whatever the roles did: the outline the author
    // drew is readable, and hiding it because a flag was written wrong
    // would hide the very picture the error line is about.
    Some(match shape? {
        ShapeAt::Rect { at, size } => Authored {
            entry: index,
            at,
            shape: AuthoredShape::Rect { size },
            solid,
            bounce,
            occludes,
            hit,
        },
        ShapeAt::Circle { at, radius } => Authored {
            entry: index,
            at,
            shape: AuthoredShape::Circle { radius },
            solid,
            bounce,
            occludes,
            hit,
        },
        ShapeAt::Poly { at, points } => Authored {
            entry: index,
            at,
            shape: AuthoredShape::Poly { points },
            solid,
            bounce,
            occludes,
            hit,
        },
    })
}

/// A geometry read together with the place it stands on: `read_entry`'s
/// private pair, split into [`Authored`] once the roles have been read
/// too. Public life knows only [`AuthoredShape`], which is the geometry
/// already standing on its entry's `at`.
enum ShapeAt {
    Rect { at: [f32; 2], size: [f32; 2] },
    Circle { at: [f32; 2], radius: f32 },
    Poly { at: [f32; 2], points: Vec<[f32; 2]> },
}

/// The polygon's outline, in the pixels it was authored in: the corners
/// absolute against the entry's `at`, handed to [`Convex::new`] — whose
/// refusal is reported as the concavity it means rather than repaired.
/// The refusal reads the same here as it did after the flip because a
/// reflection preserves convexity: the tool that draws the outline and
/// the baker that ships it must not disagree about which outlines exist.
fn read_poly(
    index: usize,
    at: [f32; 2],
    points: Option<&Val>,
    errors: &mut Vec<ShapeError>,
) -> Option<ShapeAt> {
    let Some(Val::Seq { items, .. }) = points else {
        errors.push(problem(
            Some(index),
            "a poly needs `points`: an array of [x, y] corners in the image's pixels, around the entry's `at`",
        ));
        return None;
    };
    let mut corners: Vec<[f32; 2]> = Vec::with_capacity(items.len());
    for item in items {
        let Some(p) = pair(&item.val) else {
            errors.push(problem(
                Some(index),
                "a poly's corners are arrays [x, y] of two numbers",
            ));
            return None;
        };
        corners.push(p);
    }
    if corners.len() < 3 {
        errors.push(problem(
            Some(index),
            format!(
                "a poly needs at least three corners; this outline has {}",
                corners.len()
            ),
        ));
        return None;
    }
    if corners.len() > MAX_PLANES {
        errors.push(problem(
            Some(index),
            format!(
                "that outline has {} corners and the cap is {MAX_PLANES}: the GPU pays per edge, per shadow ray, per penumbra tap, so a wide shape is the author's to simplify",
                corners.len()
            ),
        ));
        return None;
    }
    // Convexity is decided on the absolute corners — adding `at` first
    // changes nothing about the answer, it is a translation — and the
    // entry keeps the corners *relative* to `at`, the way they were
    // authored: drawing the outline and baking it each add the centre
    // when and where their own space asks for it.
    let absolute: Vec<[f32; 2]> = corners
        .iter()
        .map(|p| [at[0] + p[0], at[1] + p[1]])
        .collect();
    match Convex::new(&absolute) {
        Some(_) => Some(ShapeAt::Poly {
            at,
            points: corners,
        }),
        None => {
            errors.push(problem(
                Some(index),
                "that outline is concave or self-crossing: a polygon is convex or refused — it is not decomposed, and not hulled",
            ));
            None
        }
    }
}

/// A `true` / `false` role flag: absent is off, and anything that is not
/// a bool says so rather than being guessed at.
fn flag(value: Option<&Val>, name: &str, index: usize, errors: &mut Vec<ShapeError>) -> bool {
    match value {
        None => false,
        Some(v) => match boolean(v) {
            Some(b) => b,
            None => {
                errors.push(problem(
                    Some(index),
                    format!("`{name}` is `true` or `false`"),
                ));
                false
            }
        },
    }
}

/// A number, from the text the parser kept verbatim — underscores are
/// digit separators and belong out of the way before the parse.
fn number(v: &Val) -> Option<f32> {
    match v {
        Val::Atom(t, Kind::Num) => t.replace('_', "").parse().ok(),
        _ => None,
    }
}

/// A bool, by kind: the parser calls `true` and `false` bools, and a
/// `1.0` where a flag belongs stays the error it is.
fn boolean(v: &Val) -> Option<bool> {
    match v {
        Val::Atom(t, Kind::Bool) => match t.as_str() {
            "true" => Some(true),
            "false" => Some(false),
            _ => None,
        },
        _ => None,
    }
}

/// A quoted word, quotes off — the text arrives verbatim from the source.
fn word(v: &Val) -> Option<&str> {
    match v {
        Val::Atom(t, Kind::Str) => t.strip_prefix('"')?.strip_suffix('"'),
        _ => None,
    }
}

/// An array of exactly two numbers: every position the vocabulary takes.
fn pair(v: &Val) -> Option<[f32; 2]> {
    let Val::Seq { items, .. } = v else {
        return None;
    };
    if items.len() != 2 {
        return None;
    }
    Some([number(&items[0].val)?, number(&items[1].val)?])
}

/// Whether this is what `sprite_util` would grab as a draggable position
/// spot: a struct with exactly two numeric-atom fields, named or not.
fn is_two_number_struct(v: &Val) -> bool {
    matches!(v, Val::Struct { fields, .. }
        if fields.len() == 2 && fields.iter().all(|(_, it)| matches!(&it.val, Val::Atom(_, Kind::Num))))
}

/// One complaint, in the voice of [`crate::bake::Rejection`], addressed
/// to the whole `shapes` key or to one entry of it.
fn problem(entry: Option<usize>, body: impl Into<String>) -> ShapeError {
    let body = body.into();
    let message = match entry {
        Some(i) => format!("shapes entry {i}: {body}"),
        None => format!("shapes: {body}"),
    };
    ShapeError { entry, message }
}

/// The names an entry may use, and nothing else: the vocabulary *inside*
/// an entry is closed, because the tool preserves faithfully whatever it
/// does not know, so a `solidss:` key would be kept forever and never
/// read — the mistake has to be caught while it can still be fixed.
const KNOWN_KEYS: [&str; 8] = [
    "kind", "at", "size", "radius", "points", "solid", "occludes", "hit",
];

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bake::{Baker, Placement};
    use crate::canvas::Canvas;
    use crate::collision::{Circle, Collider, DISC_SIDES};
    use crate::objects::Transform;
    use crate::ron;

    /// Read a sidecar written as text: every test starts where the tool
    /// hands the engine off, with a parsed `Doc` and a pixel size.
    fn read(text: &str, size_px: [f32; 2]) -> Result<Vec<Feature>, ShapeErrors> {
        let doc = ron::parse_doc(text).expect("the test's sidecar parses");
        authored_shapes(&doc.root, size_px)
    }

    /// The same text through the drawing arm: what a tool would paint,
    /// and what it would read aloud, in one pass.
    fn read_px(text: &str) -> (Vec<Authored>, ShapeErrors) {
        let doc = ron::parse_doc(text).expect("the test's sidecar parses");
        read_shapes(&doc.root)
    }

    /// Read, insisting the document was fine: a test that quietly read
    /// errors instead of features would be asserting nothing.
    fn read_ok(text: &str, size_px: [f32; 2]) -> Vec<Feature> {
        read(text, size_px).unwrap_or_else(|errs| panic!("expected features, got:\n{errs}"))
    }

    /// Read, insisting it refused, and hand back the whole report.
    fn read_err(text: &str, size_px: [f32; 2]) -> ShapeErrors {
        match read(text, size_px) {
            Ok(features) => panic!("expected refusals, got {features:?}"),
            Err(errs) => errs,
        }
    }

    /// The vocabulary's triangle entry, in a 32x16 image.
    const TRIANGLE: &str =
        "(kind: \"poly\", at: [0.0, 0.0], points: [[0.0, 0.0], [12.0, 4.0], [2.0, 14.0]]";
    const SHEET: [f32; 2] = [32.0, 16.0];

    #[test]
    fn a_top_left_pixel_lands_on_the_upper_side_of_the_node() {
        // The asymmetric case the seam exists for: pixels run y down from
        // the image's top-left, the node's space runs y up about its
        // centre, so the top-left corner is the *highest* point of the
        // node, not the lowest. Drop the flip and the y lands on -8.
        assert_eq!(pixel_to_node([0.0, 0.0], [32.0, 16.0]), [-16.0, 8.0]);
        assert_eq!(pixel_to_node([32.0, 16.0], [32.0, 16.0]), [16.0, -8.0]);
        assert_eq!(pixel_to_node([16.0, 8.0], [32.0, 16.0]), [0.0, 0.0]);
    }

    #[test]
    fn a_rect_entry_bakes_as_a_solid_at_its_place_in_the_nodes_space() {
        let features = read_ok(
            "(shapes: [(kind: \"rect\", at: [8.0, 4.0], size: [8.0, 4.0], solid: true)])",
            SHEET,
        );
        // The centre is the pixel [8, 4] of a 32x16 sheet: eight pixels
        // left of the middle and four above it; `size` is full extents,
        // so the half-extents are half of those.
        assert_eq!(
            features,
            vec![Feature::solid(
                Footprint::rect([-8.0, 4.0], [4.0, 2.0]),
                0.0
            )]
        );
        let mut baker = Baker::new();
        let rejected = baker.add(Placement::IDENTITY, &features);
        assert!(rejected.is_empty(), "the bake refused {rejected:?}");
        let baked = baker.finish();
        assert_eq!(baked.solids.len(), 1);
        assert_eq!(
            baked.solids[0].shape,
            Collider::Box(crate::collision::OrientedBox::new([-8.0, 4.0], [4.0, 2.0]))
        );
    }

    #[test]
    fn a_circle_flagged_occludes_stays_round_all_the_way_to_the_field() {
        // The reader's leg of the path the bake test
        // `an_occluding_circle_stays_round_all_the_way_to_the_field`
        // walks from a hand-built disc: an authored circle must reach the
        // occluder field as the polygon of a *circle at the flipped
        // centre*, never flattened or shifted anywhere upstream.
        let features = read_ok(
            "(shapes: [(kind: \"circle\", at: [16.0, 2.0], radius: 6.0, occludes: true)])",
            SHEET,
        );
        assert_eq!(
            features,
            vec![Feature::occluding(Footprint::disc([0.0, 6.0], 6.0))]
        );
        let mut baker = Baker::new();
        let rejected = baker.add(Placement::IDENTITY, &features);
        assert!(rejected.is_empty(), "the bake refused {rejected:?}");
        let baked = baker.finish();
        assert_eq!(
            baked.disc_occluders,
            [Circle {
                center: [0.0, 6.0],
                radius: 6.0
            }]
        );
        let mut canvas = Canvas::new((64, 64), 1.0);
        baked.declare_occluders(&mut canvas, Transform::identity());
        assert_eq!(canvas.occluders.len(), 1);
        let declared = canvas.occluders[0]
            .polygon
            .expect("a disc declares itself as a polygon");
        assert_eq!(declared.planes().len(), DISC_SIDES);
        assert_eq!(
            declared.bounds_as_box().center,
            [0.0, 6.0],
            "the polygon is the circle the author drew at the node's [0, 6]"
        );
    }

    #[test]
    fn a_polygon_flagged_occludes_reaches_the_field_as_its_own_edges() {
        // Declared, not packed through the backend from here — the
        // packing of these edges is `a_polygon_record_carries_its_edges_
        // and_the_next_record_follows_them`'s claim over in
        // `src/backend/tests/fields.rs`; what this test owns is that the
        // authored triangle arrives at the field's input as itself.
        let features = read_ok(&format!("(shapes: [{TRIANGLE}, occludes: true)])"), SHEET);
        let want = Convex::new(&[[-16.0, 8.0], [-4.0, 4.0], [-14.0, -6.0]])
            .expect("the authored triangle is convex");
        let mut baker = Baker::new();
        let rejected = baker.add(Placement::IDENTITY, &features);
        assert!(rejected.is_empty(), "the bake refused {rejected:?}");
        let baked = baker.finish();
        assert_eq!(baked.poly_occluders.len(), 1);
        let mut canvas = Canvas::new((64, 64), 1.0);
        baked.declare_occluders(&mut canvas, Transform::identity());
        assert_eq!(canvas.occluders.len(), 1);
        let declared = canvas.occluders[0]
            .polygon
            .expect("a polygon occluder carries edges of its own");
        assert_eq!(
            declared.planes(),
            want.planes(),
            "the field's input holds the triangle's three half-planes, \
             flipped into node space and no one's decomposition"
        );
    }

    #[test]
    fn a_polygon_asked_for_solid_is_refused_with_the_separating_axis_reason() {
        let errs = read_err(&format!("(shapes: [{TRIANGLE}, solid: true)])"), SHEET);
        assert_eq!(errs.len(), 1, "the whole report was:\n{errs}");
        assert!(
            errs.errors[0].message.contains("separating axis"),
            "the refusal must name what is missing, got {}",
            errs.errors[0]
        );
    }

    #[test]
    fn a_concave_outline_is_refused_and_not_repaired() {
        // An L drawn in pixels: `Convex::new` refuses it and the reader
        // says so — no decomposition, no hull, nothing shipped.
        let errs = read_err(
            "(shapes: [(kind: \"poly\", at: [0.0, 0.0], points: [[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [20.0, 20.0], [20.0, 40.0], [0.0, 40.0]], occludes: true)])",
            [64.0, 64.0],
        );
        assert_eq!(errs.len(), 1, "the whole report was:\n{errs}");
        assert!(
            errs.errors[0].message.contains("concave"),
            "the refusal should say the outline is concave, got {}",
            errs.errors[0]
        );
    }

    #[test]
    fn a_tuple_position_explains_the_tools_spot_rule() {
        let errs = read_err(
            "(shapes: [(kind: \"rect\", at: (4.0, 2.0), size: [8.0, 4.0], solid: true)])",
            SHEET,
        );
        assert_eq!(errs.len(), 1, "the whole report was:\n{errs}");
        let message = &errs.errors[0].message;
        assert!(
            message.contains("spot"),
            "no spot-rule explanation: {message}"
        );
        assert!(
            message.contains("array"),
            "the explanation should say what to write instead: {message}"
        );
    }

    #[test]
    fn an_entry_with_every_flag_off_is_refused() {
        // Twice over: explicitly false, and simply saying nothing. An
        // authored shape nobody wants is a mistake, not a comment.
        let errs = read_err(
            "(shapes: [
                (kind: \"rect\", at: [8.0, 4.0], size: [8.0, 4.0], solid: false, occludes: false, hit: false),
                (kind: \"rect\", at: [8.0, 4.0], size: [8.0, 4.0]),
            ])",
            SHEET,
        );
        assert_eq!(errs.len(), 2, "the whole report was:\n{errs}");
        for err in &errs.errors {
            assert!(err.message.contains("all off"), "{err}");
        }
    }

    #[test]
    fn an_unknown_key_inside_an_entry_is_refused() {
        let errs = read_err(
            "(shapes: [(kind: \"rect\", at: [8.0, 4.0], size: [8.0, 4.0], solid: true, solidss: true)])",
            SHEET,
        );
        assert_eq!(errs.len(), 1, "the whole report was:\n{errs}");
        assert!(
            errs.errors[0].message.contains("`solidss`"),
            "the refusal should name the key, got {}",
            errs.errors[0]
        );
    }

    #[test]
    fn an_unknown_key_outside_the_shapes_list_is_left_alone() {
        // The sidecar legitimately carries keys no Rust code reads; the
        // tool preserves them and this reader has no opinion on them.
        let features = read_ok(
            "(
                flower_anchors: [(317.0, 671.0)],
                shapes: [(kind: \"rect\", at: [8.0, 4.0], size: [8.0, 4.0], solid: true)],
            )",
            SHEET,
        );
        assert_eq!(features.len(), 1);
    }

    #[test]
    fn a_capsule_is_refused_as_unimplemented_and_the_reader_does_not_panic() {
        let errs = read_err(
            "(shapes: [(kind: \"capsule\", at: [8.0, 4.0], radius: 3.0, solid: true)])",
            SHEET,
        );
        assert_eq!(errs.len(), 1, "the whole report was:\n{errs}");
        assert!(errs.errors[0].message.contains("capsule"));
    }

    #[test]
    fn every_defect_in_the_document_is_reported_not_just_the_first() {
        // One entry with two defects and a second bad entry behind it:
        // the author fixing this file wants the whole list now.
        let errs = read_err(
            "(shapes: [
                (kind: \"rect\", at: [8.0, 4.0], size: [16.0], solid: true, hit: 7.0),
                (kind: \"blob\", at: [1.0, 2.0], occludes: true),
            ])",
            SHEET,
        );
        assert_eq!(errs.len(), 3, "the whole report was:\n{errs}");
        assert!(
            errs.errors[0].message.contains("`size`"),
            "{}",
            errs.errors[0]
        );
        assert!(
            errs.errors[1].message.contains("`hit`"),
            "{}",
            errs.errors[1]
        );
        assert_eq!(errs.errors[2].entry, Some(1));
    }

    #[test]
    fn no_shapes_key_at_all_is_not_an_error() {
        // Absent and empty both mean "nothing authored" — where the
        // caller's derive-from-bounds default takes over.
        let plain = read_ok("(atlas: [4, 4])", SHEET);
        assert!(plain.is_empty());
        let empty = read_ok("(shapes: [])", SHEET);
        assert!(empty.is_empty());
        let (authored, errs) = read_px("(atlas: [4, 4])");
        assert!(authored.is_empty() && errs.is_empty());
    }

    #[test]
    fn the_drawing_arm_holds_the_pixels_the_author_wrote() {
        // The same entry the bake arm flips to [-8, 4]: the drawing arm
        // hands back the pixel the author typed, flipped by nobody, so a
        // tool painting over the texture needs no conversion of its own.
        let (authored, errs) = read_px(
            "(shapes: [(kind: \"rect\", at: [8.0, 4.0], size: [8.0, 4.0], solid: (bounce: 0.2), occludes: true)])",
        );
        assert!(errs.is_empty(), "the reading said:\n{errs}");
        assert_eq!(
            authored,
            vec![Authored {
                entry: 0,
                at: [8.0, 4.0],
                shape: AuthoredShape::Rect { size: [8.0, 4.0] },
                solid: true,
                bounce: 0.2,
                occludes: true,
                hit: false,
            }]
        );
    }

    #[test]
    fn the_drawing_arm_keeps_every_outline_it_read_and_says_the_rest() {
        // The seam the overlay exists for: a good rect, a concave L, and
        // a capsule behind it. The bake arm would refuse all three; the
        // drawing arm paints the one outline it could read while naming
        // both refusals — the author sees their shapes AND the reasons.
        let (authored, errs) = read_px(
            "(shapes: [
                (kind: \"rect\", at: [8.0, 4.0], size: [8.0, 4.0], solid: true),
                (kind: \"poly\", at: [0.0, 0.0], points: [[0.0, 0.0], [40.0, 0.0], [40.0, 20.0], [20.0, 20.0], [20.0, 40.0], [0.0, 40.0]], occludes: true),
                (kind: \"capsule\", at: [8.0, 4.0], radius: 3.0, solid: true),
            ])",
        );
        assert_eq!(authored.len(), 1, "only the readable geometry draws");
        assert_eq!(authored[0].entry, 0);
        assert_eq!(errs.len(), 2, "the whole report was:\n{errs}");
        assert!(errs.errors[0].message.contains("concave"), "{errs}");
        assert!(errs.errors[1].message.contains("capsule"), "{errs}");
    }

    #[test]
    fn a_role_that_failed_still_draws_the_shape_it_was_written_on() {
        // A polygon authored solid earns the separating-axis refusal,
        // and its outline is still the author's own drawing: it draws,
        // with its occludes role answered and the refusal reported. The
        // bake arm still refuses the document — one error, whole file —
        // and says the very same sentence.
        let (authored, errs) = read_px(&format!(
            "(shapes: [{TRIANGLE}, solid: true, occludes: true)])"
        ));
        assert_eq!(authored.len(), 1);
        assert!(!authored[0].solid, "the refused role reads off");
        assert!(authored[0].occludes);
        assert!(
            matches!(&authored[0].shape, AuthoredShape::Poly { points } if points.len() == 3),
            "the outline reaches the drawing arm as the corners it was authored with"
        );
        assert_eq!(errs.len(), 1, "the whole report was:\n{errs}");
        assert!(errs.errors[0].message.contains("separating axis"), "{errs}");
        let bake_errs = read_err(
            &format!("(shapes: [{TRIANGLE}, solid: true, occludes: true)])"),
            SHEET,
        );
        assert_eq!(
            bake_errs.errors, errs.errors,
            "both arms speak one vocabulary"
        );
    }

    #[test]
    fn a_whole_sidecar_read_end_to_end_declares_the_authored_field() {
        // The seam's proof, all four vocabulary lines at once: text as
        // an author writes it (comments and foreign keys included),
        // through `parse_doc`, through `authored_shapes`, through a real
        // `Baker`, out through `declare_occluders` — and the field the
        // GPU would receive is what the sidecar said.
        let features = read_ok(
            "// garden sprite
            (
                flower_anchors: [(317.0, 671.0)],
                shapes: [
                    (kind: \"rect\", at: [16.0, 8.0], size: [24.0, 16.0], solid: true),
                    (kind: \"circle\", at: [16.0, 2.0], radius: 6.0, occludes: true),
                    (kind: \"poly\", at: [0.0, 0.0], points: [[0.0, 0.0], [12.0, 4.0], [2.0, 14.0]], occludes: true),
                    // one wall, both jobs: bouncy, and it throws a shadow
                    (kind: \"rect\", at: [16.0, 30.0], size: [32.0, 8.0], solid: (bounce: 0.2), occludes: true),
                ],
            )",
            [32.0, 32.0],
        );
        assert_eq!(
            features.len(),
            5,
            "one solid, two round/poly shadows, and one wall both ways"
        );

        let mut baker = Baker::new();
        let rejected = baker.add(Placement::IDENTITY, &features);
        assert!(rejected.is_empty(), "the bake refused {rejected:?}");
        let baked = baker.finish();
        assert_eq!(baked.solids.len(), 2);
        let bouncy = baked.solids.iter().find(|s| s.bounce == 0.2);
        assert!(bouncy.is_some(), "the (bounce: 0.2) material rode along");
        assert_eq!(baked.occluders.len(), 1, "one occluding rectangle");
        assert_eq!(baked.occluders[0].center, [0.0, -14.0]);
        assert_eq!(baked.disc_occluders.len(), 1);
        assert_eq!(baked.poly_occluders.len(), 1);
        assert!(baked.hits.is_empty());

        let mut canvas = Canvas::new((64, 64), 1.0);
        baked.declare_occluders(&mut canvas, Transform::identity());
        assert_eq!(
            canvas.occluders.len(),
            3,
            "the box record, the cut circle, and the authored triangle"
        );
        assert!(canvas.occluders[0].polygon.is_none());
        assert_eq!(canvas.occluders[0].half, [16.0, 4.0]);
        assert_eq!(
            canvas.occluders[1].polygon.as_ref().unwrap().planes().len(),
            DISC_SIDES
        );
        let want = Convex::new(&[[-16.0, 16.0], [-4.0, 12.0], [-14.0, 2.0]])
            .expect("the authored triangle is convex");
        let triangle = canvas.occluders[2]
            .polygon
            .as_ref()
            .expect("the polygon occluder carries its own edges");
        assert_eq!(triangle.planes(), want.planes());
    }

    /// A shipped asset by name: the seam's tests read the same files a
    /// game reads, not copies of them.
    fn asset(name: &str) -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("assets")
            .join(name)
    }

    #[test]
    fn the_sidecar_bakes_at_load() {
        // The text twin of the seam — the shape of a wasm game's boot:
        // bytes in hand, nothing read from a directory. One bake, every
        // consumer's list already sorted.
        let baked = bake_sidecar_text(
            SHEET,
            "(shapes: [
                (kind: \"rect\", at: [8.0, 4.0], size: [8.0, 4.0], solid: (bounce: 0.2), occludes: true),
                (kind: \"circle\", at: [16.0, 2.0], radius: 6.0, occludes: true),
            ])",
        )
        .expect("a well-authored sidecar loads");
        assert_eq!(
            baked.occluders,
            [crate::collision::OrientedBox::new([-8.0, 4.0], [4.0, 2.0])],
            "the wall occludes, fused from its two roles into one"
        );
        assert_eq!(
            baked.disc_occluders,
            [Circle {
                center: [0.0, 6.0],
                radius: 6.0
            }]
        );
        assert_eq!(baked.solids.len(), 1);
        assert_eq!(baked.solids[0].bounce, 0.2, "the material rode along");
        assert!(baked.hits.is_empty());
    }

    #[test]
    fn a_sprite_without_a_sidecar_loads_with_nothing_authored() {
        // The seam's most common answer, and not an error.
        let baked = bake_sidecar(asset("sprites/Button.png")).expect("Button.png loads");
        assert!(baked.is_empty());
    }

    #[test]
    fn a_sidecar_without_shapes_bakes_nothing_and_stays_everything_else() {
        // A real shipped sidecar: an `atlas` key, no `shapes` — the
        // file parses, the reader finds nothing, the tool's own keys are
        // simply not the seam's business.
        let baked = bake_sidecar(asset("sprites/basic_tiles.png")).expect("the atlas loads");
        assert!(baked.is_empty());
    }

    #[test]
    fn the_halls_brick_bakes_its_authored_footprint() {
        // The asset the `authored_hall` example builds its room from,
        // pinned at the seam: the 64 x 64 picture carries one inset
        // footprint that both blocks and occludes, bouncy off the face.
        let baked = bake_sidecar(asset("sprites/brick.png")).expect("the brick loads");
        assert_eq!(
            baked.occluders,
            [crate::collision::OrientedBox::new([0.0, 0.0], [30.0, 30.0])]
        );
        assert_eq!(baked.solids.len(), 1);
        assert_eq!(baked.solids[0].bounce, 0.35);
        assert_eq!(
            baked.occluder_colliders().len(),
            1,
            "and the CPU twin agrees"
        );
    }

    #[test]
    fn a_misauthored_sidecar_fails_the_load_and_names_every_defect() {
        // The policy, at the seam: the game gets the whole list and
        // decides; nothing loads half-collidable, and nothing logs.
        let err = bake_sidecar_text(
            SHEET,
            "(shapes: [
                (kind: \"rect\", at: [8.0, 4.0], size: [16.0], solid: true),
                (kind: \"blob\", at: [1.0, 2.0], occludes: true),
            ])",
        )
        .expect_err("a mis-authored sidecar must not bake");
        match err {
            SidecarError::Shapes(errs) => {
                assert_eq!(errs.len(), 2, "the whole report was:\n{errs}");
                assert!(errs.errors[0].message.contains("`size`"), "{errs}");
                assert_eq!(errs.errors[1].entry, Some(1), "{errs}");
            }
            other => panic!("expected the shapes' own refusals, got {other:?}"),
        }
    }

    #[test]
    fn an_unparseable_sidecar_is_a_load_failure_not_a_log_line() {
        let err = bake_sidecar_text(SHEET, "(shapes: [ (kind: \"rect\" ")
            .expect_err("half a file is not a sidecar");
        assert!(
            matches!(err, SidecarError::Ron(_)),
            "the RON failure should travel, got {err:?}"
        );
    }
}
