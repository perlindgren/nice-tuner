//! The map file: painted cells on a grid, and the cargo of tilesets
//! they cut their pictures from.
//
// The format grew up inside `sprite_util`, where one program needed
// both the facts and the gestures. The demo rung asked the world to
// *boot* a map, so the facts graduated: layers, cells, the codec, and
// the road from a layer to a `TileMap` shape. The next amendment made
// the chain honest: a map no longer wears a sprite — it names
// tileset *documents* by path, and each tileset names its sheet
// (see [`crate::tileset`]). Every link is a reference resolved
// against the document that speaks it, so the chain walks identically
// against a folder of loose files and a shipped table of embedded
// ones: paths find documents, names are documents, and
// [`guard_documents`] asks the whole cargo question at build time,
// without a filesystem. The gestures stayed with the guest: brushes,
// ghosts, transform mnemonics and grid ladders are the desk's motor
// skill, and a game that reads a map has no business learning them.

use crate::objects::{Color, Shape, SpriteFilter, Tile};
use crate::ron::{self, Item, Kind, Val};
use crate::tileset;

/// One tileset's worth of painted cells, anchored on the map's origin:
/// a grid of atlas-cell indices, `EMPTY_CELL` where nothing stands.
/// A multi-tileset picture is several of these layers, each naming its
/// tileset by the name that document speaks — one hand paints a layer
/// at a time, every layer shows at once.
#[derive(Clone, Debug, PartialEq)]
pub struct MapLayer {
    /// The tileset's spoken name — the name inside the document the
    /// map's cargo carries. The layer does not repeat the cut or the
    /// sheet: those belong to the tileset, and repeating them here
    /// would be a second claim to keep true.
    pub tileset: String,
    pub cols: usize,
    pub rows: usize,
    pub cells: Vec<u32>,
    /// Each cell's orientation, painted with the cell and erased with
    /// it: bit 0 flip-x, bit 1 flip-y, bits 2..3 clockwise quarter-
    /// turns. The transform is the map's memory, not the brush's mood
    /// — which is why it lives here, inside the layer.
    pub tfms: Vec<u8>,
}

/// How many tileset cells the band shows at most; a bigger atlas waits
/// for the band's own scrolling (a later step).
/// A cell with nothing in it. Real atlas cells count from 0.
pub const EMPTY_CELL: u32 = u32::MAX;

impl MapLayer {
    /// An empty layer of `cols` by `rows`, every cell absent, naming
    /// its tileset by the name that tileset document speaks.
    pub fn blank(tileset: impl Into<String>, cols: usize, rows: usize) -> Self {
        Self {
            tileset: tileset.into(),
            cols,
            rows,
            cells: vec![EMPTY_CELL; cols * rows],
            tfms: vec![0; cols * rows],
        }
    }

    /// The storage slot of grid column/row — the map's cells are
    /// numbered from its top-left, while grid coordinates radiate
    /// from the origin, which sits at the map's center.
    pub fn at(&self, col: i32, row: i32) -> Option<usize> {
        cell_in(col, row, self.cols, self.rows)
            .then(|| (row + self.rows as i32 / 2) * self.cols as i32 + (col + self.cols as i32 / 2))
            .map(|i| i as usize)
    }

    /// Paint over one cell — picture and orientation together; true
    /// when the cell actually changed. A repaint that only turns the
    /// tile IS a change; an eraser on an empty cell is not, whatever
    /// orientation bits it carries.
    pub fn set_cell(&mut self, col: i32, row: i32, cell: u32, tfm: u8) -> bool {
        match self.at(col, row) {
            Some(i) => {
                let erased = cell == EMPTY_CELL;
                let changed = self.cells[i] != cell || (!erased && self.tfms[i] != tfm);
                if changed {
                    self.cells[i] = cell;
                    self.tfms[i] = if erased { 0 } else { tfm };
                }
                changed
            }
            _ => false,
        }
    }

    /// The layer as a `TileMap` shape sharing the tileset's sheet —
    /// or `None` when the tileset named is not the one offered (the
    /// desk's wrong-dressing guard: a layer wearing another
    /// tileset's pixels would paint the world as nonsense), when the
    /// atlas is closed (the layer waits for its sheet to open), or
    /// when the layer holds nothing. Cells past the tileset's current
    /// cut — one shrank since they were painted — are skipped.
    /// The cut is the tileset's claim, not the atlas's: the layer
    /// asks the document it names to dress its numbers.
    pub fn shape(&self, tileset: &tileset::Tileset, atlas: Option<&Shape>) -> Option<Shape> {
        if tileset.name != self.tileset {
            return None;
        }
        let (data, width, height, generation) = match atlas? {
            Shape::Sprite {
                data,
                width,
                height,
                generation,
                ..
            } => (data.clone(), *width, *height, *generation),
            _ => return None,
        };
        let (arows, acols) = (tileset.rows.max(1), tileset.cols.max(1));
        let (tw, th) = (width as f32 / acols as f32, height as f32 / arows as f32);
        let tiles: Vec<Tile> = self
            .cells
            .iter()
            .enumerate()
            .filter_map(|(i, &c)| {
                (c != EMPTY_CELL && (c as usize) < arows * acols).then_some((i, c))
            })
            .map(|(i, c)| {
                let gx = (i % self.cols) as f32 - self.cols as f32 / 2.0 + 0.5;
                let gy = (i / self.cols) as f32 - self.rows as f32 / 2.0 + 0.5;
                let t = self.tfms[i];
                Tile::new(
                    [gx * tw, -gy * th],
                    [tw, th],
                    cell_uv(c as usize, arows, acols),
                )
                .transformed((t >> 2) & 3, t & 1 != 0, t & 2 != 0)
            })
            .collect();
        if tiles.is_empty() {
            return None;
        }
        let mut shape = Shape::TileMap {
            data,
            width,
            height,
            generation,
            filter: SpriteFilter::Nearest,
            tiles: Vec::new(),
            clip: None,
            color: Color {
                r: 1.0,
                g: 1.0,
                b: 1.0,
                a: 1.0,
            },
        };
        shape.set_tiles(tiles);
        Some(shape)
    }
}

/// A map as a file knows it: the cargo of tileset documents — paths
/// **relative to the map file itself** — and the layers whose cells
/// name those documents by their spoken names. A game loads one of
/// these, resolves each reference against the map's own path, and
/// dresses its layers; a desk writes one and the tilesets it leans
/// on, and a guard test walks the whole chain through the shipped
/// table without a filesystem in sight.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Map {
    pub tilesets: Vec<String>,
    pub layers: Vec<MapLayer>,
}

/// Whether grid column/row lies on a `cols`-by-`rows` map centered on
/// the origin. Even dimensions make the grid half-open on the right
/// and bottom: it spans x in [-w/2, w/2) like the cells' own floors.
pub fn cell_in(col: i32, row: i32, cols: usize, rows: usize) -> bool {
    col + cols as i32 / 2 >= 0
        && row + rows as i32 / 2 >= 0
        && col + cols as i32 / 2 < cols as i32
        && row + rows as i32 / 2 < rows as i32
}

/// The atlas cell `idx`'s texture rectangle in a `rows`-by-`cols` grid,
/// row-major, top-left origin like the PNG's own rows.
pub fn cell_uv(idx: usize, rows: usize, cols: usize) -> [f32; 4] {
    let (row, col) = ((idx / cols) as f32, (idx % cols) as f32);
    let (rows, cols) = (rows as f32, cols as f32);
    [
        col / cols,
        row / rows,
        (col + 1.0) / cols,
        (row + 1.0) / rows,
    ]
}

/// The tileset names a map's layers ask for, first-spoken order, none
/// twice — what the desk's waiting list speaks. The cargo question is
/// the map's `tilesets` list; this is its layers' echo of it.
pub fn tileset_names(layers: &[MapLayer]) -> Vec<&str> {
    let mut names: Vec<&str> = Vec::new();
    for layer in layers {
        if !names.contains(&layer.tileset.as_str()) {
            names.push(&layer.tileset);
        }
    }
    names
}

/// Resolve `reference` — as spoken inside the document at `doc_path`
/// — against that document's folder, purely in text: no filesystem,
/// no platform separators (`\` is read as `/` for hand-edited
/// documents' sake), no root. `.` steps vanish, `..` climbs, a
/// leading `/` starts over from the root the caller lives under.
/// The result is in the same space its document was: hand it a table
/// key and it answers a table key; hand it an absolute path and it
/// answers absolute. This one function is why a chain of documents
/// walks the same through a folder and through a binary.
#[must_use]
pub fn resolve(doc_path: &str, reference: &str) -> String {
    let reference = reference.replace('\\', "/");
    let reference = reference.trim_start_matches("./");
    let doc = doc_path.replace('\\', "/");
    let mut parts: Vec<&str> = if reference.starts_with('/') {
        Vec::new()
    } else {
        match doc.rfind('/') {
            Some(i) => doc[..i]
                .split('/')
                .filter(|s| !s.is_empty() && *s != ".")
                .collect(),
            None => Vec::new(),
        }
    };
    let anchored = reference.starts_with('/');
    for c in reference.split('/') {
        match c {
            "" | "." => {}
            ".." => {
                if !anchored || !parts.is_empty() {
                    parts.pop();
                }
            }
            c => parts.push(c),
        }
    }
    // The document's own anchor survives: an absolute document
    // resolves absolutely — the doc comment's promise, pinned by
    // `an_absolute_document_keeps_its_anchor` after the desk's
    // absolute paths proved the table-key tests never asked it.
    if anchored || doc_path.replace('\\', "/").starts_with('/') {
        format!("/{}", parts.join("/"))
    } else {
        parts.join("/")
    }
}

/// The path from a folder to a file in the same space (see
/// [`resolve`]), as the relative reference its documents would
/// speak: `relative("maps", "sprites/g.ron")` is
/// `"../sprites/g.ron"`, and `resolve("maps/m.ron", of)` returns to
/// `"sprites/g.ron"`. The desk writes map and tileset files with
/// this; `resolve` is its inverse, so the chain composes by
/// construction and a round trip is a test, not a hope.
#[must_use]
pub fn relative(from_dir: &str, target: &str) -> String {
    let from = from_dir.replace('\\', "/");
    let target = target.replace('\\', "/");
    let from_parts: Vec<&str> = from
        .split('/')
        .filter(|s| !s.is_empty() && *s != ".")
        .collect();
    let target_parts: Vec<&str> = target
        .split('/')
        .filter(|s| !s.is_empty() && *s != ".")
        .collect();
    let common = from_parts
        .iter()
        .zip(target_parts.iter())
        .take_while(|(a, b)| a == b)
        .count();
    let ups = from_parts.len() - common;
    let mut parts: Vec<&str> = vec![".."; ups];
    parts.extend_from_slice(&target_parts[common..]);
    if parts.is_empty() {
        ".".to_string()
    } else {
        parts.join("/")
    }
}

/// The whole cargo question, asked of bytes and closures rather than
/// a filesystem: parse the map text, resolve every tileset reference
/// against the map's own path, read each through `read` (a shipped
/// app hands its [`crate::Assets`] table, a dev world hands a loose
/// folder, a test hands a map in memory), parse each tileset, and
/// check its sheet answers and every layer names a carried document.
/// An empty answer means the chain walks: every map in a shipped
/// app's guard test can assert this over its own table, so a
/// forgotten `include_bytes!` dies at `cargo test`, not in the field.
pub fn guard_documents(
    map_path: &str,
    map_src: &str,
    read: &dyn Fn(&str) -> Option<String>,
) -> Vec<String> {
    let (map, mut refused) = from_text(map_src);
    if !refused.is_empty() {
        return refused;
    }
    let mut names: Vec<String> = Vec::new();
    for reference in &map.tilesets {
        let path = resolve(map_path, reference);
        let Some(src) = read(&path) else {
            refused.push(format!(
                "tileset reference '{reference}' resolves to '{path}', which answers nothing"
            ));
            continue;
        };
        let (t, errs) = tileset::from_text(&src);
        match t {
            Some(t) => {
                refused.extend(errs.into_iter().map(|e| format!("'{path}': {e}")));
                names.push(t.name.clone());
                let sheet = resolve(&path, &t.sheet);
                if read(&sheet).is_none() {
                    refused.push(format!(
                        "'{path}' cuts sheet '{sheet}', which answers nothing"
                    ));
                }
            }
            None => refused.extend(errs.into_iter().map(|e| format!("'{path}': {e}"))),
        }
    }
    for layer in &map.layers {
        if !names.iter().any(|n| n == &layer.tileset) {
            refused.push(format!(
                "a layer speaks tileset '{}', which no document in the cargo names",
                layer.tileset
            ));
        }
    }
    refused
}

/// The header every map file carries: the format, spoken to whoever
/// opens it next.
const MAP_FILE_HEADER: &str = "// frost map file — painted cells on a tile grid.\n\
// tilesets: the tileset documents this map's layers cut from, by\n\
// path relative to THIS file. Each layer names its tileset by the\n\
// name that document speaks. Cells stand as named records — (x, y)\n\
// counted from the layer's top-left, tile: the cell's number in the\n\
// cut, and tfm: the tile's turn, written only when it is turned.\n\
// An absent cell is empty, and empty cells are not written.";

/// A bracketed list of values.
fn list(items: Vec<Val>) -> Val {
    Val::Seq {
        open: true,
        items: items.into_iter().map(Item::plain).collect(),
        tail: String::new(),
    }
}

/// The map file's root: the tileset cargo and every layer as one
/// `layers:` list — each entry naming its tileset by spoken name,
/// the layer's extent, and the cells that stand. Empty cells are
/// never written: an absence written is a presence faked.
fn map_value(map: &Map) -> Val {
    let tilesets = map
        .tilesets
        .iter()
        .map(|t| ron::text(t))
        .collect::<Vec<_>>();
    let layers = map
        .layers
        .iter()
        .map(|m| {
            let cells = m
                .cells
                .iter()
                .enumerate()
                .filter(|(_, c)| **c != EMPTY_CELL)
                .map(|(i, &c)| {
                    // Named, like the tileset documents: a cell
                    // says which place (x, y from the top-left),
                    // which tile, and — only when it matters — its
                    // turn. A plain tile never pays for the word.
                    let mut fields = vec![
                        ("x", ron::number((i % m.cols) as f64)),
                        ("y", ron::number((i / m.cols) as f64)),
                        ("tile", ron::number(c as f64)),
                    ];
                    if m.tfms[i] != 0 {
                        fields.push(("tfm", ron::number(m.tfms[i] as f64)));
                    }
                    ron::record_named(&fields)
                })
                .collect();
            ron::record_named(&[
                ("tileset", ron::text(&m.tileset)),
                ("cols", ron::number(m.cols as f64)),
                ("rows", ron::number(m.rows as f64)),
                ("cells", list(cells)),
            ])
        })
        .collect();
    ron::record_named(&[("tilesets", list(tilesets)), ("layers", list(layers))])
}

/// The listed entries, if this is a list.
pub(crate) fn entries(v: &Val) -> Option<&[Item]> {
    match v {
        Val::Seq { items, .. } => Some(items),
        _ => None,
    }
}

/// The nth entry of a positional record or a list — the dialect
/// spells tuples as nameless records, and a hand-written list is
/// read the same.
pub(crate) fn pos(v: &Val, n: usize) -> Option<&Val> {
    match v {
        Val::Struct { fields, .. } => fields.get(n).map(|(_, it)| &it.val),
        Val::Seq { items, .. } => items.get(n).map(|it| &it.val),
        _ => None,
    }
}

/// A bare number's text, if this is one.
pub(crate) fn number(v: &Val) -> Option<&str> {
    match v {
        Val::Atom(s, Kind::Num) => Some(s),
        _ => None,
    }
}

/// A quoted string, unwrapped, if this is one.
pub(crate) fn text(v: &Val) -> Option<&str> {
    match v {
        Val::Atom(s, Kind::Str) => Some(s.trim_matches('\"')),
        _ => None,
    }
}

/// The map file: its tileset cargo and its layers. Every readable
/// layer opens, every refusal is named — the first so a status band
/// can speak it, the rest so the count can stand behind it. A layer
/// naming a tileset the cargo does not carry is refused: the map
/// would open with a layer no sheet can dress, which is not a
/// partial truth but a broken one. Cells past their layer's extent
/// or wearing an impossible transform are skipped where they lie,
/// noted by their layer: a map opens as much as it can.
fn map_of(root: &Val) -> (Map, Vec<String>) {
    let mut refused = Vec::new();
    let mut map = Map::default();
    match ron::field(root, "tilesets").and_then(entries) {
        Some(items) => {
            for item in items {
                match text(&item.val) {
                    Some(t) if !map.tilesets.iter().any(|s| s == t) => {
                        map.tilesets.push(t.to_string());
                    }
                    Some(_) => {}
                    None => refused.push(format!(
                        "tileset reference {} is not a path in quotes",
                        map.tilesets.len() + 1
                    )),
                }
            }
        }
        None => refused.push("no `tilesets:` list".to_string()),
    }
    let mut layers = Vec::new();
    let Some(items) = ron::field(root, "layers").and_then(entries) else {
        refused.push("no `layers:` list".to_string());
        return (map, refused);
    };
    for (n, item) in items.iter().enumerate() {
        let who = format!("layer {}", n + 1);
        let Some(name) = ron::field(&item.val, "tileset").and_then(text) else {
            refused.push(format!("{who} names no tileset as tileset: \"name\""));
            continue;
        };
        if !map.tilesets.iter().any(|t| name_docs(t, name)) {
            refused.push(format!(
                "{who} speaks tileset '{name}', which the tilesets list does not carry"
            ));
            continue;
        }
        let (Some(cols), Some(rows)) = (
            ron::field(&item.val, "cols")
                .and_then(number)
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| *v > 0),
            ron::field(&item.val, "rows")
                .and_then(number)
                .and_then(|v| v.parse::<usize>().ok())
                .filter(|v| *v > 0),
        ) else {
            refused.push(format!("{who} says no honest extent"));
            continue;
        };
        let mut layer = MapLayer::blank(name, cols, rows);
        let mut skipped = 0;
        let Some(cells) = ron::field(&item.val, "cells").and_then(entries) else {
            refused.push(format!("{who} carries no `cells:` list"));
            continue;
        };
        for cell in cells {
            let ok = (|| {
                // Named records first — the current speech — and
                // bare positions after, so a file written before
                // names arrived still opens unchanged. `code:` read
                // where `tile:` speaks is the same word's former
                // name. A named cell that says nothing of turns is
                // a plain tile: tfm rests at zero.
                let spoke = |names: &[&str], at: usize| {
                    names
                        .iter()
                        .filter_map(|n| ron::field(&cell.val, n).and_then(number))
                        .next()
                        .or_else(|| pos(&cell.val, at).and_then(number))
                };
                let x = spoke(&["x"], 0)?.parse::<usize>().ok()?;
                let y = spoke(&["y"], 1)?.parse::<usize>().ok()?;
                let code = spoke(&["tile", "code"], 2)?.parse::<u32>().ok()?;
                let tfm = match spoke(&["tfm"], 3) {
                    Some(v) => v.parse::<u64>().ok()?,
                    None => 0,
                };
                (tfm < 16).then_some((x, y, code, tfm as u8))
            })();
            match ok {
                Some((x, y, code, tfm)) if x < cols && y < rows => {
                    layer.cells[y * cols + x] = code;
                    layer.tfms[y * cols + x] = tfm;
                }
                _ => skipped += 1,
            }
        }
        if skipped > 0 {
            refused.push(format!("{who} skipped {skipped} unreadable cells"));
        }
        layers.push(layer);
    }
    map.layers = layers;
    (map, refused)
}

/// Whether the cargo reference `doc` can be the document of the
/// spoken name `name`. The map file cannot open what it names —
/// this is the parse-time politeness that a layer's name must at
/// least *look* like one of the carried documents (by file stem,
/// which is how a desk names the tileset it auto-shelves beside a
/// sheet); the honest check is [`guard_documents`], which reads the
/// documents and asks their spoken names.
fn name_docs(doc: &str, name: &str) -> bool {
    let stem = doc
        .rsplit('/')
        .next()
        .unwrap_or(doc)
        .trim_end_matches(".tileset.ron")
        .trim_end_matches(".ron");
    stem == name
}

/// A map to file text, header and all. The desk writes maps with
/// this (and the tileset documents its layers lean on); a game reads
/// them with [`from_text`] — one codec on both ends, which is what
/// makes the tool's loop honest: the file means the same thing to
/// the hand that painted it and the world that boots from it.
#[must_use]
pub fn to_text(map: &Map) -> String {
    ron::to_text_doc(MAP_FILE_HEADER, &map_value(map), "")
}

/// File text to a map, read as far as the text is honest: every
/// readable layer opens, every refusal is named. A text that is not
/// RON at all is one refusal and no map.
pub fn from_text(src: &str) -> (Map, Vec<String>) {
    match ron::parse_doc(src) {
        Ok(doc) => map_of(&doc.root),
        Err(err) => (Map::default(), vec![format!("not a RON document: {err}")]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tileset::Tileset;

    /// A fixture tileset: 2 x 2, its sheet beside its own file.
    fn tileset(name: &str) -> Tileset {
        Tileset::new(name, format!("{name}.png"), 2, 2)
    }

    /// A real decoded PNG — a `w` by `h` texture, so a layer's
    /// tile sizes are the ones the file's own pixels dress.
    fn atlas(w: u32, h: u32) -> Shape {
        let img = image::DynamicImage::from(image::RgbaImage::from_pixel(
            w,
            h,
            image::Rgba([9, 9, 9, 255]),
        ));
        let mut png = std::io::Cursor::new(Vec::new());
        img.write_to(&mut png, image::ImageFormat::Png)
            .expect("a png");
        Shape::sprite_bytes(png.into_inner()).expect("a sprite")
    }

    /// A map in, text out, text in: the file means exactly what the
    /// canvas meant — cargo, names, extents, cells, orientations
    /// alike.
    #[test]
    fn the_map_round_trips_through_text() {
        let mut grass = MapLayer::blank("grass", 4, 2);
        grass.set_cell(0, 0, 3, 0);
        grass.set_cell(1, 0, 1, 7); // a turned, mirrored cell rides along
        let mut stone = MapLayer::blank("stone", 4, 2);
        stone.set_cell(-1, 1, 2, 4);
        let map = Map {
            tilesets: vec!["grass.tileset.ron".into(), "stone.tileset.ron".into()],
            layers: vec![grass, stone],
        };
        let (read, refused) = from_text(&to_text(&map));
        assert!(refused.is_empty(), "refused: {refused:?}");
        assert_eq!(read, map);
    }

    /// Cells speak names now: place, tile, and a turn only when a
    /// tile stands turned. And the era before names — bare positions,
    /// or the older `code:` word — reads to the same map, so every
    /// file written yesterday still opens today.
    #[test]
    fn named_cells_speak_and_bare_cells_still_open() {
        let mut layer = MapLayer::blank("tiles", 4, 2);
        layer.set_cell(0, 0, 3, 0);
        layer.set_cell(1, 0, 1, 7);
        let map = Map {
            tilesets: vec!["tiles.tileset.ron".into()],
            layers: vec![layer],
        };
        let text = to_text(&map);
        // Count words in the body, not in the header's own telling.
        let body = text.split_once("\n(\n").map_or("", |(_, rest)| rest);
        assert!(body.contains("x:"), "cells name their places: {text}");
        assert!(body.contains("tile:"), "cells name their tile");
        assert!(
            body.matches("tfm:").count() == 1,
            "one turned cell, one tfm, in: {body}"
        );
        let (read, refused) = from_text(&text);
        assert!(refused.is_empty(), "refused: {refused:?}");
        assert_eq!(read, map, "named speech means exactly the map");
        // The file counts cells from the layer's top-left, while the
        // grid radiates from its centre: grid (0, 0) and (1, 0) of a
        // 4 x 2 layer stand at file places (2, 1) and (3, 1).
        let bare = "(tilesets: [\"tiles.tileset.ron\"], layers: [(tileset: \"tiles\", cols: 4, rows: 2, cells: [(2, 1, 3, 0), (3, 1, 1, 7)])])";
        let (then, refused) = from_text(bare);
        assert!(refused.is_empty(), "refused: {refused:?}");
        assert_eq!(then, map, "the bare era reads to the same map");
        let named_old = "(tilesets: [\"tiles.tileset.ron\"], layers: [(tileset: \"tiles\", cols: 4, rows: 2, cells: [(x: 2, y: 1, code: 3), (x: 3, y: 1, tile: 1, tfm: 7)])])";
        let (mixed, refused) = from_text(named_old);
        assert!(refused.is_empty(), "refused: {refused:?}");
        assert_eq!(
            mixed, map,
            "code: is tile:'s former name; silence is a plain tile"
        );
    }

    /// The file carries what stands, and nothing else: a layer with
    /// three painted cells holds three entries, not a grid of zeros.
    #[test]
    fn an_empty_cell_is_never_written() {
        let mut layer = MapLayer::blank("t", 4, 2);
        layer.set_cell(-2, 0, 1, 0);
        layer.set_cell(-1, 0, 2, 0);
        layer.set_cell(0, 0, 3, 0);
        let map = Map {
            tilesets: vec!["t.tileset.ron".into()],
            layers: vec![layer],
        };
        let root = ron::parse(&ron::to_text(&map_value(&map))).unwrap();
        let entry = entries(ron::field(&root, "layers").unwrap()).unwrap();
        let cells = ron::field(&entry[0].val, "cells").unwrap();
        assert_eq!(entries(cells).unwrap().len(), 3);
    }

    /// A file is read as far as it is honest: an entry that is not a
    /// layer is refused by name, and its readable neighbours still
    /// open.
    #[test]
    fn the_reader_names_what_it_refuses() {
        let root = ron::parse(
            "(tilesets: [\"tiles.tileset.ron\"], layers: [42, (tileset: \"tiles\", cols: 2, rows: 2, cells: [(0, 0, 1, 0)])])",
        )
        .unwrap();
        let (map, refused) = map_of(&root);
        assert_eq!(map.layers.len(), 1);
        assert_eq!(refused.len(), 1);
        assert!(refused[0].contains("layer 1"), "refused: {refused:?}");
    }

    /// A layer naming a tileset the cargo does not carry is refused:
    /// a map that opens with a layer nothing can dress is not half a
    /// map, it is a contradiction written down.
    #[test]
    fn a_layer_must_name_a_carried_tileset() {
        let root = ron::parse(
            "(tilesets: [\"grass.tileset.ron\"], layers: [(tileset: \"lava\", cols: 2, rows: 2, cells: [])])",
        )
        .unwrap();
        let (map, refused) = map_of(&root);
        assert!(map.layers.is_empty());
        assert_eq!(refused.len(), 1);
        assert!(refused[0].contains("lava"), "refused: {refused:?}");
    }

    /// Cells past their layer's extent, or wearing an impossible
    /// transform, are skipped where they lie — noted, not fatal: the
    /// layer opens as large as it means to be.
    #[test]
    fn a_cell_past_the_layer_is_skipped_not_fatal() {
        let root = ron::parse(
            "(tilesets: [\"tiles.tileset.ron\"], layers: [(tileset: \"tiles\", cols: 2, rows: 2, cells: [(9, 9, 1, 0), (0, 0, 1, 0), (1, 1, 2, 99)])])",
        )
        .unwrap();
        let (map, refused) = map_of(&root);
        assert_eq!(map.layers.len(), 1);
        let painted = map.layers[0]
            .cells
            .iter()
            .filter(|&&c| c != EMPTY_CELL)
            .count();
        assert_eq!(painted, 1);
        assert_eq!(refused.len(), 1);
        assert!(refused[0].contains("2"), "refused: {refused:?}");
    }

    /// What a cell was painted with stays with the cell: the same
    /// tile, unturned, is no change; the same tile turned is one
    /// the road remembers; and the eraser takes the bits down
    /// with the tile.
    #[test]
    fn the_transform_is_the_maps_memory() {
        let mut m = MapLayer::blank("t", 4, 2);
        // The cell at grid (1, 0) — and the storage slot its centered
        // coordinates land in.
        let slot = m.at(1, 0).expect("inside the map");
        assert!(m.set_cell(1, 0, 3, 1 | (1 << 2))); // flip x, one turn
        assert_eq!(m.tfms[slot], 1 | (1 << 2));
        assert!(!m.set_cell(1, 0, 3, 1 | (1 << 2)), "nothing moved");
        assert!(m.set_cell(1, 0, 3, 1 | (2 << 2)), "a turn is a change");
        assert_eq!(m.tfms[slot], 1 | (2 << 2));
        assert!(m.set_cell(1, 0, EMPTY_CELL, 0));
        assert_eq!(m.tfms[slot], 0, "erasing clears the bits");
        assert!(!m.set_cell(1, 0, EMPTY_CELL, 0), "twice empty is empty");
    }

    /// The store decodes onto the tiles: bits out, flags in — and
    /// the size stays as authored, because the renderer, not the
    /// store, is what turns the extents. The cut now rides the
    /// tileset: a 4 x 2 sheet dressed by a tileset cut 2 x 2 makes
    /// tiles of 2 x 1.
    #[test]
    fn a_layer_hands_the_renderer_its_cells_orientations() {
        let mut m = MapLayer::blank("t", 2, 2);
        m.set_cell(-1, -1, 0, 1 | (3 << 2)); // flip x, three quarters cw
        let shape = m
            .shape(&tileset("t"), Some(&atlas(4, 2)))
            .expect("one painted cell draws");
        let Shape::TileMap { tiles, .. } = shape else {
            panic!("a tile map");
        };
        assert_eq!(tiles.len(), 1);
        assert_eq!(tiles[0].rot, 3);
        assert!(tiles[0].flip_x);
        assert!(!tiles[0].flip_y);
        assert_eq!(tiles[0].size, [2.0, 1.0]);
    }

    /// The layer wears only the tileset it names: a desk handing the
    /// wrong document — same numbers, different sheet — dresses
    /// nothing rather than painting the world as nonsense.
    #[test]
    fn a_layer_wears_only_the_tileset_it_names() {
        let mut m = MapLayer::blank("grass", 2, 2);
        m.set_cell(0, 0, 0, 0);
        assert!(
            m.shape(&tileset("stone"), Some(&atlas(4, 2))).is_none(),
            "the wrong dressing is no dressing"
        );
        assert!(
            m.shape(&tileset("grass"), None).is_none(),
            "a closed sheet waits"
        );
    }

    /// The band convention the GPU bridge measures, pinned to the
    /// demo's own formula: a cell's uv counts from the file's top
    /// row — index 4 of a rack 2 wide × 3 deep is row 2, col 0,
    /// the bottom-left third of the atlas; index 1 is row 0,
    /// col 1, the top-right third. Bands counted from the bottom
    /// would render every multi-row tileset row-swapped, a bug
    /// only a real pixel can catch; this says the map hands the
    /// bridge the formula the bridge proved.
    #[test]
    fn a_cell_uv_is_its_band_in_the_files_own_order() {
        assert_eq!(cell_uv(4, 3, 2), [0.0, 2.0 / 3.0, 0.5, 1.0]);
        assert_eq!(cell_uv(1, 3, 2), [0.5, 0.0, 1.0, 1.0 / 3.0]);
    }

    /// A map's layers speak each of their tilesets once, in the
    /// order they first speak them — the desk's waiting list speaks
    /// these names.
    #[test]
    fn the_map_names_its_tilesets_once_in_order() {
        let layers = vec![
            MapLayer::blank("grass", 4, 2),
            MapLayer::blank("bug", 4, 2),
            MapLayer::blank("grass", 8, 2),
        ];
        assert_eq!(tileset_names(&layers), ["grass", "bug"]);
        assert!(
            tileset_names(&[]).is_empty(),
            "an empty map promises nothing"
        );
    }

    /// The chain's arithmetic: every reference resolves against the
    /// document that speaks it, `relative` is `resolve`'s inverse,
    /// `..` climbs and `.` vanishes, and an absolute reference
    /// starts over from the root the caller lives under. No
    /// filesystem, no platform — the same text walks a folder and a
    /// binary.
    #[test]
    fn references_resolve_against_the_document_that_speaks_them() {
        assert_eq!(
            resolve("maps/garden.map.ron", "g.tileset.ron"),
            "maps/g.tileset.ron"
        );
        assert_eq!(
            resolve("maps/garden.map.ron", "../sprites/g.tileset.ron"),
            "sprites/g.tileset.ron"
        );
        assert_eq!(resolve("maps/garden.map.ron", "./g.ron"), "maps/g.ron");
        assert_eq!(resolve("garden.map.ron", "g.ron"), "g.ron");
        assert_eq!(resolve("/world/maps/m.ron", "/other/s.ron"), "/other/s.ron");
        assert_eq!(resolve("maps\\garden.map.ron", "..\\g.ron"), "g.ron");
        assert_eq!(resolve("maps/deep/m.ron", "../../g.ron"), "g.ron");
        // The desk's inverse: shelve a tileset anywhere, reference
        // it from any map, and resolve returns to the shelf.
        for (dir, target) in [
            ("maps", "sprites/g.tileset.ron"),
            ("maps/old", "maps/g.tileset.ron"),
            ("", "g.tileset.ron"),
            ("assets/tilesets", "assets/sprites/s.png"),
        ] {
            // A map at the table's root is `m.ron`, not `/m.ron` —
            // the second is the absolute space the next test owns.
            let doc = if dir.is_empty() {
                "m.ron".to_string()
            } else {
                format!("{dir}/m.ron")
            };
            let rel = relative(dir, target);
            assert_eq!(resolve(&doc, &rel), target, "{dir} → {target} as '{rel}'");
        }
    }

    /// The whole cargo question, asked without a filesystem: a map,
    /// its tilesets and their sheets, all answered by closures over
    /// in-memory text. The chain walks when everything ships, and
    /// every missing link is named — including the one the map
    /// itself didn't carry.
    /// The promise the doc makes and the first tests forgot: hand
    /// `resolve` an absolute document — which is every document a
    /// file dialog hands the desk — and every answer keeps the
    /// anchor. A desk joining the answer onto the document's folder
    /// must not double the road.
    #[test]
    fn an_absolute_document_keeps_its_anchor() {
        assert_eq!(
            resolve(
                "/work/assets/maps/courtyard.map.ron",
                "../sprites/basic_tiles.tileset.ron"
            ),
            "/work/assets/sprites/basic_tiles.tileset.ron"
        );
        assert_eq!(
            resolve("/work/assets/sprites/x.tileset.ron", "x.png"),
            "/work/assets/sprites/x.png"
        );
        assert_eq!(resolve("/x/a.ron", "/rooted.ron"), "/rooted.ron");
    }

    #[test]
    fn the_guard_walks_the_chain_without_a_filesystem() {
        let grass = Tileset::new("grass", "grass.png", 2, 2);
        let world = std::collections::HashMap::from([
            ("maps/garden.map.ron".to_string(), {
                let mut layer = MapLayer::blank("grass", 4, 2);
                layer.set_cell(0, 0, 1, 0);
                to_text(&Map {
                    tilesets: vec!["../tilesets/grass.tileset.ron".into()],
                    layers: vec![layer],
                })
            }),
            (
                "tilesets/grass.tileset.ron".to_string(),
                tileset::to_text(&grass),
            ),
            ("tilesets/grass.png".to_string(), "png bytes".to_string()),
        ]);
        let read = |p: &str| world.get(p).cloned();
        let map_src = read("maps/garden.map.ron").expect("fixture");
        assert!(
            guard_documents("maps/garden.map.ron", &map_src, &read).is_empty(),
            "the chain walks: {:?}",
            guard_documents("maps/garden.map.ron", &map_src, &read)
        );
        // A sheet that stayed off the ship: the map loads its layers
        // fine — this is the cargo question, and it names the hole.
        let missing: std::collections::HashMap<String, String> = world
            .iter()
            .filter(|(k, _)| *k != "tilesets/grass.png")
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let refused = guard_documents("maps/garden.map.ron", &map_src, &|p| {
            missing.get(p).cloned()
        });
        assert_eq!(refused.len(), 1, "refused: {refused:?}");
        assert!(refused[0].contains("grass.png"), "refused: {refused:?}");
        // And the map that names a tileset its cargo doesn't carry —
        // refused by the reader already, named by the guard likewise.
        let orphan = to_text(&Map {
            tilesets: vec!["other.tileset.ron".into()],
            layers: vec![MapLayer::blank("grass", 4, 2)],
        });
        let refused = guard_documents("maps/garden.map.ron", &orphan, &read);
        assert!(!refused.is_empty(), "an orphan layer must not pass");
    }
}
