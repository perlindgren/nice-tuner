//! The tileset file: which sheet, which cut, and what each tile *is*.
//
// A tileset is a thing you can open. It names one sprite sheet, says
// how the sheet is cut, and — cell by cell — says which tiles stand
// in the world's way. Before it existed, a tileset was a way of
// wearing a sprite: a name borrowed from the file stem and a cut kept
// in the sprite's sidecar, so "one sheet, two tilesets" meant one
// file in two moods, and a tile's solidity had no home at all. The
// amendment made the middle of the chain a document — map → tileset →
// sheet — where every link is a reference resolved against the
// document that speaks it, every document testifies to its own name,
// and the guard can ask, at cargo-test time and without a
// filesystem, whether a shipped world carries every path the chain
// walks.

use crate::map::{entries, number, pos, text};
use crate::ron::{self, Item, Kind, Val};

/// One cell's role, spoken by number: the atlas cell `index` (the
/// same numbers the map's cells carry, row-major from the sheet's
/// top-left) and whether it is solid. Roles absent from the file are
/// absent in the world — an unlisted tile asks nothing of a collider.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TileCell {
    pub index: usize,
    pub solid: bool,
}

/// A tileset document: the sheet it wears, the cut made into it, and
/// the cells' roles. The `name` is spoken inside the file — paths
/// find documents, names *are* documents, and a desk that shelves
/// `moss.tileset.ron` around a document that says `name: "grass"`
/// names that disagreement rather than picking a winner.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Tileset {
    pub name: String,
    /// The sheet, as a path **relative to this document's own
    /// folder** — the convention every browser and shader include
    /// already speaks. A tileset beside its sheet names it by file
    /// name; a sheet one folder up is `"../tiles/grass.png"`.
    pub sheet: String,
    pub rows: usize,
    pub cols: usize,
    /// The cells the author dressed. Reading keeps them in file
    /// order; asking never needs order.
    pub cells: Vec<TileCell>,
}

impl Tileset {
    /// A tileset from its honest parts: no cells dressed yet.
    pub fn new(
        name: impl Into<String>,
        sheet: impl Into<String>,
        rows: usize,
        cols: usize,
    ) -> Self {
        Self {
            name: name.into(),
            sheet: sheet.into(),
            rows,
            cols,
            cells: Vec::new(),
        }
    }

    /// How many cells the cut makes.
    pub fn cell_count(&self) -> usize {
        self.rows * self.cols
    }

    /// Whether cell `index` stands in the world's way: only an
    /// explicitly solid entry says so; unlisted and listed-solid-
    /// false cells both ask nothing of a collider.
    #[must_use]
    pub fn is_solid(&self, index: usize) -> bool {
        self.cells.iter().any(|c| c.index == index && c.solid)
    }

    /// Dress or clear one cell, keeping the file's honesty: setting
    /// `solid` false on an unlisted cell adds nothing, and dressing
    /// the same cell twice replaces rather than repeats.
    pub fn set_solid(&mut self, index: usize, solid: bool) {
        self.cells.retain(|c| c.index != index);
        if solid {
            self.cells.push(TileCell { index, solid });
        }
    }
}

/// The header every tileset file carries: the format, spoken to
/// whoever opens it next.
const TILESET_FILE_HEADER: &str = "// frost tileset file — one sheet, one cut, one set of roles.\n\
// sheet: the sprite file, by path relative to THIS file. cut: [rows,\n\
// cols] of cells, row-major from the sheet's top-left. cells: the\n\
// tiles that stand in the world's way, by cell number — a cell not\n\
// listed is not solid, and solid cells are the only ones written.";

/// A bracketed list of values.
fn list(items: Vec<Val>) -> Val {
    Val::Seq {
        open: true,
        items: items.into_iter().map(Item::plain).collect(),
        tail: String::new(),
    }
}

/// A tileset's tree: name, sheet, cut, and only the standing roles.
fn tileset_value(t: &Tileset) -> Val {
    let cells = t
        .cells
        .iter()
        .filter(|c| c.solid)
        .map(|c| {
            ron::record_named(&[
                ("index", ron::number(c.index as f64)),
                ("solid", ron::flag(true)),
            ])
        })
        .collect();
    ron::record_named(&[
        ("name", ron::text(&t.name)),
        ("sheet", ron::text(&t.sheet)),
        (
            "cut",
            list(vec![ron::number(t.rows as f64), ron::number(t.cols as f64)]),
        ),
        ("cells", list(cells)),
    ])
}

/// The `solid:` atom, with the reading's generosity: a bare listed
/// cell is a dressed cell, so a missing `solid:` means `true`.
fn boolean(v: &Val) -> Option<bool> {
    match v {
        Val::Atom(s, Kind::Bool) => Some(s == "true"),
        _ => None,
    }
}

/// The document's tileset, or refusals — the root either speaks name,
/// sheet and an honest cut or speaks nothing, while individual cells
/// are skipped where they lie, noted by name: a tileset dresses as
/// much of the sheet as it can.
fn tileset_of(root: &Val) -> (Option<Tileset>, Vec<String>) {
    let mut refused = Vec::new();
    let (Some(name), Some(sheet)) = (
        ron::field(root, "name").and_then(text),
        ron::field(root, "sheet").and_then(text),
    ) else {
        return (
            None,
            vec!["no name and sheet as name: \"…\", sheet: \"…\"".to_string()],
        );
    };
    let cut = ron::field(root, "cut");
    let at = |n| cut.and_then(|c| pos(c, n));
    let (Some(rows), Some(cols)) = (
        at(0)
            .and_then(number)
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0),
        at(1)
            .and_then(number)
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|v| *v > 0),
    ) else {
        return (
            None,
            vec![format!("'{name}' says no honest cut as [rows, cols]")],
        );
    };
    let mut t = Tileset::new(name, sheet, rows, cols);
    if let Some(items) = ron::field(root, "cells").and_then(entries) {
        for (n, item) in items.iter().enumerate() {
            let read = || {
                let index = pos(&item.val, 0).or_else(|| ron::field(&item.val, "index"))?;
                let index = number(index)?.parse::<usize>().ok()?;
                let solid = ron::field(&item.val, "solid")
                    .and_then(boolean)
                    .unwrap_or(true);
                Some((index, solid))
            };
            match read() {
                Some((index, solid)) if index < rows * cols => {
                    if solid {
                        t.cells.push(TileCell { index, solid });
                    }
                }
                Some((index, _)) => refused.push(format!(
                    "cell {} speaks index {index}, past the rack's {}",
                    n + 1,
                    rows * cols
                )),
                None => refused.push(format!("cell {} is not (index, solid)", n + 1)),
            }
        }
    }
    (Some(t), refused)
}

/// A tileset to file text, header and all. The desk writes these
/// beside the sheets they cut; a game reads them through
/// [`from_text`](self::from_text) at the end of a map's reference —
/// one dialect all the way down the chain.
#[must_use]
pub fn to_text(t: &Tileset) -> String {
    ron::to_text_doc(TILESET_FILE_HEADER, &tileset_value(t), "")
}

/// File text to a tileset: the whole document or named refusals —
/// name, sheet and cut are all-or-nothing, because a tileset that
/// doesn't say which sheet it cuts is not a partial fact but no
/// fact. Untruths among the cells are skipped and named; the
/// tileset still dresses the sheet as far as it is honest.
pub fn from_text(src: &str) -> (Option<Tileset>, Vec<String>) {
    match ron::parse_doc(src) {
        Ok(doc) => tileset_of(&doc.root),
        Err(err) => (None, vec![format!("not a RON document: {err}")]),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// In as a dressed tileset, out as text, back in: name, sheet,
    /// cut and every standing role survive the page.
    #[test]
    fn the_tileset_round_trips_through_text() {
        let mut t = Tileset::new("moss", "moss.png", 2, 4);
        t.set_solid(1, true);
        t.set_solid(7, true);
        let (read, refused) = from_text(&to_text(&t));
        assert!(refused.is_empty(), "refused: {refused:?}");
        assert_eq!(read.as_ref(), Some(&t));
    }

    /// A file is written in roles that stand: an undressed sheet
    /// carries an empty `cells:` list, and clearing a dressed cell
    /// dresses nothing in its place.
    #[test]
    fn only_standing_roles_are_written() {
        let bare = Tileset::new("t", "t.png", 1, 4);
        let text = to_text(&bare);
        assert!(text.contains("cells: []"), "written:\n{text}");
        let mut t = Tileset::new("t", "t.png", 1, 4);
        t.set_solid(1, true);
        t.set_solid(2, true);
        t.set_solid(2, false);
        assert_eq!(t.cells.len(), 1, "clearing takes only its own cell");
        assert!(t.is_solid(1), "the neighbours stand");
        assert!(!t.is_solid(2), "and the cleared cell asks nothing");
        t.set_solid(1, false);
        assert!(t.cells.is_empty(), "cleared in place, in any order");
    }

    /// A document that doesn't say whose sheet it cuts is refused
    /// whole, by the refusal's own words — there is no partial
    /// tileset to dress anything.
    #[test]
    fn a_tileset_must_speak_its_sheet_and_cut() {
        let (t, refused) = from_text("(name: \"grass\", cells: [])");
        assert!(t.is_none());
        assert_eq!(refused.len(), 1);
        let (t, refused) = from_text("(name: \"grass\", sheet: \"g.png\", cut: [0, 4])");
        assert!(t.is_none(), "a cut of no rows cuts nothing: {refused:?}");
        assert!(refused[0].contains("cut"), "refused: {refused:?}");
    }

    /// Cells are read as far as they are honest: an index past the
    /// rack is named and skipped, a bare `(index: n)` entry counts
    /// as dressed — why else would the author have listed it — and
    /// the tileset still dresses what it can.
    #[test]
    fn the_cells_are_read_as_far_as_they_are_honest() {
        let (t, refused) = from_text(
            "(name: \"t\", sheet: \"t.png\", cut: [2, 2], cells: [(index: 1), (index: 9, solid: true), (index: 2, solid: false)])",
        );
        let t = t.expect("the root speaks");
        assert_eq!(refused.len(), 1, "refused: {refused:?}");
        assert!(refused[0].contains("past the rack"), "refused: {refused:?}");
        assert!(t.is_solid(1), "a bare entry is a dressed cell");
        assert!(!t.is_solid(2), "solid: false dresses nothing");
        assert!(!t.is_solid(0), "and the unlisted ask nothing");
    }

    /// The numbers mean what the map says they mean: the rack counts
    /// row-major from the sheet's top-left, and solidity is asked by
    /// exactly those numbers.
    #[test]
    fn solidity_is_asked_by_the_maps_own_numbers() {
        let mut t = Tileset::new("t", "t.png", 2, 4);
        assert_eq!(t.cell_count(), 8);
        t.set_solid(5, true); // row 1, col 1
        assert!(t.is_solid(5));
        assert!(
            !t.is_solid(1),
            "the same column in another row is another tile"
        );
    }
}
