//! Column layout for the [`Ui`](crate::ui::Ui) table: the column
//! specs, the resolved geometry of an open table, and the cell
//! cursor that hands widgets out across a row.

use super::Ui;
use crate::Context;

/// Where a cell's content sits inside its table column, when the column
/// is wider than the content: at the column's left edge, its middle, or
/// its right edge. Filling widgets (a `slider_track`, a `button`) ignore
/// it — they span the cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Align {
    Left,
    Center,
    Right,
}

/// How a table column takes its share of the row's width.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ColSize {
    /// As wide as the column's widest cell. Like the rest of the UI this
    /// is one frame behind: the width comes from what the column held
    /// last frame (the frame a table first appears, auto columns start
    /// at zero width).
    Auto,
    /// A fixed width in pixels.
    Px(f32),
    /// A share of what the other columns leave over, in proportion to
    /// the weight (the weight only matters relative to the other stretch
    /// columns).
    Stretch(f32),
}

/// One column of a table: how wide it is, and how its cells align within
/// that. Build with [`Col::auto`], [`Col::px`] and [`Col::stretch`];
/// declare a row grid with [`Ui::table`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Col {
    /// The width policy.
    pub size: ColSize,
    /// How cells align within the column.
    pub align: Align,
}

impl Col {
    /// A column as wide as its widest cell.
    pub fn auto(align: Align) -> Self {
        Self {
            size: ColSize::Auto,
            align,
        }
    }

    /// A column of a fixed pixel width.
    pub fn px(width: f32, align: Align) -> Self {
        Self {
            size: ColSize::Px(width),
            align,
        }
    }

    /// A column taking the leftover width, weighted against the table's
    /// other stretch columns.
    pub fn stretch(weight: f32, align: Align) -> Self {
        Self {
            size: ColSize::Stretch(weight),
            align,
        }
    }
}

/// The running state of an open table: the resolved column geometry, the
/// cell cursor walking cells left to right and wrapping to a new row, and
/// the widest content seen in each auto column this frame.
pub(super) struct TableFrame {
    /// The size policy of each column (for the auto-width accounting).
    pub(super) sizes: Vec<ColSize>,
    /// Each column's resolved pixel width, this frame.
    pub(super) widths: Vec<f32>,
    /// Each column's alignment.
    pub(super) aligns: Vec<Align>,
    /// Each column's left edge in user space.
    pub(super) x: Vec<f32>,
    /// The index of the next cell to hand out (`widths.len()` apart cells
    /// are the same column of successive rows).
    pub(super) next: usize,
    /// The top edge of the table's first row.
    pub(super) top: f32,
    /// The widest content measured this frame, per column.
    pub(super) auto: Vec<f32>,
}

impl Ui {
    /// Feeds an auto column the width of the content just laid out in it;
    /// the column grows to it next frame, as everything else in the UI
    /// does (see [`ColSize::Auto`]).
    pub(super) fn note_cell_width(&mut self, w: f32) {
        if let Some(layout) = self.layout.last_mut()
            && let Some(t) = &mut layout.table
            && t.next > 0
        {
            let col = (t.next - 1) % t.widths.len();
            if matches!(t.sizes[col], ColSize::Auto) {
                t.auto[col] = t.auto[col].max(w);
            }
        }
    }

    /// A table: the widgets `content` declares fill cells left to right,
    /// wrapping to a new row after every `cols.len()` of them. Each column
    /// takes width per its [`ColSize`] — fixed, auto (as wide as its
    /// widest cell, from last frame) or stretch (a weighted share of the
    /// leftover) — and aligns its cells per its [`Align`]. `id` names the
    /// table within its panel: the retained auto widths live under it, so
    /// two tables in one panel keep their own column widths.
    ///
    /// ```text
    /// ui.table(ctx, "view", &[
    ///     Col::auto(Align::Left),           // the labels, hugging, left
    ///     Col::stretch(1.0, Align::Left),   // the tracks, filling
    ///     Col::auto(Align::Center),         // the readouts, centered
    /// ], |ui, ctx| {
    ///     ui.label(ctx, "zoom");
    ///     ui.slider_track(ctx, "zoom", &mut zoom, 0.25, 4.0);
    ///     ui.readout(ctx, &format!("{zoom:.2}"));
    ///     // ... the next three widgets wrap onto the second row
    /// });
    /// ```
    pub fn table(
        &mut self,
        ctx: &mut Context,
        id: &str,
        cols: &[Col],
        content: impl FnOnce(&mut Ui, &mut Context),
    ) {
        if self.table_open(id, cols) {
            content(self, ctx);
            self.table_close(id);
        }
    }

    /// The window-free heart of [`Ui::table`], split out so the column
    /// geometry is testable without a `Context`: resolves the columns and
    /// opens the table frame on the current layout. Returns `false` (and
    /// opens nothing) for an empty column list.
    pub(super) fn table_open(&mut self, id: &str, cols: &[Col]) -> bool {
        if cols.is_empty() {
            return false;
        }
        let style = self.style;
        let key = self.widget_id(id);
        let stored = self.tables.get(&key).cloned().unwrap_or_default();
        let (left, width, cursor) = {
            let layout = self.layout.last().expect("Ui layout stack is never empty");
            (layout.left, layout.width, layout.cursor)
        };
        let widths = resolve_cols(cols, width, style.col_gap, &stored);
        let n = widths.len();
        let mut x = Vec::with_capacity(n);
        let mut edge = left;
        for w in &widths {
            x.push(edge);
            edge += w + style.col_gap;
        }
        // The table's first row hangs exactly where a plain row would.
        let top = cursor - style.row_gap;
        self.layout
            .last_mut()
            .expect("Ui layout stack is never empty")
            .table = Some(TableFrame {
            sizes: cols.iter().map(|c| c.size).collect(),
            widths,
            aligns: cols.iter().map(|c| c.align).collect(),
            x,
            next: 0,
            top,
            auto: vec![0.0; n],
        });
        true
    }

    /// Closes the frame [`Ui::table_open`] opened: retains the measured
    /// auto widths and moves the layout cursor below the table's rows.
    pub(super) fn table_close(&mut self, id: &str) {
        let key = self.widget_id(id);
        let style = self.style;
        let frame = self
            .layout
            .last_mut()
            .expect("Ui layout stack is never empty")
            .table
            .take()
            .expect("table_close pairs with table_open");
        self.tables.insert(key, frame.auto);
        if frame.next > 0 {
            // The table leaves the cursor exactly where its rows of plain
            // rows would have, so the next widget stacks below normally.
            let n = frame.widths.len();
            let rows = frame.next.div_ceil(n) as f32;
            let cursor = frame.top + style.row_gap;
            let layout = self
                .layout
                .last_mut()
                .expect("Ui layout stack is never empty");
            layout.cursor = cursor - rows * (style.row_h + style.row_gap);
        }
    }
}
/// Resolve each column's pixel width across a row of `total_width`: fixed
/// columns take their width, auto columns the `stored` width measured last
/// frame (0 until first seen), and stretch columns then split what is left
/// over the gaps, weighted. Pure — `Ui::table` wraps it, and the tests
/// exercise it directly.
pub(super) fn resolve_cols(
    cols: &[Col],
    total_width: f32,
    col_gap: f32,
    stored: &[f32],
) -> Vec<f32> {
    let n = cols.len();
    let gaps = (n as f32 - 1.0).max(0.0) * col_gap.max(0.0);
    let mut widths = vec![0.0f32; n];
    let mut used = 0.0f32;
    let mut weight = 0.0f32;
    for (i, c) in cols.iter().enumerate() {
        let w = match c.size {
            ColSize::Px(w) => w.max(0.0),
            ColSize::Auto => stored.get(i).copied().unwrap_or(0.0).max(0.0),
            ColSize::Stretch(s) => {
                weight += s.max(0.0);
                0.0
            }
        };
        used += w;
        widths[i] = w;
    }
    if weight > 0.0 {
        let free = (total_width - gaps - used).max(0.0);
        for (i, c) in cols.iter().enumerate() {
            if let ColSize::Stretch(s) = c.size {
                widths[i] = free * s.max(0.0) / weight;
            }
        }
    }
    widths
}
