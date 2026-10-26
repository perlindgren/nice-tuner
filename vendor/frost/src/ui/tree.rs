//! The tree view: a two-axis-scrolling list of `key: val` rows in a panel,
//! with a close `×` in the title bar, a corner grip that resizes the view,
//! vertical and horizontal scroll thumbs, per-row `+`/`×` buttons, and
//! drag-reorder of rows that show a `×`.
//!
//! It is deliberately generic over its rows: the library knows only
//! [`TreeLine`]s (key, value, a few flags, a value color) and emits events
//! with row *indices*; the application maps them to its own data model.
//! The default metrics and colors are those of the RON sidecar tree.
//!
//! **One call per frame, all views.** [`Ui::tree`] takes *every* view's
//! [`TreeSpec`] at once and returns one [`TreeOut`] per spec, in order.
//! The views share one [`TreeState`] owned by the application. Running the
//! release resolution over all views in one pass is what makes cross-view
//! semantics exact: a release that lands inside view 0's body *and* view
//! 1's close box closes view 1 (the close box wins), which no per-view
//! call sequence could reproduce, because view 0 would have claimed the
//! press first.
//!
//! **Claim order, one press frame.** A press claims the first thing that
//! contains the pointer, in spec order: the close box (even of a folded
//! view), then — for unfolded views — the corner grip, the vertical
//! thumb, the horizontal thumb, and finally the body. The first claim
//! wins; later views see nothing. A claim is consumed on the *release*
//! frame, always: a release whose travel from the press exceeds
//! [`CLICK_TOL`] is a failed click (nothing happens), but the press is
//! gone either way, so no stale claim can block a later press.
//!
//! **Drag start and move.** While a body press is held, a drift of
//! [`CLICK_TOL`] or more over a row whose `del` flag is set starts a
//! reorder drag (`DragStart`); afterwards the current row under the
//! pointer is reported every frame — including the release frame, which
//! reports the final seat before `DragEnd` — and the pointer leaving all
//! rows reports `Drag { row: None }`. A press on a row *without* the
//! `del` flag never drags, and if the release is still (travel below
//! tolerance) it clicks that row as usual.
//!
//! **Paint.** [`Ui::tree_paint`] paints one view: rows (windowed
//! horizontally by `sx` and cullable vertically), the selection band,
//! the `+`/`×` plates and glyphs, both scroll thumbs, the corner ticks
//! (for every unfolded view, even one whose body is too small to hold a
//! row), and the close `×` (for every view, folded or not). All of a
//! view's draws sit within `spec.z`, `spec.z - 0.5` and `spec.z - 0.45`,
//! so painting the views in any order is safe.

use crate::{Color, Context};

use super::Ui;

/// The pointer's drift during a press below which a release still counts
/// as a click; at or above it the press was a drag and the release does
/// nothing (except end a reorder drag).
const CLICK_TOL: f32 = 4.0;
/// How close the release point must be to a `+`/`×` button *center* for
/// the click to hit that button rather than the row.
const BTN_NEAR: f32 = 9.0;
/// The size of the `+`/`×` glyphs in a row.
const GLYPH_SIZE: f32 = 15.0;
/// The close glyph's extra size over the row size (`size + 6`).
const CLOSE_EXTRA: f32 = 6.0;
/// The half-extent of the `+`/`×` plate rectangles.
const PLATE: f32 = 9.0;
/// The `+` button's offset from the body's right edge.
const ADD_OFF: f32 = 20.0;
/// The `×` button's offset from the body's right edge when a `+` shares
/// the row (the `×` moves left to make room).
const DEL_OFF: f32 = 42.0;
/// The selection band's inset from each body edge.
const BAND_INSET: f32 = 12.0;
/// The half-extent of a scroll track.
const TRACK_HALF: f32 = 4.0;
/// The scroll thumb's offset from the body's right (vertical) or bottom
/// (horizontal) edge.
const STRIP_OUT: f32 = 8.0;
/// The scroll thumb's offset from the body's right (vertical) or bottom
/// (horizontal) edge, on the inner side.
const STRIP_IN: f32 = 2.0;
/// The shortest a scroll thumb may be, so it stays grabbable.
const THUMB_MIN: f32 = 24.0;
/// The half-extent of the close `×` box.
const CLOSE_HALF: f32 = 11.0;
/// The corner ticks' offsets from the grip's diagonal.
const TICKS: [f32; 3] = [4.0, 8.0, 12.0];
/// The corner ticks' offset from the panel's right and bottom edges.
const TICK_IN: f32 = 3.0;

/// One row of a [`TreeSpec`]: a `key: val` pair and its interaction flags.
///
/// The flags the application sets per row: `head` marks a container head
/// (its value paints bold), `val_color` is the final value color — a type
/// color, or the palette color of a spot this row owns — `band` paints the
/// row under the selection band — one per physical row of the selected
/// logical row, which a spot can span — and `add`/`del` show the
/// `+`/`×` buttons. A row showing `×` is the draggable kind: its entries
/// can be reordered by dragging.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TreeLine<'a> {
    /// The row's key, painted in weight 700.
    pub key: &'a str,
    /// The row's value, painted in weight 700 when `head`, else 500.
    pub val: &'a str,
    /// Whether the value is a container head (painted bold).
    pub head: bool,
    /// The value's color, chosen by the application.
    pub val_color: Color,
    /// Whether the row shows the green `+` button.
    pub add: bool,
    /// Whether the row shows the red `×` button, and may be reordered by
    /// dragging.
    pub del: bool,
    /// Whether the selection band paints under this row.
    pub band: bool,
}

/// The metrics and colors of a tree view.
///
/// `Default` is the RON sidecar's look: 16 px rows 20 px high with 9.6 px
/// character advance, a 0.344 lift, 12 px body inset, 6 px row padding,
/// the usual view-size limits, and the sidecar's palette of slate, moss
/// and ember.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct TreeStyle {
    /// The row text size.
    pub size: f32,
    /// The row height.
    pub row_h: f32,
    /// The character advance (pixels per character).
    pub adv: f32,
    /// The text lift above the row center (fraction of the size).
    pub lift: f32,
    /// The body inset: the distance from the body's left edge to the text.
    pub inset: f32,
    /// The body's padding, top and bottom of the row range.
    pub pad: f32,
    /// The fewest characters the view may show, however narrow it is.
    pub min_chars: u32,
    /// The smallest view width, set by the corner grip.
    pub min_w: f32,
    /// The largest view width.
    pub max_w: f32,
    /// The smallest view height.
    pub min_h: f32,
    /// The largest view height.
    pub max_h: f32,
    /// The key text color.
    pub key: Color,
    /// The selection band color, under each row whose `band` is set.
    pub band: Color,
    /// The `+` plate color.
    pub add_bg: Color,
    /// The `+` glyph color.
    pub add_glyph: Color,
    /// The `×` plate color.
    pub del_bg: Color,
    /// The `×` glyph color.
    pub del_glyph: Color,
    /// The scroll track color.
    pub track: Color,
    /// The scroll thumb color.
    pub thumb: Color,
    /// The close `×` box color, unhovered.
    pub close_bg: Color,
    /// The close `×` box color, hovered.
    pub close_bg_hot: Color,
    /// The close glyph color, unhovered (its alpha is the resting
    /// opacity; the hover sets alpha to 1).
    pub close_glyph: Color,
    /// The corner grip's tick color.
    pub grip: Color,
}

impl Default for TreeStyle {
    fn default() -> Self {
        Self {
            size: 16.0,
            row_h: 20.0,
            adv: 9.6,
            lift: 0.344,
            inset: 12.0,
            pad: 6.0,
            min_chars: 8,
            min_w: 150.0,
            max_w: 640.0,
            min_h: 80.0,
            max_h: 560.0,
            key: Color {
                r: 0.87,
                g: 0.87,
                b: 0.90,
                a: 1.0,
            },
            band: Color {
                r: 0.30,
                g: 0.40,
                b: 0.62,
                a: 0.9,
            },
            add_bg: Color {
                r: 0.16,
                g: 0.34,
                b: 0.20,
                a: 0.95,
            },
            add_glyph: Color {
                r: 0.80,
                g: 1.00,
                b: 0.82,
                a: 1.0,
            },
            del_bg: Color {
                r: 0.42,
                g: 0.17,
                b: 0.19,
                a: 0.95,
            },
            del_glyph: Color {
                r: 0.98,
                g: 0.72,
                b: 0.68,
                a: 1.0,
            },
            track: Color {
                r: 0.10,
                g: 0.11,
                b: 0.15,
                a: 0.8,
            },
            thumb: Color {
                r: 0.44,
                g: 0.46,
                b: 0.54,
                a: 1.0,
            },
            close_bg: Color {
                r: 0.30,
                g: 0.14,
                b: 0.16,
                a: 0.95,
            },
            close_bg_hot: Color {
                r: 0.52,
                g: 0.20,
                b: 0.22,
                a: 0.95,
            },
            close_glyph: Color {
                r: 0.95,
                g: 0.68,
                b: 0.64,
                a: 0.9,
            },
            grip: Color {
                r: 0.55,
                g: 0.58,
                b: 0.66,
                a: 0.9,
            },
        }
    }
}

/// One view of a tree, for one frame: where the view lives, what it shows,
/// and the view's own state.
///
/// The application builds one per tree panel each frame, from the panel
/// layout, and collects them in slot order for [`Ui::tree`]. `body` is
/// `Some` only when the view is unfolded and tall enough to hold a row;
/// a freshly opened view can have an empty body for one frame, and a
/// folded one always does. `size` is the view's `[w, h]`, stored by the
/// application from the previous frame's [`TreeOut::size`].
#[derive(Clone, Copy, Debug)]
pub struct TreeSpec<'a> {
    /// The view's identity, echoed into [`TreeState`] so a drag started in
    /// one view reports to it alone.
    pub id: u64,
    /// The rows, top of the list first.
    pub rows: &'a [TreeLine<'a>],
    /// The panel's rect, `[left, bottom, right, top]` — used for the
    /// corner grip's box and ticks.
    pub panel: [f32; 4],
    /// The row region's rect, `[left, bottom, right, top]`; `None` when
    /// the view is too small to show a row.
    pub body: Option<[f32; 4]>,
    /// The close `×`'s center, in the panel's title bar.
    pub close: [f32; 2],
    /// Whether the view is folded: rows, body and grip are inert, only
    /// the close `×` answers.
    pub folded: bool,
    /// The vertical scroll, in pixels from the top of the row range.
    pub scroll: f32,
    /// The horizontal scroll, in pixels from the left of the row range.
    pub sx: f32,
    /// The view's size, `[w, h]`; the scroll clamps and the resize use it.
    pub size: [f32; 2],
    /// The view's metrics and colors.
    pub style: TreeStyle,
    /// The base z for everything this view paints; the band, plates,
    /// close box, ticks and tracks sit a half-step or so below it.
    pub z: f32,
}

/// The shared interaction state of a set of tree views: the current press
/// claim, thumb or grip grab, and reorder drag.
///
/// The application owns one instance — `TreeState::default()` — and passes
/// it to every [`Ui::tree`] call. All the fields are private: the state
/// moves only through the interaction itself, a press claimed on one
/// frame is always consumed by the release on a later one, and no stale
/// claim can outlive it.
#[derive(Clone, Copy, Debug, Default)]
pub struct TreeState {
    press: Option<[f32; 2]>,
    grab: Option<TreeGrab>,
    drag: Option<TreeDrag>,
}

/// What a press grabbed: a vertical thumb (`V`), a horizontal thumb (`H`)
/// or the corner grip (`R`).
#[derive(Clone, Copy, Debug)]
enum TreeGrab {
    V {
        view: u64,
        off: f32,
        len: f32,
    },
    H {
        view: u64,
        off: f32,
        len: f32,
    },
    R {
        view: u64,
        ax: f32,
        ay: f32,
        w0: f32,
        h0: f32,
    },
}

/// The in-flight reorder drag: which view started it. The source row is
/// in the `DragStart` event, and the seat in each `Drag` event, so the
/// state only needs the view to route the final `DragEnd` back to it.
#[derive(Clone, Copy, Debug)]
struct TreeDrag {
    view: u64,
}

/// One tree event, with the row *index* in the spec's rows — the
/// application maps it to its own data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum TreeEvent {
    /// The release landed on this view's close `×`.
    Close,
    /// A still click on a row: pick it (or toggle its fold).
    Row { row: usize },
    /// A still click on the row's `+` button: append an entry.
    Add { row: usize },
    /// A still click on the row's `×` button: remove the entry.
    Del { row: usize },
    /// The press drifted past tolerance over a draggable row: the
    /// reorder began.
    DragStart { row: usize },
    /// The drag's current seat: the row under the pointer, or `None` when
    /// it is over no row (the drop would be a no-op).
    Drag { row: Option<usize> },
    /// The drag ended, after the final [`TreeEvent::Drag`] report.
    DragEnd,
}

/// One view's frame output, in spec order: the events it produced, and
/// the scroll and size to store — echoed from the spec unless a thumb or
/// the corner grip drove them.
#[derive(Debug, PartialEq)]
pub struct TreeOut {
    /// The events this view produced this frame, in the order they
    /// happened (a drag reports `Drag` and `DragEnd` on the release
    /// frame).
    pub events: Vec<TreeEvent>,
    /// The vertical scroll to store next frame.
    pub scroll: f32,
    /// The horizontal scroll to store next frame.
    pub sx: f32,
    /// The view size to store next frame, after a corner drag.
    pub size: [f32; 2],
}

impl Ui {
    /// Drives one frame over *all* the given tree views, sharing
    /// `state`, and returns each view's [`TreeOut`] in spec order.
    ///
    /// Call it once per frame, in the same place the views are painted,
    /// with the specs built from that frame's panel layout. The per-frame
    /// order — press claim, grab drive, drag start, drag move, release
    /// resolution — is the sidecar's, so the views' interactions match
    /// the application that this widget replaced.
    pub fn tree(&mut self, specs: &[TreeSpec<'_>], state: &mut TreeState) -> Vec<TreeOut> {
        let mut outs: Vec<TreeOut> = specs
            .iter()
            .map(|s| TreeOut {
                events: Vec::new(),
                scroll: s.scroll,
                sx: s.sx,
                size: s.size,
            })
            .collect();

        // 1. A press claims the first thing that contains the pointer, in
        // spec order; the first claim wins.
        if self.pressed
            && let Some(p) = self.pointer
            && state.press.is_none()
            && state.grab.is_none()
        {
            for spec in specs {
                if in_rect(close_box(spec.close), p) {
                    state.press = Some(p);
                    break;
                }
                if spec.folded {
                    continue;
                }
                if in_rect(corner_box(spec.panel), p) {
                    state.grab = Some(TreeGrab::R {
                        view: spec.id,
                        ax: p[0],
                        ay: p[1],
                        w0: spec.size[0],
                        h0: spec.size[1],
                    });
                    break;
                }
                if let Some(body) = spec.body {
                    let total = spec.rows.len() as f32 * spec.style.row_h + spec.style.pad;
                    if let Some(r) = v_thumb(body, spec.scroll, total)
                        && in_rect(r, p)
                    {
                        state.grab = Some(TreeGrab::V {
                            view: spec.id,
                            off: r[3] - p[1],
                            len: r[3] - r[1],
                        });
                        break;
                    }
                    if let Some(r) = h_thumb(body, spec.sx, width_of(spec.rows), spec.style)
                        && in_rect(r, p)
                    {
                        state.grab = Some(TreeGrab::H {
                            view: spec.id,
                            off: p[0] - r[0],
                            len: r[2] - r[0],
                        });
                        break;
                    }
                    if in_rect(body, p) {
                        state.press = Some(p);
                        break;
                    }
                }
            }
        }

        // 2. A held grab drives its view's scroll or size; the result is
        // clamped against the view's dims, not the body's — and a corner
        // drag clamps against the size it just set, as the view's own
        // clamp does.
        if self.down
            && let Some(p) = self.pointer
            && let Some(grab) = state.grab
        {
            for (out, spec) in outs.iter_mut().zip(specs) {
                let (driven, size) = match grab {
                    TreeGrab::R {
                        view,
                        ax,
                        ay,
                        w0,
                        h0,
                    } if view == spec.id => {
                        let size = resize_size(w0, h0, ax, ay, p, spec.style);
                        out.size = size;
                        (true, size)
                    }
                    TreeGrab::V { view, off, len } if view == spec.id => match spec.body {
                        Some(body) => {
                            let track = body[3] - body[1];
                            let total = spec.rows.len() as f32 * spec.style.row_h + spec.style.pad;
                            let travel = (track - len).max(f32::EPSILON);
                            let at = (body[3] - p[1] - off).clamp(0.0, travel);
                            out.scroll = at / travel * (total - track).max(0.0);
                            (true, spec.size)
                        }
                        None => (false, spec.size),
                    },
                    TreeGrab::H { view, off, len } if view == spec.id => match spec.body {
                        Some(body) => {
                            let track = body[2] - body[0];
                            let travel = (track - len).max(f32::EPSILON);
                            let at = (p[0] - body[0] - off).clamp(0.0, travel);
                            let chars = chars_for(track, spec.style) as f32;
                            let wide =
                                (width_of(spec.rows) as f32 - chars).max(0.0) * spec.style.adv;
                            out.sx = at / travel * wide;
                            (true, spec.size)
                        }
                        None => (false, spec.size),
                    },
                    _ => (false, spec.size),
                };
                if driven {
                    out.scroll = out.scroll.clamp(
                        0.0,
                        (spec.rows.len() as f32 * spec.style.row_h + spec.style.pad - size[1])
                            .max(0.0),
                    );
                    out.sx = out.sx.clamp(
                        0.0,
                        ((width_of(spec.rows) as f32 - chars_for(size[0], spec.style) as f32)
                            * spec.style.adv)
                            .max(0.0),
                    );
                }
            }
        }

        // 3. A held body press that drifted past tolerance starts a
        // reorder drag when its row may be dragged. The first view whose
        // body contains the press decides; a non-draggable row leaves the
        // press in place, so a still release clicks it as usual.
        if self.down
            && state.grab.is_none()
            && state.drag.is_none()
            && let Some(from) = state.press
            && self.pointer.is_some()
            && self.click_travel() >= CLICK_TOL
        {
            for (out, spec) in outs.iter_mut().zip(specs) {
                if spec.folded || spec.body.is_none() {
                    continue;
                }
                let body = spec.body.unwrap();
                if !in_rect(body, from) {
                    continue;
                }
                if let Some(i) = row_at(from, body, spec.scroll, spec.rows.len(), spec.style)
                    && spec.rows[i].del
                {
                    state.drag = Some(TreeDrag { view: spec.id });
                    out.events.push(TreeEvent::DragStart { row: i });
                    state.press = None;
                }
                break;
            }
        }

        // 4. A drag reports the row under the pointer, every frame — the
        // release frame too, for the final seat.
        if let Some(drag) = state.drag
            && let Some(p) = self.pointer
        {
            for (out, spec) in outs.iter_mut().zip(specs) {
                if drag.view != spec.id {
                    continue;
                }
                let row = spec
                    .body
                    .and_then(|body| row_at(p, body, spec.scroll, spec.rows.len(), spec.style));
                out.events.push(TreeEvent::Drag { row });
            }
        }

        // 5. The release: any grab ends, the press is consumed whether or
        // not the release was still, and a still release resolves across
        // the views in two passes — every close box first, then the
        // first body containing the point.
        if self.released {
            state.grab = None;
            let click = state.press.take().filter(|&from| {
                self.pointer
                    .is_some_and(|p| click_dist(from, p) < CLICK_TOL)
            });
            if let Some(p) = click {
                let mut closed = false;
                for (out, spec) in outs.iter_mut().zip(specs) {
                    if in_rect(close_box(spec.close), p) {
                        out.events.push(TreeEvent::Close);
                        closed = true;
                        break;
                    }
                }
                if !closed {
                    for (out, spec) in outs.iter_mut().zip(specs) {
                        if spec.folded || spec.body.is_none() {
                            continue;
                        }
                        let body = spec.body.unwrap();
                        if !in_rect(body, p) {
                            continue;
                        }
                        if let Some(i) = row_at(p, body, spec.scroll, spec.rows.len(), spec.style) {
                            let row = &spec.rows[i];
                            let y = row_y(i, body, spec.scroll, spec.style);
                            let (ax, dx) = row_btn_boxes(row.add, row.del, body[2]);
                            let near = |x: Option<f32>| {
                                x.is_some_and(|x| {
                                    (p[0] - x).abs() <= BTN_NEAR && (p[1] - y).abs() <= BTN_NEAR
                                })
                            };
                            let event = if near(ax) {
                                TreeEvent::Add { row: i }
                            } else if near(dx) {
                                TreeEvent::Del { row: i }
                            } else {
                                TreeEvent::Row { row: i }
                            };
                            out.events.push(event);
                        }
                        break;
                    }
                }
            }
            if let Some(drag) = state.drag.take()
                && let Some(i) = specs.iter().position(|s| s.id == drag.view)
            {
                outs[i].events.push(TreeEvent::DragEnd);
            }
        }

        outs
    }

    /// Paints one tree view into `ctx`: the rows (windowed and culled),
    /// the selection band, the `+`/`×` plates and glyphs, both scroll
    /// thumbs, the corner ticks, and the close `×` — which every view
    /// carries, folded or not.
    pub fn tree_paint(&mut self, ctx: &mut Context, spec: &TreeSpec<'_>) {
        let st = spec.style;
        let z = spec.z;

        if !spec.folded {
            if let Some(body) = spec.body {
                let chars = chars_for(body[2] - body[0], st);
                let sx = (spec.sx / st.adv) as usize;
                let first = (spec.scroll / st.row_h).floor().max(0.0) as usize;
                let max_rows = (st.max_h / st.row_h).ceil() as usize;
                let left = body[0] + st.inset;
                for (i, row) in spec.rows.iter().enumerate().skip(first).take(max_rows) {
                    let y = row_y(i, body, spec.scroll, st);
                    if y < body[1] || y > body[3] {
                        continue;
                    }
                    let ty = y - st.lift * st.size;
                    if row.band {
                        ctx.rectangle(
                            (body[0] + body[2]) / 2.0,
                            y,
                            (body[2] - body[0]) / 2.0 - BAND_INSET,
                            st.row_h / 2.0,
                            st.band,
                            z - 0.5,
                        );
                    }
                    let (ax, dx) = row_btn_boxes(row.add, row.del, body[2]);
                    if let Some(x) = ax {
                        ctx.rectangle(x, y, PLATE, PLATE, st.add_bg, z - 0.45);
                    }
                    if let Some(x) = dx {
                        ctx.rectangle(x, y, PLATE, PLATE, st.del_bg, z - 0.45);
                    }
                    let (kvis, vvis) = window(row.key, row.val, sx, chars, st.adv);
                    if let Some((text, x)) = kvis {
                        ctx.text(left + x, ty, &self.font, text, st.size, 700.0, st.key, z);
                    }
                    if let Some((text, x)) = vvis {
                        let weight = if row.head { 700.0 } else { 500.0 };
                        ctx.text(
                            left + x,
                            ty,
                            &self.font,
                            text,
                            st.size,
                            weight,
                            row.val_color,
                            z,
                        );
                    }
                    if let Some(x) = ax {
                        ctx.text(
                            x,
                            y - st.lift * GLYPH_SIZE,
                            &self.font,
                            "+",
                            GLYPH_SIZE,
                            700.0,
                            st.add_glyph,
                            z,
                        );
                    }
                    if let Some(x) = dx {
                        ctx.text(
                            x,
                            y - st.lift * GLYPH_SIZE,
                            &self.font,
                            "\u{00d7}",
                            GLYPH_SIZE,
                            700.0,
                            st.del_glyph,
                            z,
                        );
                    }
                }

                let total = spec.rows.len() as f32 * st.row_h + st.pad;
                if let Some(r) = v_thumb(body, spec.scroll, total) {
                    let track = body[3] - body[1];
                    ctx.rectangle(
                        body[2] - 5.0,
                        (body[1] + body[3]) / 2.0,
                        TRACK_HALF,
                        track / 2.0,
                        st.track,
                        z - 0.4,
                    );
                    ctx.rectangle(
                        (r[0] + r[2]) / 2.0,
                        (r[1] + r[3]) / 2.0,
                        (r[2] - r[0]) / 2.0,
                        (r[3] - r[1]) / 2.0,
                        st.thumb,
                        z - 0.4,
                    );
                }
                let wide = width_of(spec.rows);
                if let Some(r) = h_thumb(body, spec.sx, wide, st) {
                    let track = body[2] - body[0];
                    ctx.rectangle(
                        (body[0] + body[2]) / 2.0,
                        body[1] + 5.0,
                        track / 2.0,
                        TRACK_HALF,
                        st.track,
                        z - 0.4,
                    );
                    ctx.rectangle(
                        (r[0] + r[2]) / 2.0,
                        (r[1] + r[3]) / 2.0,
                        (r[2] - r[0]) / 2.0,
                        (r[3] - r[1]) / 2.0,
                        st.thumb,
                        z - 0.4,
                    );
                }
            }

            // The corner ticks: every unfolded view shows them, even one
            // whose body is empty for the snap-open frame.
            let (rx, by) = (spec.panel[2], spec.panel[1]);
            for d in TICKS {
                ctx.line(
                    rx - d,
                    by + TICK_IN,
                    rx - TICK_IN,
                    by + d,
                    st.grip,
                    1.5,
                    z - 0.45,
                );
            }
        }

        // The close ×: every view, folded or not.
        let hot = self
            .pointer
            .is_some_and(|p| in_rect(close_box(spec.close), p));
        let box_color = if hot { st.close_bg_hot } else { st.close_bg };
        ctx.rectangle(
            spec.close[0],
            spec.close[1],
            CLOSE_HALF,
            CLOSE_HALF,
            box_color,
            z - 0.45,
        );
        let glyph = st.size + CLOSE_EXTRA;
        let mut color = st.close_glyph;
        if hot {
            color.a = 1.0;
        }
        ctx.text(
            spec.close[0],
            spec.close[1] - st.lift * glyph,
            &self.font,
            "\u{00d7}",
            glyph,
            700.0,
            color,
            z,
        );
    }
}

/// Whether `p` is inside the rect `[left, bottom, right, top]` —
/// inclusive on all edges, user space y-up.
fn in_rect(r: [f32; 4], p: [f32; 2]) -> bool {
    p[0] >= r[0] && p[0] <= r[2] && p[1] >= r[1] && p[1] <= r[3]
}

/// The close `×`'s hit box, around its center.
fn close_box(close: [f32; 2]) -> [f32; 4] {
    [
        close[0] - CLOSE_HALF,
        close[1] - CLOSE_HALF,
        close[0] + CLOSE_HALF,
        close[1] + CLOSE_HALF,
    ]
}

/// The corner grip's hit box: the panel's bottom-right corner, 21 wide
/// and 21 tall, reaching 3 past each edge.
fn corner_box(panel: [f32; 4]) -> [f32; 4] {
    [
        panel[2] - 18.0,
        panel[1] - 3.0,
        panel[2] + 3.0,
        panel[1] + 18.0,
    ]
}

/// Row `i`'s center y: rows count down from the body's top edge, under
/// the top padding, shifted by the scroll.
fn row_y(i: usize, body: [f32; 4], scroll: f32, st: TreeStyle) -> f32 {
    // Scrolling DOWN walks the document: every row climbs by `scroll`,
    // so the later rows enter at the bottom as the early ones leave at
    // the top. The sign here is the whole convention — `row_at` and the
    // thumbs read it the same way.
    body[3] - st.pad + scroll - (i as f32 + 0.5) * st.row_h
}

/// The row under `p`, or `None` when the point is outside the body or
/// past the row range.
fn row_at(p: [f32; 2], body: [f32; 4], scroll: f32, n: usize, st: TreeStyle) -> Option<usize> {
    if !in_rect(body, p) {
        return None;
    }
    let i = ((body[3] - st.pad - p[1] + scroll) / st.row_h).floor();
    (i >= 0.0 && (i as usize) < n).then_some(i as usize)
}

/// The character budget for a width: the characters that fit between the
/// insets, never fewer than `min_chars`.
fn chars_for(w: f32, st: TreeStyle) -> usize {
    ((w - 2.0 * st.inset) / st.adv).max(st.min_chars as f32) as usize
}

/// The widest row, in characters — the sum of its key and value lengths.
fn width_of(rows: &[TreeLine<'_>]) -> usize {
    rows.iter()
        .map(|r| r.key.chars().count() + r.val.chars().count())
        .max()
        .unwrap_or(0)
}

/// The scroll thumb's position and length on a track: `view` and `total`
/// in content units, `track` the track's length. Returns `(start, len)`
/// from the track's origin — the vertical track's origin is its top, the
/// horizontal one's its left.
fn thumb(scroll: f32, view: f32, total: f32, track: f32) -> (f32, f32) {
    if total <= view || track <= 0.0 {
        return (0.0, track.max(0.0));
    }
    let len = (view / total * track).max(THUMB_MIN).min(track);
    let start = (scroll / (total - view)).clamp(0.0, 1.0) * (track - len);
    (start, len)
}

/// The vertical thumb's rect, on the body's right edge, or `None` when
/// the rows fit.
fn v_thumb(body: [f32; 4], scroll: f32, total: f32) -> Option<[f32; 4]> {
    let track = body[3] - body[1];
    (total > track).then(|| {
        let (ty, tl) = thumb(scroll, track, total, track);
        [
            body[2] - STRIP_OUT,
            body[3] - ty - tl,
            body[2] - STRIP_IN,
            body[3] - ty,
        ]
    })
}

/// The horizontal thumb's rect, on the body's bottom edge, or `None`
/// when the rows fit (or the body has no width).
fn h_thumb(body: [f32; 4], sx: f32, total: usize, st: TreeStyle) -> Option<[f32; 4]> {
    let track = body[2] - body[0];
    let view = chars_for(track, st) as f32 * st.adv;
    let total_px = total as f32 * st.adv;
    (total_px > view && track > 0.0).then(|| {
        let (tx, tl) = thumb(sx, view, total_px, track);
        [
            body[0] + tx,
            body[1] + STRIP_IN,
            body[0] + tx + tl,
            body[1] + STRIP_OUT,
        ]
    })
}

/// The new view size from a corner grip drag: the width grows from the
/// center as the pointer moves right, the height as it moves up, both
/// clamped to the style's limits.
fn resize_size(w0: f32, h0: f32, ax: f32, ay: f32, p: [f32; 2], st: TreeStyle) -> [f32; 2] {
    [
        (w0 + 2.0 * (p[0] - ax)).clamp(st.min_w, st.max_w),
        (h0 + (ay - p[1])).clamp(st.min_h, st.max_h),
    ]
}

/// The `+`/`×` buttons' centers for a row, from the body's right edge:
/// the `+` sits `ADD_OFF` in, and a `×` beside it moves further in.
fn row_btn_boxes(add: bool, del: bool, right: f32) -> (Option<f32>, Option<f32>) {
    (
        add.then_some(right - ADD_OFF),
        del.then_some(if add {
            right - DEL_OFF
        } else {
            right - ADD_OFF
        }),
    )
}

/// The visible slice of a `key: val` row at a horizontal scroll: each
/// side is `None` when scrolled out of view, else the text and its
/// center x, relative to the body's text edge.
type Slice = Option<(String, f32)>;

fn window(key: &str, val: &str, sx: usize, maxc: usize, adv: f32) -> (Slice, Slice) {
    let klen = key.chars().count();
    let slice = |text: &str, base: usize| -> Slice {
        let a = sx.max(base);
        let b = (sx + maxc).min(base + text.chars().count());
        (b > a).then(|| {
            let t: String = text.chars().skip(a - base).take(b - a).collect();
            (t, ((a - sx) as f32 + (b - a) as f32 / 2.0) * adv)
        })
    };
    (slice(key, 0), slice(val, klen))
}

/// The distance between two points — the press drift for the click
/// tolerance.
fn click_dist(a: [f32; 2], b: [f32; 2]) -> f32 {
    f32::hypot(b[0] - a[0], b[1] - a[1])
}

#[cfg(test)]
mod tests;
