//! The menu bar: a classic strip pinned to the window's top edge whose
//! titles unfold drop-down lists — File, Edit, whatever the app declares.
//!
//! ```no_run
//! struct Editor {
//!     ui: frost::Ui,
//! }
//! impl frost::Process for Editor {
//!     fn process(&mut self, ctx: &mut frost::Context, _dt: f32) {
//!         // ...panels and widgets first: the bar must be declared last...
//!         static FILE: frost::Menu<'static> = frost::Menu {
//!             title: "File",
//!             items: &[
//!                 frost::MenuItem::new("Open", "Ctrl-O"),
//!                 frost::MenuItem::new("Quit", ""),
//!             ],
//!         };
//!         if let Some(frost::MenuEvent::Chose(menu, item)) = self.ui.menu_bar(ctx, &[FILE], &[])
//!         {
//!             log::info!("{menu} / {item}");
//!         }
//!     }
//! }
//! ```
//!
//! **Declare it last.** The bar draws at the [`z`](UiStyle::base_z) its
//! declaration order gives it, and so does its drop-down — declaring the
//! bar after every panel is what lets the open menu float above them and
//! win the clicks that land on it, by the same last-declared-wins rule
//! that orders the rest of the UI.
//!
//! **The behavior is the classic one**, down to the details a user's hand
//! expects: a click on a title opens its list, a click on the same title
//! closes it again, and while a list is open, sliding the pointer along
//! the bar switches which list is shown — menu tracking, no second click
//! needed. Hovered items highlight; an item released elsewhere cancels.
//! A click on the bar's bare strip or anywhere outside closes the open
//! list, exactly as it dismisses a real menu. Choosing an item closes the
//! list and returns `(menu title, item label)` for that frame.
//!
//! An item whose label is empty draws a separator line and swallows its
//! click without choosing anything. Shortcut texts are display-only —
//! the bar never dispatches keys; wire the shortcuts in your own input
//! handling (the accelerator beside a title is a promise to do so).
//!
//! **A list is not only commands.** A [`MenuItem::readout`] line shows a
//! live label and value (`Zoom 1.25`, say) where a command would — it
//! chooses nothing and dismisses nothing. A [`MenuItem::slider`] line
//! carries a whole slider in one row: the row is the grab area, the
//! track spans the line between label and value, and while it is held
//! the list stays open and the line reports [`MenuEvent::Slid`] every
//! frame the value moves — release commits and the list, still open,
//! awaits the next line. Sliders live where panels used to be: the
//! menus hold the settings without occupying the canvas.
//!
//! The bar and every open list register hit rects like any widget, so
//! [`Ui::hovering`](Ui::hovering) reports them: a scene that gates its
//! own mouse handling on `hovering` will not act through the bar or an
//! open menu.

use super::*;

/// What a [`Menu`]'s line is: a command to choose, a value to read, or
/// a value to set.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ItemKind<'a> {
    /// A command: released on its line, it closes the list and reports
    /// [`MenuEvent::Chose`].
    Command,
    /// A live read-only line: the label names the value, shown
    /// right-aligned where a shortcut would sit. Clicks pass it by.
    Readout { value: &'a str },
    /// A live slider in one line: the row is the grab area, the track
    /// spans between label and value, and holding it reports
    /// [`MenuEvent::Slid`] as the value travels. The list stays open.
    Slider { min: f32, max: f32, value: f32 },
}

/// One line of a [`Menu`]'s drop-down. An empty `label` is a separator:
/// a rule across the list that swallows its click and chooses nothing.
#[derive(Clone, Copy, Debug)]
pub struct MenuItem<'a> {
    /// The command's name, or `""` for a separator line.
    pub label: &'a str,
    /// The accelerator to show right-aligned on the line — display only:
    /// the bar never dispatches keys. `""` shows nothing. Ignored by
    /// readout and slider lines, which own the line's right edge.
    pub shortcut: &'a str,
    /// What the line is: see [`ItemKind`].
    pub kind: ItemKind<'a>,
    /// The tick of a mode line: the list paints a check mark in a
    /// left gutter before this line, saying the item names the state
    /// that is in effect. Dispatch does not change — a checked line
    /// is chosen like any other, and choosing the mode you are in is
    /// the application's nothing-to-do.
    pub checked: bool,
}

impl<'a> MenuItem<'a> {
    /// A named command with its (display-only) accelerator.
    pub const fn new(label: &'a str, shortcut: &'a str) -> Self {
        Self {
            label,
            shortcut,
            kind: ItemKind::Command,
            checked: false,
        }
    }

    /// Show this line as the mode in effect: a check mark in the
    /// list's left gutter.
    #[must_use]
    pub const fn checked(mut self, yes: bool) -> Self {
        self.checked = yes;
        self
    }

    /// A read-only line: the label left, the value right, and — when
    /// given (`""` shows nothing) — the accelerator that summons the
    /// value's changes, shown muted before the value.
    pub const fn readout(label: &'a str, shortcut: &'a str, value: &'a str) -> Self {
        Self {
            label,
            shortcut,
            kind: ItemKind::Readout { value },
            checked: false,
        }
    }

    /// A slider line driving a value in `min..=max`.
    pub const fn slider(label: &'a str, min: f32, max: f32, value: f32) -> Self {
        Self {
            label,
            shortcut: "",
            kind: ItemKind::Slider { min, max, value },
            checked: false,
        }
    }

    /// A separator line.
    pub const SEPARATOR: MenuItem<'static> = MenuItem {
        label: "",
        shortcut: "",
        kind: ItemKind::Command,
        checked: false,
    };
}

/// What the menu bar hands back for a frame: a chosen command, or a
/// slider line's value on the move.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum MenuEvent<'a> {
    /// An item released on its own line: `(menu title, item label)`.
    Chose(&'a str, &'a str),
    /// A slider line moved — every frame it is held, and once more on
    /// the release that commits it: `(menu title, line label, value)`.
    Slid(&'a str, &'a str, f32),
}

/// One top-level entry of the menu bar: a title and its list.
#[derive(Clone, Copy, Debug)]
pub struct Menu<'a> {
    /// The title shown in the bar.
    pub title: &'a str,
    /// The list it unfolds.
    pub items: &'a [MenuItem<'a>],
}

impl Ui {
    /// Declares the whole menu bar for the frame — call it once, after
    /// every other widget — and reports the frame's [`MenuEvent`]: a
    /// chosen command, or a slider line's value on the move. `notes` are read-only strings drawn right-aligned
    /// on the bar in the muted color, the LAST one at the bar's far end
    /// and its predecessors stepping leftwards: the last command and the
    /// current mode, say. The menu titles keep the bar's left, so a note
    /// that runs out of room to their right is cut to a `...`-led tail —
    /// a note's end is its point, so the tail is what survives — and the
    /// earlier notes fall off the left first. They are paint only —
    /// clicks there belong to the bar's strip, not to any item. See the
    /// [module](self) for the behavior.
    pub fn menu_bar<'a>(
        &mut self,
        ctx: &mut Context,
        menus: &[Menu<'a>],
        notes: &[&str],
    ) -> Option<MenuEvent<'a>> {
        let size = ctx.size();
        let (frame, choice) = self.menu_frame(menus, size);
        self.paint_menu(ctx, &frame);
        // The notes sit on the finished bar: paint them only after the
        // plate was laid, or the plate buries them.
        let (pad, font, color) = (self.style.pad, self.style.font_size, self.style.text_muted);
        let [_, cy] = frame.bar.center();
        // Letters sit on the line's optical center, not the box's.
        let cy = cy - self.style.menu_lift * font;
        // The titles own the bar up to the last one's right edge; the
        // notes are laid in what is left of that, to the bar's end.
        let limit = frame
            .titles
            .iter()
            .map(|t| t.rect.left + t.rect.w)
            .fold(frame.bar.left, f32::max)
            + pad;
        let right = frame.bar.left + frame.bar.w - pad;
        for (note, center) in self.plan_notes(notes, font, right, limit, pad) {
            self.text(ctx, [center, cy], font, color, &note);
        }
        choice
    }

    /// The paint-only heart of [`Ui::menu_bar`]'s notes, without a
    /// canvas: lay `notes` right-to-left in `(limit, right]`, answering
    /// `(text, center_x)` in the order the notes were given.
    ///
    /// Text draws centered — [`crate::Canvas::text`]'s anchor is the
    /// glyph run's middle, not its edge — so a note of width `w` ending
    /// at `x` draws at `x - w / 2`, and the next note's end starts a
    /// `pad` further left. A note wider than the room left of it keeps
    /// its tail behind a `...` lead (the tail is the half that says
    /// something), and once not even that fits, the note and every
    /// earlier one are left out.
    fn plan_notes(
        &mut self,
        notes: &[&str],
        font: f32,
        right: f32,
        limit: f32,
        pad: f32,
    ) -> Vec<(String, f32)> {
        let mut laid: Vec<(String, f32)> = Vec::with_capacity(notes.len());
        let mut edge = right;
        for note in notes.iter().rev() {
            let room = edge - limit;
            if room <= pad {
                break;
            }
            let w = self.text_width(note, font);
            let text = if w <= room {
                (*note).to_owned()
            } else if let Some(tail) = self.ellipsize(note, font, room) {
                tail
            } else {
                break;
            };
            let w = self.text_width(&text, font);
            laid.push((text, edge - w / 2.0));
            edge -= w + pad;
        }
        laid.reverse();
        laid
    }

    /// The longest `...`-led tail of `note` that measures no more than
    /// `room`, searching over char boundaries so a cut never splits a
    /// codepoint (width grows with the tail, so the fit is a binary
    /// search). `None` when not even the dots fit on their own.
    fn ellipsize(&mut self, note: &str, font: f32, room: f32) -> Option<String> {
        if room <= self.text_width("...", font) {
            return None;
        }
        let starts: Vec<usize> = note.char_indices().map(|(i, _)| i).collect();
        let (mut lo, mut hi) = (0usize, starts.len());
        while lo < hi {
            let mid = (lo + hi) / 2;
            let probe = format!("...{}", &note[starts[mid]..]);
            if self.text_width(&probe, font) <= room {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }
        if lo == starts.len() {
            None
        } else {
            Some(format!("...{}", &note[starts[lo]..]))
        }
    }

    /// The input-and-geometry heart of [`Ui::menu_bar`], without a canvas:
    /// hit-test, open, track, close, choose — and hand back everything the
    /// painter needs. Separate so the behavior is testable without a
    /// window, like [`Ui::begin_input`](Ui::begin_input).
    fn menu_frame<'a>(
        &mut self,
        menus: &[Menu<'a>],
        size: (f32, f32),
    ) -> (MenuFrame<'a>, Option<MenuEvent<'a>>) {
        let style = self.style;
        let top = size.1 / 2.0;
        let bar = Rect {
            left: -size.0 / 2.0,
            top,
            w: size.0,
            h: style.row_h,
        };
        // The bar's bare strip is a click sponge: a press that lands on
        // it hits nothing of the scene's, and closes whatever is open.
        let sponge = widget_id(0, "\u{1}menubar");
        self.interact(sponge, bar);

        // Titles, left to right.
        let mut x = bar.left + style.pad;
        let mut titles = Vec::with_capacity(menus.len());
        for menu in menus {
            let w = self.text_width(menu.title, style.font_size) + style.pad;
            let rect = Rect {
                left: x,
                top,
                w,
                h: style.row_h,
            };
            let id = widget_id(0, &format!("menu:{}", menu.title));
            let it = self.interact(id, rect);
            titles.push(TitleFrame {
                label: menu.title,
                rect,
                id,
                hot: it.hot,
                open: false,
            });
            x += w + style.pad;
        }

        // — the state machine, on this frame's click edges —
        // It runs before the list is laid: a title click must show its
        // list on the frame it opens, not the one after.
        let mut choice = None;
        let mut slid: Option<MenuEvent<'a>> = None;
        if let Some(rel) = self.released_active {
            let mut handled = false;
            // A release on the open menu's own item line chooses — and
            // anywhere else on that line's widget cancels, closing like
            // a real menu's.
            if let Some(oid) = self.menu_open
                && let Some(m) = menus
                    .iter()
                    .find(|mn| widget_id(0, &format!("menu:{}", mn.title)) == oid)
            {
                if let Some(item) = m.items.iter().find(|it| {
                    !it.label.is_empty()
                        && it.kind == ItemKind::Command
                        && widget_id(oid, it.label) == rel
                }) {
                    if self.hover == Some(rel) {
                        choice = Some((m.title, item.label));
                    }
                    self.menu_open = None;
                    handled = true;
                } else if m.items.iter().any(|it| {
                    it.kind != ItemKind::Command
                        && !it.label.is_empty()
                        && widget_id(oid, it.label) == rel
                }) {
                    // A slider line's release: the drag's last frame. The
                    // value lands in the layout below; the list stays
                    // open, ready for the next line.
                    handled = true;
                } else if rel == widget_id(oid, "\u{1}bg") {
                    // A click in the list's own padding: stays open,
                    // chooses nothing.
                    handled = true;
                }
            }
            if !handled
                && let Some(t) = titles
                    .iter()
                    .find(|t| t.id == rel && t.rect.contains(self.pointer.unwrap_or([f32::NAN; 2])))
            {
                // A title click toggles: open it, or close its own.
                self.menu_open = if self.menu_open == Some(t.id) {
                    None
                } else {
                    Some(t.id)
                };
                handled = true;
            }
            if !handled {
                self.menu_open = None; // a click on anything else dismisses
            }
        } else if self.released {
            self.menu_open = None; // a click on nothing dismisses
        }
        // Menu tracking: with a list open, gliding over another title
        // switches the list to it.
        if let Some(open) = self.menu_open
            && let Some(t) = titles.iter().find(|t| t.hot && t.id != open)
        {
            self.menu_open = Some(t.id);
        }

        // The open menu's list, laid and registered after the decision.
        let mut drop = None;
        let open_idx = self.menu_open.and_then(|oid| {
            menus
                .iter()
                .position(|m| widget_id(0, &format!("menu:{}", m.title)) == oid)
        });
        if let Some(i) = open_idx {
            let menu = &menus[i];
            let title = titles[i].id;
            // The gutter every line leaves on its left for a checked
            // line's tick: it exists for the whole list the moment any
            // line wears one, so labels stand in one column.
            let gutter = if menu.items.iter().any(|it| it.checked) {
                1.4 * style.font_size
            } else {
                0.0
            };
            let room = if menu.items.iter().any(|it| !it.shortcut.is_empty()) {
                3.0 * style.pad
            } else {
                0.0
            };
            let widest = menu
                .items
                .iter()
                .map(|it| {
                    let l = self.text_width(it.label, style.font_size);
                    match it.kind {
                        ItemKind::Command => {
                            let sc = self.text_width(it.shortcut, style.font_size);
                            style.pad + gutter + l + room + sc + style.pad
                        }
                        ItemKind::Readout { value } => {
                            let v = self.text_width(value, style.font_size);
                            let sc = self.text_width(it.shortcut, style.font_size);
                            let sc = if sc > 0.0 { sc + style.pad } else { 0.0 };
                            style.pad + gutter + l + room + sc + v + style.pad
                        }
                        ItemKind::Slider { value, .. } => {
                            // Label, a fixed-width track, the value: the
                            // list grows to hold the slider at ease.
                            let v = self.text_width(&format!("{value:.2}"), style.font_size);
                            style.pad
                                + gutter
                                + l
                                + 2.0 * style.pad
                                + MENU_TRACK
                                + 2.0 * style.pad
                                + v
                                + style.pad
                        }
                    }
                })
                .fold(titles[i].rect.w, f32::max);
            let left = titles[i].rect.left;
            let top = bar.top - bar.h;
            let h: f32 = menu
                .items
                .iter()
                .map(|it| {
                    if it.label.is_empty() {
                        style.row_gap
                    } else {
                        style.row_h
                    }
                })
                .sum();
            // The background registers FIRST: every row beats it in the
            // overlap order, and it is what catches a click inside the
            // list that lands on no row — the separator's band, the
            // rounding slack. Such a click stays open, like a real
            // menu's own padding.
            self.interact(
                widget_id(title, "\u{1}bg"),
                Rect {
                    left,
                    top,
                    w: widest,
                    h,
                },
            );
            // The table rule for the value lines: every slider shares
            // one label column — the widest of them — so the tracks
            // start in line, stretch across the slack, and end before
            // the values, the way a label-track-readout table lays its
            // rows.
            let slider_col = menu
                .items
                .iter()
                .filter(|it| matches!(it.kind, ItemKind::Slider { .. }))
                .map(|it| self.text_width(it.label, style.font_size))
                .fold(0.0f32, f32::max);
            let mut rows = Vec::with_capacity(menu.items.len());
            // `y` holds the current rect's top edge and walks down.
            let mut y = top;
            for item in menu.items {
                if item.label.is_empty() {
                    rows.push(Row::Sep {
                        rect: Rect {
                            left,
                            top: y,
                            w: widest,
                            h: style.row_gap,
                        },
                    });
                    y -= style.row_gap;
                    continue;
                }
                let rect = Rect {
                    left,
                    top: y,
                    w: widest,
                    h: style.row_h,
                };
                match item.kind {
                    ItemKind::Command => {
                        let it = self.interact(widget_id(title, item.label), rect);
                        rows.push(Row::Item {
                            label: item.label,
                            shortcut: item.shortcut,
                            rect,
                            hot: it.hot,
                            pressed: it.pressed,
                            checked: item.checked,
                        });
                    }
                    ItemKind::Readout { value } => {
                        // No registration: the list's background catches
                        // clicks on the line and keeps them inside, like
                        // the separator's band.
                        rows.push(Row::Readout {
                            label: item.label,
                            shortcut: item.shortcut,
                            value,
                            rect,
                        });
                    }
                    ItemKind::Slider { min, max, value } => {
                        let it = self.interact(widget_id(title, item.label), rect);
                        let mut v = value;
                        if it.held || it.clicked {
                            // Held: the knob follows the pointer across
                            // the track, and the line reports each frame.
                            if let Some(p) = self.pointer {
                                let track =
                                    slider_track(self, &style, &rect, slider_col, gutter, v);
                                let (lo, hi) = (
                                    track.left + style.knob_r,
                                    track.left + track.w - style.knob_r,
                                );
                                if hi > lo && max > min {
                                    let t = ((p[0] - lo) / (hi - lo)).clamp(0.0, 1.0);
                                    v = min + t * (max - min);
                                }
                            }
                            slid = Some(MenuEvent::Slid(menu.title, item.label, v));
                        }
                        let track = slider_track(self, &style, &rect, slider_col, gutter, v);
                        rows.push(Row::Slider {
                            label: item.label,
                            min,
                            max,
                            value: v,
                            rect,
                            track,
                            hot: it.hot,
                            pressed: it.pressed,
                        });
                    }
                }
                y -= style.row_h;
            }
            drop = Some(DropFrame {
                bg: Rect {
                    left,
                    top,
                    w: widest,
                    h,
                },
                rows,
            });
        }
        // The title of whatever ended up open draws pressed.
        if let Some(open) = self.menu_open
            && let Some(t) = titles.iter_mut().find(|t| t.id == open)
        {
            t.open = true;
        }
        let event = choice.map(|(t, l)| MenuEvent::Chose(t, l)).or(slid);
        (MenuFrame { bar, titles, drop }, event)
    }

    /// Lays a [`MenuFrame`] on the canvas: bar, titles, and the open list.
    fn paint_menu(&mut self, ctx: &mut Context, frame: &MenuFrame) {
        let style = self.style;
        self.fill(ctx, frame.bar, style.title_bg);
        for t in &frame.titles {
            if t.open || t.hot {
                let rect = Rect {
                    left: t.rect.left,
                    top: t.rect.top,
                    w: t.rect.w,
                    h: t.rect.h,
                };
                self.fill(
                    ctx,
                    rect,
                    if t.open {
                        style.widget_press
                    } else {
                        style.widget_hover
                    },
                );
            }
            let size = style.font_size;
            let [cx, cy] = t.rect.center();
            let cy = cy - style.menu_lift * size;
            let label = t.label.to_owned();
            self.text(ctx, [cx, cy], size, style.title_text, &label);
        }
        let Some(d) = &frame.drop else {
            return;
        };
        self.fill(ctx, d.bg, style.panel_bg);
        // A one-pixel rule under the bar and around the list's far edges:
        // the menu lifts itself off the scene by its own outline.
        let line = |ui: &mut Ui, ctx: &mut Context, x0, y0, x1, y1| {
            let z = ui.z();
            ctx.line(x0, y0, x1, y1, style.widget_hover, 1.0, z);
        };
        line(self, ctx, d.bg.left, d.bg.top, d.bg.left + d.bg.w, d.bg.top);
        line(self, ctx, d.bg.left, d.bg.top, d.bg.left, d.bg.top - d.bg.h);
        line(
            self,
            ctx,
            d.bg.left + d.bg.w,
            d.bg.top,
            d.bg.left + d.bg.w,
            d.bg.top - d.bg.h,
        );
        line(
            self,
            ctx,
            d.bg.left,
            d.bg.top - d.bg.h,
            d.bg.left + d.bg.w,
            d.bg.top - d.bg.h,
        );
        // The gutter is a fact about the rows, and the painter reads
        // the same fact the layout used: one tick in the list, and
        // every label shifts by it.
        let gutter = if d
            .rows
            .iter()
            .any(|r| matches!(r, Row::Item { checked: true, .. }))
        {
            1.4 * style.font_size
        } else {
            0.0
        };
        for row in &d.rows {
            match row {
                Row::Item {
                    label,
                    shortcut,
                    rect,
                    hot,
                    pressed,
                    checked,
                } => {
                    if *pressed {
                        self.fill(ctx, *rect, style.widget_press);
                    } else if *hot {
                        self.fill(ctx, *rect, style.widget_hover);
                    }
                    let size = style.font_size;
                    let cy = rect.center()[1] - style.menu_lift * size;
                    let lw = self.text_width(label, size);
                    let l = (*label).to_owned();
                    self.text(
                        ctx,
                        [rect.left + style.pad + gutter + lw / 2.0, cy],
                        size,
                        style.text,
                        &l,
                    );
                    if *checked {
                        // The tick, drawn as two strokes: a check-mark
                        // glyph is a font's favor to ask, and the UI
                        // font is not asked to do favors. It rides the
                        // row's own middle, not the label's lifted
                        // anchor: strokes land exactly where they are
                        // told, and it is the letters that need the
                        // lift to look centered.
                        let s = size;
                        let gx = rect.left + style.pad + gutter / 2.0;
                        let ry = rect.center()[1];
                        let z = self.z();
                        ctx.line(
                            gx - 0.30 * s,
                            ry + 0.08 * s,
                            gx - 0.02 * s,
                            ry - 0.24 * s,
                            style.text,
                            1.6,
                            z,
                        );
                        ctx.line(
                            gx - 0.02 * s,
                            ry - 0.24 * s,
                            gx + 0.42 * s,
                            ry + 0.36 * s,
                            style.text,
                            1.6,
                            z,
                        );
                    }
                    if !shortcut.is_empty() {
                        let sw = self.text_width(shortcut, size);
                        let s = (*shortcut).to_owned();
                        self.text(
                            ctx,
                            [rect.left + rect.w - style.pad - sw / 2.0, cy],
                            size,
                            style.text_muted,
                            &s,
                        );
                    }
                }
                Row::Readout {
                    label,
                    shortcut,
                    value,
                    rect,
                } => {
                    let size = style.font_size;
                    let cy = rect.center()[1] - style.menu_lift * size;
                    let lw = self.text_width(label, size);
                    let l = (*label).to_owned();
                    self.text(
                        ctx,
                        [rect.left + style.pad + gutter + lw / 2.0, cy],
                        size,
                        style.text,
                        &l,
                    );
                    let vw = self.text_width(value, size);
                    let v = (*value).to_owned();
                    let vr = rect.left + rect.w - style.pad;
                    self.text(ctx, [vr - vw / 2.0, cy], size, style.text_muted, &v);
                    if !shortcut.is_empty() {
                        // The accelerator sits just before the value.
                        let sw = self.text_width(shortcut, size);
                        let sc = (*shortcut).to_owned();
                        self.text(
                            ctx,
                            [vr - vw - style.pad - sw / 2.0, cy],
                            size,
                            style.text_muted,
                            &sc,
                        );
                    }
                }
                Row::Slider {
                    label,
                    min,
                    max,
                    value,
                    rect,
                    track,
                    hot,
                    pressed,
                } => {
                    if *pressed {
                        self.fill(ctx, *rect, style.widget_press);
                    } else if *hot {
                        self.fill(ctx, *rect, style.widget_hover);
                    }
                    let size = style.font_size;
                    let cy = rect.center()[1] - style.menu_lift * size;
                    let lw = self.text_width(label, size);
                    let l = (*label).to_owned();
                    self.text(
                        ctx,
                        [rect.left + style.pad + gutter + lw / 2.0, cy],
                        size,
                        style.text,
                        &l,
                    );
                    let v = (*value).to_owned();
                    let vs = format!("{v:.2}");
                    let vw = self.text_width(&vs, size);
                    self.text(
                        ctx,
                        [rect.left + rect.w - style.pad - vw / 2.0, cy],
                        size,
                        style.text_muted,
                        &vs,
                    );
                    // The track, the filled span, the knob: the panel
                    // slider's own dress, worn inside the list — and
                    // laid by the shared column, as a table lays it.
                    let track = *track;
                    let knob = style.knob_r;
                    let t = if *max > *min {
                        ((*value - *min) / (*max - *min)).clamp(0.0, 1.0)
                    } else {
                        0.0
                    };
                    let knob_x = track.left + knob + t * (track.w - 2.0 * knob).max(0.0);
                    self.fill(ctx, track, style.track);
                    let fill = Rect {
                        left: track.left + knob,
                        top: track.top,
                        w: (knob_x - track.left - knob).max(0.0),
                        h: track.h,
                    };
                    if fill.w > 0.0 {
                        self.fill(ctx, fill, style.accent);
                    }
                    let white = Color {
                        r: 1.0,
                        g: 1.0,
                        b: 1.0,
                        a: 1.0,
                    };
                    let knob_color = if *pressed {
                        style.accent.lerp(white, 0.35)
                    } else if *hot {
                        style.accent.lerp(white, 0.15)
                    } else {
                        style.accent
                    };
                    let z = self.z();
                    ctx.circle(knob_x, track.center()[1], knob, knob_color, z);
                }
                Row::Sep { rect } => {
                    let y = rect.center()[1];
                    let z = self.z();
                    ctx.line(
                        rect.left + style.pad / 2.0,
                        y,
                        rect.left + rect.w - style.pad / 2.0,
                        y,
                        style.widget_hover,
                        1.0,
                        z,
                    );
                }
            }
        }
    }
}

/// Everything one frame's [`Ui::menu_frame`] computed, ready to paint.
struct MenuFrame<'a> {
    bar: Rect,
    titles: Vec<TitleFrame<'a>>,
    drop: Option<DropFrame<'a>>,
}

struct TitleFrame<'a> {
    label: &'a str,
    rect: Rect,
    id: u64,
    hot: bool,
    open: bool,
}

struct DropFrame<'a> {
    bg: Rect,
    rows: Vec<Row<'a>>,
}

/// One laid line of the open list, ready to paint.
enum Row<'a> {
    Item {
        label: &'a str,
        shortcut: &'a str,
        rect: Rect,
        hot: bool,
        pressed: bool,
        checked: bool,
    },
    Readout {
        label: &'a str,
        shortcut: &'a str,
        value: &'a str,
        rect: Rect,
    },
    Slider {
        label: &'a str,
        min: f32,
        max: f32,
        /// The value as it now stands — the drag's, while held.
        value: f32,
        rect: Rect,
        track: Rect,
        hot: bool,
        pressed: bool,
    },
    Sep {
        rect: Rect,
    },
}

/// The width a menu slider's track asks for; a longer list stretches
/// it, a narrow menu is not allowed to crush it.
const MENU_TRACK: f32 = 96.0;

/// A menu slider line's track: the span between the label and the
/// value readout, centered in the row — the geometry the drag maps the
/// pointer onto and the painter draws.
fn slider_track(
    ui: &mut Ui,
    style: &UiStyle,
    rect: &Rect,
    col: f32,
    gutter: f32,
    value: f32,
) -> Rect {
    // The column is the menu's shared label column, not this line's
    // own label: the tracks all start in line and stretch to the
    // values, the way the panel's table laid them.
    let lw = col;
    let vw = ui.text_width(&format!("{value:.2}"), style.font_size);
    let x0 = rect.left + style.pad + gutter + lw + style.pad;
    let x1 = rect.left + rect.w - style.pad - vw - style.pad;
    let w = (x1 - x0).max(2.0 * style.knob_r + 1.0);
    Rect::from_center(
        (x0 + x1) / 2.0,
        rect.center()[1],
        w,
        style.track_h.max(2.0 * style.knob_r),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ui::tests::{frame, ui};

    static FILE: Menu<'static> = Menu {
        title: "File",
        items: &[
            MenuItem::new("Open", "Ctrl-O"),
            MenuItem::new("Save", ""),
            MenuItem::SEPARATOR,
            MenuItem::new("Quit", ""),
        ],
    };
    static EDIT: Menu<'static> = Menu {
        title: "Edit",
        items: &[MenuItem::new("Undo", "Ctrl-Z")],
    };
    static VIEW: Menu<'static> = Menu {
        title: "View",
        items: &[
            MenuItem::new("Markers", ""),
            MenuItem::SEPARATOR,
            MenuItem::readout("Zoom", "Ctrl +/Ctrl -", "1.00"),
            MenuItem::slider("Light", 0.0, 1.0, 0.5),
        ],
    };
    const SIZE: (f32, f32) = (800.0, 600.0);

    /// One frame of the bar with both menus, through the state machine.
    fn bar(ui: &mut Ui) -> MenuFrame<'static> {
        let (f, _) = ui.menu_frame(&[FILE, EDIT], SIZE);
        f
    }

    /// One frame of a bar carrying only the VIEW menu — the slider's.
    fn vbar(ui: &mut Ui) -> (MenuFrame<'static>, Option<MenuEvent<'static>>) {
        ui.menu_frame(&[VIEW], SIZE)
    }

    /// Click `p`: press frame, then release frame, returning the release
    /// frame's menu state.
    fn click_at(ui: &mut Ui, p: [f32; 2]) -> MenuFrame<'static> {
        frame(ui, Some(p), true);
        bar(ui);
        frame(ui, Some(p), false);
        bar(ui)
    }

    /// Move to `p` (a released frame) and return the first item line's
    /// center of the open list, panicking if none is open.
    fn open_item_center(ui: &mut Ui, p: [f32; 2], item: usize) -> ([f32; 2], MenuFrame<'static>) {
        frame(ui, Some(p), false);
        let f = bar(ui);
        let d = f.drop.as_ref().expect("no list is open");
        match &d.rows[item] {
            Row::Item { rect, .. } => (rect.center(), f),
            _ => panic!("row {item} is not an item"),
        }
    }

    #[test]
    fn a_title_click_opens_and_a_second_click_closes() {
        let mut ui = ui();
        frame(&mut ui, None, false);
        let f = bar(&mut ui);
        assert!(f.drop.is_none(), "the bar starts shut");
        let file = f.titles[0].rect.center();
        let f = click_at(&mut ui, file);
        assert!(f.drop.is_some(), "the click opened File");
        // Move away and back so the second click is a clean toggle.
        frame(&mut ui, Some([0.0, -200.0]), false);
        bar(&mut ui);
        let f = click_at(&mut ui, file);
        assert!(f.drop.is_none(), "the same title closes its own list");
    }

    #[test]
    fn an_item_released_on_its_line_is_chosen_and_the_list_closes() {
        let mut ui = ui();
        frame(&mut ui, None, false);
        let f = bar(&mut ui);
        let file = f.titles[0].rect.center();
        let f = click_at(&mut ui, file);
        let item = match &f.drop.as_ref().unwrap().rows[0] {
            Row::Item { rect, .. } => rect.center(),
            _ => panic!("first row is Open"),
        };
        frame(&mut ui, Some(item), true);
        bar(&mut ui);
        frame(&mut ui, Some(item), false);
        let (f, choice) = ui.menu_frame(&[FILE, EDIT], SIZE);
        assert_eq!(choice, Some(MenuEvent::Chose("File", "Open")));
        assert!(f.drop.is_none(), "choosing closes the list");
    }

    #[test]
    fn a_click_anywhere_else_dismisses_the_open_list() {
        let mut ui = ui();
        frame(&mut ui, None, false);
        let f = bar(&mut ui);
        let file = f.titles[0].rect.center();
        let f = click_at(&mut ui, file);
        assert!(f.drop.is_some());
        frame(&mut ui, Some([0.0, -200.0]), false);
        let f = click_at(&mut ui, [0.0, -200.0]);
        assert!(f.drop.is_none(), "a click on nothing dismisses the menu");
    }

    #[test]
    fn an_open_bar_tracks_the_titles_it_glides_over() {
        let mut ui = ui();
        frame(&mut ui, None, false);
        let f = bar(&mut ui);
        let (file, edit) = (f.titles[0].rect.center(), f.titles[1].rect.center());
        let edit_left = f.titles[1].rect.left;
        let f = click_at(&mut ui, file);
        assert!(f.drop.is_some(), "File is open");
        // Glide onto Edit — no click anywhere — and the open list follows.
        frame(&mut ui, Some(edit), false);
        let f = bar(&mut ui);
        let d = f.drop.expect("the list followed the titles");
        assert_eq!(d.bg.left, edit_left, "the list switched to Edit's column");
        // And a click now chooses from Edit's list.
        let (undo, _) = open_item_center(&mut ui, edit, 0);
        frame(&mut ui, Some(undo), true);
        bar(&mut ui);
        frame(&mut ui, Some(undo), false);
        let (f, choice) = ui.menu_frame(&[FILE, EDIT], SIZE);
        assert_eq!(choice, Some(MenuEvent::Chose("Edit", "Undo")));
        assert!(f.drop.is_none());
    }

    #[test]
    fn a_separator_clicks_for_nothing_but_still_eats_the_click() {
        let mut ui = ui();
        frame(&mut ui, None, false);
        let f = bar(&mut ui);
        let file = f.titles[0].rect.center();
        let f = click_at(&mut ui, file);
        let sep = match &f.drop.as_ref().unwrap().rows[2] {
            Row::Sep { rect } => rect.center(),
            _ => panic!("third row is the separator"),
        };
        frame(&mut ui, Some(sep), true);
        let _ = bar(&mut ui);
        frame(&mut ui, Some(sep), false);
        let (f, choice) = ui.menu_frame(&[FILE, EDIT], SIZE);
        assert_eq!(choice, None, "a separator chooses nothing");
        assert!(
            f.drop.is_some(),
            "and a separator click keeps the list open"
        );
    }

    #[test]
    fn the_open_menu_wins_the_press_over_older_widgets() {
        let mut ui = ui();
        frame(&mut ui, None, false);
        let f = bar(&mut ui);
        let file = f.titles[0].rect.center();
        let f = click_at(&mut ui, file);
        let item = match &f.drop.as_ref().unwrap().rows[0] {
            Row::Item { rect, .. } => rect.center(),
            _ => unreachable!(),
        };
        // An ordinary widget declared before the bar, under the item:
        // last declared wins, so the menu takes the press.
        frame(&mut ui, Some(item), true);
        ui.interact(
            widget_id(7, "under"),
            Rect::from_center(item[0], item[1], 400.0, 400.0),
        );
        let _ = bar(&mut ui);
        frame(&mut ui, Some(item), false);
        ui.interact(
            widget_id(7, "under"),
            Rect::from_center(item[0], item[1], 400.0, 400.0),
        );
        let (_, choice) = ui.menu_frame(&[FILE, EDIT], SIZE);
        assert_eq!(
            choice,
            Some(MenuEvent::Chose("File", "Open")),
            "the menu, not the widget, got the click"
        );
    }

    #[test]
    fn a_slider_line_travels_with_the_pointer_and_keeps_the_list_open() {
        let mut ui = ui();
        frame(&mut ui, None, false);
        let (f, _) = vbar(&mut ui);
        let view = f.titles[0].rect.center();
        // Open View the classic way.
        let f = {
            frame(&mut ui, Some(view), true);
            vbar(&mut ui);
            frame(&mut ui, Some(view), false);
            vbar(&mut ui).0
        };
        let row = match &f.drop.as_ref().expect("View is open").rows[3] {
            Row::Slider { rect, .. } => *rect,
            _ => panic!("the fourth line is the slider"),
        };
        // Glide to the line, right of centre — one frame so the row
        // holds the hover — then press and HOLD: the press resolves a
        // frame late, so the capture — and the slide — begin one frame
        // after the button falls.
        let grab = [row.left + row.w * 0.8, row.center()[1]];
        frame(&mut ui, Some(grab), false);
        vbar(&mut ui);
        frame(&mut ui, Some(grab), true);
        vbar(&mut ui);
        frame(&mut ui, Some(grab), true);
        let (f, slid) = vbar(&mut ui);
        assert!(f.drop.is_some(), "grabbing a slider keeps the list open");
        let MenuEvent::Slid(_, label, v) = slid.expect("the held press reports a value") else {
            panic!("a slide was expected, got {slid:?}");
        };
        assert_eq!(label, "Light");
        assert!(
            v > 0.5,
            "grabbing right of centre lands past the middle: {v}"
        );
        // Drag left, button still down: the value follows the pointer.
        let left = [row.left + row.w * 0.3, row.center()[1]];
        frame(&mut ui, Some(left), true);
        let (_, slid) = vbar(&mut ui);
        let MenuEvent::Slid(_, _, v2) = slid.expect("the drag reports each frame") else {
            panic!("a slide was expected");
        };
        assert!(v2 < v, "the knob followed the pointer left: {v2} < {v}");
        // Release: the last frame reports once more, and the menu — by
        // the design that makes a slider usable — stays open.
        frame(&mut ui, Some(left), false);
        let (f, slid) = vbar(&mut ui);
        assert!(matches!(slid, Some(MenuEvent::Slid(..))), "release commits");
        assert!(f.drop.is_some(), "and the list outlives the release");
    }

    #[test]
    fn a_readout_line_chooses_nothing_and_closes_nothing() {
        let mut ui = ui();
        frame(&mut ui, None, false);
        let (f, _) = vbar(&mut ui);
        let view = f.titles[0].rect.center();
        frame(&mut ui, Some(view), true);
        vbar(&mut ui);
        frame(&mut ui, Some(view), false);
        let (f, _) = vbar(&mut ui);
        let row = match &f.drop.as_ref().expect("View is open").rows[2] {
            Row::Readout { rect, .. } => *rect,
            _ => panic!("the third line is the readout"),
        };
        frame(&mut ui, Some(row.center()), true);
        vbar(&mut ui);
        frame(&mut ui, Some(row.center()), false);
        let (f, slid) = vbar(&mut ui);
        assert_eq!(slid, None, "a readout has no event to give");
        assert!(f.drop.is_some(), "and its click belongs to the list");
    }

    /// Click a one-menu bar's title open and return the open frame.
    fn opened(ui: &mut Ui, menu: &'static Menu<'static>) -> MenuFrame<'static> {
        frame(ui, None, false);
        let (f, _) = ui.menu_frame(&[*menu], SIZE);
        let title = f.titles[0].rect.center();
        frame(ui, Some(title), true);
        ui.menu_frame(&[*menu], SIZE);
        frame(ui, Some(title), false);
        ui.menu_frame(&[*menu], SIZE).0
    }

    #[test]
    fn a_checked_line_opens_a_gutter_and_ticks_its_own_row() {
        // The gutter is a property of the list, not the row: one
        // checked line shifts every label, so the marks stand in a
        // column and the plain lines line up with the ticked one.
        static MODES: Menu<'static> = Menu {
            title: "Modes",
            items: &[
                MenuItem::new("Markers", "").checked(true),
                MenuItem::new("Tile Map", ""),
            ],
        };
        static PLAIN: Menu<'static> = Menu {
            title: "Modes",
            items: &[MenuItem::new("Markers", ""), MenuItem::new("Tile Map", "")],
        };
        let mut marked = ui();
        let f = opened(&mut marked, &MODES);
        let d = f.drop.expect("the list is open");
        assert!(
            matches!(&d.rows[0], Row::Item { checked: true, .. }),
            "the checked line keeps its mark to the painter"
        );
        assert!(
            matches!(&d.rows[1], Row::Item { checked: false, .. }),
            "the plain line reports none"
        );
        // The list widened by exactly the gutter: same labels, same
        // fonts, one more column.
        let mut plain = ui();
        let g = opened(&mut plain, &PLAIN);
        let grow = d.bg.w - g.drop.expect("the plain list opens too").bg.w;
        assert!(
            (grow - 1.4 * marked.style.font_size).abs() < 1e-3,
            "the gutter is 1.4 em, not {grow}"
        );
    }

    #[test]
    fn a_checked_line_is_chosen_like_any_other() {
        // The mark reports; it does not disarm. Choosing the mode in
        // effect stays an ordinary — if idle — command.
        static MODES: Menu<'static> = Menu {
            title: "Modes",
            items: &[MenuItem::new("Markers", "").checked(true)],
        };
        let mut u = ui();
        let f = opened(&mut u, &MODES);
        let row = match &f.drop.as_ref().expect("open").rows[0] {
            Row::Item { rect, .. } => *rect,
            _ => panic!("the first line is a command"),
        };
        frame(&mut u, Some(row.center()), true);
        u.menu_frame(&[MODES], SIZE);
        frame(&mut u, Some(row.center()), false);
        let (g, chose) = u.menu_frame(&[MODES], SIZE);
        assert_eq!(chose, Some(MenuEvent::Chose("Modes", "Markers")));
        assert!(g.drop.is_none(), "and the list closes as any choice does");
    }

    #[test]
    fn the_bar_owns_the_mouse_over_its_whole_strip() {
        let mut ui = ui();
        frame(&mut ui, None, false);
        bar(&mut ui);
        // Far from every title, yet on the strip.
        frame(&mut ui, Some([350.0, 290.0]), false);
        assert!(ui.hovering(), "the bar strip is not scene");
    }

    /// Half-widths of the two notes plus a pad: the space a correct
    /// center-anchored layout leaves between two notes' centers.
    fn gap(u: &mut Ui, a: &str, b: &str, font: f32, pad: f32) -> f32 {
        (u.text_width(a, font) + u.text_width(b, font)) / 2.0 + pad
    }

    #[test]
    fn the_last_note_ends_at_the_bars_far_end() {
        let (font, pad, right, limit) = (14.0, 6.0, 500.0, 0.0);
        let mut u = ui();
        let laid = u.plan_notes(&["a", "the mode"], font, right, limit, pad);
        assert_eq!(laid.len(), 2);
        let w = u.text_width("the mode", font);
        assert!(
            (laid[1].1 + w / 2.0 - right).abs() < 0.01,
            "the last note's right edge is the bar's far end, not half a \
             note left of it: {}",
            laid[1].1
        );
    }

    #[test]
    fn notes_step_leftwards_without_touching() {
        let (font, pad, right, limit) = (14.0, 6.0, 500.0, 0.0);
        let mut u = ui();
        let laid = u.plan_notes(&["one", "two", "three"], font, right, limit, pad);
        for pair in laid.windows(2) {
            let need = gap(&mut u, pair[0].0.as_str(), pair[1].0.as_str(), font, pad);
            assert!(
                pair[1].1 - pair[0].1 >= need - 0.01,
                "`{}` and `{}` overlap: centers {:.1} and {:.1}, need {:.1} apart",
                pair[0].0,
                pair[1].0,
                pair[0].1,
                pair[1].1,
                need,
            );
        }
    }

    #[test]
    fn a_note_out_of_room_keeps_its_tail_behind_dots() {
        let (font, pad, right, limit) = (14.0, 6.0, 500.0, 380.0);
        let mut u = ui();
        let long =
            "shapes entry 3: kind `blob` is not a shape: the vocabulary is rect, circle, poly";
        let laid = u.plan_notes(&[long, "view"], font, right, limit, pad);
        assert_eq!(laid.len(), 2, "both notes still speak");
        assert_eq!(laid[1].0, "view", "the note with room is not cut");
        let cut = &laid[0].0;
        assert!(
            cut.starts_with("..."),
            "the overflow leads with dots: {cut}"
        );
        assert!(
            cut.ends_with("poly"),
            "the tail — the half that says something — survives: {cut}"
        );
        let w = u.text_width(cut, font);
        assert!(
            laid[0].1 - w / 2.0 >= limit - 0.01,
            "the cut note stays clear of the titles"
        );
    }

    #[test]
    fn when_the_room_runs_out_the_earliest_notes_fall_off_first() {
        let (font, pad, right, limit) = (14.0, 6.0, 500.0, 470.0);
        let mut u = ui();
        let laid = u.plan_notes(
            &["first", "second", "the mode that speaks last"],
            font,
            right,
            limit,
            pad,
        );
        assert!(
            !laid.iter().any(|l| l.0 == "first" || l.0 == "second"),
            "the earliest notes are the first sacrifices, got {:?}",
            laid.iter().map(|l| &l.0).collect::<Vec<_>>()
        );
        let last = &laid.last().expect("the last note keeps the far slot").0;
        assert!(
            last.ends_with("last") || last.starts_with("..."),
            "the last note holds its place, whole or cut: {last}"
        );
    }
}
