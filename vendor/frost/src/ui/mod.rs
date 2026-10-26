//! Reusable immediate-mode widgets: a [`Ui`] that draws [`button`](Ui::button),
//! [`checkbox`](Ui::checkbox), [`slider`](Ui::slider), [`label`](Ui::label),
//! draggable [`panel`](Ui::panel) chrome and a classic [`menu_bar`](Ui::menu_bar)
//! straight onto the frame's [`Canvas`], over whatever the game draws.
//! The menu bar is declared last, after every panel, so its open list
//! floats above them and wins the clicks that land on it.
//!
//! The style follows the frame flow: declare your widgets every frame from
//! [`Process::process`](crate::Process::process), exactly like the immediate
//! canvas methods. The `Ui` keeps only the interaction state the immediate
//! style cannot re-derive: which widget holds the mouse, where each panel
//! has been dragged to, and how tall each panel's content grew.
//!
//! ```no_run
//! # struct Demo { ui: frost::Ui, volume: f32, sound: bool }
//! # impl frost::Process for Demo {
//! fn process(&mut self, ctx: &mut frost::Context, _dt: f32) {
//!     self.ui.begin(ctx);
//!     self.ui.panel(ctx, "Settings", [-300.0, 150.0], 260.0, |ui, ctx| {
//!         ui.slider(ctx, "Volume", &mut self.volume, 0.0, 1.0);
//!         ui.checkbox(ctx, "Sound on", &mut self.sound);
//!         if ui.button(ctx, "Reset") {
//!             self.volume = 0.5;
//!         }
//!     });
//! }
//! # }
//! ```
//!
//! **The input model.** Every widget hit-tests with the rects it registered
//! *last* frame, so press, drag and release resolve against stable geometry
//! even while a panel is moving under the cursor, and when two widgets
//! overlap the one declared last (drawn on top) wins the press. The cost is
//! one frame of latency on the press: a press that lands on a widget arms
//! the drag on the next frame, at 60 Hz an imperceptible 16 ms.
//!
//! **Identity.** Widget ids are hashed from the widget's label and the id
//! scope of the panel it is declared in, so the same label in two panels is
//! two widgets — but the same label *twice in one panel* is one widget hit
//! twice; give interactive widgets distinct labels.
//!
//! **Order.** Every draw the `Ui` records gets a `z` counting up from
//! [`UiStyle::base_z`] in declaration order, so the UI paints over the
//! scene (whose nodes default to `z = 0.0`) and panels stack in the order
//! they are declared.

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use crate::{Color, Context, TextError};

mod menu;
mod panel;
mod style;
mod table;
mod tree;

pub use menu::{ItemKind, Menu, MenuEvent, MenuItem};
use panel::PanelState;
#[cfg(test)]
use panel::{panel_body_id, panel_id};
pub use style::UiStyle;
use table::TableFrame;
#[cfg(test)]
use table::resolve_cols;
pub use table::{Align, Col, ColSize};
pub use tree::{TreeEvent, TreeLine, TreeOut, TreeSpec, TreeState, TreeStyle};

/// The FNV-1a parameters: a cheap, deterministic byte hash. Determinism
/// matters — widget ids key the drag capture and the panel positions, so
/// the same label must hash to the same id on every frame.
const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;
/// How far the pointer may drift during a title-bar press and still count
/// as a click (fold toggle) rather than a drag.
const FOLD_DRAG_TOL: f32 = 4.0;

/// A rectangle in user space: `left` and `top` are the top-left corner
/// (`top` being the *larger* y, user space being y-up), `w` and `h` its
/// extent.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Rect {
    left: f32,
    top: f32,
    w: f32,
    h: f32,
}

impl Rect {
    fn from_center(cx: f32, cy: f32, w: f32, h: f32) -> Self {
        Self {
            left: cx - w / 2.0,
            top: cy + h / 2.0,
            w,
            h,
        }
    }

    fn center(&self) -> [f32; 2] {
        [self.left + self.w / 2.0, self.top - self.h / 2.0]
    }

    fn contains(&self, p: [f32; 2]) -> bool {
        p[0] >= self.left
            && p[0] <= self.left + self.w
            && p[1] <= self.top
            && p[1] >= self.top - self.h
    }
}

/// What one widget asks the `Ui` about its row rect: see [`Ui::interact`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Interaction {
    /// The pointer is over the widget, and no other widget holds it.
    hot: bool,
    /// This widget captured the press and is being dragged.
    held: bool,
    /// The button is down while this widget holds it.
    pressed: bool,
    /// The button was released over this widget, having been pressed on it.
    clicked: bool,
}

/// The row cursor a widget allocates from: the panel body, or the root
/// column outside any panel.
struct Layout {
    /// The content's left edge in user space.
    left: f32,
    /// The content's width: every row spans it.
    width: f32,
    /// The top edge of the next free row, moving down as rows are taken.
    cursor: f32,
    /// The id scope of the panel this layout belongs to (0 for the root).
    scope: u64,
    /// When set, this frame is a table (see [`Ui::table`]): widgets take
    /// the next cell across the row instead of a full-width row.
    table: Option<TableFrame>,
}

/// The answer a [`Ui::confirm`] box received: one of its two buttons
/// was clicked on this frame.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Answer {
    /// The affirmative button.
    Yes,
    /// The refusal.
    No,
}

/// The immediate-mode widget layer. Keep one per window, owned by your
/// [`Process`](crate::Process) alongside the state the widgets drive; call
/// [`Ui::begin`] once per frame, declare widgets, and mutate your game state
/// from their return values.
pub struct Ui {
    /// The font every label draws with, shared so all text hits one atlas.
    font: Arc<[u8]>,
    /// The look and metrics; public fields through [`Ui::style_mut`].
    pub style: UiStyle,
    // The frame's input snapshot, taken by `begin`.
    pointer: Option<[f32; 2]>,
    prev_pointer: Option<[f32; 2]>,
    down: bool,
    prev_down: bool,
    pressed: bool,
    released: bool,
    // The press bookkeeping spanning frames.
    /// Whether the previous frame saw a press edge.
    was_pressed: bool,
    /// The pointer where that press happened.
    press_pointer: Option<[f32; 2]>,
    /// Where the most recent press started, held for the press's whole
    /// life: a widget release far from here was a drag, not a click.
    hold_origin: Option<[f32; 2]>,
    /// The widget that captured the press and is being dragged.
    active: Option<u64>,
    /// The widget released this frame, if its press started on it.
    released_active: Option<u64>,
    /// The widget under the pointer, topmost first, resolved at `begin`
    /// from the rects the previous frame declared (see `hovering`).
    hover: Option<u64>,
    /// The menu bar's open menu, by its title's widget id: the one piece
    /// of menu state the immediate style cannot re-derive (see
    /// [`Ui::menu_bar`]).
    menu_open: Option<u64>,
    /// Whether a modal box stands this frame: widgets outside it are
    /// drawn but inert, and panels refuse their drags (see
    /// [`Ui::set_modal`] and [`Ui::confirm`]).
    modal_armed: bool,
    /// Whether the widget being declared right now belongs to the
    /// modal box — the exception that keeps the box clickable.
    in_modal: bool,
    /// Every interactive rect declared so far this frame, in order.
    candidates: Vec<(u64, Rect)>,
    /// Each panel's retained position and size, keyed by its title id.
    panels: HashMap<u64, PanelState>,
    /// Each table's retained auto column widths, keyed by its id (scoped
    /// to the enclosing panel), measured last frame.
    tables: HashMap<u64, Vec<f32>>,
    /// The alignment of the slot the last [`Ui::row`] call returned:
    /// a table column's, or `Center` for a plain row. Text widgets honor
    /// it.
    align: Align,
    /// The layout stack: the root column, plus one frame per open panel.
    layout: Vec<Layout>,
    /// The next draw's `z`, counting up from `style.base_z`.
    z: f32,
    /// Measured text widths, keyed by the string and the bits of the size
    /// and weight.
    text_w: HashMap<(String, u32, u32), f32>,
}

impl Ui {
    /// Loads the UI font from `path`; the bytes live behind an [`Arc`] so
    /// every label shares one glyph atlas. A missing file or non-font
    /// arrives as [`TextError`].
    pub fn from_font(path: impl AsRef<Path>) -> Result<Self, TextError> {
        let bytes = std::fs::read(path.as_ref()).map_err(TextError::Io)?;
        Self::from_bytes(&bytes)
    }

    /// Builds the UI from font bytes already in memory, for example
    /// embedded with `include_bytes!` — the way to go where there is no
    /// file system, such as a web build.
    pub fn from_bytes(font: &[u8]) -> Result<Self, TextError> {
        swash::FontRef::from_index(font, 0).ok_or(TextError::InvalidFont)?;
        Ok(Self {
            font: Arc::from(font),
            style: UiStyle::default(),
            pointer: None,
            prev_pointer: None,
            down: false,
            prev_down: false,
            pressed: false,
            released: false,
            was_pressed: false,
            press_pointer: None,
            hold_origin: None,
            active: None,
            released_active: None,
            hover: None,
            menu_open: None,
            modal_armed: false,
            in_modal: false,
            candidates: Vec::new(),
            panels: HashMap::new(),
            tables: HashMap::new(),
            align: Align::Center,
            layout: Vec::new(),
            z: 0.0,
            text_w: HashMap::new(),
        })
    }

    /// Starts a UI frame: snapshots the pointer and buttons, resolves the
    /// press that armed last frame onto its widget, drags the panel whose
    /// title bar holds the button, and resets the row cursor and the draw
    /// order. Call it once, before declaring any widget.
    pub fn begin(&mut self, ctx: &Context) {
        let input = UiInput {
            pointer: ctx.mouse_position(),
            down: ctx.mouse_button_down(crate::MouseButton::Left),
            size: ctx.size(),
        };
        self.begin_input(input);
    }

    /// Whether the pointer is over any widget of the UI: the same hit test
    /// the widgets themselves use, so a game that reads the mouse directly
    /// can tell a click on the UI from a click on its own scene. False
    /// until the frame after a panel first appears — the hit test runs on
    /// the previous frame's rects, as the press does.
    pub fn hovering(&self) -> bool {
        // A modal box owns the whole window: even empty air answers
        // true, so every game action guarded by this stands down.
        self.modal_armed || self.hover.is_some()
    }

    /// Declare the frame's modality: while a modal box stands, the UI
    /// owns the window entire — [`Ui::hovering`] answers true even in
    /// empty air, widgets outside the box are drawn but inert, and
    /// panels refuse to be dragged. The box itself, drawn between the
    /// internals of [`Ui::confirm`], keeps its input. Call once per
    /// frame right after [`Ui::begin`], with the app's live modal
    /// state; a drag captured before the box rose is dropped on the rise.
    pub fn set_modal(&mut self, armed: bool) {
        if armed && !self.modal_armed {
            self.active = None;
            self.hold_origin = None;
        }
        self.modal_armed = armed;
    }

    /// Whether a modal box stands this frame (see [`Ui::set_modal`]).
    pub fn modal(&self) -> bool {
        self.modal_armed
    }

    /// The input-free heart of [`Ui::begin`], separate so the interaction
    /// model is testable without a window.
    fn begin_input(&mut self, input: UiInput) {
        // A press seen last frame now resolves over the rects that frame
        // registered, last declared (topmost) first.
        let won = if self.was_pressed {
            self.press_pointer.and_then(|p| self.topmost(p))
        } else {
            None
        };
        // The hover resolves the same way, against the live pointer, and
        // feeds both the widget hover states and `hovering`.
        self.hover = input.pointer.and_then(|p| self.topmost(p));
        let pressed = input.down && !self.prev_down;
        let released = !input.down && self.prev_down;
        let drag = match (input.pointer, self.prev_pointer) {
            (Some(p), Some(q)) => [p[0] - q[0], p[1] - q[1]],
            _ => [0.0, 0.0],
        };
        if released {
            // The release belongs to the held widget — or, for a press of
            // a single frame, to the widget that just won the press.
            self.released_active = self.active.take().or(won);
        } else {
            self.released_active = None;
            if won.is_some() {
                self.active = won;
            }
        }
        if !input.down && !released {
            // Down state lost without a release edge (the cursor left the
            // window): drop the capture.
            self.active = None;
        }
        // A dragged panel's title bar holding the id: move with the
        // pointer. A modal box stands: panels hold still.
        if input.down
            && !self.modal_armed
            && let Some(active) = self.active
            && let Some(panel) = self.panels.get_mut(&active)
        {
            panel.pos = [panel.pos[0] + drag[0], panel.pos[1] + drag[1]];
        }
        // The frame's snapshot.
        self.prev_pointer = input.pointer;
        self.pointer = input.pointer;
        self.down = input.down;
        self.prev_down = input.down;
        self.pressed = pressed;
        self.released = released;
        self.was_pressed = pressed;
        self.press_pointer = if pressed { input.pointer } else { None };
        if pressed {
            // Remember where this press began: at release the distance
            // from here tells a click (barely moved) from a drag (moved).
            self.hold_origin = input.pointer;
        }
        // Per-frame scratch.
        self.candidates.clear();
        self.z = self.style.base_z;
        // The root column: widgets outside any panel stack from the
        // window's top-left corner downward.
        let margin = self.style.root_margin;
        let width = self.style.root_width;
        self.layout.clear();
        self.layout.push(Layout {
            left: -input.size.0 / 2.0 + margin,
            width,
            cursor: input.size.1 / 2.0 - margin,
            scope: 0,
            table: None,
        });
    }

    /// Takes the next slot from the current layout: a full-width row in a
    /// column, or — inside a table (see [`Ui::table`]) — the next cell,
    /// walking left to right and wrapping to a new row after the last
    /// column. Sets `self.align` to the slot's alignment, which the text
    /// widgets honor.
    fn row(&mut self, h: f32) -> Rect {
        let style = self.style;
        let cell = {
            let layout = self
                .layout
                .last_mut()
                .expect("Ui layout stack is never empty");
            layout.table.as_mut().and_then(|t| {
                let n = t.widths.len();
                if n == 0 {
                    return None;
                }
                let i = t.next;
                t.next += 1;
                let col = i % n;
                // Cells are uniformly a row tall, so every widget in a
                // row shares one centerline however tall its own draw.
                let top = t.top - (i / n) as f32 * (style.row_h + style.row_gap);
                Some((
                    Rect {
                        left: t.x[col],
                        top,
                        w: t.widths[col],
                        h: style.row_h,
                    },
                    t.aligns[col],
                ))
            })
        };
        if let Some((rect, align)) = cell {
            self.align = align;
            return rect;
        }
        let layout = self
            .layout
            .last_mut()
            .expect("Ui layout stack is never empty");
        // The row hangs `row_gap` below the cursor: the rect's top edge
        // sits at the gapped cursor and the cursor moves on to its bottom.
        layout.cursor -= style.row_gap;
        let rect = Rect {
            left: layout.left,
            top: layout.cursor,
            w: layout.width,
            h,
        };
        layout.cursor -= h;
        self.align = Align::Center;
        rect
    }

    /// The last (topmost) rect of the previous frame that holds `p`, with
    /// its id — the frame's single hit test, shared by the press
    /// resolution and the hover.
    fn topmost(&self, p: [f32; 2]) -> Option<u64> {
        self.candidates
            .iter()
            .rev()
            .find(|(_, r)| r.contains(p))
            .map(|(id, _)| *id)
    }

    /// Asks whether the press, hold and release belong to `id` and its
    /// rect, and registers the rect so next frame's press can find it.
    fn interact(&mut self, id: u64, rect: Rect) -> Interaction {
        let held = self.active == Some(id);
        self.candidates.push((id, rect));
        let it = Interaction {
            // Hover shows only while no other widget holds the press.
            hot: self.hover == Some(id) && (held || self.active.is_none()),
            held,
            pressed: held && self.down,
            clicked: self.released_active == Some(id) && self.hover == Some(id),
        };
        // A modal box stands: every widget outside it is drawn plainly
        // but hears nothing — no light, no press, no click.
        if self.modal_armed && !self.in_modal {
            return Interaction {
                hot: false,
                held: false,
                pressed: false,
                clicked: false,
            };
        }
        it
    }

    /// The next draw's `z`.
    fn z(&mut self) -> f32 {
        let z = self.z;
        self.z += 1.0;
        z
    }

    /// Fills `rect` on the canvas at the next `z`.
    fn fill(&mut self, ctx: &mut Context, rect: Rect, color: Color) {
        let [cx, cy] = rect.center();
        ctx.rectangle(cx, cy, rect.w / 2.0, rect.h / 2.0, color, self.z());
    }

    /// Draws centered text at `p`, at the style's label weight.
    fn text(&mut self, ctx: &mut Context, p: [f32; 2], size: f32, color: Color, s: &str) {
        let z = self.z();
        let weight = self.style.font_weight;
        ctx.text(p[0], p[1], &self.font, s, size, weight, color, z);
    }

    /// Draws `s` centered on the x `center` of the line `y`.
    fn text_at_x(
        &mut self,
        ctx: &mut Context,
        center: f32,
        y: f32,
        size: f32,
        color: Color,
        s: &str,
    ) {
        self.text(ctx, [center, y], size, color, s);
    }

    /// The advance width of `s` at the given size and the style's label
    /// weight, measured once per (string, size, weight) and remembered.
    fn text_width(&mut self, s: &str, size: f32) -> f32 {
        let weight = self.style.font_weight;
        let key = (s.to_owned(), size.to_bits(), weight.to_bits());
        if let Some(w) = self.text_w.get(&key) {
            return *w;
        }
        let w = crate::text::layout(&self.font, s, size, weight)
            .map(|l| l.width)
            .unwrap_or(0.0);
        self.text_w.insert(key, w);
        w
    }

    /// The widget id: the label hashed into the current panel's scope.
    fn widget_id(&self, label: &str) -> u64 {
        let scope = self
            .layout
            .last()
            .expect("layout stack is never empty")
            .scope;
        widget_id(scope, label)
    }

    /// A plain line of text, no interaction, in the current slot's
    /// alignment (centered outside any table column).
    pub fn label(&mut self, ctx: &mut Context, text: &str) {
        let color = self.style.text;
        self.text_slot(ctx, text, color);
    }

    /// A right-aligned numeric readout — the value half of a table-driven
    /// slider row — drawn in the muted colour and hugging its content for
    /// an auto column.
    pub fn readout(&mut self, ctx: &mut Context, text: &str) {
        let color = self.style.text_muted;
        self.text_slot(ctx, text, color);
    }

    /// The shared body of [`Ui::label`] and [`Ui::readout`]: take the next
    /// slot, place `text` at `color` per the slot's alignment, and report
    /// its width so an auto column can size to it.
    fn text_slot(&mut self, ctx: &mut Context, text: &str, color: Color) {
        let row = self.row(self.style.row_h);
        let size = self.style.font_size;
        let w = self.text_width(text, size);
        self.note_cell_width(w);
        let cx = match self.align {
            Align::Left => row.left + w / 2.0,
            Align::Center => row.left + row.w / 2.0,
            Align::Right => row.left + row.w - w / 2.0,
        };
        self.text_at_x(ctx, cx, row.center()[1], size, color, text);
    }

    /// The modal question box, drawn over everything the frame has
    /// drawn: the window dims under a veil, the question stands at the
    /// centre on a panel, and the two answers wait beneath it. The
    /// frame an answer is clicked, `Some` it returns. The app declares
    /// the box's modality with [`Ui::set_modal`] at the frame's start;
    /// this call is the box itself — the one island of input while it
    /// stands. `size` is the window's user-space size.
    pub fn confirm(
        &mut self,
        ctx: &mut Context,
        size: [f32; 2],
        question: &str,
        yes: &str,
        no: &str,
    ) -> Option<Answer> {
        let style = self.style;
        let pad = style.pad;
        // The panel sizes to the question with a courteous floor, and
        // holds three rows: the question and the two answers.
        let w = (self.text_width(question, style.font_size) + 4.0 * pad).max(240.0);
        let h = 2.0 * pad + 3.0 * style.row_h + 3.0 * style.row_gap;
        let panel = Rect {
            left: -w / 2.0,
            top: h / 2.0,
            w,
            h,
        };
        // The veil over the whole window first — under the panel, above
        // the world — then the panel's own face.
        let veil = Color {
            r: 0.0,
            g: 0.0,
            b: 0.0,
            a: 0.45,
        };
        let win = Rect {
            left: -size[0] / 2.0,
            top: size[1] / 2.0,
            w: size[0],
            h: size[1],
        };
        self.fill(ctx, win, veil);
        let bg = style.panel_bg;
        self.fill(ctx, panel, bg);
        // The rows live inside the padding, in a scope of their own:
        // a "Yes" button in the box cannot collide with one elsewhere.
        let scope = widget_id(
            self.layout
                .last()
                .expect("layout stack is never empty")
                .scope,
            question,
        );
        self.layout.push(Layout {
            left: panel.left + pad,
            width: panel.w - 2.0 * pad,
            cursor: panel.top - pad,
            scope,
            table: None,
        });
        self.in_modal = true;
        self.label(ctx, question);
        let yes_hit = self.button(ctx, yes);
        let no_hit = self.button(ctx, no);
        self.in_modal = false;
        self.layout.pop();
        if yes_hit {
            Some(Answer::Yes)
        } else if no_hit {
            Some(Answer::No)
        } else {
            None
        }
    }

    /// Spacing of `h` pixels, pushing the rows below it down.
    pub fn space(&mut self, h: f32) {
        let layout = self.layout.last_mut().expect("layout stack is never empty");
        layout.cursor -= h;
    }

    /// A push button spanning the row; returns `true` on the frame it is
    /// clicked — the button pressed on it released over it, the same arming
    /// as [`Shape`](crate::Shape)-level buttons.
    pub fn button(&mut self, ctx: &mut Context, label: &str) -> bool {
        let id = self.widget_id(label);
        let row = self.row(self.style.row_h);
        let it = self.interact(id, row);
        let style = self.style;
        let color = if it.pressed {
            style.widget_press
        } else if it.hot {
            style.widget_hover
        } else {
            style.widget_bg
        };
        self.fill(ctx, row, color);
        let size = style.font_size;
        let text_color = style.text;
        let [cx, cy] = row.center();
        let label = label.to_owned();
        self.text(ctx, [cx, cy], size, text_color, &label);
        it.clicked
    }

    /// A checkbox with its label to the right of the box, driving `value`;
    /// returns `true` on the frame it toggles. The whole row is the click
    /// target, so the label toggles it too.
    pub fn checkbox(&mut self, ctx: &mut Context, label: &str, value: &mut bool) -> bool {
        let id = self.widget_id(label);
        let row = self.row(self.style.row_h.max(self.style.check_size));
        let it = self.interact(id, row);
        if it.clicked {
            *value = !*value;
        }
        let style = self.style;
        let side = style.check_size;
        let box_rect = Rect::from_center(row.left + side / 2.0, row.center()[1], side, side);
        let box_color = if *value {
            if it.pressed {
                style.widget_press.lerp(style.accent, 0.6)
            } else if it.hot {
                style.accent.lerp(
                    Color {
                        r: 1.0,
                        g: 1.0,
                        b: 1.0,
                        a: 1.0,
                    },
                    0.15,
                )
            } else {
                style.accent
            }
        } else if it.pressed {
            style.widget_press
        } else if it.hot {
            style.widget_hover
        } else {
            style.widget_bg
        };
        self.fill(ctx, box_rect, box_color);
        if *value {
            // The tick: a short leg down-right and a long leg up-right.
            let [cx, cy] = box_rect.center();
            let z = self.z();
            let dark = Color {
                r: 0.05,
                g: 0.07,
                b: 0.10,
                a: 1.0,
            };
            ctx.line(
                cx - 0.30 * side,
                cy - 0.02 * side,
                cx - 0.10 * side,
                cy - 0.24 * side,
                dark,
                2.5,
                z,
            );
            ctx.line(
                cx - 0.10 * side,
                cy - 0.24 * side,
                cx + 0.32 * side,
                cy + 0.24 * side,
                dark,
                2.5,
                z + 0.5,
            );
        }
        let size = style.font_size;
        let text_color = style.text;
        let label = label.to_owned();
        let w = self.text_width(&label, size);
        let x = row.left + side + 8.0;
        let cy = row.center()[1];
        self.text_at_x(ctx, x + w / 2.0, cy, size, text_color, &label);
        it.clicked
    }

    /// A horizontal slider labelled on its left, driving `value` clamped
    /// into `min..=max` and shown at the row's right; returns `true` on the
    /// frames its value changed. Pressing the track jumps the value to the
    /// press point, dragging follows the pointer.
    ///
    /// This lays its own label and readout into one row. Inside a table,
    /// where the label and value are their own aligned columns, use
    /// [`Ui::slider_track`] for just the draggable track.
    pub fn slider(
        &mut self,
        ctx: &mut Context,
        label: &str,
        value: &mut f32,
        min: f32,
        max: f32,
    ) -> bool {
        let id = self.widget_id(label);
        let row = self.row(self.style.row_h);
        let style = self.style;
        let pad = style.pad;
        let knob = style.knob_r;
        let label_w = self.text_width(label, style.font_size);
        let value_s = format!("{value:.2}");
        let value_w = self.text_width(&value_s, style.font_size);
        // The track between the label column and the value readout.
        let x0 = row.left + label_w + pad;
        let x1 = row.left + row.w - value_w - pad;
        let track = if x1 > x0 {
            Rect::from_center(
                (x0 + x1) / 2.0,
                row.center()[1],
                x1 - x0,
                style.track_h.max(2.0 * knob),
            )
        } else {
            row
        };
        let changed = self.drag_track(ctx, id, track, value, min, max);
        // The readout shows the value as it now stands, after any drag.
        let value_s = format!("{value:.2}");
        let value_w = self.text_width(&value_s, style.font_size);
        let size = style.font_size;
        let text_color = style.text;
        let muted = style.text_muted;
        let label = label.to_owned();
        let cy = row.center()[1];
        self.text_at_x(ctx, row.left + label_w / 2.0, cy, size, text_color, &label);
        self.text_at_x(
            ctx,
            row.left + row.w - value_w / 2.0,
            cy,
            size,
            muted,
            &value_s,
        );
        changed
    }

    /// The draggable track of a slider on its own, filling the current row
    /// or table cell — no label, no readout (a table supplies those as
    /// separate aligned columns). Same grab-and-drag behavior as
    /// [`Ui::slider`]; returns `true` on the frames `value` changed.
    pub fn slider_track(
        &mut self,
        ctx: &mut Context,
        label: &str,
        value: &mut f32,
        min: f32,
        max: f32,
    ) -> bool {
        let id = self.widget_id(label);
        let row = self.row(self.style.row_h);
        let style = self.style;
        let knob = style.knob_r;
        let track = Rect::from_center(
            row.center()[0],
            row.center()[1],
            row.w,
            style.track_h.max(2.0 * knob),
        );
        let changed = self.drag_track(ctx, id, track, value, min, max);
        // A track in an auto column is unusual (it wants to stretch), but
        // keep a sane minimum so it does not collapse to nothing.
        self.note_cell_width(2.0 * knob + 40.0);
        changed
    }

    /// The shared heart of [`Ui::slider`] and [`Ui::slider_track`]: draw
    /// `track`, drive `value` from the pointer while it is held, and paint
    /// the filled part and the knob. Returns whether `value` changed.
    fn drag_track(
        &mut self,
        ctx: &mut Context,
        id: u64,
        track: Rect,
        value: &mut f32,
        min: f32,
        max: f32,
    ) -> bool {
        let style = self.style;
        let knob = style.knob_r;
        let it = self.interact(id, track);
        // A range (or a track) too small to travel has nothing to drive,
        // but still draws.
        let travel = max > min && track.w > 2.0 * knob;
        // While held the knob follows the pointer, clamped to the knob's
        // travel; a click (a press-and-release of a frame) sets it too.
        let mut changed = false;
        if travel
            && (it.held || it.clicked)
            && let Some(p) = self.pointer
        {
            let (lo, hi) = (track.left + knob, track.left + track.w - knob);
            let t = ((p[0] - lo) / (hi - lo)).clamp(0.0, 1.0);
            let v = min + t * (max - min);
            if v != *value {
                *value = v;
                changed = true;
            }
        }
        let t = if travel {
            ((*value - min) / (max - min)).clamp(0.0, 1.0)
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
        // The knob.
        let kc = [knob_x, track.center()[1]];
        let white = Color {
            r: 1.0,
            g: 1.0,
            b: 1.0,
            a: 1.0,
        };
        let knob_color = if it.held {
            style.accent.lerp(white, 0.35)
        } else if it.hot {
            style.accent.lerp(white, 0.15)
        } else {
            style.accent
        };
        let z = self.z();
        ctx.circle(kc[0], kc[1], knob, knob_color, z);
        changed
    }

    /// How far the pointer sits from where the current (or most recent)
    /// press began — below [`FOLD_DRAG_TOL`], a release counts as a click.
    fn click_travel(&self) -> f32 {
        match (self.hold_origin, self.pointer) {
            (Some(o), Some(p)) => ((p[0] - o[0]).powi(2) + (p[1] - o[1]).powi(2)).sqrt(),
            _ => 0.0,
        }
    }
}

/// The frame's input snapshot [`Ui::begin_input`] works from.
struct UiInput {
    pointer: Option<[f32; 2]>,
    down: bool,
    size: (f32, f32),
}

/// The id of an interactive widget: its label hashed into its panel's
/// scope (0 for the root column).
fn widget_id(scope: u64, label: &str) -> u64 {
    let mut h = FNV_OFFSET;
    for b in scope.to_le_bytes() {
        h = (h ^ b as u64).wrapping_mul(FNV_PRIME);
    }
    for &b in label.as_bytes() {
        h = (h ^ b as u64).wrapping_mul(FNV_PRIME);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The repo's UI font, read once; the test binary runs from the crate
    /// root, so the asset path is stable.
    pub(super) fn font() -> Arc<[u8]> {
        static FONT: std::sync::OnceLock<Arc<[u8]>> = std::sync::OnceLock::new();
        FONT.get_or_init(|| {
            let path = format!(
                "{}/assets/fonts/JameGem08_2026-Regular.ttf",
                std::env!("CARGO_MANIFEST_DIR")
            );
            Arc::from(
                std::fs::read(&path)
                    .unwrap_or_else(|e| panic!("failed to read {path}: {e}"))
                    .as_slice(),
            )
        })
        .clone()
    }

    pub(super) fn ui() -> Ui {
        Ui::from_bytes(&font()).expect("test font")
    }

    /// Feeds one frame of input through `begin_input`.
    pub(super) fn frame(ui: &mut Ui, pointer: Option<[f32; 2]>, down: bool) {
        ui.begin_input(UiInput {
            pointer,
            down,
            size: (800.0, 600.0),
        });
    }

    #[test]
    fn press_arms_next_frame_and_click_needs_release_over() {
        let mut ui = ui();
        // Frame 1: the widget registers its rect; a press lands on it.
        // Hover resolves from last frame's rects, so this frame it is not
        // hot yet — one frame of latency is the documented model.
        frame(&mut ui, Some([0.0, 0.0]), true);
        let id = widget_id(0, "go");
        let rect = Rect::from_center(0.0, 0.0, 100.0, 30.0);
        let it = ui.interact(id, rect);
        assert!(!it.hot, "the rect is only known next frame");
        assert!(!it.held, "a press arms with a frame of latency");
        // Frame 2: still held, no release; now hover sees the rect.
        frame(&mut ui, Some([0.0, 0.0]), true);
        let it = ui.interact(id, rect);
        assert!(it.hot);
        assert!(ui.hovering(), "the same resolution backs `hovering`");
        assert!(it.held);
        assert!(it.pressed);
        assert!(!it.clicked);
        // Frame 3: released over the widget — a click.
        frame(&mut ui, Some([0.0, 0.0]), false);
        let it = ui.interact(id, rect);
        assert!(it.clicked);
    }

    #[test]
    fn release_outside_the_widget_is_not_a_click() {
        let mut ui = ui();
        let id = widget_id(0, "go");
        let rect = Rect::from_center(0.0, 0.0, 100.0, 30.0);
        frame(&mut ui, Some([0.0, 0.0]), true);
        ui.interact(id, rect);
        frame(&mut ui, Some([900.0, 0.0]), true);
        let it = ui.interact(id, rect);
        assert!(it.held, "the drag keeps holding off-widget");
        frame(&mut ui, Some([900.0, 0.0]), false);
        let it = ui.interact(id, rect);
        assert!(!it.clicked, "release happened outside");
    }

    #[test]
    fn press_on_nothing_holds_nothing() {
        let mut ui = ui();
        frame(&mut ui, Some([500.0, 250.0]), true);
        ui.interact(widget_id(0, "go"), Rect::from_center(0.0, 0.0, 100.0, 30.0));
        frame(&mut ui, Some([500.0, 250.0]), true);
        assert_eq!(ui.active, None);
    }

    #[test]
    fn topmost_widget_wins_an_overlap_press() {
        let mut ui = ui();
        let below = widget_id(0, "below");
        let above = widget_id(0, "above");
        let r_below = Rect::from_center(0.0, 0.0, 100.0, 30.0);
        let r_above = Rect::from_center(0.0, 0.0, 40.0, 20.0);
        frame(&mut ui, Some([0.0, 0.0]), true);
        ui.interact(below, r_below);
        ui.interact(above, r_above);
        frame(&mut ui, Some([0.0, 0.0]), true);
        assert_eq!(ui.active, Some(above), "last declared (on top) wins");
    }

    #[test]
    fn a_press_of_one_frame_still_clicks() {
        let mut ui = ui();
        let id = widget_id(0, "go");
        let rect = Rect::from_center(0.0, 0.0, 100.0, 30.0);
        frame(&mut ui, Some([0.0, 0.0]), true);
        ui.interact(id, rect);
        frame(&mut ui, Some([0.0, 0.0]), false);
        let it = ui.interact(id, rect);
        assert!(it.clicked);
        assert_eq!(ui.active, None, "a never-held capture does not linger");
    }

    #[test]
    fn panel_drags_with_the_pointer_once_held() {
        let mut ui = ui();
        let id = panel_id("Settings");
        ui.panels.insert(
            id,
            PanelState {
                pos: [0.0, 0.0],
                w: 200.0,
                h: 60.0,
                folded: false,
                rect: [-100.0, -30.0, 100.0, 30.0],
            },
        );
        let title = Rect::from_center(0.0, 30.0 - 14.0, 200.0, 28.0);
        frame(&mut ui, Some([0.0, 16.0]), true);
        ui.interact(id, title);
        // The panel moves by the pointer's delta, not to the pointer.
        frame(&mut ui, Some([10.0, 20.0]), true);
        assert_eq!(ui.panel_position("Settings"), Some([10.0, 4.0]));
        frame(&mut ui, Some([17.0, 26.0]), true);
        assert_eq!(ui.panel_position("Settings"), Some([17.0, 10.0]));
        frame(&mut ui, Some([17.0, 26.0]), false);
        frame(&mut ui, Some([90.0, 90.0]), false);
        assert_eq!(
            ui.panel_position("Settings"),
            Some([17.0, 10.0]),
            "the position is retained after the release"
        );
    }

    #[test]
    fn a_modal_blinds_everything_outside_the_box() {
        let mut ui = ui();
        frame(&mut ui, Some([0.0, 0.0]), false);
        let id = widget_id(0, "go");
        let rect = Rect::from_center(0.0, 0.0, 100.0, 30.0);
        ui.interact(id, rect);
        // The box rises. A press lands on the plain widget's rect —
        // and means nothing.
        ui.set_modal(true);
        frame(&mut ui, Some([0.0, 0.0]), true);
        let it = ui.interact(id, rect);
        assert!(!it.pressed && !it.hot, "outside, the press falls dead");
        frame(&mut ui, Some([0.0, 0.0]), false);
        let it = ui.interact(id, rect);
        assert!(!it.clicked, "outside, the release clicks nothing");
        assert!(ui.hovering(), "armed, the UI owns the pointer whole");
        // Inside the box, the same press-release clicks.
        ui.set_modal(false);
        ui.set_modal(true);
        frame(&mut ui, Some([0.0, 0.0]), false);
        ui.in_modal = true;
        ui.interact(id, rect);
        ui.in_modal = false;
        frame(&mut ui, Some([0.0, 0.0]), true);
        ui.in_modal = true;
        ui.interact(id, rect);
        ui.in_modal = false;
        frame(&mut ui, Some([0.0, 0.0]), true);
        ui.in_modal = true;
        let it = ui.interact(id, rect);
        assert!(it.pressed, "inside, the press captures");
        ui.in_modal = false;
        frame(&mut ui, Some([0.0, 0.0]), false);
        ui.in_modal = true;
        let it = ui.interact(id, rect);
        assert!(it.clicked, "inside, the release clicks");
        ui.in_modal = false;
    }

    #[test]
    fn panels_hold_still_while_the_box_stands() {
        let mut ui = ui();
        let id = panel_id("Settings");
        ui.panels.insert(
            id,
            PanelState {
                pos: [0.0, 0.0],
                w: 200.0,
                h: 60.0,
                folded: false,
                rect: [-100.0, -30.0, 100.0, 30.0],
            },
        );
        let title = Rect::from_center(0.0, 30.0 - 14.0, 200.0, 28.0);
        frame(&mut ui, Some([0.0, 16.0]), false);
        ui.interact(id, title);
        // The box rises; a press lands on the title bar and drags.
        ui.set_modal(true);
        frame(&mut ui, Some([0.0, 16.0]), true);
        ui.interact(id, title);
        frame(&mut ui, Some([30.0, 40.0]), true);
        assert_eq!(
            ui.panel_position("Settings"),
            Some([0.0, 0.0]),
            "the box owns the pointer: the panel holds still"
        );
    }

    #[test]
    fn ids_are_stable_and_scoped() {
        assert_eq!(widget_id(0, "go"), widget_id(0, "go"));
        assert_ne!(widget_id(0, "go"), widget_id(1, "go"), "panels scope ids");
        assert_ne!(widget_id(0, "go"), widget_id(0, "stop"));
        assert_ne!(panel_id("go"), widget_id(0, "go"), "domains do not mix");
    }

    #[test]
    fn rect_contains_edges() {
        let r = Rect::from_center(0.0, 0.0, 100.0, 20.0);
        assert!(r.contains([50.0, 10.0]));
        assert!(r.contains([-50.0, -10.0]));
        assert!(!r.contains([50.01, 0.0]));
        assert!(!r.contains([0.0, 10.01]));
    }

    #[test]
    fn hovering_follows_the_previous_frames_rects() {
        let mut ui = ui();
        frame(&mut ui, Some([0.0, 0.0]), false);
        assert!(!ui.hovering());
        let rect = Rect::from_center(0.0, 0.0, 100.0, 30.0);
        ui.interact(widget_id(0, "go"), rect);
        // Next frame, same spot: the rect registered last frame is hit.
        frame(&mut ui, Some([0.0, 0.0]), false);
        assert!(ui.hovering());
        // A pointer outside every rect — and no pointer at all — hover
        // nothing.
        frame(&mut ui, Some([300.0, 100.0]), false);
        assert!(!ui.hovering());
        frame(&mut ui, None, false);
        assert!(!ui.hovering());
    }

    #[test]
    fn a_later_panel_blocks_the_earlier_one() {
        let mut ui = ui();
        let under = widget_id(0, "go");
        let over = panel_body_id("Cover");
        let r_under = Rect::from_center(0.0, 0.0, 100.0, 30.0);
        let r_over = Rect::from_center(0.0, 0.0, 200.0, 60.0);
        frame(&mut ui, Some([0.0, 0.0]), true);
        // Declared order: the covered widget first, the covering panel body
        // after it (panels draw over what came before).
        ui.interact(under, r_under);
        ui.interact(over, r_over);
        // The press resolves to the topmost claim — the panel body, whose
        // id is no panel's, so it captures the press and moves nothing.
        frame(&mut ui, Some([0.0, 0.0]), true);
        assert_eq!(ui.active, Some(over));
        let it = ui.interact(under, r_under);
        assert!(
            !it.hot,
            "the covered widget shows no hover through the panel"
        );
        assert!(!it.clicked);
    }

    #[test]
    fn resolve_cols_splits_fixed_auto_and_stretch() {
        let cols = [
            Col::px(40.0, Align::Left),
            Col::stretch(1.0, Align::Left),
            Col::auto(Align::Center),
        ];
        // One gap (col_gap) between each of the three columns; a stored
        // auto width of 25 for column 2 (index 2).
        let stored = [0.0, 0.0, 25.0];
        // total 200, col_gap 10 → 2 gaps = 20; fixed + auto = 40 + 25 =
        // 65; the single stretch column takes the remaining 115.
        let w = resolve_cols(&cols, 200.0, 10.0, &stored);
        assert_eq!(w.len(), 3);
        assert_eq!(w[0], 40.0);
        assert_eq!(w[2], 25.0);
        assert!((w[1] - 115.0).abs() < 1e-4, "stretch got {}", w[1]);
    }

    #[test]
    fn resolve_cols_weights_stretch_and_clamps() {
        let cols = [
            Col::stretch(3.0, Align::Left),
            Col::stretch(1.0, Align::Left),
        ];
        // No gaps counted here (col_gap 0), fixed 0 → the 100 px split 3:1.
        let w = resolve_cols(&cols, 100.0, 0.0, &[]);
        assert!((w[0] - 75.0).abs() < 1e-4, "3-weight got {}", w[0]);
        assert!((w[1] - 25.0).abs() < 1e-4, "1-weight got {}", w[1]);
        // When fixed columns already overflow the row, stretch collapses
        // to nothing rather than going negative.
        let over = [Col::px(300.0, Align::Left), Col::stretch(1.0, Align::Left)];
        let w = resolve_cols(&over, 100.0, 0.0, &[]);
        assert_eq!(w[0], 300.0);
        assert_eq!(w[1], 0.0, "no negative stretch");
    }

    #[test]
    fn table_cells_walk_right_then_wrap_and_advance_the_cursor() {
        let mut ui = ui();
        frame(&mut ui, None, false);
        // Root layout: left = -800/2 + 16 = -384, width 240. Two equal
        // stretch columns with the default 8 px gap.
        let cols = [
            Col::stretch(1.0, Align::Left),
            Col::stretch(1.0, Align::Right),
        ];
        let before = ui.layout.last().unwrap().cursor;
        assert!(ui.table_open("t", &cols));
        let row_h = ui.style.row_h;
        let gap = ui.style.row_gap;
        let col_gap = ui.style.col_gap;
        let col_w = (240.0 - col_gap) / 2.0;
        // Cell 0: top-left column, left-aligned.
        let c0 = ui.row(row_h);
        assert_eq!(c0.left, -384.0);
        assert!((c0.w - col_w).abs() < 1e-4);
        assert_eq!(ui.align, Align::Left);
        // Cell 1: to the right of column 0, right-aligned.
        let c1 = ui.row(row_h);
        assert!((c1.left - (-384.0 + col_w + col_gap)).abs() < 1e-4);
        assert_eq!(ui.align, Align::Right);
        // Cell 2: wraps to a fresh row one step below, back at column 0.
        let c2 = ui.row(row_h);
        assert_eq!(c2.left, -384.0);
        assert!(
            (c1.top - c2.top - (row_h + gap)).abs() < 1e-4,
            "a row step separates the wrapped cells"
        );
        ui.table_close("t");
        // Two rows consumed: the cursor sits two steps below where it was.
        let after = ui.layout.last().unwrap().cursor;
        assert!(
            (before - after - 2.0 * (row_h + gap)).abs() < 1e-4,
            "two rows advanced the cursor"
        );
    }

    #[test]
    fn auto_column_grows_to_its_content_next_frame() {
        let mut ui = ui();
        let cols = [Col::auto(Align::Left), Col::stretch(1.0, Align::Left)];
        // Frame 1: the auto column starts at zero and a cell reports a
        // width of 30 for it.
        frame(&mut ui, None, false);
        assert!(ui.table_open("t", &cols));
        let first = ui.row(ui.style.row_h);
        assert_eq!(first.w, 0.0, "unseen auto column is zero-wide");
        ui.note_cell_width(30.0);
        ui.row(ui.style.row_h); // the stretch cell, not measured
        ui.table_close("t");
        // Frame 2: the retained width sizes the auto column; the stretch
        // column yields to it.
        frame(&mut ui, None, false);
        assert!(ui.table_open("t", &cols));
        let grown = ui.row(ui.style.row_h);
        assert!((grown.w - 30.0).abs() < 1e-4, "auto column grew to content");
        ui.table_close("t");
    }

    // A title-bar press resolves its fold toggle from `click_travel`: a
    // release near the press point toggles (a click), one far away is a
    // drag and must leave the fold alone.
    #[test]
    fn a_still_release_reads_as_a_click() {
        let mut ui = ui();
        frame(&mut ui, Some([5.0, 5.0]), true); // press
        frame(&mut ui, Some([5.0, 5.0]), false); // release, unmoved
        assert!(
            ui.click_travel() < FOLD_DRAG_TOL,
            "an unmoved press reads as a click (would fold)"
        );
    }

    #[test]
    fn a_moved_release_reads_as_a_drag() {
        let mut ui = ui();
        frame(&mut ui, Some([0.0, 0.0]), true); // press
        frame(&mut ui, Some([30.0, 0.0]), true); // dragged far, still held
        frame(&mut ui, Some([60.0, 0.0]), false); // release far away
        assert!(
            ui.click_travel() >= FOLD_DRAG_TOL,
            "a moved press reads as a drag (must not fold)"
        );
    }
}
