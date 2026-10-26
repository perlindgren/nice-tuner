//! The [`Ui`](crate::ui::Ui) panel: a draggable, foldable titled
//! plate, plus the id domain that keeps panel state out of the
//! widget id space.

use super::{FNV_OFFSET, FNV_PRIME, FOLD_DRAG_TOL, Layout, Rect, Ui};
use crate::Context;

/// What a panel keeps between frames: where it was dragged to, how much
/// room its content asked for (the body size the next frame draws), and
/// whether the user has folded it to just its title bar.
#[derive(Clone, Copy, Debug)]
pub(super) struct PanelState {
    /// The panel's center in user space.
    pub(super) pos: [f32; 2],
    /// The panel's width as its `panel` call declared it.
    pub(super) w: f32,
    /// The panel's height: title bar plus the content height it recorded
    /// last frame (the first frame shows the title bar only).
    pub(super) h: f32,
    /// Whether a title-bar click has folded the body away.
    pub(super) folded: bool,
    /// The plate actually painted by the last `panel` call, as
    /// `[left, bottom, right, top]` — the geometry a caller can trust,
    /// since the plate draws at last frame's measured height and a
    /// same-frame request may not be it.
    pub(super) rect: [f32; 4],
}

impl Ui {
    /// The panel's current center, `[x, y]` in user space, once the panel
    /// has been declared at least once.
    pub fn panel_position(&self, title: &str) -> Option<[f32; 2]> {
        self.panels.get(&panel_id(title)).map(|p| p.pos)
    }

    /// The plate a panel PAINTED last frame, `[left, bottom, right, top]`
    /// in user space. The height is the one the drawing used — last
    /// frame's measured content, not this frame's request — so a caller
    /// laying content over the plate matches the pixels, snap-open
    /// frames included. `None` for a title never declared.
    pub fn panel_rect(&self, title: &str) -> Option<[f32; 4]> {
        self.panels.get(&panel_id(title)).map(|p| p.rect)
    }

    /// Fold a panel, or open it again: the program's hand on the
    /// title bar's click, for code that closes or restores a panel
    /// outside the UI — a no-op for a title not yet declared.
    pub fn set_folded(&mut self, title: &str, folded: bool) {
        if let Some(p) = self.panels.get_mut(&panel_id(title)) {
            p.folded = folded;
        }
    }

    /// A panel: a titled, draggable box at center `at` (user-space pixels,
    /// until the user drags it somewhere else) of the given `width`, running
    /// `content` with its rows laid out in the body.
    ///
    /// The body's height is what the content asked for *last* frame, so the
    /// first frame shows the title bar only and the body snaps open on the
    /// second; a height change keeps the title bar where it is and grows
    /// (or shrinks) the body downward. Dragging the title bar moves the
    /// panel — its position then survives every frame and `at` is ignored
    /// (see [`Ui::panel_position`] to read it back) — while a *click* on
    /// the title bar (press and release without moving) folds the panel to
    /// just its title bar, and another click unfolds it.
    ///
    /// The panel claims the space it covers: the body absorbs presses and
    /// counts as [`Ui::hovering`], so a game reading the mouse can tell a
    /// click on the panel from one on its scene, and widgets behind the
    /// panel are never hit through it.
    pub fn panel(
        &mut self,
        ctx: &mut Context,
        title: &str,
        at: [f32; 2],
        width: f32,
        content: impl FnOnce(&mut Ui, &mut Context),
    ) {
        let id = panel_id(title);
        let style = self.style;
        let title_h = style.row_h;
        let pad = style.pad;
        let state = self
            .panels
            .entry(id)
            .and_modify(|p| p.w = width)
            .or_insert(PanelState {
                pos: at,
                w: width,
                h: title_h,
                folded: false,
                rect: [
                    at[0] - width / 2.0,
                    at[1] - title_h / 2.0,
                    at[0] + width / 2.0,
                    at[1] + title_h / 2.0,
                ],
            });
        let w = state.w;
        let pos = state.pos;
        let prev_h = state.h;
        let prev_folded = state.folded;
        // The geometry the panel presented last frame, for the hit test.
        let hit_h = if prev_folded {
            title_h
        } else {
            prev_h.max(title_h)
        };
        let hit_body = Rect::from_center(pos[0], pos[1], w, hit_h);
        let title_bar = Rect {
            left: hit_body.left,
            top: hit_body.top,
            w,
            h: title_h,
        };
        // The body claims the panel's whole area first, so the title bar —
        // registered after it — wins the overlap, and neither a widget nor
        // the game behind this panel can be hit through it.
        self.interact(panel_body_id(title), hit_body);
        // The title bar is a widget too: it claims the press for the drag.
        let it = self.interact(id, title_bar);
        // A release is a click only while the pointer barely moved —
        // dragging the bar to move the panel must not fold it.
        let folded = if it.clicked && self.click_travel() < FOLD_DRAG_TOL {
            !prev_folded
        } else {
            prev_folded
        };
        self.panels
            .get_mut(&id)
            .expect("entry was made above")
            .folded = folded;
        // Draw from the title bar down: its top edge is anchored, so the
        // body opens and closes downward and the bar never jumps. A folded
        // panel is just the bar.
        let draw_h = if folded { title_h } else { prev_h.max(title_h) };
        let anchor_top = title_bar.top;
        {
            let state = self.panels.get_mut(&id).expect("entry was made above");
            state.pos[1] = anchor_top - draw_h / 2.0;
        }
        let body = Rect::from_center(pos[0], anchor_top - draw_h / 2.0, w, draw_h);
        self.fill(ctx, body, style.panel_bg);
        self.panels.get_mut(&id).expect("entry was made above").rect =
            [body.left, body.top - body.h, body.left + body.w, body.top];
        let bar = if it.held {
            style.widget_press
        } else if it.hot {
            style.widget_hover
        } else {
            style.title_bg
        };
        self.fill(ctx, title_bar, bar);
        // The fold chevron: pointing down when open, right when folded.
        let z = self.z();
        let [cx, cy] = [title_bar.left + pad + 2.0, title_bar.center()[1]];
        let s = 4.5;
        let title_color = style.title_text;
        if folded {
            ctx.line(cx - s / 2.0, cy + s, cx + s / 2.0, cy, title_color, 2.0, z);
            ctx.line(
                cx + s / 2.0,
                cy,
                cx - s / 2.0,
                cy - s,
                title_color,
                2.0,
                z + 0.5,
            );
        } else {
            ctx.line(cx - s, cy + s / 2.0, cx, cy - s / 2.0, title_color, 2.0, z);
            ctx.line(
                cx,
                cy - s / 2.0,
                cx + s,
                cy + s / 2.0,
                title_color,
                2.0,
                z + 0.5,
            );
        }
        let size = style.font_size;
        let [tcx, tcy] = title_bar.center();
        let title_string = title.to_owned();
        self.text(ctx, [tcx, tcy], size, title_color, &title_string);
        if folded {
            // Nothing below the bar: the body stays closed.
            let state = self.panels.get_mut(&id).expect("entry was made above");
            state.h = title_h;
            return;
        }
        // The content layout, and the height it uses.
        let top0 = anchor_top - title_h - pad;
        self.layout.push(Layout {
            left: body.left + pad,
            width: (w - 2.0 * pad).max(0.0),
            cursor: top0,
            scope: id,
            table: None,
        });
        content(self, ctx);
        let used = top0 - self.layout.pop().expect("just pushed").cursor;
        let new_h = (title_h + pad * 2.0 + used).max(title_h);
        let state = self.panels.get_mut(&id).expect("entry was made above");
        state.h = new_h;
        // Grow (or shrink) downward from the anchored title bar.
        state.pos[1] = anchor_top - new_h / 2.0;
    }
}
/// The id of a panel and of its draggable title bar, hashed from the title.
pub(super) fn panel_id(title: &str) -> u64 {
    let mut h = FNV_OFFSET;
    for b in b"panel".iter().chain(title.as_bytes()) {
        h = (h ^ *b as u64).wrapping_mul(FNV_PRIME);
    }
    h
}

/// The id of a panel's non-interactive body claim: its own domain, so a
/// press on the panel background captures nothing that the drag loop
/// mistakes for a panel move.
pub(super) fn panel_body_id(title: &str) -> u64 {
    let mut h = FNV_OFFSET;
    for b in b"panel-body".iter().chain(title.as_bytes()) {
        h = (h ^ *b as u64).wrapping_mul(FNV_PRIME);
    }
    h
}
