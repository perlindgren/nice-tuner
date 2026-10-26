use super::*;
use crate::ui::tests::{frame, ui};

/// The shared RON-metrics style the test rows and specs are built
/// against.
static STYLE: std::sync::LazyLock<TreeStyle> = std::sync::LazyLock::new(TreeStyle::default);

/// A row of the test view, with no buttons.
fn line<'a>(key: &'a str, val: &'a str) -> TreeLine<'a> {
    TreeLine {
        key,
        val,
        head: false,
        val_color: Color {
            r: 0.6,
            g: 0.78,
            b: 0.96,
            a: 1.0,
        },
        add: false,
        del: false,
        band: false,
    }
}

static ROWS: [TreeLine<'static>; 10] = [
    TreeLine {
        key: "alpha",
        val: "1",
        head: false,
        val_color: Color {
            r: 0.6,
            g: 0.78,
            b: 0.96,
            a: 1.0,
        },
        add: false,
        del: false,
        band: false,
    },
    TreeLine {
        key: "beta",
        val: "2",
        head: false,
        val_color: Color {
            r: 0.6,
            g: 0.78,
            b: 0.96,
            a: 1.0,
        },
        add: false,
        del: false,
        band: false,
    },
    TreeLine {
        key: "gamma",
        val: "3",
        head: true,
        val_color: Color {
            r: 0.93,
            g: 0.78,
            b: 0.44,
            a: 1.0,
        },
        add: true,
        del: false,
        band: false,
    },
    TreeLine {
        key: "delta",
        val: "4",
        head: false,
        val_color: Color {
            r: 0.6,
            g: 0.78,
            b: 0.96,
            a: 1.0,
        },
        add: true,
        del: true,
        band: false,
    },
    TreeLine {
        key: "epsilon",
        val: "5",
        head: false,
        val_color: Color {
            r: 0.6,
            g: 0.78,
            b: 0.96,
            a: 1.0,
        },
        add: false,
        del: true,
        band: false,
    },
    TreeLine {
        key: "zeta",
        val: "6",
        head: false,
        val_color: Color {
            r: 0.6,
            g: 0.78,
            b: 0.96,
            a: 1.0,
        },
        add: false,
        del: false,
        band: false,
    },
    TreeLine {
        key: "eta",
        val: "7",
        head: false,
        val_color: Color {
            r: 0.6,
            g: 0.78,
            b: 0.96,
            a: 1.0,
        },
        add: false,
        del: false,
        band: false,
    },
    TreeLine {
        key: "theta",
        val: "8",
        head: false,
        val_color: Color {
            r: 0.6,
            g: 0.78,
            b: 0.96,
            a: 1.0,
        },
        add: false,
        del: false,
        band: false,
    },
    TreeLine {
        key: "iota",
        val: "9",
        head: false,
        val_color: Color {
            r: 0.6,
            g: 0.78,
            b: 0.96,
            a: 1.0,
        },
        add: false,
        del: false,
        band: false,
    },
    TreeLine {
        key: "kappa",
        val: "10",
        head: false,
        val_color: Color {
            r: 0.6,
            g: 0.78,
            b: 0.96,
            a: 1.0,
        },
        add: false,
        del: false,
        band: false,
    },
];

/// A one-view spec: body `[0, 0, 300, 200]`, a 40 px title bar above
/// it, the close × in the title bar's right.
fn spec(id: u64) -> TreeSpec<'static> {
    let body = [0.0, 0.0, 300.0, 200.0];
    TreeSpec {
        id,
        rows: &ROWS,
        panel: [
            body[0] - 12.0,
            body[1] - 40.0,
            body[2] + 12.0,
            body[3] + 40.0,
        ],
        body: Some(body),
        close: [body[2] - 17.0, body[3] + 28.0],
        folded: false,
        scroll: 0.0,
        sx: 0.0,
        size: [body[2] - body[0], body[3] - body[1]],
        style: *STYLE,
        z: 15_000.0,
    }
}

#[test]
fn chars_for_keeps_the_character_budget() {
    // 300 px wide fits 28 characters; 20 px wide falls to the minimum.
    assert_eq!(chars_for(300.0, *STYLE), 28);
    assert_eq!(chars_for(20.0, *STYLE), 8);
    assert_eq!(chars_for(640.0, *STYLE), 64);
}

#[test]
fn width_of_counts_key_and_value_together() {
    assert_eq!(width_of(&ROWS), 8); // "epsilon" + "5"
    assert_eq!(width_of(&[line("k", "v")]), 2);
    assert_eq!(width_of(&[]), 0);
}

#[test]
fn thumb_is_absent_when_the_content_fits() {
    assert_eq!(thumb(5.0, 100.0, 80.0, 200.0), (0.0, 200.0));
    assert_eq!(thumb(5.0, 100.0, 100.0, 200.0), (0.0, 200.0));
    assert_eq!(thumb(5.0, 100.0, 200.0, 0.0), (0.0, 0.0));
}

#[test]
fn thumb_tracks_the_scroll_and_keeps_a_minimum() {
    // Half the content visible: a half-length thumb at the top for
    // scroll 0, at the end for max scroll.
    assert_eq!(thumb(0.0, 100.0, 200.0, 100.0), (0.0, 50.0));
    assert_eq!(thumb(100.0, 100.0, 200.0, 100.0), (50.0, 50.0));
    // A tiny view still gets a thumb it can grab.
    assert_eq!(thumb(0.0, 10.0, 100.0, 100.0), (0.0, 24.0));
    // The scroll clamps into the thumb's range.
    assert_eq!(thumb(1000.0, 100.0, 200.0, 100.0), (50.0, 50.0));
}

#[test]
fn row_at_maps_points_to_rows_and_guarding() {
    let body = [0.0, 0.0, 300.0, 200.0];
    // Row centers count down from the top padding: row i's center is
    // 200 - 6 - (i + 0.5) * 20.
    for i in 0..ROWS.len() {
        assert_eq!(
            row_at(
                [150.0, row_y(i, body, 0.0, *STYLE)],
                body,
                0.0,
                ROWS.len(),
                *STYLE
            ),
            Some(i)
        );
    }
    // Outside the body: nothing.
    assert_eq!(row_at([150.0, 201.0], body, 0.0, ROWS.len(), *STYLE), None);
    assert_eq!(
        row_at(
            [-1.0, row_y(0, body, 0.0, *STYLE)],
            body,
            0.0,
            ROWS.len(),
            *STYLE
        ),
        None
    );
    // Inside the body but past the last row: nothing.
    assert_eq!(row_at([150.0, 1.0], body, 0.0, 3, *STYLE), None);
}

#[test]
fn scrolling_lifts_rows_toward_the_top() {
    let body = [0.0, 0.0, 300.0, 200.0];
    // Two rows of scroll (2 x 20 px): row 2 arrives at exactly where
    // row 0 sat unscrolled — the document walks up, the window stays.
    assert!(
        row_y(2, body, 40.0, *STYLE) > row_y(2, body, 0.0, *STYLE),
        "rows climb, not sink"
    );
    assert!(
        (row_y(2, body, 40.0, *STYLE) - row_y(0, body, 0.0, *STYLE)).abs() < 1e-3,
        "one row of scroll is exactly one row of climb"
    );
}

#[test]
fn row_at_follows_the_scroll() {
    let body = [0.0, 0.0, 300.0, 200.0];
    // Scrolled down 40 px, the point at row 0's unscrolled center is
    // over row 2 now.
    let p = [150.0, row_y(0, body, 0.0, *STYLE)];
    assert_eq!(row_at(p, body, 40.0, ROWS.len(), *STYLE), Some(2));
}

#[test]
fn resize_size_clamps_to_the_style_limits() {
    // Drag far right and far up: both hit their maximums.
    assert_eq!(
        resize_size(300.0, 200.0, 0.0, 0.0, [1000.0, -1000.0], *STYLE),
        [640.0, 560.0]
    );
    // Drag far left and far down: both hit their minimums.
    assert_eq!(
        resize_size(300.0, 200.0, 0.0, 0.0, [-1000.0, 1000.0], *STYLE),
        [150.0, 80.0]
    );
    // A small drag moves the size as-is: the width grows from the
    // center (twice the pointer's rightward travel), the height with
    // the pointer's downward travel — away from the bottom grip.
    assert_eq!(
        resize_size(300.0, 200.0, 0.0, 0.0, [10.0, 5.0], *STYLE),
        [320.0, 195.0]
    );
}

#[test]
fn row_btn_boxes_place_add_and_del() {
    let right = 300.0;
    assert_eq!(row_btn_boxes(false, false, right), (None, None));
    assert_eq!(row_btn_boxes(true, false, right), (Some(280.0), None));
    assert_eq!(row_btn_boxes(false, true, right), (None, Some(280.0)));
    assert_eq!(row_btn_boxes(true, true, right), (Some(280.0), Some(258.0)));
}

#[test]
fn window_slices_and_centers_each_side() {
    // Unscrolled, both sides fully visible, centered in their halves.
    let (k, v) = window("alpha", "beta", 0, 10, 9.6);
    assert_eq!(k, Some(("alpha".into(), 24.0)));
    let (v_text, v_x) = v.expect("the value is visible");
    assert_eq!(v_text, "beta");
    assert!((v_x - 67.2).abs() < 1e-3);
    // Scrolled: the key is clipped, the value scrolled out of view.
    let (k, v) = window("abcdefgh", "", 3, 4, 9.6);
    assert_eq!(k, Some(("defg".into(), 19.2)));
    assert_eq!(v, None);
}

#[test]
fn v_thumb_rect_tracks_the_vertical_scroll() {
    let body = [0.0, 0.0, 300.0, 200.0];
    // 10 rows at 20 px plus padding: 206 of content in 200 of body.
    let total = ROWS.len() as f32 * STYLE.row_h + STYLE.pad;
    let r = v_thumb(body, 0.0, total).expect("the content overflows");
    // The thumb hugs the body's right edge, from the track's top.
    assert!((r[0] - 292.0).abs() < f32::EPSILON);
    assert!((r[2] - 298.0).abs() < f32::EPSILON);
    assert!((r[3] - 200.0).abs() < f32::EPSILON);
    let r = v_thumb(body, 6.0, total).expect("still overflowing");
    // At max scroll the thumb hugs the track's bottom: its bottom
    // edge at the body's bottom, its top just above.
    assert!((r[1] - 0.0).abs() < 1e-4);
    assert!((r[3] - 194.17).abs() < 0.01);
    assert!(v_thumb(body, 0.0, 100.0).is_none());
}

#[test]
fn h_thumb_needs_overflow_and_width() {
    let body = [0.0, 0.0, 300.0, 200.0];
    // 8 characters wide fits a 300 px body: no thumb.
    assert!(h_thumb(body, 0.0, width_of(&ROWS), *STYLE).is_none());
    // 60 characters does not: the thumb sits on the body's bottom.
    let r = h_thumb(body, 0.0, 60, *STYLE).expect("the content overflows");
    assert!((r[1] - 2.0).abs() < f32::EPSILON);
    assert!((r[3] - 8.0).abs() < f32::EPSILON);
    assert!((r[0] - 0.0).abs() < f32::EPSILON);
    // A zero-width body never thumbs.
    assert!(h_thumb([0.0, 0.0, 0.0, 200.0], 0.0, 60, *STYLE).is_none());
}

/// Runs a press at `p`, holds through `move_to`, and releases at
/// `release`, returning the outs of every frame.
fn click(ui: &mut Ui, specs: &[TreeSpec<'_>], state: &mut TreeState, p: [f32; 2]) -> Vec<TreeOut> {
    frame(ui, Some(p), true);
    let _ = ui.tree(specs, state);
    frame(ui, Some(p), true);
    let _ = ui.tree(specs, state);
    frame(ui, Some(p), false);
    ui.tree(specs, state)
}

#[test]
fn a_still_click_emits_row() {
    let mut ui = ui();
    let mut state = TreeState::default();
    let s = spec(0);
    let p = [150.0, row_y(2, s.body.unwrap(), 0.0, *STYLE)];
    let outs = click(&mut ui, &[s], &mut state, p);
    assert_eq!(outs[0].events, vec![TreeEvent::Row { row: 2 }]);
}

#[test]
fn a_click_near_a_button_emits_that_button() {
    let mut ui = ui();
    let mut state = TreeState::default();
    let s = spec(0);
    let body = s.body.unwrap();
    // Row 3 shows both buttons; the + sits at 280, the × at 258.
    let y = row_y(3, body, 0.0, *STYLE);
    let outs = click(&mut ui, &[s], &mut state, [280.0, y]);
    assert_eq!(outs[0].events, vec![TreeEvent::Add { row: 3 }]);
    let outs = click(&mut ui, &[s], &mut state, [258.0, y]);
    assert_eq!(outs[0].events, vec![TreeEvent::Del { row: 3 }]);
}

#[test]
fn the_close_box_wins_over_an_earlier_body() {
    let mut ui = ui();
    let mut state = TreeState::default();
    // View 0's body spans the origin; view 1's close × sits inside
    // it. A release at that point closes view 1, and nothing else.
    let s0 = spec(0);
    let mut s1 = spec(1);
    s1.close = [150.0, 100.0];
    s1.body = Some([400.0, 0.0, 700.0, 200.0]);
    let p = [150.0, 100.0];
    let outs = click(&mut ui, &[s0, s1], &mut state, p);
    assert!(outs[0].events.is_empty(), "the body must not win");
    assert_eq!(outs[1].events, vec![TreeEvent::Close]);
}

#[test]
fn a_drifted_press_is_not_a_click_and_leaves_no_stale_press() {
    let mut ui = ui();
    let mut state = TreeState::default();
    let s = spec(0);
    let body = s.body.unwrap();
    let y = row_y(1, body, 0.0, *STYLE);
    // Press a non-draggable row, drift 5 px — past the tolerance —
    // and release: nothing.
    frame(&mut ui, Some([150.0, y]), true);
    let outs = ui.tree(&[s], &mut state);
    assert!(outs[0].events.is_empty());
    frame(&mut ui, Some([155.0, y]), true);
    let outs = ui.tree(&[s], &mut state);
    assert!(outs[0].events.is_empty(), "row 1 cannot start a drag");
    frame(&mut ui, Some([155.0, y]), false);
    let outs = ui.tree(&[s], &mut state);
    assert!(
        outs[0].events.is_empty(),
        "a drifted release is not a click"
    );
    // The press was consumed by that release: a fresh click works.
    let y4 = row_y(4, body, 0.0, *STYLE);
    let outs = click(&mut ui, &[s], &mut state, [150.0, y4]);
    assert_eq!(outs[0].events, vec![TreeEvent::Row { row: 4 }]);
}

#[test]
fn a_drag_reorders_a_del_row_and_reports_its_seat() {
    let mut ui = ui();
    let mut state = TreeState::default();
    let s = spec(0);
    let body = s.body.unwrap();
    let y0 = row_y(3, body, 0.0, *STYLE);
    let y2 = row_y(2, body, 0.0, *STYLE);
    let y1 = row_y(1, body, 0.0, *STYLE);
    // Press on the draggable row 3.
    frame(&mut ui, Some([150.0, y0]), true);
    let outs = ui.tree(&[s], &mut state);
    assert!(outs[0].events.is_empty());
    // Drift 6 px: the drag starts, and the move phase reports the
    // current seat — the row still under the pointer (the app's
    // seq_move dedupes a same-seat move).
    frame(&mut ui, Some([156.0, y0]), true);
    let outs = ui.tree(&[s], &mut state);
    assert_eq!(
        outs[0].events,
        vec![
            TreeEvent::DragStart { row: 3 },
            TreeEvent::Drag { row: Some(3) }
        ]
    );
    // Over row 2: the current seat.
    frame(&mut ui, Some([150.0, y2]), true);
    let outs = ui.tree(&[s], &mut state);
    assert_eq!(outs[0].events, vec![TreeEvent::Drag { row: Some(2) }]);
    // Release over row 1: the final seat, then the end.
    frame(&mut ui, Some([150.0, y1]), false);
    let outs = ui.tree(&[s], &mut state);
    assert_eq!(
        outs[0].events,
        vec![TreeEvent::Drag { row: Some(1) }, TreeEvent::DragEnd]
    );
}

#[test]
fn a_press_on_a_non_del_row_never_drags_and_still_clicks() {
    let mut ui = ui();
    let mut state = TreeState::default();
    let s = spec(0);
    let body = s.body.unwrap();
    let y0 = row_y(0, body, 0.0, *STYLE);
    // Press row 0 — no del flag — drift 6 px, then release still.
    frame(&mut ui, Some([150.0, y0]), true);
    let _ = ui.tree(&[s], &mut state);
    frame(&mut ui, Some([156.0, y0]), true);
    let outs = ui.tree(&[s], &mut state);
    assert!(outs[0].events.is_empty(), "row 0 cannot start a drag");
    frame(&mut ui, Some([150.0, y0]), false);
    let outs = ui.tree(&[s], &mut state);
    assert_eq!(
        outs[0].events,
        vec![TreeEvent::Row { row: 0 }],
        "the press survived the failed drag start"
    );
}

#[test]
fn a_folded_view_claims_only_its_close_box() {
    let mut ui = ui();
    let mut state = TreeState::default();
    let mut s = spec(0);
    s.folded = true;
    let body = s.body.unwrap();
    // A press in the body claims nothing: the view is folded.
    let p = [150.0, row_y(2, body, 0.0, *STYLE)];
    let outs = click(&mut ui, &[s], &mut state, p);
    assert!(outs[0].events.is_empty());
    // A press on the close × closes the view.
    let outs = click(&mut ui, &[s], &mut state, s.close);
    assert_eq!(outs[0].events, vec![TreeEvent::Close]);
}

#[test]
fn a_grabbed_thumb_drives_the_scroll_and_clamps() {
    let mut ui = ui();
    let mut state = TreeState::default();
    let s = spec(0);
    let body = s.body.unwrap();
    let total = ROWS.len() as f32 * STYLE.row_h + STYLE.pad;
    let r = v_thumb(body, 0.0, total).expect("the content overflows");
    // Press inside the thumb, near its top.
    let p0 = [295.0, r[3] - 2.0];
    frame(&mut ui, Some(p0), true);
    let outs = ui.tree(&[s], &mut state);
    assert!(outs[0].events.is_empty());
    assert!((outs[0].scroll - 0.0).abs() < f32::EPSILON);
    // Drag the thumb down: the scroll follows and clamps at the end.
    frame(&mut ui, Some([295.0, r[1] - 50.0]), true);
    let outs = ui.tree(&[s], &mut state);
    let max_scroll = total - (body[3] - body[1]);
    assert!(
        (outs[0].scroll - max_scroll).abs() < 1e-4,
        "clamped at the end: {}",
        outs[0].scroll
    );
}

#[test]
fn a_grabbed_corner_drives_the_size() {
    let mut ui = ui();
    let mut state = TreeState::default();
    let s = spec(0);
    // The corner box is the panel's bottom-right corner.
    let p0 = corner_box(s.panel);
    let press = [(p0[0] + p0[2]) / 2.0, (p0[1] + p0[3]) / 2.0];
    frame(&mut ui, Some(press), true);
    let outs = ui.tree(&[s], &mut state);
    assert!(outs[0].events.is_empty());
    // Drag right and down: the view grows, clamped at its limits.
    // The height grows as the pointer moves down, away from the
    // bottom grip.
    frame(&mut ui, Some([press[0] + 400.0, press[1] - 360.0]), true);
    let outs = ui.tree(&[s], &mut state);
    assert_eq!(outs[0].size, [640.0, 560.0]);
    // Drag far left and up: the view shrinks, clamped at its
    // minimums.
    frame(&mut ui, Some([press[0] - 1000.0, press[1] + 1000.0]), true);
    let outs = ui.tree(&[s], &mut state);
    assert_eq!(outs[0].size, [150.0, 80.0]);
}
