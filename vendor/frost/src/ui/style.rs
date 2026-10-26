//! The look and metrics of the [`Ui`](crate::ui::Ui): colors, sizes,
//! and gaps â€” every field public, so a game can re-skin the widgets
//! without touching the code that draws them.

use super::Ui;
use crate::Color;

/// The look and metrics of a [`Ui`], all fields public: read [`Default`]
/// for the stock dark theme and overwrite what you want.
#[derive(Clone, Copy, Debug)]
pub struct UiStyle {
    /// The label text color.
    pub text: Color,
    /// How far below a menu line's center its text anchor sits, as a
    /// fraction of `font_size`: the ascent+descent box the text widget
    /// centers on reaches higher above the caps than below the
    /// baseline, so letters centered on it ride high — this nudge
    /// brings them back to the visual middle. The tree view's `lift`,
    /// in the menu's clothes.
    pub menu_lift: f32,
    /// The color of the dimmed value readout on the right of a slider.
    pub text_muted: Color,
    /// The panel body's background.
    pub panel_bg: Color,
    /// The panel title bar's background.
    pub title_bg: Color,
    /// The panel title text color.
    pub title_text: Color,
    /// The resting color of a button and a checkbox box.
    pub widget_bg: Color,
    /// The color of a hovered widget (and the title bar of a hovered panel).
    pub widget_hover: Color,
    /// The color of a widget held under the button.
    pub widget_press: Color,
    /// The accent: a checked box, a slider's filled track, its knob.
    pub accent: Color,
    /// A slider track's unfilled part.
    pub track: Color,
    /// The label and title text size in pixels per em.
    pub font_size: f32,
    /// The label and title text weight: the font's `wght` variation axis,
    /// where 400.0 is Regular and 700.0 a full bold. The default 600.0 is
    /// a SemiBold — clearly crisper than Regular at small sizes on a dark
    /// panel. A font without a weight axis (any static TTF) ignores it.
    pub font_weight: f32,
    /// The height of a widget row, a button and a title bar.
    pub row_h: f32,
    /// The vertical gap between two rows.
    pub row_gap: f32,
    /// The horizontal gap between two table columns.
    pub col_gap: f32,
    /// The padding inside a panel, between body edge and rows.
    pub pad: f32,
    /// The side of a checkbox box.
    pub check_size: f32,
    /// The height of a slider's track.
    pub track_h: f32,
    /// The radius of a slider's knob.
    pub knob_r: f32,
    /// The width of the root column the widgets outside any panel stack in.
    pub root_width: f32,
    /// The root column's inset from the window's top-left corner.
    pub root_margin: f32,
    /// The `z` of the UI's first draw; every later draw counts up from it.
    pub base_z: f32,
}

impl Default for UiStyle {
    fn default() -> Self {
        Self {
            text: Color {
                r: 0.92,
                g: 0.94,
                b: 0.98,
                a: 1.0,
            },
            text_muted: Color {
                r: 0.62,
                g: 0.66,
                b: 0.74,
                a: 1.0,
            },
            panel_bg: Color {
                r: 0.10,
                g: 0.11,
                b: 0.15,
                a: 0.94,
            },
            title_bg: Color {
                r: 0.17,
                g: 0.20,
                b: 0.28,
                a: 1.0,
            },
            title_text: Color {
                r: 0.95,
                g: 0.96,
                b: 1.0,
                a: 1.0,
            },
            widget_bg: Color {
                r: 0.21,
                g: 0.24,
                b: 0.32,
                a: 1.0,
            },
            widget_hover: Color {
                r: 0.30,
                g: 0.35,
                b: 0.47,
                a: 1.0,
            },
            widget_press: Color {
                r: 0.15,
                g: 0.17,
                b: 0.23,
                a: 1.0,
            },
            accent: Color {
                r: 0.25,
                g: 0.45,
                b: 0.85,
                a: 1.0,
            },
            track: Color {
                r: 0.14,
                g: 0.15,
                b: 0.20,
                a: 1.0,
            },
            font_size: 15.0,
            menu_lift: 0.344,
            font_weight: 600.0,
            row_h: 28.0,
            row_gap: 6.0,
            col_gap: 8.0,
            pad: 12.0,
            check_size: 20.0,
            track_h: 6.0,
            knob_r: 9.0,
            root_width: 240.0,
            root_margin: 16.0,
            base_z: 10_000.0,
        }
    }
}

impl Ui {
    /// The style, for tweaking: the colors and metrics are plain fields.
    pub fn style_mut(&mut self) -> &mut UiStyle {
        &mut self.style
    }
}
