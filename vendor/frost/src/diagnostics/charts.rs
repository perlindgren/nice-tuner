//! The chart data model: the rolling window of [`Sample`]s and the
//! pure reads that turn it into chart columns — trimming, the per-column
//! maximum, and the series accessors the panels draw from.

use super::*;

/// The charts' slots, in append order: the enabled statistics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ChartSlot {
    /// The frame-rate chart.
    Fps,
    /// The frame-time chart.
    Ft,
    /// The folded processing-time chart: total, app, and diagnostic.
    Proc,
    /// The folded draw-call chart: total, app, and diagnostic.
    Draw,
}

/// A chart's polyline points, in window user space: one per 0.1 s column,
/// at the column's center across a panel `graph_w` wide, rising from one
/// inset above the bottom edge of a panel `graph_h` tall that starts at
/// `panel_top`, the value scaled to the panel height minus the two
/// insets — the stroke the chart's [`Shape::Polyline`] draws in one draw
/// call. `inset` is the display-scaled pixel inset.
pub(super) fn chart_points(
    cols: &[f32; COLUMNS],
    x0: f32,
    panel_top: f32,
    top_value: f32,
    graph_w: f32,
    graph_h: f32,
    inset: f32,
) -> Vec<[f32; 2]> {
    let pitch = graph_w / COLUMNS as f32;
    cols.iter()
        .enumerate()
        .map(|(c, &v)| {
            [
                x0 + (c as f32 + 0.5) * pitch,
                panel_top - graph_h + inset + (v / top_value).min(1.0) * (graph_h - 2.0 * inset),
            ]
        })
        .collect()
}

/// One sample in the graphs' history: the metrics of one completed frame,
/// recorded when that frame's `process` ran — so, by construction, one
/// frame behind the frame being produced.
#[derive(Debug, Clone, Copy)]
pub(super) struct Sample {
    /// The time axis: elapsed seconds of real time (stalls included).
    pub(super) t: f32,
    /// The frame time in ms — the raw `dt`.
    pub(super) frame_ms: f32,
    /// The frame's total processing time in ms — the engine's probe.
    pub(super) proc_ms: f32,
    /// The overlay's own update cost in ms for this frame — the
    /// one-frame-behind `prev_self_ms`, the diagnostic series of the
    /// folded processing-time chart.
    pub(super) diag_ms: f32,
    /// The frame's processing time in ms with the overlay's own update
    /// cost removed.
    pub(super) app_ms: f32,
    /// The frame's GPU draw-call count — the engine's probe.
    pub(super) draw_calls: f32,
    /// The overlay's own draw calls in this frame — the engine's
    /// `Context::frame_diagnostic_draw_calls` probe, the diagnostic series
    /// of the folded draw-call chart.
    pub(super) diag_draws: f32,
    /// The frame's draw calls minus the overlay's own — the app series of
    /// the folded draw-call chart.
    pub(super) app_draws: f32,
}

/// Trims the history to the last SPAN seconds of real time ending at `now`.
pub(super) fn trim_samples(samples: &mut VecDeque<Sample>, now: f32) {
    while let Some(sample) = samples.front() {
        if sample.t < now - SPAN {
            samples.pop_front();
        } else {
            break;
        }
    }
}

/// The maximum of `value(sample)` over the samples falling in each of the
/// COLUMNS time slices of the last SPAN seconds ending at `now`, index 0 =
/// the oldest slice; a sample at exactly `now` lands in the newest slice.
pub(super) fn slice_max(
    samples: impl IntoIterator<Item = Sample>,
    now: f32,
    mut value: impl FnMut(&Sample) -> f32,
) -> [f32; COLUMNS] {
    let mut cols = [0.0f32; COLUMNS];
    let start = now - SPAN;
    for s in samples {
        let rel = (s.t - start) / SPAN;
        // rel > 1.0 is beyond the window; the epsilon absorbs the f32
        // rounding of `now - (now - SPAN)`, which can land just past SPAN.
        if rel <= 0.0 || rel > 1.0 + 1e-6 {
            continue;
        }
        let rel = rel.min(1.0);
        let c = (rel * COLUMNS as f32) as usize;
        let c = c.min(COLUMNS - 1);
        let v = value(&s);
        if v > cols[c] {
            cols[c] = v;
        }
    }
    cols
}

/// The number of series a chart shows: one for the single-value charts,
/// three — total, app, diagnostic — for the folded ones.
pub(super) fn series_count(chart: ChartSlot) -> usize {
    match chart {
        ChartSlot::Fps | ChartSlot::Ft => 1,
        ChartSlot::Proc | ChartSlot::Draw => 3,
    }
}

/// The value, in the chart's units, of `chart`'s `k`-th series for the
/// sample.
pub(super) fn series_value(chart: ChartSlot, k: usize, s: &Sample) -> f32 {
    match (chart, k) {
        (ChartSlot::Fps, _) => 1000.0 / s.frame_ms,
        (ChartSlot::Ft, _) => s.frame_ms,
        (ChartSlot::Proc, 0) => s.proc_ms,
        (ChartSlot::Proc, 1) => s.app_ms,
        (ChartSlot::Proc, 2) => s.diag_ms,
        (ChartSlot::Draw, 0) => s.draw_calls,
        (ChartSlot::Draw, 1) => s.app_draws,
        (ChartSlot::Draw, 2) => s.diag_draws,
        _ => unreachable!("k out of range for {chart:?}"),
    }
}

/// The color of `chart`'s `k`-th series.
pub(super) fn series_color(chart: ChartSlot, k: usize) -> Color {
    match (chart, k) {
        (ChartSlot::Fps, _) => FPS_COLOR,
        (ChartSlot::Ft, _) => FT_COLOR,
        (ChartSlot::Proc, 0) => PROC_COLOR,
        (ChartSlot::Proc, 1) => APP_COLOR,
        (ChartSlot::Proc, 2) => DIAG_PROC_COLOR,
        (ChartSlot::Draw, 0) => DRAW_COLOR,
        (ChartSlot::Draw, 1) => APP_DRAW_COLOR,
        (ChartSlot::Draw, 2) => DIAG_DRAW_COLOR,
        _ => unreachable!("k out of range for {chart:?}"),
    }
}

/// The top of `chart`'s scale, in the chart's units: the value that reaches
/// the panel's top edge.
pub(super) fn chart_top(chart: ChartSlot) -> f32 {
    match chart {
        ChartSlot::Fps => FPS_TOP,
        ChartSlot::Ft => FT_TOP,
        ChartSlot::Proc => PROC_TOP,
        ChartSlot::Draw => DRAW_TOP,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A test sample with the given metrics.
    #[allow(clippy::too_many_arguments)]
    fn sample(
        t: f32,
        frame_ms: f32,
        proc_ms: f32,
        diag_ms: f32,
        app_ms: f32,
        draw_calls: f32,
        diag_draws: f32,
        app_draws: f32,
    ) -> Sample {
        Sample {
            t,
            frame_ms,
            proc_ms,
            diag_ms,
            app_ms,
            draw_calls,
            diag_draws,
            app_draws,
        }
    }

    #[test]
    fn trim_keeps_only_the_last_span_seconds() {
        let mut samples = VecDeque::new();
        for t in 0..11 {
            samples.push_back(sample(t as f32, 16.0, 1.0, 0.25, 0.75, 4.0, 1.0, 3.0));
        }
        trim_samples(&mut samples, 11.0);
        // t < 11 - 10 = 1 is gone; t = 1..=11 remain.
        assert_eq!(samples.front().unwrap().t, 1.0);
        assert_eq!(samples.len(), 10);
    }

    #[test]
    fn slice_max_buckets_by_time_not_by_count() {
        // now = 10: the window is t = 0..10, 100 slices of 0.1 s.
        let samples = [
            sample(0.05, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), // oldest slice (c = 0)
            sample(5.00, 200.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), // middle of the window (c = 50)
            sample(9.95, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0), // newest slice (c = 99)
        ];
        let cols = slice_max(samples, 10.0, |s| s.frame_ms);
        assert_eq!(cols[0], 100.0);
        assert_eq!(cols[50], 200.0);
        assert_eq!(cols[99], 100.0);
        assert_eq!(cols[49], 0.0);
        assert_eq!(cols[98], 0.0);
    }

    #[test]
    fn slice_max_counts_the_newest_sample_at_now() {
        // t == now: the sample belongs to the newest slice, not beyond it.
        let samples = [sample(10.0, 300.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)];
        let cols = slice_max(samples, 10.0, |s| s.frame_ms);
        assert_eq!(cols[99], 300.0);
    }

    #[test]
    fn slice_max_ignores_samples_outside_the_window() {
        let samples = [
            sample(15.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            sample(20.0, 100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ];
        let cols = slice_max(samples, 10.0, |s| s.frame_ms);
        assert!(cols.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn slice_max_fps_is_the_inverse_of_the_frame_time() {
        // Two frames in one slice: 10 ms and 40 ms → the slice's best rate
        // is 100 fps.
        let samples = [
            sample(9.95, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            sample(9.98, 40.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        ];
        let cols = slice_max(samples, 10.0, |s| 1000.0 / s.frame_ms);
        assert_eq!(cols[99], 100.0);
    }

    #[test]
    fn slice_max_proc_ignores_the_frame_time() {
        // Two frames in one slice: 10 ms and 40 ms of frame time, 2.5 ms
        // and 1.0 ms of processing — the proc column is the max of the
        // processing times, not the frame times.
        let samples = [
            sample(9.95, 10.0, 2.5, 0.5, 2.0, 0.0, 0.0, 0.0),
            sample(9.98, 40.0, 1.0, 0.25, 0.75, 0.0, 0.0, 0.0),
        ];
        let cols = slice_max(samples, 10.0, |s| s.proc_ms);
        assert_eq!(cols[99], 2.5);
    }

    #[test]
    fn slice_max_app_ignores_the_total() {
        // Two frames in one slice: 2.5 ms and 1.0 ms of total processing,
        // 2.0 ms and 0.75 ms excluding the overlay — the app column is the
        // max of the overlay-excluded times, not the totals.
        let samples = [
            sample(9.95, 10.0, 2.5, 0.5, 2.0, 0.0, 0.0, 0.0),
            sample(9.98, 40.0, 1.0, 0.25, 0.75, 0.0, 0.0, 0.0),
        ];
        let cols = slice_max(samples, 10.0, |s| s.app_ms);
        assert_eq!(cols[99], 2.0);
    }

    #[test]
    fn slice_max_draw_ignores_the_times() {
        // Two frames in one slice: 12 and 7 total draw calls — the draw
        // column is the max of the counts, not of any time.
        let samples = [
            sample(9.95, 10.0, 2.5, 0.5, 2.0, 12.0, 4.0, 8.0),
            sample(9.98, 40.0, 1.0, 0.25, 0.75, 7.0, 2.0, 5.0),
        ];
        let cols = slice_max(samples, 10.0, |s| s.draw_calls);
        assert_eq!(cols[99], 12.0);
    }

    #[test]
    fn slice_max_projects_the_draw_split_independently() {
        // Two frames in one slice: 12 and 7 total draw calls, 4 and 2 of
        // them the overlay's own — the app column is the max of the
        // overlay-free counts (12 - 4 = 8), and the diagnostic column the
        // max of the overlay's own counts (4).
        let samples = [
            sample(9.95, 10.0, 0.0, 0.0, 0.0, 12.0, 4.0, 8.0),
            sample(9.98, 40.0, 0.0, 0.0, 0.0, 7.0, 2.0, 5.0),
        ];
        let cols = slice_max(samples, 10.0, |s| s.app_draws);
        assert_eq!(cols[99], 8.0);
        let cols = slice_max(samples, 10.0, |s| s.diag_draws);
        assert_eq!(cols[99], 4.0);
    }
}
