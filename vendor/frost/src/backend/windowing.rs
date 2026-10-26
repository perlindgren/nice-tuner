//! Opening the window and the geometry around it: the winit window
//! itself, the monitor fit that keeps an oversized request on screen,
//! and the letterbox fit the blit pass stretches content with.

use std::sync::Arc;

use crate::backend::bloom::MAX_LEVELS;
use wgpu::PresentMode;
use winit::dpi::LogicalSize;
#[cfg(not(target_arch = "wasm32"))]
use winit::dpi::PhysicalPosition;
#[cfg(not(target_arch = "wasm32"))]
use winit::dpi::PhysicalSize;
use winit::event_loop::ActiveEventLoop;
#[cfg(not(target_arch = "wasm32"))]
use winit::window::Fullscreen;
use winit::window::Window;

/// The surface presentation mode for the user's vsync request: `AutoVsync`
/// presents once per vertical blank (capped at the display's refresh rate),
/// `AutoNoVsync` presents as soon as frames are rendered. Both `Auto*`
/// modes fall back to a supported mode when their preference is unavailable,
/// so they are safe to request on every platform.
pub(crate) fn present_mode_for(vsync: bool) -> PresentMode {
    if vsync {
        PresentMode::AutoVsync
    } else {
        PresentMode::AutoNoVsync
    }
}

/// Upgrade the surface config to the sRGB flavour of its format when
/// one is supported. The shaders write linear light and rely on the
/// framebuffer to encode to sRGB on write — an `*UnormSrgb` surface
/// does that in hardware, a plain `*Unorm` one does not, and a scene
/// of linear values shown unencoded reads drastically dark. Platforms
/// disagree on which flavour `get_default_config` picks first: Metal
/// leads with the sRGB one, a browser canvas with the linear one —
/// which is why the same game can look right on the desktop and dark
/// in the browser. This makes the choice explicit and identical
/// everywhere.
pub(crate) fn prefer_srgb_format(
    config: &mut wgpu::SurfaceConfiguration,
    surface: &wgpu::Surface<'_>,
    adapter: &wgpu::Adapter,
) {
    let want = match config.format {
        wgpu::TextureFormat::Bgra8Unorm => wgpu::TextureFormat::Bgra8UnormSrgb,
        wgpu::TextureFormat::Rgba8Unorm => wgpu::TextureFormat::Rgba8UnormSrgb,
        format if format.is_srgb() => return, // already encoded
        _ => return,                          // no sRGB twin of this format
    };
    if surface.get_capabilities(adapter).formats.contains(&want) {
        config.format = want;
    }
}

/// Creates the frost window and requests its first frame.
///
/// `window_size` is the `Config`'s initial inner size in logical pixels
/// and `window_size_px` its size in physical pixels; a set physical size
/// wins, and winit converts it through the monitor's scale factor, so the
/// requested panel-pixel window opens at the same physical size on every
/// display — including Retina Macs, where a logical request would double.
/// `None` for both keeps the platform default. An oversized request is
/// clamped into the primary monitor, aspect preserved (see
/// [`fit_into_monitor`]). On the web the browser owns the window, so the
/// canvas is sized to the viewport it lands in — see [`fit_canvas`] — and
/// `window_size` survives only as the fallback when no viewport can be
/// measured, while the physical request is ignored (the canvas lives in CSS
/// pixels).
pub(crate) fn create_window(
    event_loop: &ActiveEventLoop,
    window_size: Option<[u32; 2]>,
    window_size_px: Option<[u32; 2]>,
    fullscreen: bool,
) -> Arc<Window> {
    let mut attributes = Window::default_attributes().with_title("frost");
    #[cfg(not(target_arch = "wasm32"))]
    {
        let monitor = event_loop.primary_monitor();
        if let Some([width, height]) = window_size_px {
            // Physical request: compare against the monitor in physical
            // pixels, and hand back a clamped physical size.
            let size = match monitor.as_ref() {
                Some(m) => fit_into_monitor([width, height], [m.size().width, m.size().height]),
                None => [width, height],
            };
            attributes = attributes.with_inner_size(PhysicalSize::new(size[0], size[1]));
        } else if let Some([width, height]) = window_size {
            // Logical request: the monitor's own scale factor turns its
            // physical size into the logical budget to fit into.
            let size = match monitor.as_ref() {
                Some(m) => {
                    let s = m.scale_factor().max(0.01);
                    fit_into_monitor(
                        [width, height],
                        [
                            (m.size().width as f64 / s).round().max(1.0) as u32,
                            (m.size().height as f64 / s).round().max(1.0) as u32,
                        ],
                    )
                }
                None => [width, height],
            };
            attributes = attributes.with_inner_size(LogicalSize::new(size[0], size[1]));
        }
        if fullscreen {
            // Borderless on the monitor the window lands on: on macOS
            // winit turns this into the platform's native fullscreen
            // Space. The size request above still stands — it shapes the
            // surface through the transition, and a later un-fullscreen
            // returns the window to it.
            attributes = attributes.with_fullscreen(Some(Fullscreen::Borderless(None)));
            // Two winit knobs were tried here against the notch Mac of the
            // benchmarking doc's §17, where the Space leaves the menu bar's
            // band black across the top and the surface settles 43 points
            // short of the panel: `with_borderless_game(true)`, which hides
            // the bar and dock outright, and `with_fullsize_content_view(true)`,
            // which pushes the content behind the titlebar. Each alone, each
            // measured: the settled surface stayed 4112x2572 on a 4112x2658
            // display and the band stayed visible. The second changed the
            // resize sequence and nothing else. Closing the band needs AppKit
            // past what winit exposes; it is 3.3 % of the pixels and moves no
            // measurement, so it is left as a known gap rather than a guess.
        }
    }
    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowAttributesExtWebSys;

        let _ = window_size_px;
        let _ = fullscreen;
        // The viewport, not the configured size: a 1366×682 window that boots
        // a 1920×1080 canvas draws the hill correctly and puts its bottom and
        // right third past the edge of the screen, which is what the page did
        // before this line. `window_size` remains for the case where the
        // viewport cannot be measured at all.
        let css = viewport_css().unwrap_or_else(|| {
            let [width, height] = window_size.unwrap_or([900, 600]);
            [f64::from(width), f64::from(height)]
        });
        attributes = attributes
            .with_append(true)
            .with_inner_size(LogicalSize::new(css[0], css[1]));
    }
    let window = Arc::new(
        event_loop
            .create_window(attributes)
            .expect("failed to create window"),
    );
    // A canvas has two sizes and only one of them had ever been set.
    //
    // `with_inner_size` above sizes the element's *CSS box* — what the page
    // lays out and stretches. The drawing buffer is a separate pair of
    // attributes, `canvas.width` and `canvas.height`, and an HTML canvas
    // nobody touches leaves them at the browser's default **300×150**. So
    // every frame was built in full — its 1920×1080 buffer, correct to the
    // texel, as the upload, geometry and bit-pun probes went on to establish
    // one at a time after a long detour — and then blitted down into a
    // 300×150 surface for the browser to stretch back up. That is what a
    // mosaic of blocks is, and it is why every instrument insisted the frame
    // was fine while the page insisted it was garbage: both were right about
    // different buffers.
    //
    // winit writes the CSS box and never the bitmap; wgpu reads the bitmap and
    // never writes it. Both jobs are therefore ours, at creation and on every
    // resize after it — see [`fit_canvas`] for why the box is asked of the
    // layout rather than assumed from the style just written.
    #[cfg(target_arch = "wasm32")]
    {
        // Reached through web-sys rather than as a direct dependency: the
        // engine binds no wasm-bindgen of its own, and one re-exported by the
        // crate that already carries the DOM types beats adding a second copy
        // to the graph for two traits.
        use web_sys::wasm_bindgen::JsCast;
        use web_sys::wasm_bindgen::closure::Closure;
        use winit::platform::web::WindowExtWebSys;

        if let Some(canvas) = window.canvas() {
            fit_canvas(&canvas, window.scale_factor());
            // A dragged or maximised window moves the viewport, and nothing
            // tells winit about it unless the canvas's box moves with it —
            // winit's own size only ever follows a `set_inner_size` call. So
            // the page is watched directly, and each resize refits the canvas,
            // which moves the box, which is what winit's observer is waiting
            // for before it reports `Resized` to the engine.
            let watched = canvas.clone();
            let host = Arc::clone(&window);
            let listener = Closure::wrap(Box::new(move |_event: web_sys::Event| {
                fit_canvas(&watched, host.scale_factor());
            }) as Box<dyn FnMut(web_sys::Event)>);
            if let Some(view) = web_sys::window() {
                let _ = view
                    .add_event_listener_with_callback("resize", listener.as_ref().unchecked_ref());
            }
            // Leaked on purpose: one listener per page, and the page outlives
            // the window it watches.
            listener.forget();
            // The same refit a second time, driven by the layout rather than
            // by the window. `resize` is the documented way to hear about a
            // viewport change, and it is heard once per gesture — a compositor
            // that snaps the frame to a tile, or a fullscreen transition that
            // lands without one, moves the box and says nothing. Watching the
            // root element makes the geometry itself the trigger, and
            // `fit_canvas` is idempotent, so hearing it twice costs nothing.
            let watched = canvas;
            let host = Arc::clone(&window);
            let resize = Closure::wrap(Box::new(
                move |_entries: js_sys::Array, _obs: web_sys::ResizeObserver| {
                    fit_canvas(&watched, host.scale_factor());
                },
            )
                as Box<dyn FnMut(js_sys::Array, web_sys::ResizeObserver)>);
            if let (Ok(observer), Some(root)) = (
                web_sys::ResizeObserver::new(resize.as_ref().unchecked_ref()),
                web_sys::window()
                    .and_then(|v| v.document())
                    .and_then(|d| d.document_element()),
            ) {
                observer.observe(&root);
                // Both are deliberately leaked: an observer is dropped the
                // instant its last strong handle goes, and there is no later
                // point at which the page would rather stop following its own
                // window.
                resize.forget();
                std::mem::forget(observer);
            } else {
                resize.forget();
            }
        }
    }
    // Center the window on its monitor so frost always opens in the middle
    // of the screen. Native only: on the web the page decides where the
    // canvas sits.
    #[cfg(not(target_arch = "wasm32"))]
    center_on_screen(event_loop, &window);
    // winit (Wayland) only delivers RedrawRequested after a compositor
    // frame callback, so explicitly request the first frame; otherwise
    // the window is never mapped and nothing is ever drawn.
    window.request_redraw();
    window
}

/// The browser's viewport in CSS pixels, when it can be measured at all.
#[cfg(target_arch = "wasm32")]
fn viewport_css() -> Option<[f64; 2]> {
    let view = web_sys::window()?;
    let width = view.inner_width().ok()?.as_f64()?;
    let height = view.inner_height().ok()?.as_f64()?;
    (width >= 1.0 && height >= 1.0).then(|| [width.floor(), height.floor()])
}

/// Size a canvas to the viewport, both of the two ways a canvas is sized.
///
/// The CSS box goes first, because winit's resize observer watches the
/// canvas's *device-pixel content box* and reports to the engine only when
/// that box moves: writing the style is what makes a resized window reach the
/// renderer at all. The drawing buffer follows, from the box the layout
/// actually produced rather than the style just written — a page may override
/// the style, and the laid-out box is the truth either way — scaled by
/// `devicePixelRatio` so a HiDPI page paints the pixels it will show instead
/// of asking the browser to enlarge ones we never drew.
#[cfg(target_arch = "wasm32")]
fn fit_canvas(canvas: &web_sys::HtmlCanvasElement, scale: f64) {
    let scale = scale.max(1.0);
    let style = canvas.style();
    // First take the clamps off. `set_min_inner_size` and
    // `set_max_inner_size` reach the web as CSS `min-width`/`max-width` and
    // friends on this very element, and a canvas pinned to its boot size in
    // both directions follows nothing, however its width is written: the
    // constraint beats the style, which is how a resize moves the attribute
    // and leaves the layout standing still. Clearing them here, rather than
    // only where the pin was set, makes the refit hold no matter which app
    // pinned the window or when it got around to it.
    for pin in ["min-width", "min-height", "max-width", "max-height"] {
        let _ = style.set_property(pin, "none");
    }
    if let Some([width, height]) = viewport_css() {
        let _ = style.set_property("width", &format!("{width}px"));
        let _ = style.set_property("height", &format!("{height}px"));
    }
    let laid = (canvas.client_width(), canvas.client_height());
    let css = if laid.0 > 0 && laid.1 > 0 {
        [f64::from(laid.0), f64::from(laid.1)]
    } else {
        viewport_css().unwrap_or([900.0, 600.0])
    };
    canvas.set_width((css[0] * scale).round().max(1.0) as u32);
    canvas.set_height((css[1] * scale).round().max(1.0) as u32);
}

/// Centers `window` on the monitor it sits on, so frost always opens in
/// the middle of the screen. Both the monitor's size and the window's
/// outer size (title bar and borders included) are in physical pixels, so
/// the position is computed in physical pixels too — no scale-factor
/// conversion. When the window is larger than the monitor, it is clamped
/// to the monitor's top-left.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn center_on_screen(event_loop: &ActiveEventLoop, window: &Window) {
    let Some(monitor) = window
        .current_monitor()
        .or_else(|| event_loop.primary_monitor())
    else {
        return;
    };
    let screen = monitor.size();
    let frame = window.outer_size();
    let x = ((screen.width as i64 - frame.width as i64) / 2).max(0) as i32;
    let y = ((screen.height as i64 - frame.height as i64) / 2).max(0) as i32;
    window.set_outer_position(PhysicalPosition::new(x, y));
    log::info!(
        "window centered at ({x}, {y}) px on a {}x{} monitor",
        screen.width,
        screen.height
    );
}

/// Fits a requested opening size into the space available on the primary
/// monitor, aspect preserved: a request that fits passes through
/// untouched, an oversized one shrinks to the largest fitting size of its
/// own aspect — a window that opens whole and centered beats one that
/// hangs off the screen, and squashing the shape is not the same as
/// making it smaller. `size` and `avail` are in the same unit (physical
/// or logical, as the caller chose); with no monitor info the request
/// passes through.
#[cfg(not(target_arch = "wasm32"))]
pub(crate) fn fit_into_monitor(size: [u32; 2], avail: [u32; 2]) -> [u32; 2] {
    let (req_w, req_h) = (size[0] as f64, size[1] as f64);
    let (max_w, max_h) = (avail[0].max(1) as f64, avail[1].max(1) as f64);
    if req_w <= max_w && req_h <= max_h {
        return size;
    }
    let factor = (max_w / req_w).min(max_h / req_h);
    [
        (req_w * factor).round().max(1.0) as u32,
        (req_h * factor).round().max(1.0) as u32,
    ]
}

/// Packs the present pass's whole uniform: two floats of letterbox fit, the
/// struct's padding, one bloom weight per plumbing slot in two `vec4`s, and
/// the tone-map headroom. Sixty-four bytes — the weight pair pushes the last
/// field past the old forty-eighth — which is a reminder that this function
/// and `blit.wgsl`'s `BlitUniforms` are one layout written in two languages.
pub(crate) fn fit_bytes(fit: (f32, f32), bloom: [f32; MAX_LEVELS], tone: f32) -> [u8; 64] {
    // The present pass reads the weights as two `vec4`s, so a seventh slot
    // would need the shader and this buffer widened together rather than one
    // of them silently overrunning.
    const {
        assert!(MAX_LEVELS <= 6, "blit.wgsl holds six bloom weights");
    }
    let mut bytes = [0u8; 64];
    bytes[0..4].copy_from_slice(&fit.0.to_le_bytes());
    bytes[4..8].copy_from_slice(&fit.1.to_le_bytes());
    for (i, w) in bloom.iter().enumerate() {
        bytes[16 + i * 4..20 + i * 4].copy_from_slice(&w.to_le_bytes());
    }
    bytes[48..52].copy_from_slice(&tone.to_le_bytes());
    bytes
}

/// The fraction `(width, height)` of a `surface` that a fixed
/// `render_size` buffer fills when its aspect is preserved and the result
/// is centered: `(1.0, 1.0)` when there is no fixed buffer or the aspects
/// match (the picture covers the window), and a letterbox fraction
/// otherwise — `(1.0, 0.5625)` for a 16:9 buffer on a square surface,
/// bars top and bottom included. The blit pass scales its quad by
/// exactly this, and the mouse mapping divides it back out.
pub(crate) fn content_fit(render_size: Option<[u32; 2]>, surface: (u32, u32)) -> (f32, f32) {
    let Some([bw, bh]) = render_size else {
        return (1.0, 1.0);
    };
    let (sw, sh) = (surface.0 as f32, surface.1 as f32);
    if sw <= 0.0 || sh <= 0.0 {
        return (1.0, 1.0);
    }
    if (bw as f64) * (sh as f64) > (bh as f64) * (sw as f64) {
        // The buffer is wider than the surface: width-limited, with
        // letterbox bars above and below.
        (
            1.0,
            (bh as f64 * sw as f64 / (bw as f64 * sh as f64)) as f32,
        )
    } else {
        // Taller (or exactly as wide): pillarbox bars left and right.
        (
            (bw as f64 * sh as f64 / (bh as f64 * sw as f64)) as f32,
            1.0,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn a_monitor_fit_leaves_sizes_that_fit_untouched() {
        assert_eq!(fit_into_monitor([1280, 720], [1512, 982]), [1280, 720]);
        // Exactly the budget is not an overflow.
        assert_eq!(fit_into_monitor([1512, 982], [1512, 982]), [1512, 982]);
    }

    #[cfg(not(target_arch = "wasm32"))]
    #[test]
    fn a_monitor_fit_shrinks_an_oversized_request_aspect_preserved() {
        // 1920x1080 on a 1512-wide MacBook: the width binds, and the
        // height falls by the same factor — the 16:9 shape survives.
        let [w, h] = fit_into_monitor([1920, 1080], [1512, 982]);
        assert_eq!(w, 1512);
        assert!((w as f64 / h as f64 - 16.0 / 9.0).abs() < 0.01);
        // A tall request on a wide monitor is height-bound instead.
        let [w, h] = fit_into_monitor([1080, 1920], [1512, 982]);
        assert_eq!(h, 982);
        assert!((w as f64 / h as f64 - 1080.0 / 1920.0).abs() < 0.01);
    }

    #[test]
    fn a_matching_aspect_fills_the_surface() {
        assert_eq!(content_fit(None, (100, 100)), (1.0, 1.0));
        assert_eq!(content_fit(Some([1920, 1080]), (1920, 1080)), (1.0, 1.0));
        // The same shape at a different size fills too: 800x450 is 16:9.
        let (fx, fy) = content_fit(Some([1920, 1080]), (800, 450));
        assert!((fx - 1.0).abs() < 1e-4 && (fy - 1.0).abs() < 1e-4);
    }

    #[test]
    fn a_wider_surface_than_buffer_pillarboxes_horizontally() {
        // A square window with a 16:9 buffer: the buffer is wider, so it
        // spans the full width and shrinks in height.
        let (fx, fy) = content_fit(Some([1920, 1080]), (1000, 1000));
        assert_eq!(fx, 1.0);
        assert!((fy - 1000.0 * 1080.0 / (1920.0 * 1000.0)).abs() < 1e-4);
        assert!(fy > 0.55 && fy < 0.57);
    }

    #[test]
    fn a_taller_surface_than_buffer_letterboxes_vertically() {
        // A square window with a 1:2 (portrait) buffer: the buffer is
        // taller, so it spans the full height and shrinks in width.
        let (fx, fy) = content_fit(Some([540, 1080]), (1000, 1000));
        assert_eq!(fy, 1.0);
        assert!((fx - 1000.0 * 540.0 / (1080.0 * 1000.0)).abs() < 1e-4);
        assert!((fx - 0.5).abs() < 1e-4);
    }

    #[test]
    fn a_zero_sized_surface_falls_back_to_filling() {
        // A zero surface would divide by zero; the guard returns (1, 1).
        assert_eq!(content_fit(Some([1920, 1080]), (0, 0)), (1.0, 1.0));
    }

    #[test]
    fn fit_bytes_writes_the_two_extents_and_pads() {
        let bytes = fit_bytes((0.5, 0.25), [0.0; MAX_LEVELS], 0.0);
        assert_eq!(f32::from_le_bytes(bytes[0..4].try_into().unwrap()), 0.5);
        assert_eq!(f32::from_le_bytes(bytes[4..8].try_into().unwrap()), 0.25);
        assert_eq!(&bytes[8..16], &[0u8; 8]);
        assert_eq!(&bytes[16..48], &[0u8; 32]);
        assert_eq!(&bytes[48..], &[0u8; 16]);
    }

    /// The bloom weights ride in the same bytes as the letterbox fit, at
    /// offset 16, where `blit.wgsl`'s `BlitUniforms::bloom` reads them — and
    /// the tone headroom moved when the weight array grew, so its new offset
    /// is pinned beside theirs. The layout is shared by a WGSL struct and a
    /// byte-packer in another file, so it is worth one test that says where
    /// each thing lands.
    #[test]
    fn the_bloom_weights_land_after_the_padding() {
        let bytes = fit_bytes((1.0, 1.0), [0.5, 0.3, 0.2, 0.1, 0.05, 0.05], 0.75);
        let f = |r: std::ops::Range<usize>| f32::from_le_bytes(bytes[r].try_into().unwrap());
        assert_eq!(
            (f(16..20), f(20..24), f(24..28), f(28..32)),
            (0.5, 0.3, 0.2, 0.1)
        );
        assert_eq!((f(32..36), f(36..40)), (0.05, 0.05));
        assert_eq!(f(48..52), 0.75);
    }

    /// The tone-map headroom lands at offset 48, the last field of the struct,
    /// pushed there by the second `vec4` of bloom weights. Same reason the
    /// weights needed a test: two files agree on this layout by agreement and
    /// nothing checks them but this — and this one is the test that moved when
    /// the chain grew past a single `vec4` of weights, which is exactly the
    /// failure it exists to catch.
    #[test]
    fn the_headroom_lands_at_the_end() {
        let bytes = fit_bytes((1.0, 1.0), [0.0; MAX_LEVELS], 0.15);
        assert_eq!(
            f32::from_le_bytes(bytes[48..52].try_into().unwrap()),
            0.15,
            "blit.wgsl reads `tone` at offset 48"
        );
        assert_eq!(&bytes[16..48], &[0u8; 32]);
    }
}
