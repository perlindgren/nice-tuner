//! The geometry around a frame: the letterbox fit the blit pass
//! stretches content with, the uniform it stretches it with, and the
//! format and present-mode choices a surface is configured with. The
//! window itself — and every geometry negotiation that belonged to the
//! window it used to own — left with the embedding refactor; the host
//! opens and sizes the frame's window now.

use crate::backend::bloom::MAX_LEVELS;
use wgpu::PresentMode;

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

/// Fits a requested opening size into the space available on the primary
/// monitor, aspect preserved: a request that fits passes through
/// untouched, an oversized one shrinks to the largest fitting size of its
/// own aspect — a window that opens whole and centered beats one that
/// hangs off the screen, and squashing the shape is not the same as
/// making it smaller. `size` and `avail` are in the same unit (physical
/// or logical, as the caller chose); with no monitor info the request
/// passes through. (Window opening left the engine with the embedding
/// refactor; this fit outlives it as a plain geometry answer, tested like
/// the rest of the module.)
#[allow(dead_code)]
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

    #[test]
    fn a_monitor_fit_leaves_sizes_that_fit_untouched() {
        assert_eq!(fit_into_monitor([1280, 720], [1512, 982]), [1280, 720]);
        // Exactly the budget is not an overflow.
        assert_eq!(fit_into_monitor([1512, 982], [1512, 982]), [1512, 982]);
    }

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
