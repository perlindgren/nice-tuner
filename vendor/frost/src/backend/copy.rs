//! The row-pitch law of a texture copy, and the padding that satisfies it.
//!
//! WebGPU requires `bytesPerRow` of any copy into a texture to be a multiple
//! of [`ROW_ALIGNMENT`]. wgpu-core — which is what every native backend runs
//! — accepts any pitch it is given, so an upload of tightly packed RGBA8 with
//! `bytes_per_row = width * 4` builds, uploads and renders correctly on
//! Vulkan, Metal and DX12, while on the web the browser simply refuses the
//! write. What follows is the quiet half of that failure: no error reaches
//! the pipeline, because the texture exists, its view exists and the
//! pipeline that samples it is valid. Only the texture's contents are
//! whatever its memory happened to hold, so every sprite draws noise and the
//! one piece of evidence is the picture.
//!
//! frost's pictures are authored, not padded: `ant1.png` is 966 texels wide
//! — a pitch of 3864 — and `soil.png` is 1080, a pitch of 4320. Neither is a
//! multiple of 256, and of the sprites the game ships only a handful are. So
//! every upload of texels goes through [`pad_rows`] rather than learning the
//! law per texture, and the placeholders are no exception: a 1×1 texel is a
//! pitch of 4, which the browser refuses as readily as any other.

use std::borrow::Cow;

/// The boundary a texture copy's row must start on. A law of WebGPU
/// (`GPUTextureDataLayout.bytesPerRow`), not of wgpu, and not of any GPU.
pub(crate) const ROW_ALIGNMENT: u32 = 256;

/// The bytes and the pitch to hand `write_texture` for a tightly packed
/// `width` × `height` image of `texel_bytes`-sized texels.
///
/// Returns the rows padded to [`ROW_ALIGNMENT`] and that padded pitch, or
/// borrows the caller's bytes when the pitch already satisfies the rule —
/// which is the case for a power-of-two-wide image, and for every texture
/// the engine itself renders into and copies out of.
pub(crate) fn pad_rows(
    bytes: &[u8],
    width: u32,
    height: u32,
    texel_bytes: u32,
) -> (Cow<'_, [u8]>, u32) {
    let pitch = width * texel_bytes;
    debug_assert!(
        bytes.len() as u64 >= pitch as u64 * height as u64,
        "pad_rows: {} bytes for a {width}x{height} image of {texel_bytes}-byte texels \
         needs {}",
        bytes.len(),
        pitch * height
    );
    let aligned = pitch.next_multiple_of(ROW_ALIGNMENT);
    if aligned == pitch {
        return (Cow::Borrowed(bytes), pitch);
    }
    // Rows, not a stride trick: the browser reads the pitch it is told, so
    // the gap has to exist in the buffer. This runs once per texture, at the
    // load that creates it — never per frame.
    let mut padded = vec![0u8; (aligned * height) as usize];
    for row in 0..height as usize {
        let from = row * pitch as usize;
        let to = row * aligned as usize;
        padded[to..to + pitch as usize].copy_from_slice(&bytes[from..from + pitch as usize]);
    }
    (Cow::Owned(padded), aligned)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A pitch already on the boundary is the caller's own bytes, unchanged
    /// and uncloned — the padding must not tax the textures that never
    /// needed it, which includes every render target the engine copies into.
    #[test]
    fn an_aligned_pitch_is_handed_back_untouched() {
        let bytes = vec![7u8; 64 * 4 * 2];
        let (padded, pitch) = pad_rows(&bytes, 64, 2, 4);
        assert_eq!(pitch, 256);
        assert!(matches!(padded, Cow::Borrowed(_)));
        assert_eq!(&*padded, &bytes[..]);
    }

    /// The shape the game actually ships: an ant 966 texels wide. Every row
    /// lands on its 256-aligned start, the texels of each row survive in
    /// order, and the gap between rows is zero — the browser is free to read
    /// it, and a sprite must not sample it either way.
    #[test]
    fn an_authored_picture_is_padded_row_by_row() {
        let (width, height) = (966u32, 3u32);
        let pitch = width * 4;
        let bytes: Vec<u8> = (0..pitch * height).map(|i| (i % 253) as u8).collect();
        let (padded, aligned) = pad_rows(&bytes, width, height, 4);
        // 966 * 4 = 3864, whose next multiple of 256 is 4096 — stated as a
        // number rather than recomputed with the formula under test, so the
        // test cannot agree with a wrong `pad_rows` by agreeing with itself.
        assert_eq!(aligned, 4096);
        assert_eq!(aligned % ROW_ALIGNMENT, 0);
        assert_eq!(padded.len(), aligned as usize * height as usize);
        for row in 0..height as usize {
            let src = &bytes[row * pitch as usize..(row + 1) * pitch as usize];
            let dst = &padded[row * aligned as usize..row * aligned as usize + pitch as usize];
            assert_eq!(dst, src, "row {row} lost its texels");
            assert!(
                padded[row * aligned as usize + pitch as usize..(row + 1) * aligned as usize]
                    .iter()
                    .all(|&b| b == 0),
                "row {row} left garbage in the gap"
            );
        }
    }

    /// One texel is not exempt: a placeholder's pitch of 4 is refused by the
    /// same rule as an ant's, and the engine writes two of them.
    #[test]
    fn one_texel_pays_the_same_pitch_as_a_thousand() {
        let (padded, pitch) = pad_rows(&[255, 255, 255, 255], 1, 1, 4);
        assert_eq!(pitch, ROW_ALIGNMENT);
        assert_eq!(padded.len(), ROW_ALIGNMENT as usize);
        assert_eq!(&padded[..4], &[255, 255, 255, 255]);
        assert!(padded[4..].iter().all(|&b| b == 0));
    }
}
