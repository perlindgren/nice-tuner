# Nice Tuner

A stringed-instrument chromatic tuner audio plugin (CLAP + VST3 + standalone),
built on:

| layer | crate | role |
|---|---|---|
| host protocol + audio | [nice-plug 0.4.2](https://codeberg.org/RustAudio/nice-plug) | CLAP/VST3, params, DSP↔UI plumbing |
| window embedding | [baseview `=0.3.3`](https://github.com/RustAudio/baseview) | child view inside the DAW window, host-driven events & frame timer |
| rendering | `vendor/frost` (fork of [perlindgren/frost](https://github.com/perlindgren/frost)) | scene graph + wgpu 30 2D renderer, winit run loop stripped (`frost::embedded::Embedded`) |
| DSP | `crates/pitch` | allocation-free YIN pitch detection + note mapping |

See [docs/VIABILITY.md](docs/VIABILITY.md) for why the GUI stack looks like this
(winit cannot embed into a host; baseview is nice-plug's sanctioned BYO-GUI path),
and [docs/SCENE.md](docs/SCENE.md) for the UI design.

## Layout

```
src/lib.rs        plugin: TunerPlugin (params, process), TunerEditor (baseview spawn)
src/gui.rs        baseview WindowHandler driving `frost::embedded::Embedded` (surface, inputs)
src/tuner_ui.rs   frost scene: gauge, needle, note readout, input meter (`Process` impl)
src/main.rs       standalone target (`--features standalone`)
crates/pitch      pure-Rust YIN pitch detector (unit-tested, no deps)
vendor/frost      frost fork: embedded wgpu renderer, no event loop, no rodio
```

## Building

```sh
cargo check                          # plugin lib (cdylib) — CLAP/VST3 body
cargo test                           # unit tests (pitch detector, frost logic)
cargo run --features standalone      # standalone app for testing without a DAW
# mono input (e.g. the built-in "MacBook Pro Microphone") needs the second,
# 1-in/1-out audio layout, since standalone matches input channel counts
# exactly and defaults to layout 1 (stereo):
./target/debug/nice-tuner -l 2 --input-device "MacBook Pro Microphone"

# Windows + ASIO (e.g. a Komplete Audio interface): build with the ASIO
# feature, then select the backend and driver. The device name must match
# the ASIO driver name exactly; passing a bogus name prints the full list.
# ASIO is duplex, so omitting --output-device plays back on the same box.
cargo build --features standalone-asio --bin nice-tuner
target/debug/nice-tuner --backend asio --input-device "Komplete Input 1/2"
```

### Producing plugin binaries

```sh
cargo xtask bundle nice-tuner --release   # → target/bundled/  (Nice Tuner.clap / .vst3 on macOS)
```

This uses `nice-plug-xtask` (wired up as the `xtask` workspace member with
`bundler.toml`). The full verification suite lives in `scripts/check.sh`.

## Notes

- **No vsync**: the GUI surface runs `PresentMode::AutoNoVsync`; plugins must
  never block the host main thread.
- **Real-time safety**: `process()` is guarded by nice-plug's
  `assert_process_allocs` in debug builds; the detector preallocates.
- **Winit is absent from the runtime path**: `vendor/frost` keeps winit only for
  event *data types*; constructing an `EventLoop` is grep-gated out of the tree.

## Licensing

- This plugin: MIT OR Apache-2.0.
- `vendor/frost`: fork of perlindgren/frost, whose upstream is unlicensed
  ("all rights reserved" pending a license decision). This repository is
  maintained by one of the upstream authors, so use in this project is
  authorized; redistribute the fork only under terms agreed with both upstream
  authors. The bundled FiraCode font is OFL.

## Status

- [x] M1 plugin skeleton (baseview window, wgpu surface, host adapters)
- [x] M2 pitch detector (`crates/pitch`) — 17/17 tests green
- [x] M3 frost embedded engine (`frost::embedded::Embedded`) — 313 fork tests green,
      winit event loop grep-gated out
- [x] M4 baseview→frost adapter (`src/gui.rs`) + tuner scene (`src/tuner_ui.rs`)
- [x] M5 DSP→GUI wiring (analysis gated on editor open; shared atomics, zero RT allocs)
- [x] M6 `scripts/check.sh` all gates green, incl. `Nice Tuner.clap`/`.vst3` bundling

## Development

Full gate (fmt, clippy `-D warnings`, tests, linker, bundling): `bash scripts/check.sh`.
It also runs automatically on `git push` once the hook is installed per clone:

```sh
git config core.hooksPath .githooks
```
