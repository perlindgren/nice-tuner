# Integration notes (staged M4/M5 → src/)

Apply once M2 (`crates/pitch`) and M3 (`vendor/frost` `frost::embedded`) are green.

## Cargo.toml (root) — workspace flip
- [ ] root now has `[workspace] members=["xtask"] exclude=[pitch, frost]`;
      when the path deps go live: drop the `exclude` line and remove the
      `[workspace]` table from crates/pitch/Cargo.toml (it self-detaches today).
- [ ] bundle check: `cargo xtask bundle nice-tuner --release` → dist/ (.clap/.vst3 on macOS).

## Cargo.toml (root)
- [ ] uncomment `pitch = { path = "crates/pitch" }` and `frost = { path = "vendor/frost" }`
- [x] `keyboard-types = "0.8.3"` added to root Cargo.toml already (baseview 0.3.3 does NOT re-export it;
      `Event::Keyboard` carries `keyboard_types::KeyboardEvent{key,state,text,modifiers}`)
- [x] baseview 0.3.3 event model verified from cached source:
      Event::{Mouse,Keyboard,Window}; MouseEvent::{CursorMoved(physical px),
      ButtonPressed, ButtonReleased, WheelScrolled(ScrollDelta::Lines|Pixels)};
      WindowEvent::{Focused, Unfocused, WillClose}. stage/gui.rs + tuner_ui.rs
      rewritten to match (2026-10-26).

## M2 result (DONE — integrate as reported)
- API confirmed: `Config{sample_rate,min_freq,max_freq,downsample_factor,min_clarity}`
  (defaults 48000/60/2000/4/0.85), `Detection{freq_hz,clarity,rms}`,
  `PitchDetector::{new,push_samples,set_min_freq,set_max_freq,reset}`,
  `Note{name,octave,midi,cents,nearest_freq}`, `freq_to_note(f64,f64)`. 17/17 tests.
- **Deviation to honor in M5:** `push_samples` returns `Some` only when a window
  completes AND clarity ≥ min_clarity (audio-thread gate). Therefore UI needs
  FRESHNESS: shared `last_detection_ms` (epoch ms, written on Some); UI signal =
  fresh(<500ms) && rms_db ≥ gate. MIN_CLARITY removed from tuner_ui (done).

## src/lib.rs
- [x] `TunerSharedState`: `gate_db` mirror already added to the live file.
- [ ] `mod gui; mod tuner_ui;` — gui.rs replaces the in-file `TunerWindow`
      (delete the triangle renderer + `Surface` struct; `TunerEditor::spawn`
      keeps using `TunerWindow::new` with the same signature).
- [ ] `TunerPlugin`: add `detector: Option<pitch::PitchDetector>` field.
- [ ] `activate()`: build `pitch::PitchDetector::new(Config { sample_rate:
      buffer_config.sample_rate as f64, min_freq: 30.0 /* 5-string bass B0 */,
      max_freq: 2100.0, ..Default::default() })`.
- [ ] `process()`: per block, if `self.shared.is_open()` and detector exists —
      downmix main input to mono into a small stack buffer (nice-plug gives
      `&mut [f32]` frames; iterate `buffer.iter_frames()`), call
      `push_samples`, on `Some(d)` store `freq_hz/clarity/rms` AND
      `last_detection_ms` (SystemTime epoch ms) into shared;
      if not open, `detector.reset()` (cheap) so a reopened UI starts clean.
      Also mirror `gate_db` param → shared.gate_db next to the a4 mirror.
- [ ] keep passthrough (do not touch output channels).

## M2/M3 API reconciliation
- [ ] `pitch`: confirm names `Config{sample_rate,min_freq,max_freq,downsample_factor,min_clarity}`,
      `Detection{freq_hz,clarity,rms}`, `Note{name,octave,midi,cents,nearest_freq}`,
      `freq_to_note(f64,f64)`. Adjust `tuner_ui.rs` imports if renamed.
- [ ] `frost::embedded::Embedded`: verify `new(...)` argument order,
      `FrameError` variants (`Skipped` may not exist — map whatever "nothing to
      draw" is), `resize((u32,u32), f64)`, `replace_surface(Surface, format)`,
      input setters, `Scene`/`SceneNode`/`Shape`/`Color` paths still re-exported.
      `Context` must still Deref to `Canvas` (size/line/rect/text) — if M3
      changed that, go through an accessor.
- [ ] frost `KeyCode`/`MouseButton` re-exports exist (winit types kept).
- [ ] font path: `vendor/frost/assets/fonts/FiraCode-VariableFont_wght.ttf`
      present in vendored copy (it is; OFL — keep its license file next to it).

## Gates (M6)
```sh
export CARGO_HOME=$PWD/.cargohome
cargo check && cargo check --features standalone
cargo test            # pitch + frost logic tests
cargo run --features standalone   # manual: hum at it / play a string
```
Also: run the EventLoop grep on vendor/frost/src once more; add the root
workspace [workspace] members table ONLY if we want `cargo test` at root to
cover sub-crates — otherwise document `cargo test --manifest-path crates/pitch/Cargo.toml`.

## Known upstream drift to record in README when done
- baseview `=0.3.3` pinned; nice-plug main tracks baseview 0.4.1 — bump both
  together at the next nice-plug release.
- Linux: nice-plug 0.4.2 needs `futures-intrusive` patched (PR #1365) — CI note.
