# Viability: frost + winit inside a nice-plug (CLAP/VST3) plugin

**Date checked:** against `nice-plug 0.4.2` (crates.io / tag `v0.4.2`), `frost@main`
(commit from 2026-10), `baseview 0.3.x`, `wgpu 30.0.1`, `winit 0.30.13`, macOS host,
rustc 1.99.

## Verdict

| Question | Answer |
|---|---|
| Do frost + nice-plug coexist in one dependency tree? | **Yes, verified by compiling** — single `wgpu 30.0.1`, single `winit 0.30.13`, single `raw-window-handle 0.6.2` |
| Can frost be used **as-is** (`frost::run`) inside a plugin? | **No.** It owns the process: own window, own winit event loop, blocks, and hijacks `NSApplicationDelegate` on macOS |
| Can **winit** windows embed into a DAW host window? | **Not on macOS** (no NSView embedding; `with_parent_window` is Windows/X11-only). Even where it works it is the wrong tool |
| Can frost's **renderer** (wgpu/scene graph) run inside nice-plug's GUI host? | **Yes, via a fork + baseview adapter** — nice-plug's own `byo_gui_wgpu` example proves the wgpu-in-host path; frost uses the same wgpu major version |
| Licensing | **⚠️ frost has no license** — all rights reserved. Needs written permission from the authors before shipping anything |

**Bottom line:** viable as a *rendering engine via a fork*, not viable as a drop-in GUI
dependency. The winit half of frost gets discarded at the seam; baseview (nice-plug's
host-integration layer) replaces it.

---

## Evidence

### 1. Dependency coexistence — verified by compile

Probe crate (`nice-plug 0.4.2` default-off + `editor`, `baseview 0.3.x`, `wgpu 30`,
`frost` git) **compiles clean**, including the exact seam nice-plug uses:

```rust
// wgpu 30 accepts baseview's WindowContext handle as a surface target —
// exactly what nice-plug's byo_gui_wgpu example does:
fn surface_target(window: &baseview::WindowContext) -> wgpu::SurfaceTarget<'static> {
    wgpu::SurfaceTarget::Window(Box::new(window.platform_handle()))
}
```

Resolved tree (duplicates only the usual bitflags/objc2 0.5-vs-0.6 noise):

```
baseview 0.3.7 · frost 0.1.0 (git) · wgpu 30.0.1 · winit 0.30.13 · raw-window-handle 0.6.2
```

frost pulls the only winit in the tree; baseview does **not** use winit (it uses
`objc2`/win32/x11 directly).

### 2. frost's public API cannot host-embed

- `frost::run()` / `run_configured()` (src/lib.rs ~913–1009): creates wgpu
  instance/adapter/device, then `EventLoop::new()` and **blocks in
  `event_loop.run_app(&mut app)`** until the window closes. No parent-window
  parameter, no external event pump, no external surface.
- The entire windowing/render-loop backend is **private**: `src/backend/mod.rs` says
  “Everything here is internal to the crate”; `Frost` is `pub(crate)`. Public surface
  is `Canvas`, `Context`, `Process`, `run`. Embedding = fork.
- Config knobs that do help once forked: `Config::vsync` (must be `false` — nice-plug's
  byo example warns “Plugins should not use vsync as it blocks the host's main thread”),
  `Config::resizable`, `window_size_px`.

### 3. winit-inside-a-plugin analysis (winit 0.30.13)

- **macOS:** `EventLoop::new` panics unless on the main thread
  (`platform_impl/macos/event_loop.rs:221`) — and the host already owns that thread's
  run loop. Worse, it calls `NSApplication::setDelegate(...)` with winit's own delegate
  (event_loop.rs ~227–240), **replacing the DAW's app delegate**. `pump_events` does
  exist publicly on macOS in 0.30 (`platform/mod.rs` cfg includes `macos_platform`), so
  an externally-driven loop is mechanically possible — but the delegate swap and app
  setup make this hostile in a plugin.
- **Embedding:** `WindowAttributes::with_parent_window` (window.rs ~475) is documented
  for **Windows (WS_CHILD) and X11 reparent only**; the macOS impl ignores it; Wayland
  unsupported. winit creates NSWindows, never embeddable NSViews. Proper in-host
  embedding is exactly why nice-plug uses **baseview** instead.
- nice-plug `main` even carries a workaround note (byo example Cargo.toml) warning about
  baseview coexisting with **winit in the same binary** (RustAudio/baseview#321,
  standalone mode) — another reason the fork should drop winit from the tree entirely.

### 4. What baseview gives us (the target the fork must fit)

- macOS: plugin content is an `NSView` added as a **subview** of the host view; keyboard/
  mouse/tracking arrive as native responder callbacks and are delivered as
  `baseview::Event` to `WindowHandler::on_event` on the host main thread. Frames are
  paced by an internal repeating **~15 ms timer** that calls `WindowHandler::on_frame`
  (`platform/macos/view.rs:142`). Continuous 60 fps-ish polling; fine for a tuner.
- Handler trait on crates.io 0.3.x (the one nice-plug **0.4.2** targets):
  `on_frame()`, `resized(WindowSize)`, `on_event(Event)` — no `poll/draw/on_timer` and
  no `request_redraw` on `WindowContext` (those exist only in the unreleased baseview
  that nice-plug `main` examples use). **Pin `baseview = "=0.3.3"`** (what v0.4.2
  examples were built against; 0.3.7 already shifted the API surface).
- Resize: host calls `EditorHandle::set_size` → `window.resize()` → `resized()` callback
  → reconfigure the wgpu surface. Standard for every nice-plug GUI.

### 5. Licensing — the real blocker

frost has **no LICENSE file and no `license` field**; README: “Copyright to the
Developers (Per Lindgren/Iris Hulsman) … hopefully we can find a license allowing the
sources to be open.” Default copyright ⇒ **not legally safe to build/ships as a
product dependency** without asking. Before investing in the fork: contact the authors
(they're responsive — active repo) and get MIT/Apache-2.0 or explicit permission.

---

## Recommended architecture if proceeding with frost

Fork frost, surgically split renderer from windowing:

1. **Keep**: `Scene`/`Node`/`Shape`/`Canvas` scene graph, `shaders.rs`, `frame.rs`
   draw-list pipeline, `text.rs` (swash), `bake/collision` if needed.
2. **Delete in fork**: `backend/app.rs`, `backend/windowing.rs`, `run()`, and the
   `rodio`/`Audio` module (a plugin must not open its own audio device — the host owns
   audio). winit dep drops out of the tree entirely; replace re-exported
   `KeyCode`/`MouseButton` with baseview equivalents.
3. **Add `FrostEmbedded` adapter** implementing `baseview::WindowHandler`:
   - create: wgpu surface via `SurfaceTarget::Window(Box::new(ctx.platform_handle()))`,
     `PresentMode::AutoNoVsync`, surface config from `ctx.size()`;
   - `on_frame`: build draw list from the `Scene`, run `Process`-style tick with the
     latest params/pitch state, render + present;
   - `on_event`: translate `baseview::Event` → frost's input state (port the input
     handling out of `windowing.rs`);
   - `resized`: reconfigure surface.
4. **nice-plug side**: copy the `byo_gui_wgpu` example (v0.4.2 tag) as the skeleton —
   `Editor::spawn` + `EditorHandle` + host-callback adapters are ~all boilerplate.
5. **Audio path for the tuner**: nice-plug's `process()` does the DSP (pitch detect,
   e.g. YIN/FFT); publish detected pitch via an `Arc<AtomicCell<...>>` like the examples'
   `WgpuEditorState`; `on_frame` reads it and redraws. Gate detection on
   `editor_state.is_open()`.

Effort: the seam is concentrated in `backend/app.rs` + `backend/windowing.rs`
(the scene/render pipeline itself stays untouched). Realistic: a few days of
implementation + ongoing fork maintenance.

### Alternatives considered

- **Run frost unmodified in a floating winit window** ("GUI sidecar window"): blocked on
  macOS (main-thread event loop + delegate hijack; spawn() must return to the host);
  fragile on Windows/X11; UX wrong (window never follows the host). Rejected.
- **Skip frost, use nice-plug's official GUI paths** (`gain_egui`, `gain_iced`,
  raw `byo_gui_wgpu`): zero fork risk, but loses frost's scene-graph aesthetic. If the
  goal is "a good-looking tuner" rather than "frost specifically", `byo_gui_wgpu` +
  frost-inspired 2D drawing, or egui/iced, is far cheaper.

## Open risks for the frost route

- Fork maintenance while frost upstream keeps evolving (active repo, LLM-driven churn).
- baseview is in flux (API moved again post-0.3.7; nice-plug main already tracks an
  unreleased version) — pin strictly now, plan a bump when nice-plug releases next.
- DPI/scale-factor handling must use `WindowSize.scale_factor` (host-provided), not the
  monitor scale frost reads today.
- Legal clearance (above).
