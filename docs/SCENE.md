# Nice Tuner — GUI scene design (M4/M5 spec)

Target API: `frost::embedded::Embedded<TunerUi>` (vendor/frost, post-M3) driving
frost's scene graph + immediate `Canvas` from inside `baseview::WindowHandler::on_frame`.

frost coordinate space: origin at window **center**, **y up**, units = logical px;
`Context` derefs to `Canvas` (immediate `line/circle/rectangle/text(..., z)`);
`Process` = `FnMut(&mut Context, dt: f32)`; text via `Canvas::text(cx, cy, &Arc<[u8]> font, s, size, weight, color, z)`.
Font: `vendor/frost/assets/fonts/FiraCode-VariableFont_wght.ttf` (OFL) — load once via `include_bytes!` into `Arc<[u8]>` in our crate.

## Layout (logical px, default window 640×400, min 480×320)

```
 y
 ↑                     note label   "A2"        text @(0, 95)  size 96 w=700
   ░░░░░░░░░░░░░▓▓▓░░░░░░░░░         cents bar  rect @(0, 25), half 160×5,
                  ▲ needle                        needle line @x = cents·2.4, ±120 halfwidth
                     +4.1¢          cents readout text @(0, -5)  size 28
  440.0 Hz · A4=440                 freq + ref readout text @(0, -45) size 22 (dim)
  ▂▃▅▇ level meter                  rms bar: rect from x=-160, halfw = rms-scaled, @(0,-95), h=6
```

Everything scales with window size by a `fit = min(w/640, h/400)` factor applied
in the Process closure (multiply sizes/positions), so resizing needs no rebuild.

## State machine (in the Process closure, audio thread untouched)

Read from `TunerSharedState` (atomics): `freq_hz`, `clarity`, `rms`, `a4_ref`.

- **signal present** iff `clarity ≥ min_clarity(0.80)` and `rms ≥ gate` param
  (gate stored on shared state too, M5).
- **stable** iff note == previous stable note for ≥ 400 ms → latch display color.
- **needle position**: EMA over 80 ms of `clamp(cents, -50, 50)`; springs back to
  center with 150 ms decay when signal lost.
- **colors**: base amber `#E8A33D`; in-tune green `#4ADE80` when stable && |cents| ≤ 3
  (200 ms color lerp); dim gray `#6B7280` when no signal.

## Scene graph vs immediate

Static structure (background, tick marks at −50..50 step 10 as small rectangles,
meter track, panel rectangles): plain `Scene` built once at spawn.
Dynamic items (needle, note/cents/freq text, meter fill, highlight zone):
drawn **immediately** through `Canvas` every frame inside Process — no node
surgery, no text-shape churn. (`Canvas::draw_scene` already composites the
static scene; order z: scene default, overlays z ≥ 0.1.)

## Input

None needed beyond niceties: window focus (redraw gating) handled by adapter.
Future: click-drag on ref readout to change A4 (set_key/scroll → frost input state).

## baseview → frost input mapping (adapter, mostly unused now but wired)

| baseview | frost Embedded |
|---|---|
| `KeyboardInput{keycode,..}` | `set_key(map(virtual_keycode → winit KeyCode), down)` |
| `CursorMoved{position}` (physical px) | `set_cursor(x/scale, logical_h − y/scale)` |
| `MouseInput{button, state}` | `set_mouse_button(map(button), down)` |
| `MouseWheel{delta}` | `scroll(dx, dy)` |
| `ReceivedCharacter(ch)` | `add_text(ch)` |
| `WindowEvent::Focused(b)` | `set_focused(b)` + redraw |

## Notes for M3 review

- `frame()` must tolerate dt spikes (host pauses) and `SizeInvalid` (zero-size
  windows during hide) → return Ok, skip frame.
- Surface `Lost` → `replace_surface` path must also re-configure at current size.
- Text expansion cost: only redraw strings that changed; strings ≤ 16 chars @ 64 Hz, negligible.
