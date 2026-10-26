# ARCHITECTURE

A map of the `frost` crate for getting up to speed quickly: what each file is for,
how a frame flows through the library, and the invariants the code relies on.
The doc comments in the source are the ground truth — this file is the index and
the "why".

## What frost is

A **minimal winit + wgpu immediate-mode drawing library** for games in Rust
(edition 2024). You give it a `Scene` (a tree of nodes) and a `Process` (a
per-frame callback), and it runs an event loop: each frame your `Process` mutates
the scene and issues immediate draws, then the scene tree is updated and drawn.

Design stance (see `README.md`): a game-engine experiment by N65 Game Research
Station. Deliberately **not** Bevy-style ECS with shared mutability — plain Rust
ownership, a scene tree, and immediate drawing. Goals: a realistic reasoning
challenge for a local-hosted AI workflow, human learning of engine internals, and
(if it works) the core mechanics of a library-based engine.

## Crate layout

```
Cargo.toml            deps: gilrs 0.11 (gamepads), image 0.25 (png),
                      log 0.4 (release_max_level_off), swash 0.2 (text),
                      wgpu 30.0.1, winit 0.30.13; native-only: rodio 0.22
                      (audio); wasm32-only: web-sys,
                      console_error_panic_hook; dev-deps: arboard 3.6 (the
                      text_input example's clipboard), clap, env_logger,
                      naga 30.0.0 (shader validation, no GPU needed),
                      serde + ron 0.8 (the immortal example's save file)
src/lib.rs            crate docs + public API: Canvas, Context, Process, Config,
                      run / run_configured, draw_scene, paint_order, expand_text,
                      gamepad re-exports (gilrs Axis / Button / Gamepad)
src/objects.rs        Color, Transform, Shape, Light, ParticleShape,
                      SpriteFilter, Node, SceneNode, Layer, Scene, NodePath,
                      world_at / camera_world
src/tween.rs          Tween<T> (f32 / [f32;2]) with Repeat modes
src/particles.rs      Particle, ParticleSystem (pure simulation)
src/collision.rs      OrientedBox, Circle, Collider, Convex, push_out, reflect
                      (pure math; Convex is the occluder's polygon, half-planes only)
src/physics.rs        Body (Collider + velocity + weight), resolve_pair,
                      resolve_static, kinetic_energy (elastic, pure math)
src/bake.rs           Feature, Role, Material -> Baker -> Baked (authored geometry)
src/shapes.rs         authored-shapes reader: the sidecar `shapes:` vocabulary,
                      one pass over the parsed ron tree, both arms —
                      read_shapes hands the entries back in the texture's
                      pixels (what sprite_util draws over the sprite),
                      authored_shapes hands Features for a Baker to eat in
                      the node's space; bake_sidecar is the load-time seam:
                      a sprite's <name>.ron found beside its png, parsed,
                      read and baked once, the load refused loudly on any
                      defect — the pixel-to-node flip, every error returned
                      not logged
src/ron.rs            hand-written RON subset: Val/Item tree, parse_doc /
                      to_text_doc, lossless by preservation (comments and
                      verbatim atoms ride through); shared by ron_view,
                      sprite_util and tilemap — namespaced, not globbed
src/rng.rs            Rng — seedable splitmix64, state get/set for save games
src/diagnostics.rs    Diagnostics (flag-gated FPS/FT/PROC/DRAW HUD overlay
                      with folded strip charts), DiagnosticsFlags
src/ui/               Ui immediate-mode widget layer split by concern:
                      mod.rs (Ui: begin, button, checkbox, slider, label,
                      space, table), panel.rs (draggable folding panels),
                      table.rs (grid columns), tree.rs (two-axis scrollable
                      tree-view widget), style.rs — re-exports Ui, UiStyle
                      and the TreeLine/TreeEvent/TreeSpec/TreeState/
                      TreeStyle/TreeOut tree-view API
src/audio.rs          Audio (device + loop player), Sound (decoded buffer),
                      AudioError — native-only, rodio-based
src/text.rs           CPU text shaping/rasterization on swash, glyph shelf atlas
src/shaders.rs        include_str! of the 7 WGSL sources + naga layout tests
src/backend/mod.rs    `app`, `frame`, `wasm` (wasm32 only), `tests` (test only)
src/backend/app.rs    Frost<P>: winit app handler + GPU state + render loop
src/backend/frame.rs  Draw list model: Draw enum, scissor rects, uniform writers
src/backend/wasm.rs   WebFrost: deferred async GPU setup + fallback DOM helpers
src/backend/tests/    Backend tests, one file per area (uniforms, scissor,
                      nodes, layers, camera, repeat, context, config, …) —
                      all GPU-free but the few that render for real
                      (sprites, tilemap, shadows) and say so when they can't
shaders/*.wgsl        nine pipelines (line, polyline, circle, rectangle, shape,
                      sprite, particles, tilemap, blit) plus lighting.wgsl, the
                      lighting vocabulary the four lit ones are built on
examples/             40 runnable demos, filed into folders (see table below)
assets/               sprites/*.png (+ .pxo sidecars; immortal art, worm
                      crops, …), fonts/ (FiraCode Variable, JameGem08,
                      Leofont + licenses), audio/*.wav (swoof, waterflow,
                      bug/wasp clips, …), sprites/plant1..5.ron (the
                      immortal plant's anchor files), ron/garden.ron
                      (ron_view)
tests/                integration tests over the dogs_name flipbook art
                      (headless Shape::sprite reads: frame fit, ink stats)
```

The library uses only the `log` facade; consumers wire up their own logger
(`env_logger` in the examples). No randomness crate either: `frost::Rng`
(`src/rng.rs`) is a splitmix64 — pure 64-bit arithmetic, identical on every
target; `Rng::new()` seeds from the clock, `Rng::with_seed(seed)` fixes the
stream for reproducible runs and tests, and `state()` / `set_state()` let a
save file carry the stream across (`immortal` does exactly that).

## The public API (src/lib.rs)

- `frost::run(scene: Scene, process: P) -> Result<(), ...>` — the entry point.
  `run_configured` takes a `Config { vsync: bool, window_size: Option<[u32;2]>,
  window_size_px: Option<[u32;2]> }` (vsync default on, window size default
  platform) to turn vsync off for uncapped frame rates or open the window at
  a specific inner size — `window_size` in logical pixels, `window_size_px`
  in **physical** pixels and winning over it. The engine's user space is
  physical, so `window_size_px` pins an identical design space on every
  display whatever scaling the OS applies (this is why `immortal`,
  `dogs_name` and `centipede` open at a true 1920×1080 on all four dev
  machines); it is ignored on web, where CSS sizes the canvas. The user can
  still resize afterwards. Both functions exist for native and wasm32.
- `Process` — `fn process(&mut self, ctx: &mut Context, dt: f32)`; `dt` is
  seconds since the previous frame (`0.0` on the first, clamped to 1.0s).
  **Any `FnMut(&mut Context, f32)` closure is a `Process`.**
- `Context` — derefs to the frame's `Canvas` (immediate draws), and:
  - `scene() -> &mut Scene` — mutable access to the scene passed to `run`;
    mutate it here to animate it; it is drawn after the process returns.
  - `key_down(KeyCode) -> bool` — held physical keys (scancodes, layout-free).
  - `char_down(char) -> bool` — whether a held key typed that character when
    it went down, under the user's keyboard layout: the layout-following
    twin of `key_down` (a physical code names a US-ANSI position — the
    Swedish `+` key reports the code US calls `Minus`; `char_down('+')`
    finds the `+` key on any layout). The character carries the modifiers
    held at press (a macOS Option rewrites them); a release removes by
    physical code, so held state never sticks.
  - `mouse_position() -> Option<[f32;2]>` — user coords, `None` when the cursor
    is outside the window (last known position is deliberately dropped).
  - `mouse_button_down(MouseButton) -> bool`
  - `mouse_wheel() -> f32` — vertical wheel movement since the previous
    frame, in lines (positive = up). It sums every wheel event of the frame
    (`LineDelta` as-is, `PixelDelta` at 120 px/line) and is reset after the
    frame's `process`, so an unread frame discards its delta.
  - `window() -> Option<&Window>` — the engine's winit `Window` (`None` only
    before the first frame exists): parent native dialogs to it (e.g.
    `rfd::FileDialog::set_parent`, as `sprite_util` does) and read any window
    state the platform exposes — `inner_size()` / `scale_factor()` for
    pixel-exact math (the `worm` example wraps at the window's *physical*
    edges converted through the scale factor).
  - `gamepads() -> impl Iterator<Item = Gamepad>` — the connected
    controllers (gilrs): each reports its pressed buttons and axis values,
    updated with the frame's events; empty when none is connected or the
    platform's input devices could not be opened at startup. `Axis`,
    `Button`, and `Gamepad` are re-exported from gilrs.
  - `expected_fps() -> Option<f32>` — monitor refresh rate when vsync is on.
  - `frame_processing_ms() -> f64` — the engine's CPU time for the last
    completed frame (its own probe; see "Diagnostics" below).
  - `frame_draw_calls() -> u32` — the engine's GPU draw calls for the last
    completed frame (its own probe; see "Diagnostics" below).
  - `frame_diagnostic_draw_calls() -> u32` — the subset of those spent on
    the `Diagnostics` overlay's own nodes (its own probe; see "Diagnostics"
    below).
  - `size() -> (f32, f32)` — window size in pixels.
- Re-exports: `KeyCode`, `MouseButton`, `Axis`/`Button`/`Gamepad` (gilrs),
  `Rng`, `Tween`/`Repeat`, `ParticleSystem`/`Particle`, `Ui`/`UiStyle`, and
  everything from `objects` (`Scene`, `SceneNode`, `Node`, `Shape`, `Light`,
  `Color`, `Transform`, `Layer`, `Canvas`).
- Native only (no wasm32): `Audio`, `Sound`, `AudioError` (see
  "Audio (native only)" below).

**Coordinate system.** User space is in **pixels**, origin at the **window
center**, **y pointing up**: top-left = `(-w/2, h/2)`. The canvas works in
physical pixels internally; the only conversion is the cursor flip done when
building the `Context` (physical y-down → user y-up).

### Frame flow (the whole library in one pass)

`Frost::render()` (src/backend/app.rs) is the heart:

1. Acquire the surface texture (`Timeout` → log and skip the frame).
2. Build a fresh `Canvas` for the pixel size; compute `dt` from a monotonic
   millisecond clock (`Instant` native, `performance.now()` wasm), clamped to
   1.0s.
3. Build the `Context` (canvas + scene + held keys + typed characters +
   mouse state) and call `process.process(&mut ctx, dt)`.
4. `scene.visit()` — every node's `Node::process`, **children before parent**
   (post-order), on the root tree and every layer tree.
5. `canvas.draw_scene(&scene)` — walks each group depth-first, composing
   transforms/modulates/orders, and records `Draw`s (see below).
6. `canvas.expand_text(&mut text_atlases)` — lays out every `Draw::Text` and
   splices one `Draw::Sprite` per glyph **in place**, so glyphs keep their
   node's position in the paint order.
7. `canvas.paint_order()` — merges groups (base group first by implicit order
   0.0, layers by `Layer::order`, higher on top; stable) and **stable-sorts
   each group by `z`** (lower first; ties keep call order, so the last drawn is
   on top).
8. One render pass for the whole frame: clear with the frame's background color,
   then per draw: set the scissor to the draw's **tight bounding box**, build a
   fresh uniform buffer + bind group, draw a full-screen triangle.
9. `queue.submit`, `queue.present`. `RedrawRequested` re-requests the next
   frame, so animation runs continuously.

## The scene graph (src/objects.rs)

- `Scene { root: SceneNode, layers: Vec<Layer>, camera: Option<NodePath>,
  ambient: Color }` — `ambient` is the light floor every lit receiver adds
  under itself (a dim neutral gray by default; see "Lighting" below).
  `Scene::visit()` updates the root subtree and every layer subtree.
- `SceneNode` — a `transform` + `scale` + `modulate` + `order`, an optional
  `shape`, and `children`. All four are **relative to the parent**. Five
  per-node lighting flags (`lit`, `occludes`, `emission`, `glow`, `additive`)
  join them; unlike these four they apply to the node's own shape only and never
  propagate (see "Lighting" below):
  - transform/scale: `local = scale(node.scale).compose(&node.transform)`,
    `world = local.compose(&parent)`. **Scale is innermost** — it applies to the
    node's own shape *and* composes onto descendants.
  - `modulate` multiplies channel-wise into the node's shape color and composes
    onto descendants' (a descendant's color is the product of modulates on its
    root-to-leaf path). `Color` fields are 0..=1; `WHITE` is the identity.
  - `order` adds to the subtree's draw `z` (inherited `order + node.order`).
  - A node without a shape is a pure group/pivot; `..SceneNode::default()`
    fills identity transform, no scale, white modulate, order 0.
- `Node` trait — `visit()` (post-order walk) and `process()` (default no-op),
  so custom node types can participate; `SceneNode` implements it.
- `Layer { order, speed, root }` — a **hard draw partition** painted as a whole
  at `order` (its internal `z` ordering only matters inside the layer), with a
  parallax `speed` under a camera. `Layer::repeat` tiles the layer's content
  with a period ≥ the window size, splitting boundary-crossing objects into
  wrapping slices (drawing the same object twice aborts with an error).
- `camera: Option<NodePath>` — when set, every group is drawn from the camera
  node's coordinate space: the scene is transformed by the **inverse of the
  camera's world transform**, with the translation scaled per group (base 1.0,
  each layer by its `speed`) — that's the parallax: higher speed reads as
  closer. `NodePath` names a node by (group index?, child indices).
  `Scene::world_at(path)` / `Scene::camera_world()` compute the composed world
  transform (`pub(crate)` `world_at` / `camera_world` in `src/objects.rs`).
- `Transform` — `p' = m * p + t` with rows `m[0]`, `m[1]`; `rotate(angle)` is
  CCW from +x toward +y; `a.compose(&b)` means **a first, then b**
  (matrix `b.m * a.m`); `invert()` for the render path; `scales()` for the AA
  band.
- `Shape` (attached to a node, drawn in the node's local space before its
  children, so parents paint under descendants):
  - `Circle { center, radius, color }`, `Rectangle { center, extent (half
    widths), color }` — the node's world transform applies.
  - `Polyline { points, width, color }` — straight segments through the
    points (local space), stroked at `width`, one draw call for the whole
    chain (max 128 points; fewer than two draws nothing). Width scales with
    the node's geometric-mean scale axis.
  - `Particles { system, color, shape: ParticleShape }` — an instanced
    particle batch (one draw per batch; the game updates `system` each
    frame). `ParticleShape` picks the primitive each instance draws:
    `Circle` (radius = the particle's size), `Rectangle { aspect }`
    (`size × 2` wide, `size × aspect × 2` tall), or a sprite. Each
    `Particle` carries its own `angle`, so instances rotate individually —
    spinning squares just advance it (see
    `examples/particles/square_fountains.rs`).
  - `Background { color }` — fills the window, **ignores all transforms**,
    drawn at the very back: at render time it becomes the frame's clear color
    (the last one in depth-first call order wins; default is a dark blue-gray
    when none exists). It costs no draw call.
  - `Sprite { data, width, height, color, alpha, filter }` — RGBA8 PNG from
    `Shape::sprite(path)` / `sprite_bytes(bytes)` / `sprite_nearest` /
    `sprite_bytes_nearest` (decoded eagerly; `SpriteError` on failure).
    `filter: SpriteFilter::{Linear, Nearest}` picks the GPU sampler —
    Nearest keeps pixel art crisp when scaled.
    Centered on the node origin, 1 texture pixel per scene pixel. `color` tints
    every pixel (white = unchanged); overall opacity = texture alpha × `alpha`
    × `color.a`.
  - `Light { light: Light }` — contributes the node's [`Light`] to the
    frame's light field; never drawn itself (see "Lighting" below).
  - `Text { text, font: Arc<[u8]>, size, weight, color, alpha }` — from
    `Shape::text`/`text_bytes` (swash, CPU-only), thickened with
    `Shape::with_weight` (the `wght` variation axis; 400 Regular, 700 bold —
    inert on static fonts). Centered on the node origin;
    each glyph becomes a tinted quad over a shared atlas.
  - Color model: plain shapes emit `coverage * a`; sprites/text emit
    `texture_alpha * opacity * a` — all composited with `ALPHA_BLENDING`.

## Lighting (src/objects.rs + the backend)

A per-pixel light field, evaluated inside the *existing* shape/sprite/particle
pipelines — no extra pass, pipeline, or draw call, and skipped light draws
never count (`Diagnostics` excludes them like backgrounds).

- A `Shape::Light` node contributes a `Light { color, intensity, radius,
  direction, spread, softness, penumbra }` at the node's local origin: the
  node's transform (and ancestors') places it, rotates its beam, and tints it
  through the composed modulate. Three kinds: **omni** (`spread ≥ TAU`, the
  default — lights every pixel within `radius`, fading to zero at the edge),
  **cone** (narrower `spread` around `direction`, edges feathered by
  `softness` — a torch), and **directional** (`Light::directional`, marked by
  a negative `radius` sentinel: parallel rays, no falloff, position ignored —
  a sun; a scene may carry any number). `penumbra` is the light disk's radius:
  shadow rays sample it (nine taps, rotated per pixel), so shadows fade
  across an edge instead of dropping hard.
- The tilemap was the one surface pipeline kept out of the vocabulary —
  *scenery, not a scene actor*, its doc said — until a demo wanted a
  shadow to fall on a tile floor and exposed what that phrase had been
  silently conceding: the floor is exactly where a shadow falls. Maps
  receive like everything else now: `SceneNode::lit` multiplies the
  light mix per tile pixel and occluders cast across them. Pinned on
  the GPU, not asserted in prose — the test paints one white map under
  one lamp behind one declared wall and reads the bytes: bright on the
  lamp's side, dark behind the wall, full brightness when the node
  stays unlit. The flag is no tax; an unlit map draws exactly as it
  always did, demo for demo.
- **Vocabulary, because the word was wrong once.** `SceneNode::emission` is a
  surface's own light. **Bloom** is the renderer spilling bright pixels onto
  their neighbours — an effect of the frame, not a property of a node — and
  frost does not have it yet. The field was `SceneNode::glow` until
  2026-10-15: a name that promised the second thing and delivered the first,
  renamed the week bloom was scheduled rather than after it landed, so nobody
  wrote code on top of the ambiguity.
- Receivers opt in per node, flag applies to that shape only:
  `SceneNode::lit` multiplies the shape's color by `ambient + Σ lights +
  emission` per pixel; `SceneNode::emission` (a `Color`, alpha-scaled) is the
  shape's own emission — visible in the dark, whether or not any light reaches
  it, never shadowed, never spilling onto neighbors (attach a `Shape::Light`
  child for that); `SceneNode::occludes`
  makes a shape's transformed silhouette block every light's path to pixels
  behind it — a shadow caster. A wall can be `lit + occludes` at once. A
  painted rectangle reaches the field as itself and a painted circle as the
  sixteen-gon inscribed in it, because a curve still has to be told to
  hardware as flat edges; `Shape::Sprite` has no silhouette to pack at all, so
  anything else that must cast, a sprite body, one tile of a map, a physics hit
  box, declares itself with
  `Canvas::occluder(world, center, half)`: a box in its own space plus a world
  transform, joining the same field after the painted occluders. Both channels
  keep the shape local, which is why a non-uniformly scaled caster still throws
  square-cornered shadows — the shader pulls the pixel into the shape's space
  rather than pushing a sheared box out. A shape that is neither a rectangle nor
  aligned in its own space declares with `Canvas::occluder_polygon(world,
  &Convex)` instead, and a *disc* occludes that way too: a curved edge has to be
  told to hardware as flat ones, so a disc ships the sixteen-gon inscribed in it
  (its corners on the circle, within two percent of its radius) while physics
  keeps the round `Circle` and its exact normal.
- The backend packs the frame into two **storage buffers** bound to every
  draw: the light field (32-byte header — count and the scene's ambient —
  plus one 64-byte record per light) and the occluder field (16-byte header
  plus a 48-byte record per occluder, each followed by one 16-byte half-plane
  per edge of a polygon). Both start at their binding minimum and grow
  only to the peak count, so a lighter frame writes a shorter slice into the
  same buffer; the shaders read ambient and the lists from these fields.
- **Light layers.** A `LightLayers(u32)` set — 32 named layers, plain
  bitmasks — filters *who lights whom* without touching draw order (that
  remains `Layer` and `SceneNode::order`'s). Three ends wear a set: a
  `Light::layers` is the set it shines on, a `SceneNode::lit_layers` is the
  set a lit surface listens to, and an occluder's set — `SceneNode::occludes_layers`
  for a painted shape, `Canvas::occluder_layers` /
  `Canvas::occluder_polygon_layers` for a declared one — answers the
  receiver's question, not the light's. A light reaches a receiver when
  `light.layers.intersects(receiver.lit_layers)`; and — the half that looks
  asymmetric until you picture it — an occluder's shadow lands on a receiver
  when `occluder.layers.intersects(receiver.lit_layers)`, *regardless of
  which light the shadow is cut from*. The filter asks "whose surfaces do I
  shadow", so one wall can shade the far background it stands in front of
  while the tilemap floor it stands on, lit from a different layer, never
  notices it. Ambient belongs to no layer and always applies.
  Every set defaults to `LightLayers::default()` — the set holding only
  layer 0 — and every light constructor seeds layer 0, so a scene that
  names no layers is one layer-0 world and behaves exactly as it did before
  layers existed; the antifrost far background is the first real user,
  shadowed by the tiles' declared occluders on a layer the tiles themselves
  do not listen on. The wire carries the sets as raw bits through
  `f32::from_bits`, read back with `bitcast<u32>` — a mask is never a float
  to do arithmetic on, and every set from `NONE` to `ALL` round-trips exact:
  the light record grew to four vec4s whose last is `(layers, 0, 0, 0)`,
  the occluder record's second vec4 took its zero padding slot —
  `(hx, hy, planes, layers)` — and each receiver's uniform gained one
  `lit_layers` f32 in padding it already had (`shape` @76, `sprite` @56,
  `particles` @44, `tilemap` @12; no struct grew a byte).

## The draw list (src/backend/frame.rs)

`Canvas` holds `draws` (the **base group**: immediate methods + every scene's
root subtree, mixed) plus one list per explicit layer (`layer_draws`,
`layer_orders`).

- Immediate methods: `line`, `circle`, `rectangle` (pixel args converted to
  pixel space via `user_to_pixels`), `text` (records a `Draw::Text` with a
  translate-only user-space transform, expanded by `expand_text` like a
  node's text), and `draw_scene`.
- `Draw` enum: `Line`, `Polyline` (the batched line: `points: Vec<[f32; 2]>`
  in pixel space, `width`, `color` — consecutive pairs joined by straight
  segments, stroked by one full-screen-triangle draw against a fixed
  128-point uniform array — the points are packed as vec4s (x, y, 0, 0)
  because the uniform address space requires an array stride that is a
  multiple of 16 bytes; fewer than two points draws nothing, more than 128
  draw the first 128), `Circle`, `Rectangle`, `Shape` (transformed SDF
  circle/rectangle: `world`, `center`, `params` (circle `[r,0]`, rect `[hx,hy]`),
  `kind` 0/1, `aa`, `color`), `Sprite` (`world`, `data: Arc<[u8]>`, `size`,
  `texture_size`, `aa`, `tint`, `alpha`, `uv_rect`, `z`), `Text` (expanded
  before rendering — never reaches the render loop), `Background` (becomes the
  clear color — never drawn).
- **Scissoring**: every draw's fragment work is clipped to its tight pixel
  bounding box (`scissor_rect`, including the AA band), so **GPU cost scales
  with on-screen object area, not window size**. Rotated shapes/sprites get the
  AABB of their four transformed corners (`aabb_of_box`).
- **Anti-aliasing**: `AA_BAND = 0.75` px on the CPU; the same constant appears
  as `aa` in the shader sources — keep them in sync. Under node scaling the
  band is divided by the transform's scale so it stays a constant screen width.
- **Uniform writers** (`line_uniform_data`, `polyline_uniform_data`,
  `circle_uniform_data`, `rect_uniform_data`, `shape_uniform_data`,
  `sprite_uniform_data`) hand-write
  little-endian bytes at explicit offsets matching the WGSL **uniform-space**
  layout (e.g. ShapeUniforms: `to_local` mat2x2 @0 (16B, 8-aligned columns),
  `translation` @16, `center` @24, `params` @32, `color` vec4 @48, `misc` @64;
  total 80. Sprite the same with `size`/tint/`alpha`/`uv_rect`). Line/circle
  are 48 bytes, rectangle 32. The polyline writer packs 2080 bytes:
  `points` @0 (128 × vec4 — 2048 bytes), `color` @2048, `count` @2064 (u32),
  `width` @2068.

## The GPU side (src/backend/app.rs, shaders/)

- `Frost<P>` holds the wgpu instance/adapter/device/queue, `vsync`,
  `window_size` and `window_size_px` (the `Config` values), the window
  (`Arc<Window>`), logical
  size + scale factor, the surface, the seven pipelines
  (line/polyline/circle/rect/shape/sprite/particles), `sprite_resources` (a
  `HashMap<(u64, u64), (TextureView, Sampler)>` keyed by the sprite pixel-data
  `Arc`'s pointer plus its generation — `0` for static images, bumped by the
  glyph atlas on every repack — so a freed-and-recycled buffer address can
  never hit a stale texture; sprites from the same file, and glyph quads from
  the same atlas, **share one uploaded texture**), `text_atlases`
  (`HashMap<(font ptr, size bits), text::Atlas>`, kept between frames so
  unchanged text never re-rasterizes), held keys/mouse, and the user's
  `process` + `scene`.
- All seven pipelines are the same shape: a **full-screen triangle** vertex
  shader (no vertex buffer), one `@binding(0)` uniform (plus texture + sampler
  for sprites, plus an instance buffer for particles), alpha blending, no
  depth. `present_mode_for(vsync)` maps to `AutoVsync`/`AutoNoVsync`.
- **One uniform arena per frame.** Every uniform the frame draws with is
  packed before the render pass by `frame::uniform_bytes` into aligned runs of
  one grow-only buffer, uploaded in a single `queue.write_buffer`, and reaches
  its draw as a dynamic offset into that pipeline's bind group — of which the
  frame builds one per image and pipeline, not one per draw. The old rule it
  replaces was a real one: `write_buffer` copies all land *before any draw
  executes*, so a buffer shared at one offset would make every draw read the
  last parameters. Sharing it in *runs* is what makes the fact survivable.
  Particle batches and tile maps keep their per-draw buffers, because the bulk
  of what they pass is a storage buffer of instance data, not a uniform.
  Measured: 7.7 µs of CPU per draw per-draw against 0.31 µs with the arena, on
  an Apple M4 Max ([`docs/benchmarking.md`](docs/benchmarking.md) §6).
- **Frames render into a float buffer where the machine offers one.** The
  scene writes linear light, and an 8-bit sRGB buffer rounds it on the way in
  and clips whatever the lights summed past 1.0 — a lamp's core, two lights
  landing on one pixel, an emission pushed past white. `float_target` asks the
  adapter once whether `Rgba16Float` can be an attachment *and* a blit source;
  where it can, every scene pipeline is built against that format and the
  frame lands in a float buffer that the blit pass carries to the window, so
  the 8-bit clamp happens once, at the end, instead of at every store. Only the
  blit pipeline is built against the surface format — the two disagreeing is a
  validation error on every frame, which is why the question gets asked in one
  function. **It changes nothing you can see**, and that was measured rather
  than argued: the two buffers agree on every pixel of a plain blend and move
  at most one 8-bit step on the pixels under twenty stacked washes (the
  `frame_buffer` test), and the 4K demo's drained encode span is 1.044 ms
  against 1.042 ms (§7 of the benchmarking notes). What the buffer buys is the
  right to *keep* a value above 1.0 — worth something to a bloom bright-pass or
  a tone map, and nothing at all until something reads it.
- **The scene writes what it emits into a second target.** Where the frame
  buffer can hold a value past 1.0, every scene pipeline is built with two
  color outputs — the shaders' `fs_emissive` entry point instead of `fs_main`,
  chosen by the same `Option` that builds the texture, because a pipeline
  declaring one target cannot draw into a pass offering two (measured; wgpu
  rejects the mismatch). The composited pixel goes to the first, the light the
  surface puts out to the second, and both are alpha-blended on the same terms
  by one rule the shaders state once: **rgb is radiance, alpha is coverage, the
  blend does the rest.** That is what makes an opaque shape drawn over a lamp
  take the lamp's glow away, a translucent one only dim it, and a fragment that
  covers nothing change nothing. One pipeline answers differently, and it is the
  only variant the renderer carries: the sprite pipeline's additive twin, chosen
  for a node flagged `additive`. Its rgb is added, scaled by its own alpha, and
  its coverage adds too, so an authored glow sprite deposits its falloff instead
  of cutting a bell-shaped hole in the light behind it. Nothing about the two
  channels changes and the shader cannot tell which pipeline it was drawn with —
  only the arithmetic the blend performs differs.
  Shapes and sprites carry emission; lines,
  rectangles, particles and tile maps write zero radiance at their own coverage
  — which is what makes them occlude glow rather than punch holes in it.
  Emission does two jobs at once — it is the surface's own color, and it is what
  the chain reads — so a lamp's core and its halo have always arrived in the
  same hue. `SceneNode::glow` takes the second job away from it: set, the
  emissive attachment carries that color instead and the surface keeps its own,
  which is how a filament reads white while the room it glows into goes amber.
  Unset — every scene in the tree, and every draw the tests build — the two
  fields hold the same bytes and the frame is what it was, bit for bit.
  It costs about 1% of a 4K frame (§8 of the benchmarking notes) and is pinned
  by the `emission` test — the number it holds is `color × glow`, with no light
  field in it, and still unclamped at three times white.
- **Bloom is what reads it, when the app asks.** `Config::bloom` (0.0, off)
  thresholds the emissive target with a soft knee, folds it into four levels
  each half the last, blurs each across and down with a thirteen-tap Gaussian
  folded into six bilinear fetches, and hands the four to the blit pass, which
  adds them — weighted 0.40 / 0.28 / 0.20 / 0.12, scaled by strength — in linear space,
  before the encode to the window's format. Two things the tests hold onto: a
  frame with no emission comes back *bit-identical*, because it adds exactly
  zero at every pixel, and the glow falls off (measured eight to one between
  five pixels out and twenty-one), because a glow that is uniform across the
  frame is a haze, not a glow. The chain was nine render passes and costs
  **+0.21 ms** at antifrost's default 1920×1080 buffer, **+0.5 ms** at a
  4112×2572 one: part of it is overhead a pass carries whatever it draws into —
  six more passes over the same texels measured +0.10 ms — and part is fill,
  which is why the kernel is wide and iterated once instead of narrow and run
  twice (§9 of the benchmarking notes, and §11 for the correction this session
  made to how the size half of that was measured). The chain has since grown a
  fourth level, which is twelve passes and about a third more chain than was
  measured — call it +0.07 ms at 1920×1080 by the same per-pass law, though
  that one is arithmetic on a measurement rather than a measurement; §12 has
  the reason levels are the width control and what the fourth did and did not
  change. No look is decided by any of this: the threshold, the knee, the four
  weights and the level count are four numbers to argue with, all of them at the
  top of `bloom.rs`, and every one of them is a guess today.
  `Context::set_bloom` is the live version of the strength, and the strength is
  what decides whether the chain runs at all: at zero the present pass already
  ignores the levels, so `Bloom::run` declines and the off state costs no passes
  — an app may toggle a glow per frame. Asking for a glow on a frame that was
  built without a chain builds one there and then, for about 0.1 ms of texture
  and bind-group bookkeeping that does not grow with the window (§11).
- **The tone map folds light above white into the display's range, and nothing
  else.** `Config::tone_map` (0.0, off) is a *headroom*: the share of the display
  range handed to light above white. Below the knee at `1 - headroom` the curve
  is the identity with matching slope, and that choice is the whole design — it
  means no tone map in the engine can change how an ordinary pixel looks, and the
  test asserts exactly that as bit-equality rather than tolerance. Above the knee
  the curve is rational, approaching white as 1/x, evaluated on the pixel's
  brightest channel and scaled back over all three so a light keeps its colour on
  the way in. (The exponential shoulder it was written instead of reaches white
  so fast that a lamp at 2.0 and a lamp at 20.0 arrive as the same pixel, and the
  difference between those two is most of what an over-bright frame has to say.)
  What it buys and what it charges are the same number: at a headroom of 0.15
  display white itself lands at 0.925, and everything from 1.0 to 16 of radiance
  arrives inside the last three percent of the display's range — nothing clips,
  hue survives, order survives, and the separation between bright things is paid
  for out of the room they were given. A filmic curve buys that separation the
  other way round, by remapping the whole tone scale and with it the whole look,
  which is a look decision and is not made here. It runs last in the present
  pass, after the bloom is added, so the total is what fits the display; its cost
  is the one number in this whole area that cannot be measured — +0.007 ms against
  a ±0.01 ms drift, and a paired per-frame instrument that is blind to it at three
  buffer sizes too (§10, §11 of the benchmarking notes).
  `Context::set_tone_map` is the live version, and being one float in a uniform
  the present pass rewrites every frame, changing it costs nothing and lands the
  frame it is asked on — which is what lets an app pick a headroom per shot rather
  than per program. It is still one number for the whole window: there is no
  per-layer fold, because the fold happens once over the composed frame, and
  keeping an interface out of it would need a subtree that renders into a buffer
  of its own, which the scene tree cannot express today.
- Sprite textures are `Rgba8UnormSrgb` (samples land in linear space; sRGB
  blending matches authored colors), linear-filtered, clamp-to-edge.
- The shape/sprite shaders evaluate an **SDF in the object's local space**: the
  fragment's pixel coordinate is transformed by the inverse world transform
  (uploaded as `to_local` + `translation`; a non-invertible transform is
  skipped), the signed distance is measured in local units, and
  `smoothstep(-aa, aa, d)` gives the coverage. Rotation and scaling "apply for
  free".
- Events (native): keyboard held-set and per-key typed characters (Escape
  exits), cursor position, mouse
  buttons, mouse wheel (accumulated in lines per frame, reset after the
  frame's `process`), `CursorLeft`/`Focused(false)` **clear** the held state
  and the typed characters (no stuck controls), resize/scale-factor
  reconfigure the surface (pipelines
  rebuild only if the format changed), `RedrawRequested` → `render()` +
  re-request.
  The window opens at the `Config`'s `window_size_px`/`window_size` (or the
  platform default),
  centered on the monitor, and requests its first redraw explicitly (Wayland
  won't deliver one otherwise).
- `block_on` (native only) drives the adapter/device futures with a no-op
  waker — the browser can't block on them, hence `WebFrost`.

### wasm32 (src/backend/wasm.rs)

In the browser `request_adapter`/`request_device` are JS promises that only
resolve once the main thread is free, so `run` defers GPU setup: `WebFrost`
creates the canvas immediately (`Config::window_size` in CSS pixels, 900×600
default — `window_size_px` is native-only), appended to the page), keeps the
scene/process in a `Core`, and on every `resumed` polls the in-flight setup
future (`GpuInit::{Idle, Init, Failed}`); when it completes it hands everything
to a real `Frost`. `hide_fallback`/`show_fallback` swap the page's "Loading
frost…" placeholder with the first frame or a red error message.

## Text (src/text.rs)

CPU-only, built on **swash** (pure-Rust Fontation shaping/rasterization):

- `layout(font: &[u8], text, size, weight) -> Option<TextLayout>` shapes the string into
  `PlacedGlyph { id, x, y }` (pen origin at baseline left, y up) and returns
  metrics `{ ascent, descent, width, glyphs }`.
- Each unique glyph is rasterized **once per (font, size, weight)** into a shared 512px
  shelf atlas (`ATLAS_SIZE: u32 = 512`) — first-fit row packing; oversize
  glyphs are skipped but the cursor still advances. Output is RGBA8 with alpha
  replicated across channels.
- `Canvas::expand_text` (src/lib.rs) consumes the `Draw::Text` entries, lays
  them out, and replaces each with per-glyph `Draw::Sprite`s sampling the
  atlas sub-rectangle (`uv_rect`) — this is why sprite `uv_rect` exists.

## Tween (src/tween.rs)

`Tween<T: Tweenable>` for `f32` and `[f32;2]`: `new(from, to, duration)`
(default `Repeat::PingPong`), `repeat(Repeat::{Once, Loop, PingPong})`,
`tick(dt) -> T`. Durations are floored at 1e-6. **There is no re-target API** —
the established pattern (e.g. `examples/inout/cursor.rs`) is to **rebuild the
tween**
on the press/release edge with the current value as the new `from`.

## Particles (src/particles.rs)

`Particle { pos, vel, life, max_life, size, angle }`; `ParticleSystem { particles: Vec }`
with `spawn(p)` and `update(dt, gravity)` — semi-implicit Euler
(`vel += g*dt`, then `pos += vel*dt`), `life -= dt`, `retain(|p| p.life > 0.0)`.
Pure simulation, renderer-independent; **drawing is the caller's job**
(typically `ctx.circle` per particle, alpha ∝ remaining life). Emission is
accumulated in the caller (`acc += rate*dt; while acc >= 1.0 { acc -= 1.0; spawn }`)
so the spawn rate is independent of frame rate.

## Collision (src/collision.rs)

Pure math — no winit/wgpu, identical on wasm, no dependencies. Intentionally
**minimal by design**; the documented escape hatch when the game outgrows it is
`rapier2d` (the scene bridge and rendering would stay the same):

- `OrientedBox { center, half, angle }` (full size = 2×half, CCW angle) with
  `axis(i)` and `corners()` (CCW from lower-left); `Circle { center, radius }`;
  `Collider::{Box, Circle}`.
- `push_out(&other) -> Option<PushOut { dir, depth }>` — the minimum separating
  translation to resolve the overlap (SAT for boxes, direct for circles).
- Segment queries, the CPU twin of the shaders' `occluded` so gameplay and
  pixels answer the same question: `Collider::cast(from, to) -> Option<t>`
  (the entering `t` of the open segment) and `contains(p)`, plus the two
  policies over a field of shapes — `occluded(from, to, &[..])`, where a
  receiver *inside* an occluder is lit, and `visible(from, to, &[..])` for line
  of sight, where a shape holding either endpoint blocks — and `first_hit` for
  the index of the nearest. That inside-case split is the whole difference
  between the two policies.
- `reflect(vel, normal, restitution)` — kills the normal component (scaled by
  restitution), leaves the tangent; no-op for velocity already leaving the
  surface.
- The intended usage pattern: the game's `Process` owns all state, builds
  colliders from node transforms each frame, applies `push_out` + `reflect`,
  and writes the results back into the node transforms.

## Physics (src/physics.rs)

The collision module answers static questions and owns no state; physics
puts those shapes in motion. A `Body` is a `Collider` (which carries the
position, as it does for every shape) plus a velocity and a weight, and the
module's one contract is that contacts trade momentum without spending
energy.

- `Body::advance(dt)` integrates the center by the velocity — position is a
  `Collider::translate`, so the shape stays the shape.
- `resolve_pair(a, b)` stands two overlapping bodies apart along the
  `push_out` normal, splitting the penetration by inverse weight (the lighter
  does most of the moving), then reverses their approach with an equal-and-
  opposite impulse along that normal. The impulse is `e = 1` and touches only
  the normal component, so the closing speed becomes the separating speed:
  energy and momentum both hold. A pair that overlaps yet is already parting
  gets the positional fix only, never a second kick.
- `resolve_static(body, wall)` is the same with an immovable partner — it
  reduces exactly to `reflect(vel, dir, 1.0)`, so a wall takes nothing.
- `kinetic_energy` and `momentum` read the two ledgers; a test watches the
  first stay flat across a three-body round and every wall bounce.

It is deliberately small: discrete integration (a body moving farther than a
wall is thick in one frame passes through — `Collider::cast` is the tool for
that day), no angular momentum, friction, or gravity. The same
rapier2d-upgrade note from the collision section covers outgrowing it.

## Bake (src/bake.rs)

Authored geometry — a sprite's body, a tile's solid top, a hurt box — turns
into engine shapes here. Pure math like collision: no winit, no wgpu, no state,
nothing but data in and flat lists out.

- `Feature { role, shape, material }` is one authored fact, in the space of the
  sprite frame or atlas cell it describes: `Footprint::{rect, tilted, disc,
  poly}` for the shape, `Role::{Occlude, Solid, Hit}` for what it is for, and
  `Material { bounce }` — the `e` of `reflect` — for the roles that care.
- `Baker` takes a `Feature` list per `Placement` (a sprite frame at its own
  origin, one atlas cell per tile that uses it) and `finish()`es the whole set
  into `Baked`. Collecting before fusing is what lets a tile map bake at all:
  two tiles of one wall fuse because they end up adjacent in the map's space,
  which nothing knows while the cell is still one cell.
- In and out, everything stays in **the node's own space**. A baked set rides
  the node's `Transform`, one bake serves every instance, and a stretched or
  turned sprite still throws square-cornered shadows because nothing was baked
  into world coordinates and nothing has to approximate a sheared box.
- `Baked { occluders, disc_occluders, poly_occluders, solids, hits }` is one
  authored set seen per consumer,
  because the consumers want different things from the same shape. `occluders`
  are fused rectangles — the GPU reads one record per shape and tests every lit
  pixel against every record, so a floor of four thousand tiles is four thousand
  shadow tests per pixel per light, and fusing also retires the shared edge, the
  one seam a light ray can slip along. `solids` fuse too, but only within one
  material, so a crate touching a trampoline does not become one bouncing wall.
  `hits` are never fused, because the game has to answer *which box was
  struck*. `disc_occluders` are the circles authored to cast — kept circles,
  counted rather than merged, since no fusion of two circles is a circle anyone
  drew, and cut into the polygon the light field speaks only on the way out, so
  a bake never holds an approximation of a shape somebody authored.
  `poly_occluders` are the convex outlines authored to cast, and they need no
  last-moment work at all: the outline is already the edges the field speaks.
- A fusion is taken only when the result is still one rectangle covering
  exactly the points its parts covered: agree across one axis, touch or overlap
  along the other, repeated until a pass changes nothing. An L stays two boxes
  and a gap stays a gap, while a 4×5 block becomes one box. Boxes fuse only
  with others turned alike, and a turn within a hair of a quarter turn is
  snapped to it — which is what lets a tile laid down sideways, or turned three
  quarters, fuse with its upright neighbours. Away from a quarter turn "alike"
  means *the same number*: a turn reached by two different calculations costs an
  extra record and never a wrong box, because a fusion group takes one of its own
  members as its frame, and that is what makes the fusion exact rather than
  nearly so. Widening the grouping past a rounding error buys the records back
  and charges the difference to the edges.
- Nothing is dropped quietly. `add` returns `Rejection { index, role, why }`,
  `#[must_use]`, for geometry the bake cannot honour: a shape with no area, or a
  corner that is not a number. A painted sprite that silently cannot cast a
  shadow is exactly the failure this path exists to end, so it says so out loud
  instead of leaving a list that looks fine. What it *can* honour has grown: an
  occluding circle used to be a refusal, and is now the round list above.
- Two exits close the loop: `declare_occluders(canvas, world)` feeds the fused
  boxes to `Canvas::occluder` under the node's transform, each box's own turn
  folded into that transform so the shape stays local, and the round and
  polygonal lists to `Canvas::occluder_polygon` — the discs as `Convex::disc`,
  the authored outlines as drawn — the tessellation happens here,
  at the border and once, which is why nothing upstream of it is an
  approximation; and `occluder_colliders()` hands the same shapes — the discs
  still round, because asking the CPU about the polygon would fold the answer
  into the question — to `occluded` / `visible`, so the gameplay answer and the
  pixel answer come from one bake.
- Not here yet: capsules; convex polygons as `solids` or `hits`, which is a
  step of its own — they occlude already, but pushing two polygons apart needs
  the separating axis of a curved boundary and a contact normal worth the name,
  while occlusion only ever asks whether a segment crossed an edge; the reader
  that builds these features from the `.ron` sidecar which already rides
  each sprite has arrived as `src/shapes.rs`, whose vocabulary and constraints
  are written up in `docs/authored-shapes.md`; and a tile-grid broadphase for
  line queries over very large maps — fused strips keep a linear scan cheap
  enough until a real map says otherwise.

## Shaders (shaders/*.wgsl, src/shaders.rs)

Nine pipelines, all full-screen-triangle + SDF/sampling; seven are tabulated:

| file        | pipeline   | uniforms (byte layout)                                              |
|-------------|------------|---------------------------------------------------------------------|
| line.wgsl   | line       | `a`@0 `b`@8 `color`@16 `width`@32 → 48                             |
| polyline.wgsl | polyline | `points`@0 (128 × vec4, 2048B) `color`@2048 `count`@2064 (u32) `width`@2068 → 2080 |
| circle.wgsl | circle     | `center`@0 `color`@16 `radius`@32 → 48                             |
| rectangle.wgsl | rectangle | `center`@0 `extent`@8 `color`@16 → 32                            |
| shape.wgsl  | shape      | `to_local`@0 `translation`@16 `center`@24 `params`@32 `color`@48 `misc`@64 → 80 |
| sprite.wgsl | sprite     | `to_local`@0 `translation`@16 `size`@24 `tint`@32 `alpha`@48 `uv_rect`@64 → 80 |
| particles.wgsl | particles | `size`@0 `color`@16 `misc`@32 `lit`@40 → 48 (one instanced draw per batch) |

`lighting.wgsl` is in no table because it is not a pipeline: it is WGSL text
`src/shaders.rs` prepends to `shape.wgsl`, `sprite.wgsl` and `particles.wgsl`,
the three pipelines that read the frame's light and occluder fields. It holds
what those three had as byte-identical copies — the two field record types, the
`occluded` slab test, the shadow taps, the `light_mix` loop — and it declares no
`@group` or `@binding`, because those two storage bindings sit at different
indices per pipeline, so each host keeps its own `field` and `occluders`
declarations around it. The shared text therefore reads two names declared after
it, which WGSL allows: a module-scope name is in scope across the whole program.
`occluded` walks the occluder records with a running offset rather than a
stride, because a polygon's record is `3 + edges` `vec4`s and a box's is still
exactly three — the count riding in a component that was padding, so a field
packed before polygons still reads as boxes. The naga tests below run on the
*assembled* sources, and
`the_lighting_vocabulary_is_shared_not_copied` keeps a copy from creeping back
into a fourth shader, where it would compile happily and disagree with the rest.

The polyline's fragment shader walks its 128-point uniform array and takes the
minimum point-to-segment distance — a stroke of any length is still one draw
call; its scissor is the point bounding box plus half the stroke width and the
AA band, so the fragment loop only runs over the pixels the stroke can write.

`src/shaders.rs` includes them, hosts the watched-shader table `SOURCES`
and the runtime `validate` (naga's parse + validator, the same two stages
`create_shader_module` runs — the watching `Live` refuses a batch before the
GPU ever sees it), and has tests that **parse all nine with
`naga::front::wgsl::parse_str`** (the same frontend wgpu uses, so validity is
pinned without a GPU) and assert the `ShapeUniforms`/`SpriteUniforms` member
offsets are `[0,16,24,32,48,64]` spanning 80 bytes, the `ParticlesUniforms`
offsets are `[0,16,32,40]` spanning 48, and the `PolylineUniforms` offsets are
`[0,2048,2064,2068]` spanning 2080 with a fixed 128-point vec4 array — the **CPU
uniform writers must mirror these layouts**; the GPU-side mirror tests live in
`src/backend/tests/`. If you change a shader uniform struct, update the
writer and both test sides together.

## Audio (native only)

`src/audio.rs` plays sounds through **rodio** on the host's default output
device. The module is `#[cfg(not(target_arch = "wasm32"))]`-gated in
`lib.rs` (rodio's cpal device layer is native-only), so wasm builds never
compile it; `Audio`/`Sound`/`AudioError` are re-exported from the root only
on native targets.

- `Sound::load(path)` / `Sound::load_bytes(bytes)` — **load-time decode**:
  the file (WAV, FLAC, MP3, OGG Vorbis, AAC, M4A) is read and decoded once,
  into an in-memory `rodio::buffer::SamplesBuffer` of f32 samples, so a bad
  path or format is an `AudioError` up front, not at play time. The samples
  sit behind an `Arc`, so re-triggering a `Sound` shares one buffer and
  clones are cheap. Accessors: `channels()`, `sample_rate()`, `duration()`.
- `Audio::new()` — opens the default output device
  (`DeviceSinkBuilder::open_default_sink`), silences the drop log
  (`log_on_drop(false)`), and builds the single loop `Player` on the device's
  mixer. `Audio` owns the device; dropping it stops everything.
- `Audio::play_once(&sound)` — **one-shot, parallel overlap**: a copy of the
  buffer is `amplify`-ed by the master volume and `add`-ed to the mixer, so
  the same sound re-triggers freely and copies overlap. The volume is frozen
  at trigger time.
- `Audio::play_loop(&sound)` / `stop_loop()` — **loop, single sequential
  player**: `play_loop` `clear()`s, `append`s `buffer.repeat_infinite()`,
  `play()`s (a `clear` leaves the player paused, so the `play()` is
  load-bearing), and sets the loop's volume. Only one sound loops at a time;
  `play_loop` on a running loop restarts with the new sound. `stop_loop`
  `clear()`s. The master volume applies to the loop live.
- `Audio::set_volume(v)` — clamps to `0.0..=1.0` and stores the f32 bits in
  an `AtomicU32` (no cast needed: rodio's `Float`/`Sample` are `f32`); it
  updates the loop player immediately and one-shots from the next trigger.
- `AudioError` — hand-rolled `Debug` enum with a manual `Display`/`source()`
  in the project's `TextError`/`SpriteError` style: `Io(std::io::Error)`,
  `Decode(rodio::decoder::DecoderError)` (held by value; `Clone`), and
  `Device(Box<rodio::DeviceSinkError>)` (boxed because the device error is
  not `Clone`).

Tests are **device-free** (no `DeviceSinkBuilder`/`Audio::new()`), so
`cargo test` runs headless: a synthetic in-code PCM16 WAV decodes to the
right channel/rate/length, the bundled `swoof.wav` decodes, a missing path is
`AudioError::Io`, garbage bytes are `AudioError::Decode`, and the
`clamp01`/f32-bits helpers round-trip. `Sound::load_bytes` carries a doctest
that decodes an in-code WAV.

## Diagnostics (src/diagnostics.rs)

`Diagnostics` is a HUD overlay whose statistics are each gated by a bit of
the `DiagnosticsFlags` set passed to the constructor. The window size line
(`"{w}x{h}"`) is always shown, and below it the enabled statistics in fixed
order, all at a base 32 px that the display scale (Alt+'+' / Alt+'-', under
*Keyboard shortcuts*) grows or shrinks together: the smoothed frame rate (`"FPS {fps}"`, the `FPS`
bit), the current frame time (`"FT {ms}ms"`, the `FT` bit), the last
frame's total processing time (`"PROC {ms}ms"`, the `PROC` bit), that time
excluding the overlay's own update cost (`"APP {ms}ms"` — the companion line
to `PROC`, on the same bit), and the last frame's GPU draw-call count
(`"DRAW {n}"`, the `DRAW` bit). Under the lines, one scrolling ten-second
strip chart per enabled statistic group, top to bottom: frame rate (orange),
frame time (green), the folded processing-time chart, and the folded
draw-call chart. The two folded charts show three polylines on one shared
scale — the total (blue / magenta), the app's share (violet / pink), and
the overlay's own share (cyan / deep rose) — so the overlay's cost is
readable as the vertical gap between the total and the app lines; the FPS
and FT charts are single series, and the three time charts carry a
reference line at 60 fps / 16.7 ms (the draw-call chart has none — draw
counts have no universal budget). The processing times and the draw-call
counts are the engine's own probes; the APP lines and the diagnostic series
are the overlay subtracting its own measured costs from those probes — see
the probe bullets below. The whole integration is one struct in the demo
state plus one `self.diag.process(ctx, dt)` call per frame — the overlay
itself is a `Process`.

- `DiagnosticsFlags` is a hand-rolled four-bit set — `FPS (1 << 0)`,
  `FT (1 << 1)`, `PROC (1 << 2)`, `DRAW (1 << 3)` — with `ALL`, `NONE`,
  `none()` / `all()`, `contains()` / `is_empty()` / `insert()` /
  `remove()`, and `|` / `|=` (no dependency). `Diagnostics::new(path,
  flags)` reads the font file up front (checked with swash, so a bad path
  or file is a `TextError` at construction); `from_bytes(&[u8], flags)` is
  the no-filesystem twin (embedded `include_bytes!`, the wasm path). The
  bytes live behind an `Arc` shared with the readout's `Shape::Text`. The
  engine's probes always run; the flags gate only what the overlay
  displays, and an off flag costs nothing on the display side.
- `Process::process` smooths the frame rate as an exponential moving average
  of `1.0 / dt` (time constant 0.25 s, snapping to the first real
  measurement so the first frame does not flash a zero; `dt` 0.0 on the
  first frame is skipped; `dt` past a 0.25 s stall is skipped by the
  smoothing), records the raw frame time for the `FT` line, reads the
  engine's processing-time and draw-call probes (total and diagnostic),
  derives the `APP` line and the diagnostic series from the previous call
  (see the probe bullets), and appends a `Sample` (elapsed time, frame
  time, total/diagnostic/app processing times, total/diagnostic/app
  draw-call counts) to a history trimmed to the last 10 s. Each line's
  shape is rebuilt only when its text changes — the frame-time line shows
  the raw last gap, so it usually changes every frame.
- The processing-time probe lives in the backend, because the demo's
  `process` callback runs *inside* the frame — a demo cannot time its own
  frame from inside it. `Frost` measures `now_millis()` around its frame
  work in `render` (from just after the surface acquire — the acquire may
  wait on the display's back buffer, and that wait is not processing — to
  the `queue.submit` of the frame's command buffer; the present handoff and
  the wait for the next frame are excluded) and stores the result as
  `frame_processing_ms`, which the next frame's `Context` reports as
  `Context::frame_processing_ms()`. The draw-call probe counts in the same
  pass: one increment per `pass.draw` / `pass.draw_indexed` that actually
  executes (seven sites in the render loop — line, polyline, circle,
  rectangle, shape, sprite, particle batch), so draws skipped as off-screen,
  background, light, or empty (a degenerate transform, an empty particle
  batch) never count. It keeps two counters — `frame_draw_calls` and
  `frame_diagnostic_draw_calls`, reported as
  `Context::frame_draw_calls()` and `Context::frame_diagnostic_draw_calls()`
  — and the diagnostic one increments only when the draw's node carries the
  `SceneNode::diagnostic` marker, which the overlay sets on the nodes it
  appends to the scene. The marker is a public but engine-managed field
  (public so node literals with `..Default::default()` keep working from
  outside the crate; demo code leaves it at the default `false`), and it
  rides onto every draw the tagged nodes produce — the per-glyph quads of
  the text lines included — so the diagnostic count is a true subset of
  the total and the app's share is `total − diagnostic`. The overlay
  therefore always shows the last *completed* frame's numbers — a one-frame
  lag, one 0.1 s column at 60 fps, invisible in the graphs. `0.0` / `0`
  until the first frame completes (one `PROC 0.00ms` / `DRAW 0` flash, same
  as the `FT 0.0ms` flash).
- The APP value is the one probe the overlay makes itself, and it must be
  one frame behind for the same reason: it times its own `process` call
  with the engine's `now_millis` (crate-internal, shared with the backend)
  and stores the duration as `prev_self_ms`; the next call subtracts it from
  the engine's total probe of the *same* completed frame, clamped at zero
  against timer jitter, so PROC and APP always describe the same frame. It
  removes the overlay's per-frame *update* work (readout layout, sample
  binning, node placement); the overlay's own shape draws still count in
  the total — they go through the engine's per-draw path like everything
  else — which is what the diagnostic series on the folded charts shows.
- The strip charts are the history binned by time: `slice_max` takes the max
  of a per-sample projection over each 0.1 s column across the last 10 s
  (the fps column is the max of `1000 / frame_time`, i.e. the frame's
  fastest rate; the other columns project their own field), so a spike is
  visible as a high point, and the newest frame lands in the newest column.
  Each series' 100 column maxima become one `Shape::Polyline` (a point per
  column center, the value scaled to the panel height with a 1 px inset) —
  one draw call per series instead of 200 per-frame rectangle draws, because
  the cost of a draw in this engine is on the CPU side (a fresh uniform
  buffer + bind group + scissor/pipeline state per draw), not in the
  fragment work. Each folded chart keeps one scale: processing time is
  0–20 ms (the reference line at 16.7 ms), draw calls a fixed 0–128 (clipped
  above it — the readout always shows the true count).
- Node management: the slot lists are built once at construction from the
  flags — the lines in order (the size line first, then the enabled
  statistics), one panel per enabled chart, one reference line per chart
  except the draw-call chart, one polyline per series (one for the FPS and
  FT charts, three for the folded ones) — so all four flags give at most 21
  nodes (six lines, four panels, three references, eight polylines), and
  `none()` gives a single line. The first `process` creates a **dedicated
  topmost `Layer`** — ordered one past the highest existing layer (or above
  the root subtree when there are none) — and appends the nodes to that
  layer's root, remembering the indices; later calls update each node's
  shape and transform in place. A stale index (the demo rebuilt the scene's
  layers) recreates the layer and re-appends the nodes. Because a `Layer` is
  a hard draw partition, the overlay can never be buried under a
  high-order app node, and the demo is free to reorder its root's children.
  Placement uses crate-internal `text::layout` for each line's width; the
  lines' vertical extent is measured once, at construction, over every
  character a readout line can display, because neither the current text's
  ink (Fira Code's "8" and "9" carry a pixel of antialiasing below the
  baseline where the "4" does not) nor the font's metrics (both bundled
  fonts report degenerate ones — Leofont's ascent plus descent is about a
  pixel at 32 px) is a stable extent — so the topmost *visible* line's ink
  sits 20 px in from the window's top-left corner, and the visible parts
  reflow into that corner whenever one of them is hidden.
- Keyboard shortcuts: the overlay reads eight key combinations from the
  engine's key state on every `process`, acting on press edges only (a
  held combination toggles once): Alt-0 is the master switch — while off,
  every node draws nothing and the per-part toggles are remembered as-is —
  Alt-1..Alt-4 toggle the charts by position among the enabled charts (top
  chart first; the extra keys are no-ops when fewer than four charts are
  enabled), Alt-T toggles all the readout lines, the window-size line
  included, and Alt+'+' / Alt+'-' grow or shrink the whole overlay — font,
  margins, gaps and chart panels together — stepped by a quarter of the
  current size and clamped between a quarter and four times the base. The
  zoom pair matches **typed characters** (`+`/`=`/`±`/`−` grow,
  `-`/`_`/`–`/`—` shrink, numpad included) so it follows the user's layout:
  physical codes name US positions, where the Swedish `+` key sits on the
  US `-`; the line extent re-measures at the new font size, so the stacked
  lines stay even and every laid-out line re-lays out on its next refresh.
  The charts are independent of the text toggle. The layout
  reflows around whatever is hidden — the visible lines and charts re-pack
  into the corner — while the node indices stay fixed to the construction
  slot order; a hidden node keeps its place as a bare pivot (`shape:
  None`). While the text is hidden, the lines are not rebuilt, and a
  re-shown line re-lays out its last text the next visible frame. The
  shortcuts are read, not consumed: a demo that wants the same
  combinations still sees them. They are also available as the methods
  `toggle_all()`, `toggle_chart(index)`, `toggle_text()`, `scale()`, and
  `set_scale(scale)`.
- Pure CPU work (no device I/O), so it compiles on `wasm32` too — no cfg
  gate, unlike `audio`.

`examples/diagnostics.rs` shows it over a swaying circle, set in
`assets/fonts/FiraCode-VariableFont_wght.ttf`, with `DiagnosticsFlags::all()`;
`examples/particles/square_fountains.rs` runs it over a live particle scene.

## UI (src/ui/)

`Ui` is an **immediate-mode widget layer** over the frame's `Canvas`: the
app owns one `Ui` in its `Process` struct, calls `Ui::begin(ctx)` at the top
of `process` (snapshot the mouse, resolve last frame's press, move the
dragged panel), then declares widgets each frame — `button` (armed-release
click, as in `examples/button.rs`), `checkbox` (drives `&mut bool`),
`slider` (label left, value readout right, drag-follows-pointer, drives
`&mut f32`), `label`, `space`, and `panel`. Returns: `true` on the frame a
button clicks, a checkbox toggles, a slider's value changes.

- **Panels** are titled boxes anchored at a center `at` until the user
  **drags them by the title bar**; the dragged position is retained per
  title (hashed id). The body is as tall as the content asked for *last*
  frame (first frame shows the title bar only, then it snaps open), and it
  grows **downward from the anchored title bar**. A **click** on the title
  bar (a release that barely moved — measured from where the press began,
  so dragging never folds) **folds** the panel to just its bar and folds it
  back; a chevron on the bar shows the state. Widgets outside any panel
  stack in a root column at the window's top-left. A
  panel **claims its whole area** (body included): presses land on the
  panel, never on a widget or the game behind it, and `Ui::hovering()` —
  the pointer over any widget or panel — lets a raw-input game tell UI
  clicks from scene clicks (see `tools/sprite_util/process.rs`). Two accessors
  reach a panel from outside: `Ui::panel_position(title)` reads where it
  sits, `Ui::panel_rect(title)` reports the plate as it was PAINTED last
  call (the height the drawing used — content laid over a plate must
  trust this, not its own same-frame request, or it floats a frame
  ahead of the pixels; see `tools/sprite_util/process.rs`), and
  `Ui::set_folded(title, folded)` is the program's hand on the title-bar
  click — code that closes or restores a panel uses it so a reopened
  title never greets you folded.
- **The input model** is one frame delayed: a press is resolved at the next
  `begin` against the rects the previous frame registered, last-declared
  (topmost) first — so overlapping panels route presses to the visible one,
  and a press that leaves a widget keeps its drag until release (sliders,
  panel drags). Hover, click and hold all read this resolved state.
- **Tables** (`Ui::table(ctx, id, cols, content)`) lay the widgets declared
  in `content` into a grid: each consumes the next cell, wrapping to a new
  row after `cols.len()`. Each `Col` has a [`ColSize`] — `Px` (fixed),
  `Auto` (as wide as its widest cell, from last frame, matching the UI's
  one-frame model) or `Stretch(w)` (a weighted share of the leftover) — and
  an [`Align`] (`Left`/`Center`/`Right`) honored by text. `slider_track` and
  `readout` are the label-less pieces for a `[label | track | value]` row
  (see `tools/sprite_util/process.rs`); `id` scopes the retained auto widths so
  two tables in one panel keep their own columns.
- **Tree view** (`tree.rs`) is the same widget family's two-axis-scrolling
  list: `TreeLine` rows (key, value, a few flags, a value color) in a
  resizable, scrollable panel — close `×`, corner grip, both scroll thumbs,
  per-row fold/delete buttons, drag-reorder of deletable rows — driven
  through `TreeSpec`/`TreeState` and answering `TreeEvent`s **by row index**
  only: the widget knows nothing of the app's data model. `sprite_util`'s
  sidecar panels and `ron_view` are its two applications (there
  `frost::ron` and `examples/ron_view/view.rs` are the data and row
  sides, here it is the widget).
- **Identity**: widget ids are FNV-1a hashes of the label scoped by the
  containing panel's id, so the same label in two panels is two widgets.
- **Text** goes through `Canvas::text` (see above) with one shared
  `Arc<[u8]>` font (`Ui::from_font` / `from_bytes`, validated at
  construction), so all UI text shares one cached glyph atlas; measured
  widths are cached per `(string, size, weight)`; the weight is
  `UiStyle::font_weight` (default 600 — a SemiBold on a variable font like
  FiraCode, inert on static fonts).
- **Order**: every UI draw gets a `z` counting up from
  `UiStyle::base_z` (default 10 000) in declaration order — the UI paints
  over the scene, panels stack in declaration order.
- `UiStyle` is one `Copy` struct of colors and metrics (dark default),
  reachable through `ui.style`.

## Testing

Baseline: **412 lib tests + 3 integration tests (`tests/`) + 7 doctests**
passing — six ignored probe tests ride beside the lib count —
`cargo build` (lib + the `sprite_util` and `noisefloor` tools) clean.
The tool's 93
tests — 68 that moved with `examples/sprite_util.rs`, each living in the
subject module it pins, the asset-path guard, the three
authored-shapes overlay tests, and the nine that pin the status band's
log (capture by change, the cap, the window and thumb arithmetic, the
dock's geometry) — ride along in the same
`cargo test` now that it is a binary target. So does `noisefloor`'s 7 plus a
doctest, which pin the percentile ranks, the two gate readings, and that a
live canary wakes and reports — never *how late* the machine was allowed to
be, because that is the quantity it exists to hand back rather than one it
may assert. Notable test areas:

- `src/shaders.rs` — naga parse + device-side validation (the
  `Validator` stage wgpu runs in `create_shader_module` — this is what
  catches uniform-address-space layout rules the parser never checks) +
  uniform-offset assertions (above).
- `src/backend/tests/` — mostly GPU-free: uniform-layout mirrors, scissor
  math, `paint_order`, `expand_text` (splicing + atlas reuse across frames),
  `context_reports_held_mouse_button`, `context_reports_mouse_wheel_delta`,
  and the context's typed-character read (`char_down`).
- `src/backend/tests/shadows.rs` — the exception, and the reason the rule is
  stated: it renders the real packed occluder field through the real
  `shape.wgsl` on whatever adapter the machine has, and compares every pixel
  with `collision::occluded`. That agreement cannot be checked from either
  side alone, so this is the one place a drift between the CPU's slab test and
  the shader's can be caught; with no adapter it prints that the shadow went
  unverified rather than passing quietly. A box is required to agree exactly,
  away from where its shadow edge crosses between neighbouring pixels; a disc
  gets the tessellation's own bound instead, which needs no invented epsilon
  because containment does the work — the sixteen-gon contains the circle it
  inscribes and is contained in the one it is inscribed in, and a shadow grows
  with the shape casting it, so its shadow must sit between theirs (outside the
  disc, where the receiver-inside rule is silent for all three), with the
  disagreement counted to show the band is as thin as `DISC_SIDES` claims.
- `src/diagnostics.rs` — press-edge detection for all eight shortcuts, the
  display-scale clamp, and the line-extent re-measure on scale change.
- `src/ui/menu.rs` — the bar's state machine without a canvas, and the
  notes' paint geometry: right-aligned from the bar's far end with
  center-anchored text (a note's center, never its edge, steps by half
  widths), a `...`-led tail kept when a note overruns the room the titles
  leave — the tail is the half that says something — and the earliest
  notes falling off first.
- `tests/` — headless `Shape::sprite` reads of the `dogs_name` art (frame
  fit, ink statistics), through the same public entry point the examples use.
- `src/physics.rs` — the elastic impulse's two conserved ledgers (energy and
  momentum, checked against closed-form one-dimensional results and a
  three-body round), the inverse-weight split of a shared penetration, and
  the anti-stick rule that a separating pair is separated but never sped up.
- `src/collision.rs` — push-out separation, reflect restitution semantics,
  and the convex occluder polygon: what it refuses rather than repair, the
  receiver-inside rule, and the two-percent band a tessellated disc keeps from
  the round one.
- `src/bake.rs` — each role's fusion policy, quarter-turn equivalence, what
  gets rejected and why, and a sampled coverage property: a fused set covers
  exactly the points the authored boxes covered, on random piles and on a
  half-filled tile lattice.
- `src/shapes.rs` — the sidecar's `shapes:` vocabulary end to end: sidecar
  text through `ron::parse_doc`, the reader, a real `Baker` and
  `declare_occluders`; the pixel-to-node flip pinned by an asymmetric case;
  and the refusals — every error reported not just the first, a tuple
  position answered with the tool's spot rule, capsules and concave outlines
  refused by name. The two arms agree: the drawing arm keeps every outline
  whose geometry read while the refusals pile up beside them, and a role the
  bake refuses reaches the tool and the baker as one and the same sentence.
  The load-time seam too: `bake_sidecar` against the real shipped files —
  a missing sidecar bakes nothing, an `atlas`-only one bakes nothing,
  `brick.ron` bakes the hall's one bouncy footprint — while a mis-authored
  file fails the load with the whole report and an unparseable one travels
  as a RON failure, never a log line.
- `src/ron.rs` — the RON subset itself: atoms, numbers and enum paths kept
  verbatim, comments as noise that still survives the writer, errors that carry
  a line number, and `assets/ron/garden.ron` round-tripped to a fixed point —
  the property that makes the file safe for a tool to rewrite at all. (The
  sprite sidecar fixture is round-tripped over in `sprite_util`'s own tests,
  where the tool that rewrites it lives.)
- `src/tween.rs`, `src/particles.rs`, `src/text.rs` — behavior unit tests.
- `src/rng.rs` — a fixed seed replays its stream; `state()`/`set_state()`
  round-trips continue it exactly.
- `src/audio.rs` — device-free decode tests (synthetic WAV, bundled
  `swoof.wav`, error variants) + the `load_bytes` doctest.
- `src/ui/` — the interaction model driven frame by frame through
  `begin_input` (press arming, armed release, hold past the rect, topmost
  overlap wins, one-frame click, panel drag moves by the pointer delta and
  retains), plus id hashing and rect math.

## Examples (examples/)

48 demos, filed into subject folders — `shapes/`, `lighting/`, `inout/`,
`layers/`, `particles/`, `physics/`, `text/`, the `immortal/` game, and the
tools (`worm`, `tilemap`, …) at the top level. Cargo only auto-discovers
`examples/*.rs` and `examples/*/main.rs`, so every folder member is named
with an explicit `[[example]]` entry in `Cargo.toml` — the short names keep
working: `cargo run --example parallax`. `ron_view/` is the exception that
proves the rule: a `main.rs` folder target needs no entry, and its RON-subset
parser, tree model and writer (every value carries its source comments
through parse, edit and save) now live in the crate as `src/ron.rs`, so
both examples parse sidecars and samples with the crate's own
dependency-free parser; what stayed in the example is the row flattener
(`view.rs`, path-included by `sprite_util` too), which flattens a tree
into the folded rows both viewers paint. Most load their
assets from disk at runtime, pinned to the crate root via
`CARGO_MANIFEST_DIR`; `immortal` and `text_web` embed their assets into the
binary with `include_bytes!` instead, so they run with no asset files on
disk. Run with `cargo run --example <name>`
(`RUST_LOG=info` for logs; on Windows PowerShell:
`RUST_LOG=info cargo run --example gizmos 2>&1 | Out-String`).

`sprite_util` is no longer in this folder or this package: the tool lives
at [`tools/sprite_util/`](tools/sprite_util/main.rs) as its own workspace
package — run with `cargo run -p sprite_util`, tested by plain `cargo test`
(see [`docs/tooling.md`](docs/tooling.md)). Its `ron_view` row flattener is
still the one in `examples/ron_view/view.rs`, path-included across the
directory line: a known debt until a second tool proves the view is shared
tool surface. The tool's one `main.rs` became a crate of its own subjects:
`theme` (colors, z-orders, pool ids), `layout` (zoom, pan, selection),
`world` (sprites, the bench, the undo history), `art` (PNG codec, asset
paths), `strip` (the slot strip), `journal` (the status band's log),
`band` (the tileset rack), `spots` and `sidecar` and `ron_panels` (the
positions, the `.ron` files, their foldable trees), `map` and `clip` (the
tile-map desk and the clipboard block), `desk` (the `Demo` state and its
frame sync), `draw` and `process` (the frame: painters read one
`FrameState` geometry record the handlers fill once), and `actions` —
where a planned verb is born: a menu line and its method, side by side.
Each module's `mod tests` holds the tests that pin its subject. The
`sprite_util` row below stays as the description.

| example      | shows                                                              |
|--------------|--------------------------------------------------------------------|
| one_rect     | smallest scene: background + one static rectangle                  |
| circles      | immediate circle draws                                              |
| gizmos       | the README's canonical demo (mixed immediate + scene draws)        |
| box_z        | overlapping squares with explicit `z` values (z-ordering)          |
| tween        | one scalar `Tween` driving a line                                   |
| sub_zero     | text shapes ("SUB0") through the scene tree                        |
| text         | text shapes ("Sub") through the scene tree                         |
| text_web     | the `text` scene for the wasm target                               |
| text_leo     | the same text set in `Leofont-Regular.ttf` — one of the three     |
|              | bundled faces (with JameGem08 and FiraCode Variable)                  |
| player       | WASD drives a Button sprite from `(0, 0)`                          |
| layers       | obstacles + rendering layers                                       |
| parallax     | layers + camera: infinite parallax scrolling                       |
| repeat       | a layer tiling itself (`Layer::repeat`)                            |
| scene        | an in-place animated scene tree (orbiting children)                |
| satellite    | the `scene` tree refactored into reusable nodes                    |
| sprites      | PNG sprites through the scene tree                                 |
| tomato       | a plant assembled from three sprite slices                         |
| grow         | the tomato plant growing slice by slice                            |
| dogs_name    | the four-frame 1-2-1-4 flipbook monster (frame 3 deliberately repeats |
|              | frame 1), bob/sway/breath, a true physical 1920x1080 window via    |
|              | `Config::window_size_px`                                            |
| centipede    | a whole animal assembled from the same monster's cut-out parts and |
|              | walked seen from above: rigid one-pivot sprite legs whose planted- |
|              | foot geometry IS the gait (pure clock, nothing keyframed), a      |
|              | metachronal step wave, and one segment grown on screen per         |
|              | crossing; all geometry in `rig.rs` (unit-tested), the folder's    |
|              | README is the anatomy doc                                           |
| collision    | the player with `push_out`/`reflect` collision                     |
| cone_collider| the `cone` hall with physics added: the walls that cut shadows out  |
|              | of the beam also push the player back (lighting × collision)        |
| bodies       | three elastic `Body` rectangles rattling in a box of four           |
|              | 50 px walls, under the diagnostics overlay: random starts,          |
|              | and a kinetic-energy readout that does not move                     |
| authored_hall| a dark brick hall whose shadows and collisions come                 |
|              | from `brick.ron`, baked once at load: one `Baked`                   |
|              | rides every brick — occluders declared per instance                 |
|              | cut the light and the mouse-aimed torch beam into shadows,              |
|              | the baked solids bounce the player with the sidecar's               |
|              | own restitution (`docs/authored-shapes.md`)                         |
| lighting     | a lit floor, a lit ball with an ember plume, three lights sweeping  |
|              | across them — the feature tour                                      |
| lit_unlit    | two same-color rectangles — one `lit: true` — and a light on the    |
|              | mouse: the receiver flag in isolation                               |
| bouncing_lights | three lights ricocheting around the window like billiard balls  |
| cone         | a square explorer carries a torch (a cone `Light`) through a dark   |
|              | hall of occluding walls                                             |
| dawn         | one directional light as a day cycle: sunrise, climb, sunset, the   |
|              | world warming and cooling with it                                   |
| near_sun     | the same day told by a *near* sun — a point light on the visible    |
|              | disk, so shadows fan out radially                                   |
| occlusion    | lit panels behind tall occluder walls, one lamp orbiting through    |
|              | them: hard vs. penumbra shadows                                     |
| twostars     | a planet under two stars: two directional lights, both on stage     |
| eyes         | self-lit eyes drifting through the dark — a tour of `SceneNode::emission`|
| eyes_light   | the companion: some eyes are real emitters (`Shape::Light` too)     |
| eyes_light_bloom | both halls with the glow and the fold on: a lamp-eye's halo   |
|              | crosses the wall its own light cannot                                 |
| particles    | a `ParticleSystem` fountain with life-fraction alpha               |
| fountains    | three overlapping fountains + a full-screen waterfall, each one    |
|              | CPU-simulated batch (one instanced draw per `Shape::Particles` node)  |
| square_fountains | the middle fountain spits rotating squares, each in its own hue: |
|              | per-particle `Particle.angle` advanced by a side `spins` vec kept in |
|              | lockstep with the sim's dead-particle removal; runs the full        |
|              | `Diagnostics` overlay over the scene                                |
| cursor       | a watering-can cursor: mouse-following sprite, press-to-tilt tween, |
|              | spout-emitted water particles over a full-screen grass field       |
| torch        | two kinds of particles side by side: embers that ride the torch and |
|              | sparks that keep flying once they leave it                          |
| worm         | one sprite crawling by peristalsis — a stretch anchored at the rear |
|              | and a contract anchored at the front (matched so the center glides  |
|              | smoothly), a 2 px bob at twice the beat, wrapping at the window's   |
|              | physical edges (scale-factor honest); 960×540 via `Config`          |
| sound        | one-shot + looping playback of a decoded WAV: Space re-triggers     |
|              | (pulsing circle), L toggles the loop (wobbling ring), +/- the bar   |
| audio_test   | a raw rodio scratchpad: plays WAV/MP3 straight through rodio,      |
|              | beside (not through) frost's `Audio` wrapper                        |
| button       | a "play" button: hovering tweens the node's `scale` to 1.1 (rebuilt |
|              | on every hover edge) so rect and label grow together; a press edge  |
|              | over the button arms it, and an armed release over the button plays |
|              | the swoosh once                                                     |
| tomato_sprite| the tomato sprite with its `tomato_fg.png` overlay, both siblings   |
|              | under a shapeless dummy parent node, both at the same draw order:   |
|              | the tie resolves in tree order, so the later sibling (the overlay)  |
|              | draws on top; each child puts the image px `(308, 411)` on          |
|              | the parent's origin; the parent carries the inverse, so the         |
|              | image's center sits on the window's center                          |
| immortal     | 1920x1080 (Config window_size); four tools ride the shelf items.png |
|              | paints: the sprite's four bays ARE the four slots - BAYS holds each |
|              | bay's opening centre and its board-top rest line, read off the      |
|              | sprite's luminance bands; a bay's ceiling is its click cell's own   |
|              | cut, so the bays stand 214/213.5/214/207 px tall and every tool is  |
|              | drawn at `tool_scale` - the largest size that fits every bay, so    |
|              | nothing changes size as it moves along the shelf; tweezers, spade,  |
|              | spray can, watering can top to bottom, click a bay to swap          |
|              | with it, right button trades held for stored; a fresh left click    |
|              | digs one spade stroke - `dig_offset`: shoved DIG_PUSH_DIST px along |
|              | its own drawn line (34.9 deg below horizontal, read off the art),   |
|              | held buried DIG_HOLD s, then arched up and left over the hole it    |
|              | made and home to the cursor - no rotation, the tool is carried, not |
|              | turned; docked and held tools are centred on their drawing rather   |
|              | than their canvas, so the off-centre tweezers fit inside the dock;  |
|              | the tweezers stay inert until pinch frames exist;                   |
|              | plants grow one at a time, each slice opening its flowers once fully|
|              | grown (the four lower slices' spawn points, the top slice bearing   |
|              | none, populated with `flower.png`); one wasp per fully grown plant  |
|              | layer orbits the row; bug and louse swarms pop up out of the grass in|
|              | batches of three per plant each (the bugs module is generic over a  |
|              | Species: the lice of Lice1/Lice2.png are smaller, untinted, left-   |
|              | facing art whose facing flip mirrors the bugs'), growing over 0.75 s;|
|              | each pop-up - a batch spawn or a respawn - plops on a random plopp  |
|              | clip (bugs_plopp1/2/3.wav); spray mist touching a bug wounds it -   |
|              | three starting hits, one per 0.2 s per bug, a bug at its park spot  |
|              | regaining one hit every 5 s, a louse two hits every 10 s (lice start|
|              | at two health), both capped at six, a white pip above the bug per hit|
|              | it can still take; a bug reaching its park spot while another bug is|
|              | within 50 px of it chatters - a random chatter clip (ChatterLow/Mid/|
|              | High.wav) plays - and the killing blow starts the two-phase death:  |
|              | over 0.5 s it bounces up off the grass, flipping upside down in the |
|              | air (y scale to fully inverted, position/growth/facing/frame frozen),|
|              | landing on its back; then over 0.5 s the inverted sprite evaporates,|
|              | shrinking to nothing while its center sinks through the grass; after|
|              | a random 5-10 s delay the bug pops back up at its spawn spot, fully |
|              | healed; the five slices' joints and flower anchors load at startup  |
|              | from the embedded `assets/sprites/plant1..5.ron` files (serde + ron —|
|              | the hand-picked constants are gone); a 14-slot worm swarm burrows up|
|              | and walks the bench on the `worm` example's stride — the rear grips |
|              | while the body reaches, the front holds while the contract slides it|
|              | up — two poses riding the stride's halves, hull-picked spawn/target |
|              | spots off a seedable `Rng`, and digs back in; every draw order      |
|              | composes through the `zorder` module's bands (ground = −y, bench    |
|              | slots on top, grass/UI/overlay pinned outside); F5 / F9 save & reload|
|              | the whole game as versioned RON under `~/.frost/immortal/` — the    |
|              | snapshot carries the random streams, so a reload continues each     |
|              | stream exactly (`--load` loads at startup)                          |
| sprite_util  | up to seven picked PNGs (rfd dialogs): the window splits into a     |
|              | work area and a bottom strip of 100x100 slots holding the           |
|              | minimized originals (click a slot to activate, drag one onto        |
|              | another to swap). The active sprite shows over a checkerboard with  |
|              | `Ui` View and Operations panels (Open / Crop / Save / Save As /     |
|              | Close): a left-drag draws a crop selection (a still click logs the  |
|              | texture pixel), Ctrl-O opens into the next slot, Ctrl-Z undoes      |
|              | crops, Save/Save As write PNG over a native confirm. An Animation   |
|              | panel builds frames — each a set of layers with its own hold-time —  |
|              | played on a clock that loops or ping-pongs (space plays), previewed |
|              | live in the work area. A sprite with a sidecar `<name>.ron` beside  |
|              | its PNG gets its own `Ui` panel titled by the file — one view per   |
|              | open file, cascading from the lower right, each dragging, folding,  |
|              | scrolling by its own pair of handles (wheel too, Shift-wheel         |
|              | sideways) and resizing by a corner grip; the boxed × in a view's    |
|              | bar closes file and panel together — laid out on the plate the      |
|              | UI actually painted, never a frame ahead of it                      |
|              | Every `(x, y)` the file names is marked on the sprite in the colour |
|              | its rows carry: pick a row and the sprite centres the point;        |
|              | right-click a marker to pick it back — the tree unfolds to the      |
|              | row and scrolls it to centre. Left-click the sprite to move the     |
|              | position, Escape lets go; other rows fold on click. Save /          |
|              | Save As write the tree back beside the PNG with its comments —      |
|              | the parser keeps leads, trails, tails and the header, the writer    |
|              | puts them back. Sequences wear + / × row buttons and entries        |
|              | reorder by press-drag; a list entry's marker shows its seat         |
|              | number on the sprite. A `shapes:` key draws over the                |
|              | sprite as amber outlines — rects by their edges,                    |
|              | circles as the 16-gon the light field shadows with,                 |
|              | polygons as authored (View ▸ Shapes toggles the                     |
|              | drawing, never the validation); the bar's notes then                |
|              | voice the first refusal of a mis-authored entry.                    |
| widgets      | the `Ui` layer: a draggable Tomato panel (three color sliders, a    |
|              | Spin checkbox, a Speed slider, a Reset button) and an About label   |
|              | panel drive a spinning face's color and rotation                    |
| ron_view     | a foldable, scrollable tree view of a `.ron` file (`view.rs`,       |
|              | the rows shared with `sprite_util`'s sidecar panels): click         |
|              | `[-]/[+]` rows to fold, drag the two scroll handles (or wheel       |
|              | / shift-wheel), text in FiraCode Variable — keys & heads at         |
|              | weight 700, values at 500; reads through the crate's own            |
|              | RON-subset parser (`frost::ron`: comments, enums, maps,             |
|              | escapes; line-numbered errors); `assets/ron/garden.ron` sample      |
| vrr          | the pacing instrument: keys, never the mouse (pointer motion        |
|              | re-clocks the display to its ceiling under ProMotion), driving a    |
|              | target rate the loop meets on an absolute deadline; Target, Actual  |
|              | and Mode read back together; `0` runs free; `--modes` lists the     |
|              | panel's modes. See *Frame pacing and variable refresh* below.       |

`cursor.rs` is the most complete reference demo: `CAN_IMAGE [331,247]` scaled
to 100 px, a 90° CCW tilt tween (0.5 s, rebuilt on press/release edges),
spout at image px (0,25), a 120/s emission accumulator, semi-implicit water
droplets under gravity 420, life-fraction alpha, and a splitmix64 `Rng` for
spread/life/size randomization.

## Frame pacing and variable refresh (examples/vrr)

*The instrument measures itself first.* vrr drives a target frame rate
through the keys and nothing else: the slider is gone because pointer
motion is compositor work, and dragging it on an M4 Max driving an ASUS
PG42UQ pumped the panel to 120 Hz while the app went on presenting at its
target — a control reached with the mouse is a control that changes the
thing it is set to measure. The panel reads the target beside the rate
actually held and the mode the system reports, because an instrument that
cannot see its own error is only guessing, and this guess had been lying.
The old pacer re-anchored each deadline on the instant its sleep woke, and
macOS wakes a coalesced timer about three milliseconds late, so the
overshoot was written into every frame after it: measured, a 120 fps target
ran 100.7 and a 60 target ran 51.2. An absolute schedule with a
one-millisecond spin tail lands within 0.3 % of the request, which makes
every macOS refresh reading taken before it a reading of the loop's own
shortfall rather than the display's. Counting frames across a window matters
as much as pacing them: averaging instantaneous rates reads 121.2 for a
cadence that is exactly 120, because short frames carry more rate than long
ones carry lack of it — and that was the number first noticed as impossible,
sitting above a ceiling vsync was visibly enforcing.

*What the swap chain can ask for is settled from the vendored source.* The
Metal backend advertises `[Fifo, Immediate]` and its `configure` calls
`unreachable!` on anything else, so `AutoVsync` resolves to `Fifo` and
`FifoRelaxed` — the mode that means *present by my deadline* — does not
exist there at all; DX12 advertises `[Mailbox, Fifo, Immediate]` with `Fifo`
becoming `SyncInterval = 1`. The same word with different owners of the
scanout: DXGI stretches the refresh to the game's deadline, Metal latches to
a cadence WindowServer chose, and the engine asks for variable refresh in
neither case.

*VRR engages on macOS anyway*, measured — at `Fifo`, with vsync on, and only
inside a native fullscreen Space (winit's `toggleFullScreen:`, a window in a
Space of its own rather than a large borderless window). There the panel's
counter follows the app's rate and wanders a few hertz around it while the
body glides, which is the display's sampling and the server's ramp rather
than the app's cadence: read the average, never the instantaneous figure.
Above the mode nothing the app wants matters — free at any target, the chain
holds 120.1 — while with vsync off the loop runs genuinely uncapped, 289 fps
into a 120 Hz mode, so macOS offers the latency of running free and refuses
its cadence. *And the refusal is worse than the cadence it refuses*: held at
the mode with the latch on, the box glides, and unlatched at that same rate it
stutters badly. Nothing ties the app's clock to the display's once the latch is
off, so every scanout catches a frame at an arbitrary phase — paced at 120 into
a 120 Hz mode it drifts in and out of step, and free at the 289 measured above,
289 ÷ 120 being 2.4, the surviving frames advance two, then three, then two, an
irregular beat with no common divisor to settle it. Every frame correct, the
selection a lottery. Running free is worth it only while the rate stays *under*
the mode, where something might follow it — and on a Mac something does, in its
mode-stepping way, but never above the panel's maximum and never for a loop
running unlatched: free run is a probe, not a way to play.

*ProMotion follows the app, and the proof is in the width of the band.* An
earlier paragraph here had the built-in's rate driven off content with the app
looking on, and that had the causality backwards: no API lets an app *name* a
rate — System Settings offers ProMotion or a fixed rate, `system_profiler`
prints nothing at all for the built-in panel — yet under ProMotion the panel
chases the rate the app presents at and locks to it, one scan per frame.
Targets of 48, 50, 60, 90 and 120 each hold that lock; 45 and 40 are refused,
clamped to a floor of 48 scans, and judder at 1.07 and 1.2 scans a frame. The
tell is the width of the reading, not its value: ±0.1 fps is the half-second
rate window closing a fraction early, while ±1 is a slipped scan — one slip at
a 90 target would drag the reading to 88 — so a ±0.1 band means one frame per
scan with nothing slipping. And it scanned at 90 for a 90 target though
`--modes` lists no 90: the modes CoreGraphics publishes are not the ladder the
compositor walks, which is why enumerating them says what the panel can be
*pinned* to and nothing about what it will *choose*. What a low lock costs is
dwell, and dwell is the smear: a clean 48 blurs like 48, because what smears is
the frame's lifetime, not the panel's refresh. One remainder, half settled:
pointer motion re-clocks this panel to its maximum, and that turns out to be a
ProMotion effect rather than a compositor one — brisk movement at a pinned 60
changes nothing — so the only thing it can disturb is the chase, and a reading
taken a second after touching anything may still catch the panel on its way
back down. That remains the likeliest account of an early ±1 at a 60 target
that now reads ±0.1, and the settle time behind it is still unmeasured.

*A pinned mode is the opposite bargain: it does not chase, and it does not
listen.* The menu for
pinning is short — `--modes` offers 120.00, 60.00, 59.94, 50.00, 48.00 and
47.95 on the built-in panel, with no 30 and no 24 — and none of it is a
promise about what the panel will scan at on its own. Run with the pacer stood
down and the latch on, a pinned 60 holds 60.0 exactly and does not flinch at
the pointer, which is as deterministic a presentation as an app can arrange
here: the display is the whole clock, the app owes nothing to it, and a missed
frame costs exactly one frame with no debt left to repay. Every burden moves
onto the frame's own tail, so the budget that matters is the worst frame in a
minute rather than the median one, which is what the overlay's envelope of
maximums is for. On a fixed 60 the
clean paces are its divisors, 60, 30, 20 and 15, and 50 sits at 1.2 scans a
frame and judders hard — the same 50 that locks cleanly one setting away, which
is the cleanest demonstration that the ladder and the chase are different
mechanisms. So for a pinned mode the old rule stands: **when the pace is not a
divisor of the mode, change the mode rather than the pace**, and only
`Fullscreen::Exclusive(video_mode)` sets one, with winit asserting its way
through the call, so an engine that ever does this enumerates first and never
guesses. Measured on the M4 Max with the PG42UQ and the built-in panel over the
14th of 2026-10 into the 15th; the LG 48CX under Windows 11 still wants
re-running with the corrected pacer, its substance — windowed VRR, the latch on
capping at max, the latch off uncapped and followed — expected to survive at
better numbers.

## Load-bearing invariants (don't break these)

1. **Node transform order**: `scale` innermost, then the node's `transform`,
   then the parent's world — `world_at` and `draw_node` both follow it, and
   the `can_transform`-style compositions in examples match it (e.g.
   `cursor.rs` matches the unrotated pointer-follow exactly at angle 0).
2. **Uniform layout**: CPU byte offsets in `frame.rs` ↔ WGSL uniform structs ↔
   naga tests must all agree (`[0,16,24,32,48,64]`, span 80 for shape/sprite).
3. **AA band**: `AA_BAND = 0.75` in `frame.rs` and the `aa` constant in the
   shader sources must stay equal, or scissor boxes stop covering every pixel
   the shaders can write.
4. **`Tween` has no re-target** — rebuild on edges.
5. **One uniform arena per frame** (the `write_buffer` batch semantics are why
   it is one arena and not one buffer per draw). The contract has two halves
   that must agree: each pipeline that reads a uniform has a hand-written bind
   group layout — `has_dynamic_offset: true`, `min_binding_size` equal to the
   `*_UNIFORM_BYTES` constant beside its packer in `frame.rs` — and that packer
   allocates its buffer at the same length. A uniform that outgrows its
   constant fails at bind-group creation, a layout that stops matching its
   shader fails when the pipeline is built, and every test frame builds every
   pipeline.
6. **Sprite/atlas texture sharing** is keyed by the `Arc<[u8]>` pointer plus
   the buffer's generation (0 for static images, bumped by `text::Atlas` on
   every repack) — the pixel buffer must be the same allocation (clone the
   `Arc`, don't copy bytes) for the sharing to work. When an atlas repacks,
   its stale texture is evicted from `sprite_resources` right after
   `expand_text`, so the fresh buffer uploads in the same frame.
7. **Text expansion happens before sorting**, so glyphs inherit the node's
   paint position.
8. The **last** `Background` in call order wins as the clear color.
9. **Authored shapes stay local.** `Baker`'s output and `Canvas::occluder`'s
   arguments are a shape in its own space plus a transform, never world-space
   corners — which is also why a `Convex` occluder ships half-planes and not
   corners: a turned or mirrored instance needs no second polygon and no
   re-derived normals, and the y-flip to the framebuffer reverses a winding
   that was never asked to mean anything: a turned, non-uniformly scaled rectangle is not an oriented box, so
   baking it into coordinates would force an approximation of the shear — on the
   GPU, which pulls the pixel into the shape's space rather than pushing a
   sheared box out, and on the CPU alike.

## Direction

The queue moved into the game it serves: `examples/immortal/TODO.md` carries
the immortal wish-list — graphics (shop icon, progression toward immortality,
level indicators), sound effects, music, and the fruit's ripen-rot-fall cycle.
The library's last queue entry, the `Body` type, shipped as `src/physics.rs`
with the `bodies` example as its demo; the physics section's `rapier2d` note
stays the documented escape hatch if it ever outgrows itself.

**Asset identity moves from paths to names.** The key finding: a shipped app
is one self-contained binary — assets arrive via `include_bytes!` at compile
time (`immortal`'s `assets_load.rs` is the pattern), so a saved file that
names `assets/sprites/basic_tiles.png` refers to a place that does not exist
in the binary. The one noun that survives all three worlds — repo (resolved
by `asset_path` beside the crate root), loose runtime (an `assets/` folder
next to the binary), shipped (a `const ASSETS: &[(&str, &[u8])]` name table
of `include_bytes!` entries) — is the **asset name**: the file stem. The
house already treats the stem as the key (`<stem>.ron` rides `<stem>.png`);
what was sloppy was carrying it around inside a path.

**Two tilesets can share a sprite — and the desk used to be blind to it
(fixed 2026-10-10).** The atlas grid is per-loaded-instance state, so the
same PNG can sit in two slots cut two ways (1×4 here, 2×8 there).
`MapLayer` stamped provenance by path alone and the tile desk dressed with
first-match — so a layer painted against one grid got dressed from a
same-path slot with the other. The identity is now what it always
meant: `TilesetRef { name, rows, cols }` — the file's stem plus the cut
made into it, which fully determines a cell's picture and code — and the
map's save format inherits that meaning from the field it stamps.
`one_file_cut_two_ways_stays_two_tilesets` pins it, and the storage
language it introduced (`asset_name` in `art.rs`: a path says where a
file was found, a name says what it is) is the first rung of the name
architecture above.

**Maps get a file: `.map.ron`, names throughout.** The tile desk paints
`Vec<MapLayer>` and nothing persists it — an evening of tiling evaporates
at window close. The format follows the identity above —
`(tileset: ("basic_tiles", 2, 8), cols: .., rows: .., cells: [(col, row,
code, tfm), ..])` — sparse (cell 0 is not written), the orientation byte
riding with the cell because the transform is the map's memory, layers
keeping the order the desk renders. `Save map` / `Open map` are born in
`actions.rs` as the first planned verbs on that seam; opening a map pulls
the tilesets it names that the bench lacks, setting the recorded grid with
the existing `set_atlas` machinery; a tileset the file names that nothing
can find gets the house treatment — first refusal named, the rest counted,
spoken on the status band like `shapes_note`.

Landed 2026-10-10, with one remainder on the record: the codec and both
verbs shipped — cells written `(x, y, code, tfm)` from the layer's
top-left, empties never written, refusals named by the reader and spoken
by the opener — but the opener does not yet *pull* missing tilesets from
disk. Their layers wait dressed-as-nothing, which the desk has always
known how to do; pulling them by house convention is the increment that
closes this paragraph.

**The name table is also the hot-reload seam — the shipping model and the
dev loop turn out to share one interface.** Game code never says
`include_bytes!`; it says *give me `basic_tiles`*, and one resolver sits
behind that request: the const table in release, the same names read from
disk under a watcher in debug. Same code, same bytes-to-GPU path — the
decode-from-bytes entry points already serve both worlds — so a tileset
saved in `sprite_util` lands in a running game the next frame, process
state untouched (immediate mode keeps UI state nearly stateless, and not
restarting keeps the game's own state: the state preservation Godot buys
with a scene graph, frost gets for free). The ladder, cheapest first:
**data** (a gameplay `.ron` re-parsed on mtime change — the app rebuilds
config from the root; more than Godot offers for data-driven play, and
free); **textures, audio, fonts** (mtime → decode → `write_texture` →
rebind by name); **maps** (re-parse and rebuild the tilemap shapes — free
once data exists); **shaders** (`create_render_pipeline` is a runtime
call and naga is already a dependency — watching the `.wgsl` files is
surprisingly easy, and it is where frost can embarrass the big engines);
**code**, honestly last: incremental `cargo run` is the floor, and a true
hot swap means a cdylib game crate swapped by a dylib reloader with state
held host-side across an ABI-stable seam — a tax the API would carry
forever, so if it ever comes it comes as a documented *app pattern*, never
a frost API.

Landed 2026-10-10 as `frost::Assets` (src/assets.rs): a `Table` of
name-and-bytes compiled into the binary, loose directories admitted
above it by stem, and one request answering both worlds — immortal's
44 assets all flow through it, and its release binary boots from
`/tmp` with nothing on disk but itself. One finding the road
produced: **a name is the full request only when the format is
known** — the sprite directories keep their `.ron` sidecars beside
the PNGs, and `Getingeye1.ron` shadowed its own sprite's bytes until
the ask learned to name its formats (`asset(name, exts)`) — the sprite
is `Waspeye1` now, and the finding stands under either name; name and
format together dress an ask, exactly as name and cut identify a
tileset. The watcher (mtime → decode → rebind) remains the tiers' to
raise; today's resolver reads loose files when asked.

The data tier climbed that ladder first, landed the same day:
`frost::Watch` (src/watch.rs) is the shared sense — a file's stamp,
polled once per frame, `changed()` true once per rewrite, appearing
and vanishing counted as rewrites, silence on targets without a
filesystem — and immortal's `Config` (config.rs) is the pattern it
serves: four of the garden's paces read from
`assets/game/gameplay.ron` through the store, rebuilt whole from the
new root on change, every refusal keeping the last honest config
with its name in the log. The live proof is in the rung's check:
touch, and the running game answers the same second. Godot's
equivalent reloads scripts; the numbers that make the game *play*
the way it plays live in its scene files, behind an editor. Here the
crown jewel is a file and a watch.

Keeping landed 2026-10-17 as `frost::store` (src/store.rs): text kept
under a name until the next launch — `Assets` turned around, the same
two worlds answering one request. A file beside the launch folder on
the desktop, the browser's `localStorage` on the web; antifrost's
settings and progress retired their cfg twin pairs, the web build's
silent no-op and its four dead-code warnings went with them, and the
browser now remembers the hill it last stood on.

The media tier climbed next, and it asked nothing of the engine:
the texture cache has keyed textures by `(pointer, generation)`
since the first frame, so new bytes were always a new texture
waiting to be noticed — only the sense was missing, and `Watch` was
built for it. immortal's `rebind` is the pattern: a table of watched
names, and one arm per name because the *wearer* is what an engine
cannot know. The arms differ only in where the name is worn — the
grass on a node shaped once, a walk frame in a field the swarm
re-reads every frame (so a field swap *is* the rebind), a sound
heard from a field on the next press, and the overlay's text
derived from its font and so derived anew. A file that stops
decoding leaves the old asset worn; the log speaks; the garden does
not blink. A shipped world watches paths that never answer.

The shader tier was the ladder's last surprise, and it was a
surprise in the doctrine's favour: the WGSL had lived in `shaders/`
files all along (the consts in `src/shaders.rs` are `include_str!`,
and three sources are assembled — `lighting.wgsl` plus their host),
so the tier needed a sense and a seam, not a restructure. The sense
is `shaders::Live` — one `Watch` per file, debug builds only,
polled at the head of `render` before the surface is even asked,
because a saved shader should land whether or not a window is
compositing. The seam is the batch: any touched file re-assembles
*every* source from disk — which is how a `lighting.wgsl` edit
reaches the four lit pipelines at once — validates the whole table
through naga's parse and validator (the very two stages
`create_shader_module` runs), and swaps only if all nine compile.
A broken file refuses the batch: the old shaders stand, the log
carries naga's words, and the next save gets its chance. The swap
sets `format` to `None` — the pipelines rebuild by the exact road a
surface-format change always took, so the tier rides a path that
existed and was tested long before it. One test pins the seam
between the watched world and the shipped one: disk assembles
byte-exactly what `include_str!` embedded — drift there would mean
a game running different shaders after a save than after a rebuild.
Unity would call this a domain reload; here it is ten stats and a
veto.

**And one guard the name table makes possible:** a small reader lists the
names a map file asks for, so an app's own test can assert every one of
them sits in its `ASSETS` table — a forgotten `include_bytes!` dies at
`cargo test`, not in the field. The reader landed 2026-10-10, and the
finding is *where*: not at the paragraph's guessed `frost::map`, but in
the guest — the guard follows the codec, and that morning the codec
was the tool's. The same day's next rung corrected the address: a demo
must *boot* a map, and a format the world must read is not a tool's
dialect but a file format, so the facts graduated to `frost::map` —
identity, layers, cells, the codec, and the road from a layer to a
`TileMap` shape. That is the paragraph's original guess, reached by
the honest road: the guard moved with the codec, because it never
followed anything else, and it now stands at the predicted post,
`frost::map::tileset_names`. What stayed in the guest is motor skill
that knows `Sprite` — `tref_of`, `dressed`, `layer_shape`, the brush,
the ghost, the grid ladder — and the desk's verbs write and read maps
through the engine's `to_text` and `from_text`: one codec, both ends
of the tool's loop, which is what makes *edit in the desk, the world
follows* honest rather than hopeful. Its desk-side face
`waiting_names` decides waiting by full identity, name and cut together,
exactly as dressing does, and `open_map` names the waiters in its note
where it once counted them. The app-side pin needs an app that ships a
map; none does yet — the box stays open, and the demo rung owes it.

**The same day's second amendment: identity walks back to paths — the
honest ones.** The professor's question was structural: *shouldn't a
tileset be its own file, referring to a sprite sheet?* It should, and
the pressure that proved it was the tile's own role — solidity had no
home while a tileset was a way of wearing a sprite; it lived as pixel
rectangles in the sheet's sidecar, with a grouping helper still owed
to translate them into cell numbers. The old objection to paths — that
shipped code resolving `CARGO_MANIFEST_DIR` at runtime is flaky —
dissolved under inspection: nothing has to resolve anything at runtime.
The root is an **input, never a question**: loose reading is opt-in
(`with_dir`), the wasm store is table-only by cfg, and `env!` is a
compile-time constant in dev code. So the chain became three documents
where it had been two: `map → tileset → sheet`, with a **tileset as a
first-class document** (`frost::tileset`: name, sheet, cut, and the
cells' roles spoken *by cell number, in the tileset's own dialect* —
the grouping helper retired unwritten). References are
**document-relative** — the convention every browser and shader
include already speaks — resolved by `frost::map::resolve`, a pure
string operation whose inverse `relative` the desk writes with: the
chain walks a folder of loose files and a shipped table of embedded
bytes with the same arithmetic, and `guard_documents` asks the whole
cargo question — map, every tileset it points at, every sheet every
tileset cuts — of bytes and a closure, no filesystem in sight, at
`cargo test` time. The table's keys are now the same strings the
documents speak and the `include_bytes!` arguments name — key, embed
and reference in one dialect — which also kills the stem days' real
weakness: two `grass.png` in two folders were one name; they are two
paths now. Case became a dialect *law*: paths are `/`-joined and
case-exact, enforced by a `read_dir` walk that declines the forgiving
filesystem's nod (`a_case_slip_refuses_rather_than_forges`), so dev
and release refuse a wrong case at the same moment. Identity by speech
survives the move: paths *find* documents, names *are* documents, and
a desk that shelves `moss.tileset.ron` around a document that says
`name: "grass"` says so. The map file's root now carries a `tilesets:`
cargo list; a layer naming a document the cargo does not carry is a
contradiction refused, not a partial map; the desk writes each
tileset's document beside its sheet (`<stem>.tileset.ron`) *before*
the map that points at them, and refuses a layer whose tileset can be
neither found nor written. `waiting_names` keeps its post with new
eyes: a shelf document whose sheet closed waits exactly as a name
nobody wears — half a dressing dresses nothing. Immortal's 45 entries
key by path; its `key_of` is housekeeping over *its own* table (a stem
worn twice there is a build error the app refuses to guess), so call
sites keep their names while the store hears only paths. And the
guard's box? The reader moved one final post — `guard_documents` is
the whole cargo question now, two levels deep — and the box stays
open for the one thing it always waited on: the first app that ships
a map, whose guard test is then `guard_documents` over its own table.
That hour came the same evening: `examples/courtyard.rs` carries its
map, tileset and sheet in one table and asks the guard the whole
chain of it — the box closed, and the demo's remaining rungs (lamp,
marble, watchers, telling) light a floor the guard has certified. The
resolver core, the watcher tick, and the rebind-mid-swap semantics are
still to be designed; this paragraph is the finding, not the plan. The
engines this bet is made against, and the scoreboard tracking it, live in
`COMPARISON.md`.

**Tools are guests: the app is the truth, so design the telling.** In
Godot the scene tree *is* the application — a file (`*.tscn`) that the
editor mutates as ground truth and the game merely renders. In frost the
inverse holds: truth is a running program, and the scene tree is a frame
artifact — `process()` manufactures it from app state every frame and
discards it — so no tool can *extract* an application's state of affairs;
it can only be *told*. The architecture that follows has two clocks, one
mechanism, and one discipline.

*Document truth* serves the selected, sleeping application: whatever the
author wants tools to see lives in the RON dialect on purpose — save
games already do (immortal carries its `Rng` stream in `.ron`), maps will
(`.map.ron`), gameplay values will (the data tier above). Opening an
application in `sprite_util` then means: open its world document, read
which tilesets it names and at which cuts (`("basic_tiles", 2, 8)`), and
rebuild exactly those on the bench via the `set_atlas` machinery. The
document set is the app's identity while it sleeps.

*Live truth* serves the running application: it announces itself with a
small card in a well-known folder (say `target/frost/live/` — name, pid,
asset root, where its live file is; deleted on exit, stale entries pruned
by pid liveness), which is what "attach to the *selected* application"
means concretely — a tool enumerates the folder like a debugger lists
processes. The app then publishes `fn state_of_affairs(&self) ->
frost::ron::Val` — its own being in its own words, one hand-written
function, zero cost where absent, no reflection tax and no shared Rust
types across the tool boundary; tools read the tree with the parser they
already have.

The snapshot is *transition-driven, not clock-driven, and it carries
structure only*. Streaming data (tweens, lerps, anything without
transitions) does not join the telling — a tool that wants continuous
values wants a socket, someday, not a file. What is published is the
app's static structure, so the file is not a stream at all but a
document with an author who lives in a process: idle apps write nothing,
ever. The clock is a dirty check, and the house already owns both idioms
it needs: value-key dirtiness (the checker backdrop's
`if key != self.checker_key` — serialize once, compare against last
published, and since the serialization *is* the snapshot, the check is
nearly free and a forgotten flag cannot hide a change) and set-and-flush
(the journal collapsing repeats: mutation sites set a bit, one flush at
the end of `process()` coalesces a burst — an erase touching forty cells
publishes once, not forty watcher fires). Every publish carries a
monotonic `rev` (tools that coalesce watcher events still know something
happened, and in what order; an unchanged `rev` says the write was a
heartbeat, not a change), and the first write happens at boot — a
freshly loaded, never-mutated app still has a structure worth seeing,
because coming into existence is a transition. Liveness stays the
announce card's pid question; change and life are different questions
and do not mix in one file. What this converges on: a dirty-driven
structure publish is an *autosave with manners* — same serialization,
same clock, same dialect, different folder — which folds the
sleeping/running duality shut: the save a dead app leaves and the
snapshot a living app keeps become one format from one function, so
opening a sleeping application and attaching to a running one is one
reader, two folders.

*The telling landed 2026-10-10 as `frost::Teller`* (src/tell.rs) —
cards, live files, `rev`, and the self-healing folder exactly as
the paragraph sketches them, with one refinement the writing
demanded: instead of mutation sites setting a bit, the app's key is
*derived* from the discrete facts its snapshot holds (which beds are
planted, whether the garden was ever planted, whether it is over,
what pace it reads) — a derived key cannot forget to be dirty, and
the whole garden needed zero mark-calls to become tellable.
immortal's `state_of_affairs` speaks `told, config, watches, ships,
beds` — structure, never clocks; the boot publish proved *coming
into existence is a transition* (rev 1 before a single frame), and
the live run proved the rest: `grow_slowdown` saved to 2 re-told the
being as rev 2 three seconds later, reverted to 3 as rev 3, and the
idle minutes between wrote nothing. The folder's manners were proven
too: cards die with their apps (`Drop`), and a fresh boot pruned a
nightmare's worth of stale cards, orphaned documents, and crashed
pids by signal-zero liveness — leaving exactly one card: the app
that is actually running. The reader landed the same
day: `Teller::announce` lists the folder as a debugger lists
processes — pruning the departed, reading each surviving card in
the house dialect — and `Teller::tiding` returns a telling's rev
and being, resolving relative paths through the card's own answers
so guest and app need no shared working directory. `frost::ron`
gained `field`, the spoken lookup: readers who care about `ships`
ask by name, which is what makes the telling a dialect and not a
coincidence of order. `sprite_util` wears it as `App → Attach`
(cycling the living tellings, status band speaking each rev, card
absence meaning death); `examples/attune.rs` wears it as a console
— and proved the whole protocol live beside a running immortal:
attach at rev 1, re-told as rev 2 and rev 3 with the pace file's
edit and revert, and at the end, the card's absence spoken aloud:
*pid 40428 is gone — its card died with it*. Birth, change, and
death, over files, with the folder as the bus.

*One mechanism, both directions:* hot reload has the tool writing a
document while the app's watcher pulls; introspection has the app writing
a snapshot while the tool's watcher pulls. The watched-file primitive
designed for the reload ladder *is* the inter-tool bus — roles swap,
machinery doesn't. Sockets can wait until someone needs sub-frame
latency; files speak at editing speed. The name spine closes the loop:
snapshots name assets by name, so `sprite_util`, attached to a running
game, can answer "where does this tile I am editing appear?" — the
draw-list echo (a frame trace, if draws ever carry a debug tag) is a
later rung that has to earn its keep against the logic-side snapshot.

*One discipline:* tools write documents, never memories. There is
deliberately no push-into-the-running-app channel — Godot's bidirectional
scene editing is exactly the coupling frost refuses, an editor that must
own the truth or become a second author of a book it cannot read. The app
owns its being; the tools own the documents; the watcher is the only
courier. Summed up: Godot's editor edits the truth; frost's app *is* the
truth, so tools read documents (sleeping state) and snapshots (running
state), write only documents, and one watched-file mechanism carries
every word in both directions.

**Sprite minification: an invoice on file, unpaid on purpose.** Observed
rather than derived — the centipede's legs (`dogs_name`) and the garden's
wasps (`immortal`'s bugs) flicker as they move, and the drifting glow-eyes
(`eyes`) — the scene that minifies almost as hard — flicker not at all.
That pairing is the whole law in one sentence: a fetch that must squash a
dozen texels into one pixel and is granted four samples changes which
four every frame, but only artwork with fine detail to lose can show it —
every frame's wrong four average to the same answer where the art is
soft, so `Waspeye1`'s glow scales cleanly at `0.22 + 0.2·hash` while
detail-carrying sprites shimmer and motion makes the shimmer crawl. The
witnesses minify knowingly: the centipede scales its `250×448` legs to
`BODY_SPAN`, a fifth of the window's height, rotating every foot; the
wasp is drawn 40 px wide — `species.size / w` at `BUG_SIZE` — from a
258-px `Bug1a`, bobbing on its gait's scale as it walks, and the lice a
gentler 2× at `LICE_SIZE`. Neither flicker is a bloom failure: the chain
blurs what it is given, and its own pyramid-averaging keeps the wide halo
calm while the fetch-aliased radiance dances in the fine core — the fetch
sits upstream of both passes, and the fetch law is not a ledger bloom can
settle. When a design truly wants minification, the fix is Godot's move
cut to frost's size: a mip pyramid built at load for sprites that declare
they shrink, `mip_level_count` grown to match, and no change to
`sprite.wgsl` at all — its `textureSample` already asks which level fits
the quad (the discard-quad comment guards exactly that question's
derivatives); it can today answer only 0. Two debts come with the
pyramid, and neither is new to this house: the averaging must happen in
linear space — a box filter over sRGB codes is exactly the mistake the
sRGB envelope probe exists to keep out of the frame — and the
third-of-a-texture's memory, paid per shrinking sprite rather than per
engine, and asked of none the eye's art would decline. Until a design
pays them, the integer-scaling discipline holds: the shipped look
minifies nothing that shows it, and the invoice stays on file rather than
lost.


## Verification loop

```
cargo build              # expect EXIT 0; covers the lib AND the tool —
                           # both are default members of the workspace
cargo build --examples   # expect EXIT 0; no longer covers the tool
cargo build -p sprite_util   # expect EXIT 0; the tool on its own, now in
                               # tools/sprite_util/ as its own package
cargo test               # expect 412 lib (plus six ignored probes) + 3 integration
                           # + 7 doctests + 93 sprite_util + 7 noisefloor
                           # (the tools' unit tests run here now: a binary
                           # target is tested by default)
cargo test --examples    # expect 195 passed (unit tests inside the examples,
                           # centipede's rig included; ron_view/view.rs compiles
                           # into both ron_view and sprite_util, so its 2 fold
                           # tests run twice; the parser tests run in the lib;
                           # the tool's 68 moved tests are NOT here anymore —
                           # they are in the plain `cargo test` above)
cargo check --target wasm32-unknown-unknown -p frost
                           # expect EXIT 0; scoped to -p frost so the native-only
                           # tool is not built for wasm; dev-dependencies (arboard, clap,
                           # env_logger, ...) serve examples and tests only, so a lib-only
                           # check never compiles them
cargo run --example cursor   # visual check; closing the window exits 0
cargo run --example worm     # peristaltic crawl, edge wrap; exits 0
cargo run --example ron_view # fold rows, drag both scroll handles; exits 0
cargo run --example sound    # Space/L/+/- check; closing the window exits 0
cargo run --example button   # hover-scale + click-swoosh check; window exits 0
cargo run --example authored_hall   # bricks' shadows and bounces come from
                           # brick.ron (RUST_LOG=info logs the one-bake line);
                           # closing the window exits 0
cargo run --example widgets  # panels drag, widgets drive the face; exits 0
cargo run --example tomato_sprite   # overlay check; window exits 0
cargo run --example immortal   # 1920x1080 window: grass fills it 1:1; exits 0
cargo run -p sprite_util     # the tool, run from the workspace root; -i <png|folder>
```
