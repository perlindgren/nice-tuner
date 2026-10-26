# nice-tuner — working conventions

Stringed-instrument tuner: frost GUI (vendored fork) + nice-plug CLAP/VST3 backend.
Owner: perlindgren. Works on macOS (dev) and Windows (Reaper + Komplete Audio ASIO).

## Build & gates (all required before every commit)

- Cargo needs `CARGO_HOME=$PWD/.cargohome`.
- Gate: `bash scripts/check.sh > /tmp/gate.log 2>&1; echo "gate: $?"` — then check
  the printed code. NEVER pipe the gate into `tail`/`head` inside `&&` chains:
  pipeline exit codes mask failures and have landed broken commits here.
  Only chain `git commit` after a verified `gate: 0`.
- Windows build: `cargo build --release --features standalone-asio --bin nice-tuner-standalone`.

## Git

- Commits: `git -c commit.gpgsign=false commit`. Push: `GIT_TERMINAL_PROMPT=0 git push origin main`.
- The pre-push hook re-runs the gate — a refused push after a "green" gate means
  the gate result was masked; inspect `git log` before continuing.

## Editing discipline

- Use the edit/write tools for file changes; never python/sed heredoc patches
  (stale anchors + formatter rewrites → silent wrong edits; hit repeatedly in
  this repo). Re-read exact current text before anchoring.
- Simple state machines here get unit tests, not visual iteration (see
  src/meter.rs: pure struct + fabricated-clock tests replaced five rounds of
  eyeball churn).

## Architecture facts (non-obvious)

- RT path is allocation-free by contract (gated by `assert_process_allocs`);
  analysis runs on the UI thread via an rtrb ring; metering truth comes from
  RT atomics (per-channel pre-mix), never from the mono downmix.
- nice-plug 0.4.2: pinned, FloatParam has no setter (host owns params); its
  standalone always opens 2-channel input; its clap wants kebab-case backend
  keys (`core-audio`, not `coreaudio` — the settings panel writes these).
- Settings JSON lives in CWD; params (a4/gate/channel) are live atomics,
  audio settings respawn the process (500 ms teardown pause is intentional).
- frost fork: `ctx.text` centers the INK rect (not em box); wght axis real
  (FiraCode Variable max 700).

## Owner's taste (established, don't relitigate)

- Big type, bold-ish, minimal chrome; dashes instead of zeros; green only
  for in-tune; dropdowns not sliders; continuous over quantized motion.
- "Trust the driver": buffer options come from driver-reported ranges only.
- Verdict is strict ±3 cents; the gauge scale (±50¢) is the wide part.
- Do NOT file or comment upstream nice-plug issues.
