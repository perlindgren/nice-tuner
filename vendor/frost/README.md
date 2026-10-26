# FROST

A game engine experiment by N65 Game Research Station (N65 GRS).

Why?

- Primarily to gain experience of local hosted AI workflows.
- Sufficiently complex problem to challenge ability to reason on realistic use-cases.
- To gain human experience on inner workings of a games engine, starting from very limited, surface level understanding.
- If successful to provide the core mechanics of a library based games engine, following the Rust paradigms (unlike e.g., Bevy, that introduces an additional ECS abstraction to offer shared mutability.)

See [SETUP.md](SETUP.md) for AI based workflow.

## Examples

Run the examples with:

```shell
RUST_LOG=info cargo run --example gizmos
```

or under Win11 with Powershell:

```powershell
$env:RUST_LOG="info"; cargo run --example gizmos
```

## Repository hooks

The versioned hooks in `.githooks/` gate commits (fmt + clippy) and pushes
(tests, plus a `wasm32-unknown-unknown` check of the web target — every other
gate compiles the host target only, so a `cfg`-gated import in the backend
walks through all of them and breaks `web/build.sh` later). Every push runs
every check, `main` included, and a clone missing the wasm target is refused
rather than warned, with the remedy printed beside the refusal:

```shell
rustup target add wasm32-unknown-unknown
```

There is no CI behind this repository, so the hook is the only gate the history
has — and a check that runs only when someone remembered was not a check. Git
runs them only once this clone points at them — the trigger is a per-clone
setting, never cloned itself. After every fresh clone, once:

```shell
sh .githooks/install.sh
```

(or clone with `git clone --config core.hooksPath=.githooks <url>` and skip
the step). On Windows from PowerShell or cmd, where `sh` is usually not on
the PATH, the script's whole job is the one command:
`git config core.hooksPath .githooks` — run it from any shell (from Git
Bash, `sh .githooks/install.sh` works as everywhere). They are a gate you
opt into, not a wall: `--no-verify` skips them by design.

## License

The `assets/fonts/FiraCode-VariableFont_wght.ttf` is a redistribution under OFL.

Copyright to the Developers (Per Lindgren/Iris Hulsman), for other assets and code. We will keep development sources available for now, hopefully we can find a license allowing the sources to be open and available (giving back to the game dev community), while protecting all rights to releases and distributions.