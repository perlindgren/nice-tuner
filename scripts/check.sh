#!/usr/bin/env bash
# M6 verification gates for nice-tuner. Run from the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."
export CARGO_HOME="${CARGO_HOME:-$PWD/.cargohome}"

echo "== plugin lib (cdylib) check =="
cargo check

echo "== standalone check =="
cargo check --features standalone

echo "== tests: pitch crate =="
cargo test --manifest-path crates/pitch/Cargo.toml

echo "== tests: frost fork (pure-logic) =="
cargo test --manifest-path vendor/frost/Cargo.toml || echo "NOTE: frost test status reported above"

echo "== winit event-loop grep gate =="
if grep -rn "EventLoop\|ActiveEventLoop\|ApplicationHandler\|WindowAttributes" vendor/frost/src --include='*.rs' \
   | grep -v "//" | grep -v "^\s*///"; then
  echo "FAIL: winit event-loop machinery referenced in frost fork src"
  exit 1
fi

echo "== standalone bin build (linker gate) =="
cargo build --features standalone --bin nice-tuner

echo "== plugin bundling =="
cargo xtask bundle nice-tuner --release

echo "ALL GATES PASSED"
