#!/usr/bin/env bash
# M6 verification gates for nice-tuner. Run from the repo root.
set -euo pipefail
cd "$(dirname "$0")/.."
export CARGO_HOME="${CARGO_HOME:-$PWD/.cargohome}"

echo "== cargo fmt =="
cargo fmt --all -- --check

echo "== clippy (workspace + standalone/ASIO configs, deny warnings) =="
cargo clippy --workspace --all-targets -- -D warnings
cargo clippy -p nice-tuner --features standalone --all-targets -- -D warnings
# Inert on non-Windows (cpal gates its ASIO host by target); real there.
cargo clippy -p nice-tuner --features standalone-asio --all-targets -- -D warnings

echo "== plugin lib (cdylib) check =="
cargo check

echo "== standalone check =="
cargo check --features standalone

echo "== windows target typecheck (cfg(windows) code the macOS gate never compiles) =="
# `standalone` WITHOUT asio here: ASIO's C++ build script cannot
# cross-compile from macOS. This still type-and-name-checks all the
# Win32 window-placement code, which no macOS pass ever sees.
if rustup target list --installed 2>/dev/null | grep -q x86_64-pc-windows-msvc; then
  cargo clippy -p nice-tuner --features standalone --target x86_64-pc-windows-msvc -- -D warnings
else
  echo "skipped: rustup target add x86_64-pc-windows-msvc"
fi

echo "== tests: workspace (all members + doctests) =="
cargo test --workspace

echo "== winit event-loop grep gate =="
if grep -rn "EventLoop\|ActiveEventLoop\|ApplicationHandler\|WindowAttributes" vendor/frost/src --include='*.rs' \
   | grep -v "//" | grep -v "^\s*///"; then
  echo "FAIL: winit event-loop machinery referenced in frost fork src"
  exit 1
fi

echo "== standalone bin build (linker gate) =="
cargo build --features standalone --bin nice-tuner-standalone

echo "== plugin bundling =="
cargo xtask bundle nice-tuner --release

echo "ALL GATES PASSED"
