#!/bin/sh
# Activate the versioned hooks for THIS clone.
#
# Git only runs hooks it finds in .git/hooks/ — and .git/ is never cloned,
# so the link must be made once per clone: a single local config value
# redirects git at the versioned .githooks/ folder. Nothing else in the
# repo can do this from the tree side (git ignores hook settings in
# shared configs on purpose), so this file is the opt-in ritual:
#
#     sh .githooks/install.sh
#
# A fresh clone can skip the ritual and state it at creation time instead:
#
#     git clone --config core.hooksPath=.githooks <url>
#
# Idempotent: safe to re-run any time; it just re-asserts the same value.
# On Windows: run it from Git Bash; from PowerShell or cmd (no `sh` on the
# PATH), the equivalent is just `git config core.hooksPath .githooks` —
# the chmods below are no-ops there, since Windows git executes hooks
# through its bundled sh.exe whatever the file's mode says, and the
# scripts themselves stay LF via the repo's .gitattributes.
# Mind the honest limit: hooks are a gate you opt into, not a wall —
# `git commit --no-verify` skips them by design. The wall, if you ever
# want one, is server-side (a CI check or a pre-receive hook).
set -e
git config core.hooksPath .githooks
chmod +x "$(git rev-parse --show-toplevel)/.githooks/pre-commit" \
         "$(git rev-parse --show-toplevel)/.githooks/pre-push"
echo "hooks active for this clone: $(git rev-parse --show-toplevel)"
echo "  core.hooksPath = $(git config core.hooksPath)"
