#!/usr/bin/env bash
#
# Smoke check that nothing inside ``src/`` carries a hard-coded credential
# or non-localhost URL. Run by CI on every PR.

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

violations=0

# API key prefixes we never want in source.
patterns=(
    "sk-[A-Za-z0-9]\{10,\}"
    "pa-[A-Za-z0-9]\{10,\}"
    "AKIA[0-9A-Z]\{16\}"
    "ASIA[0-9A-Z]\{16\}"
    "AIDA[0-9A-Z]\{16\}"
)

for pattern in "${patterns[@]}"; do
    if grep -REn "$pattern" src/ 2>/dev/null; then
        echo "ERROR: forbidden secret pattern matched: $pattern"
        violations=$((violations + 1))
    fi
done

# DATABASE_URL must never be hard-coded with a real-looking host.
if grep -RnE 'postgresql\+[a-z]+://[a-zA-Z0-9_]+:[a-zA-Z0-9_]+@(?!localhost|127\.0\.0\.1)' src/ 2>/dev/null; then
    echo "ERROR: hard-coded non-localhost Postgres URL detected in src/"
    violations=$((violations + 1))
fi

if [[ "$violations" -gt 0 ]]; then
    exit 1
fi

echo "OK: no hard-coded secrets detected"
