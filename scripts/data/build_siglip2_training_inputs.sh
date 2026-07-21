#!/usr/bin/env bash
# Stage 4 convenience wrapper: build the pinned SigLIP2 Giant training inputs.
set -euo pipefail

CB_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CB_PYTHON="${CONCEPTBASIS_PYTHON:-${CB_REPO_ROOT}/.venv/bin/python}"

exec "${CB_PYTHON}" "${CB_REPO_ROOT}/scripts/data/build_training_inputs.py" \
  --encoder siglip2-giant \
  "$@"
