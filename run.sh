#!/usr/bin/env bash
set -euo pipefail

python rpcontact_cli.py \
  -i examples/input.zip \
  -o output \
  --top-k 100
