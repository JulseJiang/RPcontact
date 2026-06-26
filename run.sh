#!/usr/bin/env bash
set -euo pipefail

python rpcontact_cli.py \
  -i examples/rpcontact_example_input.zip \
  -o output \
  --top-k 100
