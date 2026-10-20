#!/usr/bin/env bash
# Job "inventory" (runpod-job workflow in Osirus): read-only listing of the network volume.
set -euo pipefail
python3 "$(dirname "$0")/inventory.py" /workspace "$1"
