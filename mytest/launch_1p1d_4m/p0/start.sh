#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python launch_online_dp.py \
  --dp-size 1 \
  --tp-size 16 \
  --dp-size-local 1 \
  --dp-rank-start 0 \
  --dp-address 10.246.63.45 \
  --dp-rpc-port 16600 \
  --vllm-start-port 8004 \
  2>&1 | tee /home/z_offload/mytest/logs/p0.log
