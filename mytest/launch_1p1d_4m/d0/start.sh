#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python launch_online_dp.py \
  --dp-size 8 \
  --tp-size 2 \
  --dp-size-local 8 \
  --dp-rank-start 0 \
  --dp-address 10.246.63.43 \
  --dp-rpc-port 16600 \
  --vllm-start-port 8005 \
  2>&1 | tee /home/z_offload/mytest/logs/d0.log
