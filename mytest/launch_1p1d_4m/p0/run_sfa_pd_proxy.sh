#!/bin/bash
# =============================================================================
# SFA PD-disaggregated proxy / metaserver launcher.
#
# This connector only needs the proxy for the metaserver rendezvous (D posts
# remote_block_ids / remote_host / remote_port; proxy relays them to P). The
# per-layer base-addr + te_rpc_port exchange happens over a ZMQ side channel
# directly between P and D, NOT through the proxy.
#
# The existing layerwise proxy is protocol-agnostic (it transparently forwards
# kv_transfer_params), so we reuse it unchanged:
#   load_balance_proxy_layerwise_server_example.py
#
# GOTCHA: the layerwise proxy forbids --host 0.0.0.0 (it builds the metaserver
# URL from {host}:{port} that D must be able to reach). Use a concrete IP:
# single-box -> 127.0.0.1; multi-host -> the proxy's reachable IP.
#
# START ORDER: 1) D (run_sfa_pd_decode.sh)  2) P (run_sfa_pd_prefill.sh)
#              3) this proxy  4) send requests to the proxy's /v1/completions
# =============================================================================
export http_proxy=""
export https_proxy=""
export no_proxy="localhost,127.0.0.1"

set -euo pipefail

# export PYTHONPATH=/mnt/share_space/l00933205/code/mte_fused/vllm-ascend:$PYTHONPATH

# HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------- CONFIG (edit me) -------------------------------
PROXY_HOST="10.246.63.45"                  # MUST be a concrete IP (not 0.0.0.0); reachable from D
PROXY_PORT=8593                            # clients send requests here
P_HOST="10.246.63.45"; P_PORT=8421         # matches run_sfa_pd_prefill.sh SERVE_*
D_HOST="10.246.63.45"; D_PORT=7450         # matches run_sfa_pd_decode.sh SERVE_*
# ----------------------------------------------------------------------------

exec python "load_balance_proxy_layerwise_server_example.py" \
  --host "$PROXY_HOST" \
  --port "$PROXY_PORT" \
  --prefiller-hosts 10.246.63.45 10.246.63.43 \
  --prefiller-ports 8421 8421 \
  --decoder-hosts 10.246.63.47 10.246.63.47 10.246.63.47 10.246.63.47 10.246.63.47 10.246.63.47 10.246.63.47 10.246.63.47 10.246.63.48 10.246.63.48 10.246.63.48 10.246.63.48 10.246.63.48 10.246.63.48 10.246.63.48 10.246.63.48 \
  --decoder-ports 7450 7451 7452 7453 7454 7455 7456 7457 7450 7451 7452 7453 7454 7455 7456 7457
