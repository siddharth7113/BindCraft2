#!/usr/bin/env bash
# Run the Protenix validation batches. Two processes so their MSA-server queues overlap:
# the remote MSA queue, not the GPU, is the bottleneck (~3.8 GB used of 8 GB per job).
#
#   ./run_protenix.sh [colabfold|protenix]     (default colabfold)
#
# MSA server choice matters. ByteDance's default (protenix-server.com) sat PENDING indefinitely
# on 2026-09-23; api.colabfold.com is an independent MMseqs2 service and is the fallback.
# Check depth first:  curl -s https://api.colabfold.com/queue
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env-protenix.sh
MSA_MODE="${1:-colabfold}"

run () {  # $1=input json  $2=out dir  $3=gpu  $4=log
  CUDA_VISIBLE_DEVICES="$3" protenix pred \
      -i "$1" -o "$2" \
      -n protenix-v2 \
      --use_default_params True \
      --use_msa True \
      --msa_server_mode "$MSA_MODE" \
      > "$4" 2>&1
  echo "done: $1 -> $2 (exit $?)" >> tmp/protenix_batch.status
}

mkdir -p analysis/protenix/out tmp
: > tmp/protenix_batch.status
echo "MSA server mode: $MSA_MODE"
run analysis/protenix/rbp4cyc_input.json analysis/protenix/out/cyclic 0 tmp/protenix_cyclic.log &
run analysis/protenix/rbp4lin_input.json analysis/protenix/out/linear 1 tmp/protenix_linear.log &
wait
echo "ALL DONE" >> tmp/protenix_batch.status
