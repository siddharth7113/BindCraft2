#!/usr/bin/env bash
# Add Protenix seeds to the EXISTING validation outputs, for error bars on px_iptm/jaccard.
#
#   ./run_protenix_seeds.sh [seeds]          default 202,303,404,505
#
# No MSA server involved: rbp4*_input.json carry absolute pairedMsaPath/unpairedMsaPath into
# analysis/protenix/msa_cache, so this is pure GPU. ~29 s per job-seed after a ~100 s model load.
# Protenix writes <job>/seed_NNN/ alongside the existing seed_101, so nothing is overwritten.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
source ./env-protenix.sh
export TORCH_CUDA_ARCH_LIST=8.6          # skip the 4-minute five-architecture JIT rebuild
SEEDS="${1:-202,303,404,505}"

run () {  # $1=input json  $2=out dir  $3=gpu  $4=log
  CUDA_VISIBLE_DEVICES="$3" protenix pred \
      -i "$1" -o "$2" -n protenix-v2 \
      --use_default_params True --use_msa True -s "$SEEDS" \
      > "$4" 2>&1
  echo "done: $1 seeds=$SEEDS -> $2 (exit $?)" >> tmp/protenix_seeds.status
}

mkdir -p tmp
: > tmp/protenix_seeds.status
echo "seeds: $SEEDS"
run analysis/protenix/rbp4cyc_input.json analysis/protenix/out/cyclic 0 tmp/protenix_seeds_cyclic.log &
run analysis/protenix/rbp4lin_input.json analysis/protenix/out/linear 1 tmp/protenix_seeds_linear.log &
wait
echo "ALL DONE" >> tmp/protenix_seeds.status
