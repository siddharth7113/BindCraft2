# BC2 environment for AIMecoSPECS. Usage: source env.sh
# Keeps weights, compile cache and scratch inside the repo instead of ~/.cache and /tmp.
BINDCRAFT_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export BINDCRAFT_HOME
export BINDCRAFT_WEIGHTS="$BINDCRAFT_HOME/cache"          # AlphaFold params -> cache/alphafold
export JAX_COMPILATION_CACHE_DIR="$BINDCRAFT_HOME/cache/compile"
export TMPDIR="$BINDCRAFT_HOME/tmp"
mkdir -p "$TMPDIR" "$JAX_COMPILATION_CACHE_DIR"

# This node carries three 8 GB Ampere cards plus a 6 GB GTX 1660 Ti. The 1660 Ti cannot hold a
# worker for a 174-residue target, so select the >=8 GB cards by memory rather than by index.
if command -v nvidia-smi > /dev/null 2>&1; then
  good_gpus=$(nvidia-smi --query-gpu=index,memory.total --format=csv,noheader,nounits \
    | awk -F', *' '$2 >= 8000 {printf "%s%s", sep, $1; sep=","}')
  [ -n "$good_gpus" ] && export BINDCRAFT_GPU_IDS="$good_gpus"
fi

source "$BINDCRAFT_HOME/.venv/bin/activate"
