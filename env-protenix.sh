# Protenix environment. Usage: source env-protenix.sh
# Separate from env.sh: BC2 is JAX, Protenix is Torch - they must not share a venv.
PROTENIX_HOME="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROTENIX_HOME
export TMPDIR="$PROTENIX_HOME/tmp"
# Checkpoints live at $PROTENIX_ROOT_DIR/checkpoint (default $HOME) - keep them in the repo,
# matching how env.sh keeps the AlphaFold params under cache/.
export PROTENIX_ROOT_DIR="$PROTENIX_HOME/cache/protenix_root"
# Only sm_86 is needed here (3070/3070 Ti). Without this, the JIT layer-norm kernel
# rebuilds for sm_80/86/89/90/100 - five architectures on four cores.
export TORCH_CUDA_ARCH_LIST="8.6"
export MAX_JOBS=4
mkdir -p "$TMPDIR" "$PROTENIX_ROOT_DIR/checkpoint"
source "$PROTENIX_HOME/.venv-protenix/bin/activate"
