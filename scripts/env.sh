# Sourced by every script: repo on PYTHONPATH, data root, conda environments.
#   use_env carla | cwm
# Cluster-specific settings (e.g. RTCWM_EXCLUDE, a comma-separated list of nodes to avoid, used as
# `sbatch --exclude=$RTCWM_EXCLUDE`) go in ops/cluster.sh, which is not in git.
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH=$REPO${PYTHONPATH:+:$PYTHONPATH}
export RTCWM_DATA=${RTCWM_DATA:-$REPO}
export RTCWM_EXCLUDE=${RTCWM_EXCLUDE:-}
[ -f "$REPO/ops/cluster.sh" ] && source "$REPO/ops/cluster.sh"
use_env() {
  source "${CONDA_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
  conda activate "$1"
}
