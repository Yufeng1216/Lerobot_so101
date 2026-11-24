#!/bin/bash
# Submit a training job to Slurm with this project's settings.
#
#   bash cloud/slurm/submit.sh diffusion          # new run, or continue it if it was cut off
#
# Account, partition and time limit come from cloud/slurm/site.env (git-ignored, see
# site.env.example), so nothing user-specific is stored in the repository.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
POLICY="${1:?usage: bash cloud/slurm/submit.sh <policy>   (diffusion)}"

if [ ! -f "$HERE/site.env" ]; then
    echo "Missing $HERE/site.env: copy site.env.example to site.env and fill it in." >&2
    exit 1
fi
source "$HERE/site.env"

# Refuse to submit with a non-allowed account (checked again inside the job).
case "$ACCOUNT" in
    "$ALLOWED_ACCOUNT_PREFIX"*) ;;
    *) echo "ERROR: ACCOUNT=$ACCOUNT does not start with ALLOWED_ACCOUNT_PREFIX=$ALLOWED_ACCOUNT_PREFIX" >&2; exit 1 ;;
esac

mkdir -p logs                                   # Slurm does not create the log directory
export ALLOWED_ACCOUNT_PREFIX                   # passed into the job's environment
sbatch --account="$ACCOUNT" --partition="$PARTITION" --time="$TIME" \
       --job-name="$POLICY" "$HERE/train.sbatch" "$POLICY"
