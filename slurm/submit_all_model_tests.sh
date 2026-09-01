#!/bin/bash
# Submit independent test jobs for every detector in parallel.
set -euo pipefail

REPO="${REPO:-$(pwd)}"
DATA="${DATA:-$REPO/data/harness_large}"
MANIFEST="${MANIFEST:-wildfake_benchmark.csv}"
OUT_ROOT="${OUT_ROOT:-$REPO/results/parallel_evaluation}"
BASELINE="${BASELINE:-$REPO/checkpoints/patchhead/pooled/baseline/checkpoint.pt}"
AWARE="${AWARE:-$REPO/checkpoints/patchhead/pooled/distortion_aware/checkpoint.pt}"
FILTER="${FILTER:-$REPO/filter_based_approach/models/mask_classifier.pt}"
DID="${DID:-$REPO/checkpoints/did/pooled_sd15_resnet18.pt}"

for required in "$DATA/$MANIFEST" "$BASELINE" "$AWARE" "$FILTER" "$DID"; do
  if [ ! -f "$required" ]; then
    echo "missing required evaluation input: $required" >&2
    exit 2
  fi
done

baseline_job=$(sbatch --parsable --export=ALL,REPO="$REPO",DATA="$DATA",MANIFEST="$MANIFEST",MODELS=patchhead_baseline,BASELINE="$BASELINE",OUT="$OUT_ROOT/patchhead" "$REPO/slurm/evaluate_patchhead.sh")
aware_job=$(sbatch --parsable --export=ALL,REPO="$REPO",DATA="$DATA",MANIFEST="$MANIFEST",MODELS=patchhead_distortion_aware,AWARE="$AWARE",OUT="$OUT_ROOT/distortion_aware" "$REPO/slurm/evaluate_patchhead.sh")
physics_job=$(sbatch --parsable --export=ALL,REPO="$REPO",DATA="$DATA",MANIFEST="$MANIFEST",OUT="$OUT_ROOT/physics" "$REPO/slurm/evaluate_physics.sh")
did_job=$(sbatch --parsable --export=ALL,REPO="$REPO",DATA="$DATA",MANIFEST="$MANIFEST",CKPT="$DID",OUT="$OUT_ROOT/did" "$REPO/slurm/evaluate_did_manifest.sh")
filter_job=$(sbatch --parsable --export=ALL,REPO="$REPO",DATA="$DATA",MANIFEST="$MANIFEST",CHECKPOINT="$FILTER",OUT="$OUT_ROOT/filter" "$REPO/slurm/evaluate_filter.sh")

echo "Submitted shared-manifest evaluations:"
echo "  PatchHead baseline:          $baseline_job"
echo "  PatchHead distortion-aware:  $aware_job"
echo "  Physics:                     $physics_job"
echo "  DID:                         $did_job"
echo "  filter:                      $filter_job"
