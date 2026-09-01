# Independent harness

The harness owns large-run orchestration without changing the Physics or
PatchHead packages. It reuses the existing cached provider fetcher, creates a
deterministic quick subset, and normalizes current model entrypoints into one
JSONL result stream.

## Fetch

```bash
python -m harness fetch \
  --output-dir data/harness_large \
  --quick-output-dir data/harness_quick \
  --base-per-class 3000 \
  --sid-per-class 3000 \
  --wildfake-per-source 3000 \
  --quick-per-class 200 \
  --seed 42
```

The current provider implementation uses the base quota for SID's three
labels, so `--base-per-class` and `--sid-per-class` should match for now.

For a network-enabled CPU SLURM job:

```bash
sbatch --export=ALL,REPO="$HOME/DistortAware" \
  slurm/fetch_harness_data.sh
```

## Quick before/after training

```bash
python -m harness train-patchhead \
  --data-dir data/harness_quick \
  --mode both \
  --epochs 1 \
  --output-dir runs/quick_training
```

The baseline uses `--no-distortion-aware`; the second run uses the current
distortion-aware model. Both use `--ds pooled` so the merged PatchHead
inference contract accepts the checkpoints.

## Evaluation

```bash
python -m harness evaluate \
  --data-dir data/harness_quick \
  --baseline-checkpoint runs/quick_training/baseline/checkpoint.pt \
  --aware-checkpoint runs/quick_training/distortion_aware/checkpoint.pt \
  --output-dir results/harness/quick_evaluation
```

Physics records retain aggregate/cue confidence and automatic proposal
confidence. These are physical-evidence confidence values, not AIGC
probabilities. Each evaluation also writes `records.csv`, `records.jsonl`,
`metrics.json`, a combined `report.md`, and per-model CSV/Markdown reports
under `models/<model>/`.
