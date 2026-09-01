#!/bin/bash
# Symlink two (or more) feature caches into one pooled cache — no re-extraction.
#   slurm/merge_cache.sh cache/feat_pooled_sd15_r256s10 \
#       cache/feat_wildfake_sd15_r256s10 cache/feat_sid_set_sd15_r256s10
# Then: sbatch --export=ALL,DS=pooled,SKIP_EXTRACT=1 slurm/pipeline.sh
set -e
dest=$1; shift
rm -rf "$dest"; mkdir -p "$dest"
for src in "$@"; do
  cp -rs --update=none "$(readlink -f "$src")/." "$dest/"
done
echo "pooled cache at $dest:"
for s in train test; do
  for d in "$dest/$s"/*/; do echo "  $s/$(basename "$d"): $(find -L "$d" -name '*.npz' | wc -l)"; done
done
