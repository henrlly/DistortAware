"""Download a handful of SID_Set validation parquet shards into the HF cache
(login node: file download only, parsing happens later in a job)."""
import sys
from huggingface_hub import hf_hub_download
n = int(sys.argv[1]) if len(sys.argv) > 1 else 8
for i in range(n):
    f = f"data/validation-{i:05d}-of-00034.parquet"
    print(hf_hub_download("saberzl/SID_Set", f, repo_type="dataset"), flush=True)
