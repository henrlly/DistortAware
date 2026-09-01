# Method entry points

Each method exposes a `run(input_path, ...) -> dict` function for one image.
The returned dictionary has the shared keys `method`, `image_path`, `score`,
`score_kind`, `confidence`, `threshold`, `decision`, and `details`.

```python
from patchhead.entrypoint import run

result = run("image.jpg", checkpoint="patchhead/checkpoints/patchhead.pt")
```

The equivalent modules are:

- `did.entrypoint`: trained DID classifier plus its diffusion reconstructor.
- `patchhead.entrypoint`: trained PatchHead detector.
- `filter_based_approach.entrypoint`: trained residual/filter baseline.
- `physics_engine.entrypoint`: Physics evidence sidecar; run with
  `PYTHONPATH=physics/src`.

The same modules provide batch CLIs used by the harness. They accept one
materialized image directory, load the model once, and write a JSON list using
the same output keys:

```bash
python -m patchhead.entrypoint --image-dir images --checkpoint checkpoint.pt --output predictions.json
python -m patchhead.entrypoint --image-dir images --checkpoint aware.pt --distortion-aware --output predictions.json
python -m filter_based_approach.entrypoint --image-dir images --checkpoint filter_based_approach/models/mask_classifier.pt --output predictions.json
PYTHONPATH=physics/src python -m physics_engine.entrypoint --image-dir images --output predictions.json
python -m did.entrypoint --image-dir images --checkpoint checkpoints/did/pooled_sd15_resnet18.pt --output predictions.json
```

The legacy `patchhead/infer.py` one-logit production contract remains separate;
three-class benchmark checkpoints are evaluated through
`patchhead.entrypoint`.

Only DID, PatchHead, and the residual/filter baseline require training. The
Physics engine has no learned checkpoint. DID's diffusion reconstructor and
PatchHead's DINOv3 backbone are pretrained components; their detector heads
still require trained checkpoints.

The detector packages are independent: DID does not import PatchHead or
Physics, PatchHead does not import DID or Physics, and the filter baseline does
not import another detector. The root `harness/` and `infer.py` modules are
orchestration code and may intentionally call multiple methods.
