#!/bin/bash
#SBATCH -G a100-40
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --output=job_%j.out
#SBATCH --error=job_%j.err
set -e
source "${REPO:-$HOME/DistortAware}/slurm/_env.sh"
python - <<'PY'
import torch, torchvision, diffusers, transformers
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("device count", torch.cuda.device_count(), torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device="cuda")
y = (x @ x).sum().item()
print("matmul ok", y)
from diffusers import AutoencoderKL
vae = AutoencoderKL.from_pretrained("stable-diffusion-v1-5/stable-diffusion-v1-5", subfolder="vae").cuda().eval()
with torch.no_grad():
    z = vae.encode(torch.randn(1,3,192,192, device="cuda")).latent_dist.mean
print("vae encode ok", tuple(z.shape))
from torchvision.models import resnet18, ResNet18_Weights
resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
print("resnet18 weights ok")
print("ALL GOOD")
PY
