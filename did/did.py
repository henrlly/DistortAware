"""Difference-in-Difference (DID) reconstruction features for AIGC detection.

Follows "A Difference-in-Difference Approach to Detecting AI-Generated Images"
(arXiv:2602.23732): a diffusion model reconstructs an image via DDIM inversion +
resampling; we take the first-order reconstruction error (DIRE, |x - R(x)|) and
the second-order error  |x - x'| - |x' - x''|  where x' = R(x), x'' = R(x').

Reconstructor: Stable Diffusion v1.5 latent diffusion (UNet 0.86B + VAE 0.08B +
text-encoder 0.12B  ~= 1.07B params  < 2B constraint). Reconstruction is
unconditional (empty-prompt embedding).
"""
import numpy as np
import torch
from PIL import Image

MODEL_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
SCALE = 0.18215


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class DIDReconstructor:
    def __init__(self, res=192, steps=6, device=None, dtype=None):
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from transformers import CLIPTextModel, CLIPTokenizer

        self.device = device or get_device()
        self.dtype = dtype or (torch.float16 if self.device != "cpu" else torch.float32)
        self.res = res
        self.steps = steps

        v16 = dict(variant="fp16") if self.dtype == torch.float16 else {}
        # VAE kept in fp32 (fp16 VAE decode is numerically unstable on MPS)
        self.vae = AutoencoderKL.from_pretrained(MODEL_ID, subfolder="vae").to(
            self.device, torch.float32).eval()
        self.vae_dtype = torch.float32
        self.unet = UNet2DConditionModel.from_pretrained(
            MODEL_ID, subfolder="unet", **v16).to(self.device, self.dtype).eval()
        tok = CLIPTokenizer.from_pretrained(MODEL_ID, subfolder="tokenizer")
        te = CLIPTextModel.from_pretrained(
            MODEL_ID, subfolder="text_encoder", **v16).to(self.device, self.dtype).eval()
        with torch.no_grad():
            ids = tok("", padding="max_length", max_length=tok.model_max_length,
                      return_tensors="pt").input_ids.to(self.device)
            self.null_ctx = te(ids)[0]  # (1, 77, 768)
        del te

        betas = torch.linspace(0.00085 ** 0.5, 0.012 ** 0.5, 1000,
                               dtype=torch.float64) ** 2
        acp = torch.cumprod(1.0 - betas, dim=0)
        self.acp = acp.to(self.device, torch.float32)
        # evenly spaced timesteps, ascending: [t0, t1, ... t_{steps-1}]
        self.timesteps = torch.linspace(0, 999, steps + 1, dtype=torch.long)[:-1]

    # ---- image <-> latent -------------------------------------------------
    @torch.no_grad()
    def encode(self, x):  # x: (B,3,H,W) in [-1,1]
        x = x.to(self.device, self.vae_dtype)
        z = self.vae.encode(x).latent_dist.mean * SCALE
        return z.to(torch.float32)

    @torch.no_grad()
    def decode(self, z):  # -> (B,3,H,W) in [0,1]
        img = self.vae.decode(z.to(self.vae_dtype) / SCALE).sample
        return (img.to(torch.float32) / 2 + 0.5).clamp(0, 1)

    def _eps(self, z, t):
        ctx = self.null_ctx.expand(z.shape[0], -1, -1)
        tt = torch.full((z.shape[0],), int(t), device=self.device, dtype=torch.long)
        return self.unet(z.to(self.dtype), tt, encoder_hidden_states=ctx).sample.to(
            torch.float32)

    @torch.no_grad()
    def reconstruct_latent(self, z0):
        acp = self.acp
        ts = self.timesteps
        z = z0
        # --- DDIM inversion: ascend t0 -> t_{steps-1} ---
        for i in range(len(ts)):
            t = ts[i]
            t_next = ts[i + 1] if i + 1 < len(ts) else torch.tensor(999)
            a_t = acp[t]
            a_next = acp[t_next]
            eps = self._eps(z, t)
            x0 = (z - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            z = a_next.sqrt() * x0 + (1 - a_next).sqrt() * eps
        # --- DDIM sampling (eta=0): descend back down ---
        for i in reversed(range(len(ts))):
            t = ts[i + 1] if i + 1 < len(ts) else torch.tensor(999)
            t_prev = ts[i]
            a_t = acp[t] if i + 1 < len(ts) else acp[torch.tensor(999)]
            a_prev = acp[t_prev]
            eps = self._eps(z, t)
            x0 = (z - (1 - a_t).sqrt() * eps) / a_t.sqrt()
            z = a_prev.sqrt() * x0 + (1 - a_prev).sqrt() * eps
        return z

    @torch.no_grad()
    def reconstruct_image(self, x):  # x in [-1,1] (B,3,H,W) -> x' in [0,1]
        return self.decode(self.reconstruct_latent(self.encode(x)))

    @torch.no_grad()
    def did_features(self, x):
        """x: (B,3,H,W) in [0,1]. Returns d1 (B,3,H,W) in [0,1] and d2 signed."""
        x01 = x.to(self.device)
        xm = x01 * 2 - 1
        x1 = self.reconstruct_image(xm)               # x'
        x2 = self.reconstruct_image(x1 * 2 - 1)       # x'' = R(x')
        d1 = (x01 - x1).abs()
        d2 = (x01 - x1).abs() - (x1 - x2).abs()
        return d1.cpu(), d2.cpu()


SANA_MODEL_ID = "Efficient-Large-Model/Sana_1600M_512px_diffusers"


class SanaReconstructor:
    """DID reconstructor built on SANA-1.6B (DiT + DC-AE, rectified-flow).

    The DDIM inversion of the SD path becomes deterministic rectified-flow
    inversion: integrate  dz/dsigma = v_theta(z, sigma)  from data (sigma=0) up to
    noise (sigma=1) with Euler steps, then integrate back down to sigma=0.  No
    stochastic noise is added, so the map is a genuine reconstruction operator.

    Note: SANA's Gemma text encoder (~2B) is loaded only to cache one null-prompt
    embedding and is then freed; the resident reconstructor is DC-AE (~0.3B) +
    the 1.6B DiT.  This exceeds the <2B budget of the SD path and is run purely
    as a cross-reconstructor ablation.
    """

    def __init__(self, res=512, steps=10, device=None, dtype=None):
        import glob, os
        from diffusers import AutoencoderDC, SanaTransformer2DModel
        from transformers import AutoModel, AutoTokenizer
        from huggingface_hub.constants import HF_HUB_CACHE

        self.device = device or get_device()
        self.dtype = dtype or (torch.float16 if self.device != "cpu" else torch.float32)
        self.res = res
        self.steps = steps

        # resolve the local cache dir directly; loading by repo-id + subfolder
        # makes transformers/diffusers phone home for repo-root metadata even
        # offline, and snapshot_download rejects our partial (fp16-only) cache
        cache = os.environ.get("HF_HUB_CACHE") or HF_HUB_CACHE
        snaps = glob.glob(os.path.join(
            cache, "models--" + SANA_MODEL_ID.replace("/", "--"), "snapshots", "*"))
        if not snaps:
            raise RuntimeError(f"SANA weights not cached under {cache}; run slurm/dl_sana.py")
        root = max(snaps, key=os.path.getmtime)

        self.vae = AutoencoderDC.from_pretrained(
            os.path.join(root, "vae")).to(self.device, torch.float32).eval()
        self.vae_dtype = torch.float32
        self.vae_scale = float(getattr(self.vae.config, "scaling_factor", 0.41407))

        # single-file fp16 weights: sharded checkpoints hit the HF Hub even in
        # offline mode (diffusers _get_checkpoint_shard_files bug)
        self.transformer = SanaTransformer2DModel.from_pretrained(
            os.path.join(root, "transformer"), variant="fp16",
        ).to(self.device, self.dtype).eval()

        tok = AutoTokenizer.from_pretrained(os.path.join(root, "tokenizer"))
        te = AutoModel.from_pretrained(
            os.path.join(root, "text_encoder")).to(self.device, self.dtype).eval()
        with torch.no_grad():
            ins = tok("", padding="max_length", max_length=300, truncation=True,
                      return_tensors="pt").to(self.device)
            emb = te(ins.input_ids, attention_mask=ins.attention_mask).last_hidden_state
        self.null_ctx = emb.to(self.dtype)
        self.null_mask = ins.attention_mask
        del te

        # SANA uses rectified-flow (DPMSolver w/ flow_prediction, flow_shift=3.0).
        self.shift = 3.0
        # sigma schedule: ascending 0 -> 1 with the flow-matching resolution shift
        s = torch.linspace(0, 1, steps + 1, dtype=torch.float32)
        self.sigmas = (self.shift * s / (1 + (self.shift - 1) * s)).to(self.device)

    @torch.no_grad()
    def encode(self, x):
        z = self.vae.encode(x.to(self.device, self.vae_dtype)).latent
        return (z * self.vae_scale).to(torch.float32)

    @torch.no_grad()
    def decode(self, z):
        img = self.vae.decode((z / self.vae_scale).to(self.vae_dtype)).sample
        return (img.to(torch.float32) / 2 + 0.5).clamp(0, 1)

    def _v(self, z, sigma):
        t = torch.full((z.shape[0],), float(sigma) * 1000.0, device=self.device,
                       dtype=self.dtype)
        ctx = self.null_ctx.expand(z.shape[0], -1, -1)
        mask = self.null_mask.expand(z.shape[0], -1)
        out = self.transformer(hidden_states=z.to(self.dtype), timestep=t,
                               encoder_hidden_states=ctx,
                               encoder_attention_mask=mask).sample
        return out.to(torch.float32)

    @torch.no_grad()
    def reconstruct_latent(self, z0):
        sig = self.sigmas
        z = z0
        # inversion: data (sigma=0) -> noise (sigma=1), Euler on dz = v * dsigma
        for i in range(len(sig) - 1):
            v = self._v(z, sig[i])
            z = z + (sig[i + 1] - sig[i]) * v
        # resample: noise -> data
        for i in reversed(range(len(sig) - 1)):
            v = self._v(z, sig[i + 1])
            z = z + (sig[i] - sig[i + 1]) * v
        return z

    @torch.no_grad()
    def reconstruct_image(self, x):  # x in [-1,1]
        return self.decode(self.reconstruct_latent(self.encode(x)))

    @torch.no_grad()
    def did_features(self, x):
        x01 = x.to(self.device)
        xm = x01 * 2 - 1
        x1 = self.reconstruct_image(xm)
        x2 = self.reconstruct_image(x1 * 2 - 1)
        d1 = (x01 - x1).abs()
        d2 = (x01 - x1).abs() - (x1 - x2).abs()
        return d1.cpu(), d2.cpu()


def make_reconstructor(name="sd15", **kw):
    name = (name or "sd15").lower()
    if name in ("sd15", "sd", "sd1.5"):
        return DIDReconstructor(**kw)
    if name in ("sana16", "sana", "sana1.6b", "sana-1.6b"):
        kw.setdefault("res", 512)
        return SanaReconstructor(**kw)
    raise ValueError(f"unknown reconstructor {name!r}")


def load_image(path, res=256):
    im = Image.open(path).convert("RGB").resize((res, res), Image.Resampling.BICUBIC)
    return torch.from_numpy(np.asarray(im)).permute(2, 0, 1).float() / 255.0
