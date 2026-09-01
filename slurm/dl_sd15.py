from huggingface_hub import snapshot_download
p = snapshot_download(
    'stable-diffusion-v1-5/stable-diffusion-v1-5',
    allow_patterns=['vae/diffusion_pytorch_model.safetensors', 'vae/config.json',
                    'unet/diffusion_pytorch_model.fp16.safetensors', 'unet/config.json',
                    'text_encoder/model.fp16.safetensors', 'text_encoder/config.json',
                    'tokenizer/*', 'model_index.json'],
    max_workers=3)
print('SD1.5 at', p)
