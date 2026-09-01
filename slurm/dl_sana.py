from huggingface_hub import snapshot_download
p = snapshot_download(
    'Efficient-Large-Model/Sana_1600M_512px_diffusers',
    allow_patterns=[
        'model_index.json', 'scheduler/*', 'tokenizer/*',
        'vae/config.json', 'vae/diffusion_pytorch_model.safetensors',
        'transformer/config.json',
        'transformer/diffusion_pytorch_model-*-of-*.safetensors',
        'transformer/diffusion_pytorch_model.safetensors.index.json',
        'text_encoder/config.json',
        'text_encoder/model-*-of-*.safetensors',
        'text_encoder/model.safetensors.index.json',
    ],
    max_workers=3)
print('SANA at', p)
