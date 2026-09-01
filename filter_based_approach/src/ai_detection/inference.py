"""Inference API for the trained segmentation and classification model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .model import ResidualUNet, image_tensor


class MaskPredictor:
    def __init__(self, checkpoint_path: str | Path = "models/mask_classifier.pt") -> None:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        self.size = int(checkpoint["size"])
        self.class_names = list(checkpoint["class_names"])
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = ResidualUNet().to(self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

    @torch.inference_mode()
    def predict_pil(self, image: Image.Image) -> tuple[dict[str, object], np.ndarray]:
        mask_logits, class_logits = self.model(
            image_tensor(image, self.size).unsqueeze(0).to(self.device)
        )
        probabilities = torch.softmax(class_logits, dim=1)[0].cpu().numpy()
        mask = torch.sigmoid(mask_logits)[0, 0].cpu().numpy()
        winner = int(probabilities.argmax())
        result = {
            "prediction": self.class_names[winner],
            "binary_prediction": "authentic" if winner == 0 else "ai",
            "confidence": float(probabilities[winner]),
            "ai_probability": float(probabilities[1] + probabilities[2]),
            "class_probabilities": {
                name: float(probability)
                for name, probability in zip(self.class_names, probabilities)
            },
            "predicted_mask_area": float((mask >= 0.5).mean()),
        }
        return result, mask

    def predict_path(self, path: str | Path) -> tuple[dict[str, object], np.ndarray]:
        with Image.open(path) as image:
            return self.predict_pil(image.convert("RGB"))
