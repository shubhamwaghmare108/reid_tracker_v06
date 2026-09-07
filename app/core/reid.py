"""Body appearance embeddings used to re-identify a person across frames."""
from __future__ import annotations
import numpy as np
import torch


class FeatureExtractor:
    """Adapter around TorchReID that returns one normalized vector per person crop."""
    def __init__(self, model_name: str = 'osnet_x1_0', model_path: str = '', device: str = 'cpu'):
        """Support both TorchReID import layouts used by different releases."""
        try:
            from torchreid.utils import FeatureExtractor as TorchExtractor
        except ModuleNotFoundError:
            from torchreid.reid.utils import FeatureExtractor as TorchExtractor
        self.extractor = TorchExtractor(
            model_name=model_name,
            model_path=model_path if model_path else None,
            device=device,
            verbose=False
        )

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Convert OpenCV BGR input to RGB, run the model, and L2-normalize its output."""
        if image is None or image.size == 0:
            raise ValueError('Empty crop')
        if image.ndim == 3 and image.shape[2] == 3:
            rgb = image[:, :, ::-1].copy()
        else:
            rgb = image
        # Inference does not need gradients, which saves memory and computation.
        with torch.no_grad():
            feature = self.extractor(rgb)
        feature = feature.cpu().numpy().flatten().astype(np.float32)
        norm = np.linalg.norm(feature)
        if norm > 1e-12:
            feature /= norm
        return feature
