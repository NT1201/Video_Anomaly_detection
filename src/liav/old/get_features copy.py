import torch
import timm
import torchvision.transforms as transforms
import numpy as np
import cv2
from typing import List


class FeatureExtractor:
    """
    Feature extractor using pretrained ViT and Swin models from timm.
    These operate per-frame; temporal modeling must be added separately.
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        # Using image-based transformer models as fallback
        self.timesformer = timm.create_model('vit_base_patch16_224', pretrained=True).to(device).eval()
        self.swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True).to(device).eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def _prepare_frame_tensor(self, frame: np.ndarray) -> torch.Tensor:
        """
        Prepare a single image frame.
        """
        return self.transform(frame).unsqueeze(0).to(self.device)

    def extract_timesformer_features(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Average features of multiple frames using ViT model (simulated TimeSformer).
        """
        features = []
        for frame in frames:
            tensor = self._prepare_frame_tensor(frame)
            with torch.no_grad():
                features.append(self.timesformer.forward_features(tensor))
        return torch.mean(torch.stack(features), dim=0)

    def extract_swin_features(self, frames: List[np.ndarray]) -> torch.Tensor:
        """
        Average features of multiple frames using Swin model.
        """
        features = []
        for frame in frames:
            tensor = self._prepare_frame_tensor(frame)
            with torch.no_grad():
                features.append(self.swin.forward_features(tensor))
        return torch.mean(torch.stack(features), dim=0)