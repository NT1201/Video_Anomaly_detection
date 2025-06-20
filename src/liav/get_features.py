import torch
import torch.nn.functional as F
import warnings
from transformers import logging, VideoMAEModel, VideoMAEImageProcessor


# Silence all warnings from Hugging Face Transformers
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

class FeatureExtractor:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        from transformers import TimesformerModel, AutoImageProcessor        
        self.processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
        self.model = TimesformerModel.from_pretrained("facebook/timesformer-base-finetuned-k400")
        # Load pre-trained TimeSformer
        # self.processor = VideoMAEImageProcessor.from_pretrained("MCG-NJU/videomae-large")
        # self.model = VideoMAEModel.from_pretrained("MCG-NJU/videomae-large")

        self.model.eval().to(self.device)

    def extract_timesformer_features(self, segment: torch.Tensor) -> torch.Tensor:
        """
        Extract features using TimeSformer model.
        Input: [T, C, H, W] tensor (e.g., [16, 3, 224, 224])
        Output: L2-normalized feature vector [1024]
        """
        if isinstance(segment, list):
            segment = torch.stack(segment)

        segment = segment.permute(0, 2, 3, 1)  # [T, H, W, C]
        segment = torch.clamp(segment, 0.0, 1.0).cpu().numpy()
        inputs = self.processor([frame for frame in segment], return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            features = outputs.last_hidden_state[:, 0, :]  # CLS token output

        features = F.normalize(features, p=2, dim=1)
        #print(f" TimeSformer feature shape: {features.shape}")
        #print("First segment features:", features)  # [1024] vector
        return features.squeeze(0).cpu()