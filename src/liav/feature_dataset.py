import os
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from loading_videos import load_video_segments
from get_features import FeatureExtractor

class VideoFeatureDataset(Dataset):
    def __init__(self, base_path):
        self.entries = []
        self.base_path = Path(base_path)
        self.feature_extractor = FeatureExtractor()

        for category in ['Normal', 'Anomaly']:
            category_path = self.base_path / category
            label = 0 if category.lower() == 'normal' else 1
            save_dir = self.base_path / f"{category}_NPY"
            save_dir.mkdir(parents=True, exist_ok=True)

            video_files = list(category_path.glob("*.mp4"))
            for video_file in tqdm(video_files, desc=f"Processing {category}", unit="video"):
                segments = load_video_segments(str(video_file))
                features = [
                    self.feature_extractor.extract_timesformer_features(segment).numpy()
                    for segment in tqdm(segments, desc=f"{video_file.name}", leave=False)
                ]
                features_array = np.stack(features)
                print(f"{video_file.name} → Segments: {len(segments)}, Feature shape: {features_array.shape}")

                save_path = save_dir / f"{video_file.stem}.npy"
                np.save(save_path, features_array)
                print(f" Saved features to {save_path} with shape {features_array.shape}")

                self.entries.append({
                    "filename": video_file.name,
                    "label": label,
                    "features": features_array
                })

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        return self.entries[idx]
