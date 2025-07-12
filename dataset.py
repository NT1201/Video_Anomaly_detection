import os, glob, numpy as np, torch
from torch.utils.data import Dataset


class VideoFeatureDataset(Dataset):
    """
    Loads fixed-length segment features for MIL-style video anomaly detection.

    Folder layout (case-insensitive):

        data/
            Training/
                Anomaly_I3D_RGB_NPY/
                Anomaly_TimesFormer_NPY/
                Normal_ ... etc.
            Testing/
                Anomaly_...
                Normal_ ...

    Supported feat_type:
        "timesformer"  → (768-D)
        "i3d"          → (1024-D)
        "both"         → concatenate (1792-D)
    """

    def __init__(self, root_dir="data", mode="train", label="normal",
                 feat_type="timesformer", num_seg=32, l2_norm=True):

        mode  = mode.lower();  label = label.lower();  feat_type = feat_type.lower()
        assert mode  in ("train", "test")
        assert label in ("normal", "anomaly")
        assert feat_type in ("timesformer", "i3d", "both")

        self.num_seg, self.l2_norm, self.feat_type = num_seg, l2_norm, feat_type

        # --- resolve Training/Traning typo ---------------------------------
        mode_dir = "Training" if mode == "train" else "Testing"
        if not os.path.isdir(os.path.join(root_dir, mode_dir)):
            mode_dir = "Traning"   # fall back to miss-spelled folder

        # choose a *base* directory to list npy files from
        suffix = "I3D_RGB" if feat_type in ("i3d", "both") else "TimesFormer"
        sub    = f"{label.capitalize()}_{suffix}_NPY"
        self.files = sorted(glob.glob(os.path.join(root_dir, mode_dir, sub, "*.npy")))
        if not self.files:
            raise RuntimeError(f"No .npy files under {os.path.join(root_dir, mode_dir, sub)}")

        self._feat_dim = None  # discovered lazily

    # ------------------------------------------------------------------ #
    @staticmethod
    def _sample_fixed(arr, n_seg):
        """Uniform subsample / pad to n_seg rows."""
        if arr.shape[0] == n_seg:
            return arr
        idx = np.linspace(0, arr.shape[0] - 1, n_seg).astype(int)
        return arr[idx]

    # ------------------------------------------------------------------ #
    def __len__(self):  return len(self.files)

    def __getitem__(self, idx):
        base_path = self.files[idx]                                  # I3D file
        vid_i3d   = np.load(base_path).astype(np.float32)            # (T,1024)

        if self.feat_type == "i3d":
            x = vid_i3d
        elif self.feat_type == "timesformer":
            tsf_path = base_path.replace("_I3D_RGB_NPY", "_TimesFormer_NPY")
            x = np.load(tsf_path).astype(np.float32)                 # (T,768)
        else:  # "both"
            tsf_path = base_path.replace("_I3D_RGB_NPY", "_TimesFormer_NPY")
            vid_tsf  = np.load(tsf_path).astype(np.float32)          # (T,768)
            x = np.concatenate([vid_tsf, vid_i3d], axis=1)           # (T,1792)

        # (S, D) after uniform sampling
        x = self._sample_fixed(x, self.num_seg)

        if self.l2_norm:
            x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8

        if self._feat_dim is None:
            self._feat_dim = x.shape[1]

        return torch.from_numpy(x)                                   # (S,D)

    # ------------------------------------------------------------------ #
    @property
    def feat_dim(self):
        if self._feat_dim is None: _ = self[0]
        return self._feat_dim
