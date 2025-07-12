# dataset.py
import os, glob, numpy as np, torch
from torch.utils.data import Dataset


class VideoFeatureDataset(Dataset):
    """
    Loads fixed-length segment features for MIL-style video anomaly detection.

    Expected directory layout (case-insensitive):

        data/
          Training/  or  Traning/      ← spelling in some releases
              Normal_I3D_RGB_NPY/
              Normal_TimesFormer_NPY/
              Anomaly_I3D_RGB_NPY/
              Anomaly_TimesFormer_NPY/
          Testing/
              ... same sub-folders ...

    Supported feat_type
        "timesformer"  -> 768-D vectors
        "i3d"          -> 1024-D vectors
        "both"         -> [TimesFormer; I3D] concatenation (1792-D)
    """

    def __init__(self,
                 root_dir="data",
                 mode="train",
                 label="normal",
                 feat_type="timesformer",
                 num_seg=32,
                 l2_norm=True,
                 gt_file="gt_windows.npy"  # optional test-only annotation
                 ):
        mode  = mode.lower()
        label = label.lower()
        feat_type = feat_type.lower()

        assert mode  in ("train", "test")
        assert label in ("normal", "anomaly")
        assert feat_type in ("timesformer", "i3d", "both")

        self.num_seg   = num_seg
        self.l2_norm   = l2_norm
        self.feat_type = feat_type

        # ------------------------------------------------------------------
        # Fix the Training / Traning typo used by the original repo
        mode_dir = "Training" if mode == "train" else "Testing"
        if not os.path.isdir(os.path.join(root_dir, mode_dir)):
            mode_dir = "Traning"          # fallback

        # build file list using the *I3D* folder as anchor (easier to rename)
        suffix = "I3D_RGB" if feat_type in ("i3d", "both") else "TimesFormer"
        sub    = f"{label.capitalize()}_{suffix}_NPY"
        self.files = sorted(
            glob.glob(os.path.join(root_dir, mode_dir, sub, "*.npy"))
        )
        if not self.files:
            raise RuntimeError(
                f"No .npy files under {os.path.join(root_dir, mode_dir, sub)}"
            )

        # ------------------------------------------------------------------
        # Ground-truth anomaly windows (only loaded when available)
        self._gt_windows = None
        if mode == "test":        # only the test split needs them
            gt_path = os.path.join(root_dir, gt_file)
            if os.path.isfile(gt_path):
                self._gt_windows = np.load(gt_path)      # (N_test, 2)
                if self._gt_windows.shape[0] != len(self.files):
                    print("[dataset] WARNING:",
                          "gt_windows.npy length mismatch; ignoring.")
                    self._gt_windows = None

        self._feat_dim = None     # discovered lazily at first __getitem__

    # ------------------------------------------------------------------
    @staticmethod
    def _sample_fixed(arr, n_seg):
        """Uniformly subsample / pad so the output has *n_seg* rows."""
        if arr.shape[0] == n_seg:
            return arr
        idx = np.linspace(0, arr.shape[0] - 1, n_seg).astype(int)
        return arr[idx]

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        """Return a tensor of shape (num_seg, feat_dim)."""
        base_path = self.files[idx]                      # always I3D anchor
        vid_i3d   = np.load(base_path).astype(np.float32)  # (T,1024)

        if self.feat_type == "i3d":
            x = vid_i3d

        elif self.feat_type == "timesformer":
            tsf_path = base_path.replace(
                "_I3D_RGB_NPY", "_TimesFormer_NPY"
            )
            x = np.load(tsf_path).astype(np.float32)     # (T,768)

        else:  # "both"
            tsf_path = base_path.replace(
                "_I3D_RGB_NPY", "_TimesFormer_NPY"
            )
            vid_tsf  = np.load(tsf_path).astype(np.float32)  # (T,768)
            x        = np.concatenate([vid_tsf, vid_i3d], axis=1)  # (T,1792)

        # Sample exactly self.num_seg rows
        x = self._sample_fixed(x, self.num_seg)

        # Optional per-segment ℓ2 normalisation
        if self.l2_norm:
            x /= np.linalg.norm(x, axis=1, keepdims=True) + 1e-8

        if self._feat_dim is None:
            self._feat_dim = x.shape[1]

        return torch.from_numpy(x)                       # (S, D)

    # ------------------------------------------------------------------
    def get_gt_window(self, idx):
        """
        Return (start_seg, end_seg) for shading the red region in timeline
        plots.  Only works when gt_windows.npy was provided **and** this
        dataset was created with mode='test'.  Otherwise raises ValueError.
        """
        if self._gt_windows is None:
            raise ValueError("Ground-truth windows not loaded for this split.")
        return tuple(int(v) for v in self._gt_windows[idx])

    # ------------------------------------------------------------------
    @property
    def feat_dim(self):
        if self._feat_dim is None:
            _ = self[0]          # triggers lazy discovery
        return self._feat_dim
