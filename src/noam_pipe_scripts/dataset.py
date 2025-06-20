import torch
from torch.utils.data import Dataset
import numpy as np
import os
import random


class Normal_Loader(Dataset):
    """
    Loader for normal video features.
    is_train = 1 → training set
    is_train = 0 → testing set
    Assumes npy features are precomputed from TimeSformer and stored under 'features/'.
    """
    def __init__(self, is_train=1, path='features/', use_debug=False):
        super(Normal_Loader, self).__init__()
        self.is_train = is_train
        self.path = path
        self.use_debug = use_debug

        if self.is_train == 1:
            list_path = os.path.join(path, 'train_normal.txt')
        else:
            list_path = os.path.join(path, 'test_normal.txt')

        if not os.path.exists(list_path):
            raise FileNotFoundError(f"Missing file list: {list_path}")

        with open(list_path, 'r') as f:
            self.data_list = f.readlines()

        self.data_list = [x.strip() for x in self.data_list]

        if not self.is_train:
            random.shuffle(self.data_list)
            self.data_list = self.data_list[:-10]  # Optional, remove last 10 for debug

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        name = self.data_list[idx]
        full_path = os.path.join(self.path, name + '.npy')
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"[Normal] Feature not found: {full_path}")

        features = np.load(full_path)  # shape [T, 1024]
        if self.use_debug:
            print(f"[Normal] Loaded {name} shape: {features.shape}")
        return torch.tensor(features, dtype=torch.float32)


class Anomaly_Loader(Dataset):
    """
    Loader for anomaly video features.
    is_train = 1 → training set
    is_train = 0 → testing set
    """
    def __init__(self, is_train=1, path='features/', use_debug=False):
        super(Anomaly_Loader, self).__init__()
        self.is_train = is_train
        self.path = path
        self.use_debug = use_debug

        if self.is_train == 1:
            list_path = os.path.join(path, 'train_anomaly.txt')
        else:
            list_path = os.path.join(path, 'test_anomaly.txt')

        if not os.path.exists(list_path):
            raise FileNotFoundError(f"Missing file list: {list_path}")

        with open(list_path, 'r') as f:
            self.data_list = f.readlines()

        self.data_list = [x.strip() for x in self.data_list]

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        line = self.data_list[idx]

        if self.is_train:
            name = line
            full_path = os.path.join(self.path, name + '.npy')
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"[Anomaly] Feature not found: {full_path}")
            features = np.load(full_path)
            if self.use_debug:
                print(f"[Anomaly] Loaded {name} shape: {features.shape}")
            return torch.tensor(features, dtype=torch.float32)

        else:
            name, frame_str, gts_str = line.split('|')
            frames = int(frame_str)
            gts = [int(x) for x in gts_str.strip()[1:-1].split(',')]  # string "[0,1,0,...]"
            full_path = os.path.join(self.path, name + '.npy')
            if not os.path.exists(full_path):
                raise FileNotFoundError(f"[Anomaly] Feature not found: {full_path}")
            features = np.load(full_path)
            if self.use_debug:
                print(f"[Anomaly] Loaded {name} shape: {features.shape}, frames: {frames}, gts: {gts[:3]}...")
            return torch.tensor(features, dtype=torch.float32), gts, frames


if __name__ == '__main__':
    normal_train = Normal_Loader(is_train=1, path='features/', use_debug=True)
    anomaly_test = Anomaly_Loader(is_train=0, path='features/', use_debug=True)

    print(f"Normal train size: {len(normal_train)}")
    print(f"Anomaly test size: {len(anomaly_test)}")
