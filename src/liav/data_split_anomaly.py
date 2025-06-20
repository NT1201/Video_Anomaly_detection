import shutil
from pathlib import Path
import os

def move_videos_by_split(anomaly_base, train_txt, test_txt):
    # Create Training and Testing folders under data directory
    data_root = anomaly_base.parents[1]  # points to 'project/data'
    root_train_dir = data_root / "Training"
    root_test_dir = data_root / "Testing"
    root_train_dir.mkdir(parents=True, exist_ok=True)
    root_test_dir.mkdir(parents=True, exist_ok=True)

    train_dir = root_train_dir / "Anomaly"
    test_dir = root_test_dir / "Anomaly"
    train_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)

    # Load video names from training split
    with open(train_txt, "r") as f:
        train_videos = [Path(line.strip()).name for line in f if line.strip()]

    # Load video names from testing split
    with open(test_txt, "r") as f:
        test_videos = [line.strip().split()[0] for line in f if line.strip()]

    train_moved = []
    test_moved = []

    # Move training videos
    for root, _, files in os.walk(anomaly_base):
        for file in files:
            if file in train_videos:
                src = Path(root) / file
                dst = train_dir / file
                if not dst.exists():
                    shutil.move(str(src), str(dst))
                    train_moved.append(file)

    # Move testing videos
    for root, _, files in os.walk(anomaly_base):
        for file in files:
            if file in test_videos:
                src = Path(root) / file
                dst = test_dir / file
                if not dst.exists():
                    shutil.move(str(src), str(dst))
                    test_moved.append(file)

    print(f"Moved {len(train_moved)} training videos to {train_dir}")
    print(f"Moved {len(test_moved)} testing videos to {test_dir}")

if __name__ == '__main__':
    anomaly_base = Path("project/data/Anomaly/Anomaly-Videos")
    train_txt = Path("project/data/Training_split.txt")
    test_txt = Path("project/data/Testing_split.txt")

    if not train_txt.exists():
        raise FileNotFoundError(f"Training split file not found: {train_txt}")
    if not test_txt.exists():
        raise FileNotFoundError(f"Testing split file not found: {test_txt}")

    move_videos_by_split(anomaly_base, train_txt, test_txt)