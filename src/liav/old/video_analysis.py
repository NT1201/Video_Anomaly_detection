import os
import cv2
import random
import matplotlib.pyplot as plt
import statistics
import numpy as np
from scipy.stats import norm

# Global settings
RESULTS_DIR = "code/upload"
SAMPLE_SIZE_PER_CATEGORY = 1000

os.makedirs(RESULTS_DIR, exist_ok=True)

class VideoFile:
    def __init__(self, path, category):
        self.path = path
        self.category = category
        self.frames = 0
        self.duration = 0
        self.size_mb = 0
        self.resolution = ""
        self.get_video_info()

    def get_video_info(self):
        cap = cv2.VideoCapture(self.path)
        if not cap.isOpened():
            print(f"⚠️ Failed to open: {self.path}")
            return
        self.frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.duration = self.frames / fps if fps else 0
        self.size_mb = os.path.getsize(self.path) / (1024 * 1024)

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resolution = f"{width} x {height}"
        cap.release()

def get_subfolders_with_mp4s(base_dir, category_name):
    subfolders = []
    for root, _, files in os.walk(os.path.join(base_dir, category_name)):
        if any(file.endswith(".mp4") for file in files):
            subfolders.append(root)
    return subfolders

def collect_videos(base_dir, sample_size=SAMPLE_SIZE_PER_CATEGORY):
    random.seed(42)
    selected_videos = []

    for category in ['Anomaly', 'Normal']:
        subfolders = get_subfolders_with_mp4s(base_dir, category)
        all_video_paths = []

        for folder in subfolders:
            mp4_files = [f for f in os.listdir(folder) if f.endswith('.mp4')]
            for f in mp4_files:
                full_path = os.path.join(folder, f)
                all_video_paths.append(full_path)

        print(f"📁 Found {len(all_video_paths)} {category.lower()} videos")

        actual_sample_size = min(sample_size, len(all_video_paths))
        sampled_paths = random.sample(all_video_paths, actual_sample_size)

        for path in sampled_paths:
            selected_videos.append(VideoFile(path, category))

    # Save selected video names
    with open(os.path.join(RESULTS_DIR, "selected_videos.txt"), "w") as f:
        for v in selected_videos:
            f.write(f"{v.category} | {os.path.basename(v.path)} | {v.resolution} | {v.frames} frames | {v.duration:.2f} sec | {v.size_mb:.2f} MB\n")

    return selected_videos


def plot_video_stats(videos, category):
    filtered = [v for v in videos if v.category == category]
    frames_list = [v.frames for v in filtered]
    names = [os.path.basename(v.path) for v in filtered]

    if not frames_list:
        return

    # Plot only bars
    plt.figure(figsize=(16, 6))
    plt.bar(names, frames_list, color='skyblue')
    plt.xticks(rotation=90)
    plt.title(f'{category} Videos - Frame Counts')
    plt.xlabel('Video Name')
    plt.ylabel('Frame Count')
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, f"{category.lower()}_frame_counts.png")
    plt.savefig(plot_path)
    plt.close()

    # Save detailed stats
    stats_path = os.path.join(RESULTS_DIR, f"{category.lower()}_stats.txt")
    with open(stats_path, 'w') as f:
        f.write(f"📊 Statistics for {category}:\n")
        f.write(f"  Count           : {len(frames_list)}\n")
        f.write(f"  Mean            : {statistics.mean(frames_list):.2f}\n")
        f.write(f"  Median          : {statistics.median(frames_list):.2f}\n")
        f.write(f"  Min             : {min(frames_list)}\n")
        f.write(f"  Max             : {max(frames_list)}\n")
        f.write(f"  Range           : {max(frames_list) - min(frames_list)}\n")
        if len(frames_list) > 1:
            f.write(f"  Std Deviation   : {statistics.stdev(frames_list):.2f}\n")
            f.write(f"  Variance        : {statistics.variance(frames_list):.2f}\n")
        else:
            f.write(f"  Std Deviation   : N/A\n")
            f.write(f"  Variance        : N/A\n")


def plot_combined_stats(videos):
    anomaly_videos = [v for v in videos if v.category == 'Anomaly']
    normal_videos = [v for v in videos if v.category == 'Normal']

    all_videos = normal_videos + anomaly_videos
    all_names = [os.path.basename(v.path) for v in all_videos]
    all_frames = [v.frames for v in all_videos]
    all_labels = ['Normal'] * len(normal_videos) + ['Anomaly'] * len(anomaly_videos)
    colors = ['blue' if label == 'Normal' else 'red' for label in all_labels]

    # Plot only bars
    plt.figure(figsize=(18, 6))
    plt.bar(all_names, all_frames, color=colors)
    plt.xticks(rotation=90)
    plt.title('Combined Video Frame Counts (Normal vs Anomaly)')
    plt.xlabel('Video Name')
    plt.ylabel('Frame Count')
    plt.tight_layout()

    plot_path = os.path.join(RESULTS_DIR, "combined_frame_comparison.png")
    plt.savefig(plot_path)
    plt.close()

    # Save detailed stats
    stats_path = os.path.join(RESULTS_DIR, "combined_stats.txt")
    with open(stats_path, 'w') as f:
        f.write(f"📊 Statistics for Combined (Normal + Anomaly):\n")
        f.write(f"  Count           : {len(all_frames)}\n")
        f.write(f"  Mean            : {statistics.mean(all_frames):.2f}\n")
        f.write(f"  Median          : {statistics.median(all_frames):.2f}\n")
        f.write(f"  Min             : {min(all_frames)}\n")
        f.write(f"  Max             : {max(all_frames)}\n")
        f.write(f"  Range           : {max(all_frames) - min(all_frames)}\n")
        if len(all_frames) > 1:
            f.write(f"  Std Deviation   : {statistics.stdev(all_frames):.2f}\n")
            f.write(f"  Variance        : {statistics.variance(all_frames):.2f}\n")
        else:
            f.write(f"  Std Deviation   : N/A\n")
            f.write(f"  Variance        : N/A\n")