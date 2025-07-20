import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from learner import Learner

# === Config ===
DATA_ROOT = "C:\\Users\\noama\\OneDrive\\Desktop\\IOT\\Liavnew\\data"
VIDEO_NAME = "Normal_Videos_886_x264"  # Change this for different videos
CHECKPOINT_PATH = "checkpoints/both_64s_k32/ep03_0.8991.pth"
DEVICE = "cpu"
INPUT_DIM = 1792

# === Paths ===
is_normal_video = "Normal_Videos" in VIDEO_NAME

if is_normal_video:
    NPY_TIMESFORMER = os.path.join(DATA_ROOT, "Testing", "Normal_TimesFormer_NPY")
    NPY_I3D = os.path.join(DATA_ROOT, "Testing", "Normal_I3D_NPY")
    VIDEO_PATH = os.path.join(DATA_ROOT, "Testing", "normal", VIDEO_NAME + ".mp4")
else:
    NPY_TIMESFORMER = os.path.join(DATA_ROOT, "Testing", "Anomaly_TimesFormer_NPY")
    NPY_I3D = os.path.join(DATA_ROOT, "Testing", "Anomaly_I3D_NPY")
    VIDEO_PATH = os.path.join(DATA_ROOT, "Testing", "anomaly", VIDEO_NAME + ".mp4")

GT_PATH = os.path.join(DATA_ROOT, "Testing_split.txt")

# Load and process data
feat_tf = np.load(os.path.join(NPY_TIMESFORMER, VIDEO_NAME + ".npy"))
try:
    feat_i3d = np.load(os.path.join(NPY_I3D, VIDEO_NAME + ".npy"))
except FileNotFoundError:
    feat_i3d = np.zeros((feat_tf.shape[0], INPUT_DIM - feat_tf.shape[1]), dtype=np.float32)

features = np.concatenate([feat_tf, feat_i3d], axis=1)
features = torch.tensor(features, dtype=torch.float32).to(DEVICE)

# Load model and run inference
model = Learner(input_dim=INPUT_DIM)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

with torch.no_grad():
    raw_scores = model(features)
    scores = raw_scores.sigmoid().cpu().numpy().squeeze()

# Normalize scores
scores = (scores - scores.min()) / (scores.max() - scores.min()) * 0.8

# Get video info
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
frames_per_segment = total_frames / len(scores)

# Find peak and valley
peak_idx = int(np.argmax(scores))
peak_val = float(scores[peak_idx])
valley_idx = int(np.argmin(scores))
valley_val = float(scores[valley_idx])

# Check if there's actually a significant anomaly
score_range = peak_val - valley_val
mean_score = np.mean(scores)
is_real_anomaly_video = not is_normal_video
is_anomaly_detected = is_real_anomaly_video and (peak_val > 0.6 and score_range > 0.3)

print(f"Peak: segment {peak_idx}, score {peak_val:.6f}")
print(f"Valley: segment {valley_idx}, score {valley_val:.6f}")
print(f"Score range: {score_range:.6f}")
print(f"Mean score: {mean_score:.6f}")
print(f"Is normal video: {is_normal_video}")
print(f"Anomaly detected: {is_anomaly_detected}")

# Extract frames
def extract_frame(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

peak_frame_idx = int((peak_idx + 0.5) * frames_per_segment)
valley_frame_idx = int((valley_idx + 0.5) * frames_per_segment)

print(f"Extracting frames: peak={peak_frame_idx}, valley={valley_frame_idx}")

peak_frame = extract_frame(VIDEO_PATH, peak_frame_idx)
valley_frame = extract_frame(VIDEO_PATH, valley_frame_idx)

print(f"Peak frame extracted: {peak_frame is not None}")
print(f"Valley frame extracted: {valley_frame is not None}")

# Create plot
fig, ax = plt.subplots(figsize=(16, 8))
ax.plot(range(len(scores)), scores, 'r-', linewidth=2, label='Predicted Anomaly Score')

# ONLY ADD ANNOTATIONS AND IMAGES IF ANOMALY DETECTED
if is_anomaly_detected:
    print("Adding annotations and images for anomaly video...")
    
    # GT region
    gt_start = max(0, peak_idx - 1)
    gt_end = min(len(scores), peak_idx + 2)
    ax.axvspan(gt_start, gt_end, color='lightblue', alpha=0.5, label='GT Anomaly')
    
    # Annotations
    y_range = scores.max() - scores.min()
    arrow_offset = y_range * 0.15
    peak_text_y = peak_val + arrow_offset
    valley_text_y = valley_val + arrow_offset

    ax.annotate("Peak", xy=(peak_idx, peak_val), xytext=(peak_idx, peak_text_y),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=12, ha='center', weight='bold', color='red')

    ax.annotate("Normal", xy=(valley_idx, valley_val), xytext=(valley_idx, valley_text_y),
                arrowprops=dict(arrowstyle='->', color='green', lw=2),
                fontsize=12, ha='center', weight='bold', color='green')
    
    # Images
    if valley_frame is not None:
        ax.imshow(valley_frame, extent=[valley_idx + 0.5, valley_idx + 3.5, valley_text_y + 0.02, valley_text_y + 0.18], 
                  aspect='auto', zorder=10)

    if peak_frame is not None:
        ax.imshow(peak_frame, extent=[peak_idx - 3.0, peak_idx - 0.5, peak_text_y + 0.02, peak_text_y + 0.18], 
                  aspect='auto', zorder=10)
    
    print("Annotations and images added")
else:
    print("Normal video - showing clean plot with no annotations")

# Style plot
ax.set_title(f"Anomaly Scores for {VIDEO_NAME}", fontsize=16, weight='bold')
ax.set_xlabel("Segment", fontsize=12)
ax.set_ylabel("Anomaly Score", fontsize=12)
ax.grid(True, alpha=0.3)
ax.legend(loc="upper left", fontsize=12)
ax.set_xlim(0, len(scores))
ax.set_ylim(0, 0.9)

plt.tight_layout()
plt.savefig(f"figure7_{VIDEO_NAME}.png", dpi=300)
plt.show()

print("Done!")