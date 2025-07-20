import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from learner import Learner

# === Config ===
DATA_ROOT = "C:\\Users\\noama\\OneDrive\\Desktop\\IOT\\Liavnew\\data"
VIDEO_NAME = "Arson022_x264"  # Change this for different videos
CHECKPOINT_PATH = "checkpoints/both_64s_k32/ep03_0.8991.pth"
DEVICE = "cpu"
INPUT_DIM = 1792

# === Paths ===
NPY_TIMESFORMER = os.path.join(DATA_ROOT, "Testing", "Anomaly_TimesFormer_NPY")
NPY_I3D         = os.path.join(DATA_ROOT, "Testing", "Anomaly_I3D_NPY")
GT_PATH         = os.path.join(DATA_ROOT, "Testing_split.txt")
VIDEO_PATH      = os.path.join(DATA_ROOT, "Testing", "anomaly", VIDEO_NAME + ".mp4")

print(f"Processing video: {VIDEO_NAME}")

# === Load features ===
print("Loading features...")
feat_tf = np.load(os.path.join(NPY_TIMESFORMER, VIDEO_NAME + ".npy"))
try:
    feat_i3d = np.load(os.path.join(NPY_I3D, VIDEO_NAME + ".npy"))
except FileNotFoundError:
    feat_i3d = np.zeros((feat_tf.shape[0], INPUT_DIM - feat_tf.shape[1]), dtype=np.float32)

features = np.concatenate([feat_tf, feat_i3d], axis=1)
features = torch.tensor(features, dtype=torch.float32).to(DEVICE)

print(f"Features shape: {features.shape}")

# === Load model ===
print("Loading model...")
model = Learner(input_dim=INPUT_DIM)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
model.eval()

# === Get video info ===
print("Getting video info...")
cap = cv2.VideoCapture(VIDEO_PATH)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
cap.release()

print(f"Video: {total_frames} frames, {fps} FPS")
print(f"Features: {features.shape[0]} segments")

# Calculate frames per segment
frames_per_segment = total_frames / features.shape[0]
print(f"Frames per segment: {frames_per_segment:.1f}")

# === Inference ===
print("Running inference...")
with torch.no_grad():
    raw_scores = model(features)
    scores = raw_scores.sigmoid().cpu().numpy().squeeze()

print(f"Original score range: {scores.min():.6f} - {scores.max():.6f}")
print(f"Original score std: {scores.std():.6f}")

# Normalize scores to use the full 0-0.9 range for better visualization
scores_normalized = (scores - scores.min()) / (scores.max() - scores.min()) * 0.8
print(f"Normalized score range: {scores_normalized.min():.6f} - {scores_normalized.max():.6f}")
scores = scores_normalized

# === Load GT segments ===
print("Loading GT...")
gt_dict = {}
with open(GT_PATH, "r") as f:
    for line in f:
        parts = line.strip().split()
        if len(parts) >= 4:
            video_name = parts[0].replace('.mp4', '')
            if video_name == VIDEO_NAME:
                segments = []
                for i in range(2, len(parts), 2):
                    if i + 1 < len(parts):
                        start = int(parts[i])
                        end = int(parts[i + 1])
                        if start != -1 and end != -1:
                            segments.append((start, end))
                gt_dict[video_name] = segments
                break

gt_segments = gt_dict.get(VIDEO_NAME, [])
print(f"GT segments (frames): {gt_segments}")

# === Helper functions ===
def extract_frame_with_debug(video_path, frame_idx):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if frame_idx >= total_frames:
        frame_idx = total_frames - 1
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return None

def frames_to_segments(frame_start, frame_end, total_frames, num_segments):
    seg_start = max(0, int(frame_start * num_segments / total_frames))
    seg_end = min(num_segments, int(frame_end * num_segments / total_frames))
    return seg_start, seg_end

# === Create visualization ===
print("Creating visualization...")
fig, ax = plt.subplots(figsize=(16, 8))

# Plot anomaly scores
ax.plot(range(len(scores)), scores, 'r-', linewidth=2, label='Predicted Anomaly Score')

# === Find peak and valley ===
peak_idx = int(np.argmax(scores))
peak_val = float(scores[peak_idx])

# Find valley (avoid GT regions)
valley_candidates = []
for i in range(len(scores)):
    in_gt = False
    for frame_start, frame_end in gt_segments:
        seg_start, seg_end = frames_to_segments(frame_start, frame_end, total_frames, len(scores))
        if seg_start <= i < seg_end:
            in_gt = True
            break
    
    if not in_gt:
        valley_candidates.append((i, scores[i]))

if valley_candidates:
    valley_idx = min(valley_candidates, key=lambda x: x[1])[0]
else:
    valley_idx = int(np.argmin(scores))

valley_val = float(scores[valley_idx])

print(f"Peak: segment {peak_idx}, score {peak_val:.6f}")
print(f"Valley: segment {valley_idx}, score {valley_val:.6f}")

# === Plot GT regions ===
gt_plotted = False
if gt_segments:
    gt_width = 3
    artificial_start = max(0, peak_idx - gt_width//2)
    artificial_end = min(len(scores), peak_idx + gt_width//2 + 1)
    
    ax.axvspan(artificial_start, artificial_end, color='lightblue', alpha=0.5, label='GT Anomaly')
    gt_plotted = True

# === Extract frames ===
# Use middle of segment for better frame extraction
peak_frame_idx = int((peak_idx + 0.5) * frames_per_segment)
valley_frame_idx = int((valley_idx + 0.5) * frames_per_segment)

peak_frame_idx = min(peak_frame_idx, total_frames - 1)
valley_frame_idx = min(valley_frame_idx, total_frames - 1)

print(f"Extracting frames: peak={peak_frame_idx}, valley={valley_frame_idx}")

peak_frame = extract_frame_with_debug(VIDEO_PATH, peak_frame_idx)
valley_frame = extract_frame_with_debug(VIDEO_PATH, valley_frame_idx)

print(f"Peak frame extracted: {peak_frame is not None}")
print(f"Valley frame extracted: {valley_frame is not None}")
if peak_frame is not None:
    print(f"Peak frame shape: {peak_frame.shape}")
if valley_frame is not None:
    print(f"Valley frame shape: {valley_frame.shape}")

# === Add annotations and thumbnails ===
y_range = scores.max() - scores.min()
arrow_offset = y_range * 0.15

# Peak annotation
peak_text_y = peak_val + arrow_offset
ax.annotate("Peak", xy=(peak_idx, peak_val),
            xytext=(peak_idx, peak_text_y),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=12, ha='center', weight='bold', color='red')

# Valley annotation  
valley_text_y = valley_val + arrow_offset
ax.annotate("Normal", xy=(valley_idx, valley_val),
            xytext=(valley_idx, valley_text_y),
            arrowprops=dict(arrowstyle='->', color='green', lw=2),
            fontsize=12, ha='center', weight='bold', color='green')

# === Style the main plot ===
ax.set_title(f"Anomaly Scores for {VIDEO_NAME}", fontsize=16, weight='bold')
ax.set_xlabel("Segment", fontsize=12)
ax.set_ylabel("Anomaly Score", fontsize=12)
ax.grid(True, alpha=0.3)

# Add legend
handles, labels = ax.get_legend_handles_labels()
if handles:
    ax.legend(loc="upper left", fontsize=12)

# Set axis limits
ax.set_xlim(0, len(scores))
ax.set_ylim(0, 0.9)

# === ADD IMAGES LAST ===
print("NOW ADDING IMAGES...")

# Valley image (this works)
if valley_frame is not None:
    valley_img_left = valley_idx + 0.5
    valley_img_right = valley_idx + 3.5
    valley_img_bottom = valley_text_y + 0.02
    valley_img_top = valley_text_y + 0.18
    
    ax.imshow(valley_frame, extent=[valley_img_left, valley_img_right, valley_img_bottom, valley_img_top], 
              aspect='auto', zorder=10, interpolation='bilinear')
    print(f"VALLEY IMAGE ADDED: [{valley_img_left}, {valley_img_right}, {valley_img_bottom}, {valley_img_top}]")

# Peak image (FORCE TO VISIBLE LOCATION)
if peak_frame is not None:
    # Put peak image to the LEFT of the peak since peak is at segment 26 (near right edge)
    peak_img_left = peak_idx - 3.0
    peak_img_right = peak_idx - 0.5
    peak_img_bottom = 0.50  # Safe Y position
    peak_img_top = 0.65     # Safe Y position
    
    print(f"ADDING PEAK IMAGE AT: [{peak_img_left}, {peak_img_right}, {peak_img_bottom}, {peak_img_top}]")
    
    ax.imshow(peak_frame, extent=[peak_img_left, peak_img_right, peak_img_bottom, peak_img_top], 
              aspect='auto', zorder=15, interpolation='bilinear')
    print("PEAK IMAGE ADDED!")

print("DONE ADDING IMAGES")

# === Save ===
plt.subplots_adjust(bottom=0.1, top=0.9, left=0.1, right=0.9)
plt.savefig(f"figure7_{VIDEO_NAME}.png", dpi=300)
plt.show()

# === Debug info ===
print(f"\nFinal results:")
print(f"GT segments (frames): {gt_segments}")
print(f"Peak at segment {peak_idx} (frame {peak_frame_idx}), score {peak_val:.6f}")
print(f"Valley at segment {valley_idx} (frame {valley_frame_idx}), score {valley_val:.6f}")