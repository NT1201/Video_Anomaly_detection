# loading_videos.py
import cv2
import torch
from torchvision.transforms.functional import to_tensor, resize
from torchvision.transforms import InterpolationMode

def load_video_segments(video_path, segment_length=16, target_size=(224, 224), fps=30):
    """
    Load a video, resample at fixed FPS, resize, and segment it.
    Returns: List of segments, where each segment is a [T, 3, 224, 224] tensor
    """
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"[INFO] Original video FPS: {original_fps}")

    frame_interval = int(round(original_fps / fps)) if original_fps > 0 else 1

    frames = []
    segments = []

    frame_count = 0
    sampled_frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0  # [C, H, W], scaled to [0, 1]
            frame = resize(frame, target_size, interpolation=InterpolationMode.BILINEAR, antialias=True)
            frames.append(frame)
            sampled_frame_count += 1
            if len(frames) == segment_length:
                segment = torch.stack(frames)  # [T, C, H, W]
                segments.append(segment)
                ##print(f"[INFO] Segment {len(segments)} created with {len(frames)} frames")
                frames = []
        frame_count += 1

    cap.release()
    print(f"[INFO] Total sampled frames: {sampled_frame_count}")
    print(f"[INFO] Total segments created: {len(segments)}")
    return segments
