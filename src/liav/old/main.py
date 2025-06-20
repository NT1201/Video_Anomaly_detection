import os
import cv2
from get_features import FeatureExtractor
import os
from video_analysis import collect_videos, plot_video_stats, plot_combined_stats



def load_video_segments(video_path, segment_length=32):
    """
    Loads a video and splits it into fixed-length segments.

    :param video_path: Path to the video file
    :param segment_length: Number of frames per segment (default=16)
    :return: List of frame segments, each a list of 16 RGB frames
    """
    cap = cv2.VideoCapture(video_path)
    segments = []
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
        if len(frames) == segment_length:
            segments.append(frames)
            frames = []
    cap.release()
    return segments


if __name__ == '__main__':
    # Initialize feature extractor
    feature_extractor = FeatureExtractor()

    # Path to a test video file
    video_path = r'code/data/Anomaly/Anomaly-Videos-Part-2/Burglary/Burglary001_x264.mp4'  # You can change this
    print(os.path.exists(video_path))

    # Load video and segment it
    segments = load_video_segments(video_path)

    # Process each segment
    for i, segment in enumerate(segments):
        print(f"Segment {i + 1}/{len(segments)}")

        tf_feat = feature_extractor.extract_timesformer_features(segment)
        print(f"  TimeSformer features shape: {tf_feat.shape}")

        swin_feat = feature_extractor.extract_swin_features(segment)
        print(f"  Swin Video features shape: {swin_feat.shape}")
    base_dir = 'code/data'
    videos = collect_videos(base_dir)

    plot_video_stats(videos, 'Anomaly')
    plot_video_stats(videos, 'Normal')
    plot_combined_stats(videos)