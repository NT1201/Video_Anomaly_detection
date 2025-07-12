import numpy as np, pandas as pd, matplotlib.pyplot as plt, os, argparse

def plot_video(seg_scores, start, end, out_png, title=""):
    x = np.arange(len(seg_scores))
    plt.figure(figsize=(6,3))
    plt.plot(x, seg_scores, lw=2, color="royalblue")
    plt.axvspan(start, end, color="salmon", alpha=.3)
    plt.ylim(0,1.05); plt.xlabel("segment #"); plt.ylabel("anomaly score")
    if title: plt.title(title, fontsize=10)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scores", required=True)       # runs/.../seg_scores.npy
    ap.add_argument("--gt_csv", required=True)       # ground_truth.csv
    ap.add_argument("--video_idx", type=int, default=0)   # which test clip
    ap.add_argument("--out", default="qual.png")
    args = ap.parse_args()

    all_scores = np.load(args.scores, allow_pickle=True)
    gt = pd.read_csv(args.gt_csv)

    row = gt.iloc[args.video_idx]
    plot_video(all_scores[args.video_idx],
               row.start_seg, row.end_seg,
               args.out,
               title=f"Video {row.video_id}")
