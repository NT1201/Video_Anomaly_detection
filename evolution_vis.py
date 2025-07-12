# evolution_vis.py -----------------------------------------------------
"""
Create a 2×2 panel like Fig-8: anomaly-score curves at several epochs.

Requirements:
• .npy files saved by main.py with --save_scores
• one test video is chosen (default vid_idx=0)
• ground-truth anomalous window (start,end) in segment indices
"""

import os, glob, argparse, numpy as np, matplotlib.pyplot as plt
plt.rcParams.update({"font.size": 11})

# ----------- customise here ------------------------------------------
EPOCHS_TO_SHOW = [1, 20, 40, 60]      # epoch numbers to plot
GT_WINDOW      = (20, 35)             # anomalous region in [seg idx]
ITER_TEXTS     = {1: "iteration =1000",
                  20: "iteration =3000",
                  40: "iteration =5000",
                  60: "iteration =8000"}   # label per subplot
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", default="runs",
        help="folder that contains scores_epXX.npy files")
    ap.add_argument("--num_segments", type=int, default=64)
    ap.add_argument("--vid_idx", type=int, default=0,
        help="which test video to visualise")
    ap.add_argument("--out_png", default="plots/evolution_figure.png")
    args = ap.parse_args()

    # load the requested epochs
    curves = {}
    for ep in EPOCHS_TO_SHOW:
        f = os.path.join(args.runs_dir, f"scores_ep{ep:02d}.npy")
        if not os.path.isfile(f):
            raise FileNotFoundError(f"missing {f} (run training with --save_scores)")
        y = np.load(f)            # shape (N_test, )
        curves[ep] = y[args.vid_idx]

    # --- plotting -----------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), dpi=120, sharex=True, sharey=True)
    axes = axes.flatten()

    for ax, ep in zip(axes, EPOCHS_TO_SHOW):
        y = curves[ep]
        ax.plot(np.arange(len(y)), y, lw=1.4, c="tab:blue")
        ax.axvspan(GT_WINDOW[0], GT_WINDOW[1], color="sandybrown", alpha=.4,
                   label="GT anomaly" if ep == EPOCHS_TO_SHOW[0] else "")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Training {ITER_TEXTS.get(ep, f'epoch {ep}')}")

        if ep in (EPOCHS_TO_SHOW[2], EPOCHS_TO_SHOW[3]):      # bottom row
            ax.set_xlabel("segment")
        if ep in (EPOCHS_TO_SHOW[0], EPOCHS_TO_SHOW[2]):      # left column
            ax.set_ylabel("anomaly score")

    axes[0].legend(frameon=False, loc="upper left")
    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out_png), exist_ok=True)
    plt.savefig(args.out_png, bbox_inches="tight")
    print("saved ▶", args.out_png)

if __name__ == "__main__":
    main()
