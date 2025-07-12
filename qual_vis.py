# qual_vis.py
# ----------------------------------------------------------------------
# Make the per-video qualitative plots that appear in many papers:
#  • anomaly–score curve
#  • shaded ground-truth anomalous window
#  • (optional) threshold line at best-F1 decision boundary
# Generates one PNG + one interactive HTML per chosen test clip under
# plots/<run_name>/qualitative/.
# ----------------------------------------------------------------------
import os, argparse, numpy as np, torch, matplotlib.pyplot as plt
import plotly.graph_objects as go
from dataset import VideoFeatureDataset
from learner  import Learner
from metrics_utils import compute_all_metrics

# ----------------------------------------------------------------------
def score_video(model, clip, device):
    """Return 1 anomaly-score per segment (sigmoid logit)."""
    with torch.no_grad():                                # ① turn grad off
        x   = clip.view(-1, clip.size(-1)).to(device)    # (S,D)
        log = model(x)                                   # (S,1)
    return torch.sigmoid(log).squeeze(1).detach().cpu().numpy()  # ② detach


# ----------------------------------------------------------------------
def make_plot(scores, gt_start, gt_end, save_root, best_thr=None):
    """Save both PNG & HTML."""
    S = len(scores); x = np.arange(S)

    # ── static (PNG) ──────────────────────────────────────────────────
    plt.figure(figsize=(8,2.8), dpi=120)
    plt.plot(x, scores, lw=1.5, label="score")
    plt.axvspan(gt_start, gt_end, color="salmon", alpha=.35, label="GT anomaly")
    if best_thr is not None: plt.axhline(best_thr, ls="--", c="grey", lw=1)
    plt.ylim(0,1); plt.xlim(0,S-1)
    plt.xlabel("segment"); plt.ylabel("anomaly score"); plt.tight_layout()
    plt.legend(loc="upper right")
    plt.savefig(save_root + ".png"); plt.close()

    # ── interactive (HTML) ───────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=scores, mode="lines",
                             name="anomaly score"))
    fig.add_shape(type="rect", x0=gt_start, x1=gt_end, y0=0, y1=1,
                  fillcolor="tomato", opacity=.3, line_width=0)
    if best_thr is not None:
        fig.add_hline(y=best_thr, line_dash="dash", line_color="grey",
                      annotation_text=f"thr={best_thr:.2f}",
                      annotation_position="bottom right")
    fig.update_layout(height=300, width=800,
                      xaxis_title="segment", yaxis_title="score")
    fig.write_html(save_root + ".html", include_plotlyjs="cdn")

# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",    required=True)
    ap.add_argument("--feat_type",    choices=["timesformer","i3d","both"],
                    default="both")
    ap.add_argument("--num_segments", type=int, default=64)
    ap.add_argument("--checkpoint",   required=True,
                    help="Path to trained *.pth (mandatory).")
    # pick clips –– either explicit indices OR a range
    ap.add_argument("--video_idx",  type=int, nargs="*",
                    help="Explicit list of test-video indices (0-based).")
    ap.add_argument("--video_start", type=int,
                    help="Start index (inclusive) if you want a range.")
    ap.add_argument("--video_end",   type=int,
                    help="End index   (inclusive) if you want a range.")
    args = ap.parse_args()

    # ­­­resolve which indices ------------------------------------------------
    if args.video_idx is None:
        if args.video_start is None or args.video_end is None:
            raise SystemExit("❌  Need either --video_idx  OR  both --video_start/--video_end.")
        args.video_idx = list(range(args.video_start, args.video_end+1))

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feat_dim = {"timesformer":768,"i3d":1024,"both":1792}[args.feat_type]

    # loader for *all* test clips (we’ll sub-index)
    test_set = VideoFeatureDataset(args.data_root,"test","anomaly",
                                   args.feat_type,args.num_segments)
    test_set.files += VideoFeatureDataset(
        args.data_root,"test","normal",
        args.feat_type,args.num_segments).files
    test_ld  = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False)

    # model ------------------------------------------------------------------
    model = Learner(feat_dim).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    # output dir -------------------------------------------------------------
    run_name = f"{args.feat_type}_{args.num_segments}s"
    out_dir  = os.path.join("plots", run_name, "qualitative")
    os.makedirs(out_dir, exist_ok=True)

    # optional threshold (best-F1 over *all* test clips)
    #  – we reuse existing metrics helper
    labels, scores_all = [], []
    with torch.no_grad():
        for clip in test_ld:
            lbl = 1 if "Anomaly" in os.path.basename(test_set.files[len(labels)]) else 0
            labels.append(lbl)
            scores_all.append(score_video(model, clip[0], device).max())
    labels = np.asarray(labels); scores_all = np.asarray(scores_all)
    best_thr = compute_all_metrics(labels, scores_all)['best_thr']

    # iterate over requested indices ----------------------------------------
    for vid_idx in args.video_idx:
        clip = test_set[vid_idx][None]         # add batch dim
        scr  = score_video(model, clip[0], device)

        # fake GT window: here we *don’t* have segment-level GT – for demo,
        # mark centre ⅓ as anomaly if the clip is from anomaly folder.
        is_anom = 1 if "Anomaly" in os.path.basename(test_set.files[vid_idx]) else 0
        S       = len(scr); g0, g1 = (S//3, 2*S//3) if is_anom else (0, -1)

        root = os.path.join(out_dir, f"vid{vid_idx:03d}")
        make_plot(scr, g0, g1, root, best_thr)

        print(f"✔ saved   vid{vid_idx:03d} → {root}.png / .html")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    np.random.seed(0); torch.manual_seed(0)
    main()
