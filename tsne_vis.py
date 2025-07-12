# tsne_vis.py  ─────────────────────────────────────────────────────────
# 3-D / 2-D t-SNE visualisation of segment-level features BEFORE and
# AFTER MIL training.  Produces an interactive HTML + a static PNG in
#     plots/<run_name>/tsne/
# ─────────────────────────────────────────────────────────────────────
import os, argparse, random
import numpy as np, torch
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.express as px

from dataset import VideoFeatureDataset
from learner  import Learner


# ╭──────────────────── helper: get a feature extractor ──────────────╮
def _get_backbone(model: torch.nn.Module):
    # order: backbone ▸ net ▸ first part of cls_mlp ▸ identity
    if hasattr(model, "backbone"):        # custom attr (e.g. ViT, ResNet…)
        return model.backbone
    if hasattr(model, "net"):             # many toy models keep the stem here
        return model.net
    if hasattr(model, "cls_mlp"):         # fallback: drop the final logit layer
        return torch.nn.Sequential(*list(model.cls_mlp.children())[:-1])
    return torch.nn.Identity()            # last-resort no-op
# ╰────────────────────────────────────────────────────────────────────╯


def extract_features(model, loader, label_idx, device, max_videos=60):
    """
    Returns
        feats  – (N, D)  stacked segment embeddings
        labels – (N,)    0 = normal, 1 = anomaly
    """
    feats, labs = [], []
    backbone = _get_backbone(model).to(device).eval()

    with torch.no_grad():
        for i, (v,) in enumerate(loader):
            if i >= max_videos:
                break
            x = v.view(-1, v.size(-1)).to(device)           # (S, D)
            z = backbone(x).cpu().numpy()                   # (S, D)
            feats.append(z)
            labs.extend([label_idx] * z.shape[0])

    return np.vstack(feats), np.asarray(labs, np.int32)


# ╭──────────────────── helper: run & save t-SNE ──────────────────────╮
def run_tsne(feats, labels, out_dir, tag, dim, perp):
    print(f"[t-SNE] {tag:9s} on {feats.shape[0]:,}×{feats.shape[1]}  "
          f"(dim={dim}, perp={perp})")

    tsne = TSNE(n_components=dim, learning_rate="auto",
                init="random", perplexity=perp, random_state=0)
    emb  = tsne.fit_transform(feats)

    # Interactive HTML (Plotly)
    if dim == 3:
        fig_i = px.scatter_3d(
            x=emb[:, 0], y=emb[:, 1], z=emb[:, 2],
            color=["Anomaly" if l else "Normal" for l in labels],
            opacity=0.75, title=f"{dim}-D t-SNE – {tag}"
        )
    else:  # 2-D
        fig_i = px.scatter(
            x=emb[:, 0], y=emb[:, 1],
            color=["Anomaly" if l else "Normal" for l in labels],
            opacity=0.75, title=f"{dim}-D t-SNE – {tag}"
        )

    html_path = os.path.join(out_dir, f"tsne_{tag}.html")
    fig_i.write_html(html_path)
    print("  ↳ HTML :", html_path)

    # Static PNG (matplotlib)
    fig = plt.figure(figsize=(6, 5), dpi=120)
    if dim == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(*emb[labels == 0].T, s=6, c="tab:green", label="Normal")
        ax.scatter(*emb[labels == 1].T, s=6, c="tab:red",   label="Anomaly")
    else:
        ax = fig.add_subplot(111)
        ax.scatter(emb[labels == 0, 0], emb[labels == 0, 1],
                   s=6, c="tab:green", label="Normal")
        ax.scatter(emb[labels == 1, 0], emb[labels == 1, 1],
                   s=6, c="tab:red",   label="Anomaly")

    ax.set_title(f"{dim}-D t-SNE ({tag})")
    ax.legend(loc="best")
    png_path = os.path.join(out_dir, f"tsne_{tag}.png")
    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)
    print("  ↳ PNG  :", png_path)
# ╰────────────────────────────────────────────────────────────────────╯


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",    required=True)
    ap.add_argument("--feat_type",    choices=["timesformer", "i3d", "both"],
                    default="both")
    ap.add_argument("--num_segments", type=int, default=64)
    ap.add_argument("--checkpoint",   default="",
                    help="Path to trained *.pth (optional)")
    ap.add_argument("--max_videos",   type=int, default=60,
                    help="Limit videos per class for speed")
    ap.add_argument("--dim",          type=int, default=2, choices=[2, 3],
                    help="t-SNE output dimension (2 or 3)")
    ap.add_argument("--perp",         type=int, default=30,
                    help="t-SNE perplexity (5-50)")
    args = ap.parse_args()

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ╭──── loaders ───────────────────────────────────────────────────╮
    ld_n = torch.utils.data.DataLoader(
        VideoFeatureDataset(args.data_root, "test", "normal",
                            args.feat_type, args.num_segments),
        batch_size=1, shuffle=False)
    ld_a = torch.utils.data.DataLoader(
        VideoFeatureDataset(args.data_root, "test", "anomaly",
                            args.feat_type, args.num_segments),
        batch_size=1, shuffle=False)
    # ╰────────────────────────────────────────────────────────────────╯

    # model
    feat_dim = {"timesformer": 768, "i3d": 1024, "both": 1792}[args.feat_type]
    model = Learner(feat_dim).to(device)

    # output directory
    run_name = f"{args.feat_type}_{args.num_segments}s"
    out_dir  = os.path.join("plots", run_name, "tsne")
    os.makedirs(out_dir, exist_ok=True)

    # ───────── UNTRAINED ─────────────────────────────────────────────
    f_n, l_n = extract_features(model, ld_n, 0, device, args.max_videos)
    f_a, l_a = extract_features(model, ld_a, 1, device, args.max_videos)
    run_tsne(np.vstack([f_n, f_a]), np.hstack([l_n, l_a]),
             out_dir, tag="untrained", dim=args.dim, perp=args.perp)

    # ───────── TRAINED  (optional) ───────────────────────────────────
    if args.checkpoint and os.path.isfile(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        f_n, l_n = extract_features(model, ld_n, 0, device, args.max_videos)
        f_a, l_a = extract_features(model, ld_a, 1, device, args.max_videos)
        run_tsne(np.vstack([f_n, f_a]), np.hstack([l_n, l_a]),
                 out_dir, tag="trained", dim=args.dim, perp=args.perp)
    else:
        print("⚠  trained t-SNE skipped – checkpoint not found.")


if __name__ == "__main__":
    main()
