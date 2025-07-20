import os, argparse, random
import numpy as np, torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap

import plotly.express as px
from dataset import VideoFeatureDataset
from learner import Learner


def _get_backbone(model: torch.nn.Module):
    if hasattr(model, "backbone"):
        return model.backbone
    if hasattr(model, "net"):
        return model.net
    if hasattr(model, "cls_mlp"):
        return torch.nn.Sequential(*list(model.cls_mlp.children())[:-1])
    return torch.nn.Identity()


def extract_features(model, loader, label_idx, device, max_videos=60):
    feats, labs = [], []
    backbone = _get_backbone(model).to(device).eval()
    with torch.no_grad():
        for i, (v,) in enumerate(loader):
            if i >= max_videos:
                break
            x = v.view(-1, v.size(-1)).to(device)
            z = backbone(x).cpu().numpy()
            z = z / np.linalg.norm(z, axis=1, keepdims=True)
            feats.append(z)
            labs.extend([label_idx] * z.shape[0])
    return np.vstack(feats), np.asarray(labs, np.int32)


def run_visualization(feats, labels, out_dir, tag, dim=2, method='tsne', perp=30):
    print(f"[{method.upper()}] {tag:9s} on {feats.shape[0]:,}×{feats.shape[1]}")

    if method == "tsne":
        feats = PCA(n_components=50).fit_transform(feats)
        reducer = TSNE(
            n_components=dim,
            learning_rate="auto",
            init="pca",
            perplexity=perp,
            n_iter=1500,
            random_state=0
        )
    elif method == "umap":
        reducer = umap.UMAP(n_components=dim, random_state=0)
    else:
        raise ValueError(f"Unknown method: {method}")

    emb = reducer.fit_transform(feats)

    # Save plot (matplotlib)
    fig = plt.figure(figsize=(6, 5), dpi=120)
    ax = fig.add_subplot(111)
    ax.scatter(emb[labels == 0, 0], emb[labels == 0, 1],
               s=10, alpha=0.6, c="tab:green", edgecolors='k', linewidths=0.2, label="Normal")
    ax.scatter(emb[labels == 1, 0], emb[labels == 1, 1],
               s=10, alpha=0.6, c="tab:red",   edgecolors='k', linewidths=0.2, label="Anomaly")
    ax.set_title(f"{dim}-D {method.upper()} ({tag})")
    ax.legend(loc="best")
    fig.tight_layout()
    img_path = os.path.join(out_dir, f"{method}_{tag}.png")
    fig.savefig(img_path)
    plt.close(fig)
    print("  ↳ PNG  :", img_path)

    # Save interactive HTML (Plotly)
    fig_i = px.scatter(
        x=emb[:, 0], y=emb[:, 1],
        color=["Anomaly" if l else "Normal" for l in labels],
        opacity=0.75, title=f"{dim}-D {method.upper()} – {tag}"
    )
    html_path = os.path.join(out_dir, f"{method}_{tag}.html")
    fig_i.write_html(html_path)
    print("  ↳ HTML :", html_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True)
    ap.add_argument("--feat_type", choices=["timesformer", "i3d", "both"], default="both")
    ap.add_argument("--num_segments", type=int, default=64)
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--max_videos", type=int, default=60)
    ap.add_argument("--dim", type=int, default=2, choices=[2, 3])
    ap.add_argument("--perp", type=int, default=30)
    ap.add_argument("--method", choices=["tsne", "umap"], default="tsne")
    args = ap.parse_args()

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    ld_n = torch.utils.data.DataLoader(
        VideoFeatureDataset(args.data_root, "test", "normal", args.feat_type, args.num_segments),
        batch_size=1, shuffle=False)
    ld_a = torch.utils.data.DataLoader(
        VideoFeatureDataset(args.data_root, "test", "anomaly", args.feat_type, args.num_segments),
        batch_size=1, shuffle=False)

    feat_dim = {"timesformer": 768, "i3d": 1024, "both": 1792}[args.feat_type]
    model = Learner(feat_dim).to(device)

    run_name = f"{args.feat_type}_{args.num_segments}s"
    out_dir = os.path.join("plots", run_name, args.method)
    os.makedirs(out_dir, exist_ok=True)

    # UNTRAINED
    f_n, l_n = extract_features(model, ld_n, 0, device, args.max_videos)
    f_a, l_a = extract_features(model, ld_a, 1, device, args.max_videos)
    run_visualization(np.vstack([f_n, f_a]), np.hstack([l_n, l_a]),
                      out_dir, tag="untrained", dim=args.dim,
                      method=args.method, perp=args.perp)

    # TRAINED
    if args.checkpoint and os.path.isfile(args.checkpoint):
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
        f_n, l_n = extract_features(model, ld_n, 0, device, args.max_videos)
        f_a, l_a = extract_features(model, ld_a, 1, device, args.max_videos)
        run_visualization(np.vstack([f_n, f_a]), np.hstack([l_n, l_a]),
                          out_dir, tag="trained", dim=args.dim,
                          method=args.method, perp=args.perp)
    else:
        print("⚠  trained visualization skipped – checkpoint not found.")


if __name__ == "__main__":
    main()
