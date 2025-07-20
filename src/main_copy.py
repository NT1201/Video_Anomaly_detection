import os, argparse, random, itertools, numpy as np, torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import sklearn.metrics as skm

from dataset        import VideoFeatureDataset
from learner        import Learner
from loss           import topk_mil_bce
from metrics_utils  import compute_all_metrics
from vis            import epoch_summary_plots, save_epoch_csv

def make_loaders(root, feat, batch, seg):
    n_tr = VideoFeatureDataset(root, "train", "normal",  feat, seg)
    a_tr = VideoFeatureDataset(root, "train", "anomaly", feat, seg)
    n_te = VideoFeatureDataset(root, "test",  "normal",  feat, seg)
    a_te = VideoFeatureDataset(root, "test",  "anomaly", feat, seg)

    g = torch.Generator().manual_seed(0)
    return (
        DataLoader(n_tr, batch_size=batch, shuffle=True,  drop_last=True, generator=g),
        DataLoader(a_tr, batch_size=batch, shuffle=True,  drop_last=True, generator=g),
        DataLoader(n_te, batch_size=1,     shuffle=False),
        DataLoader(a_te, batch_size=1,     shuffle=False),
    )

def evaluate(model, ld_a, ld_n, device):
    model.eval()
    scores, labels = [], []
    with torch.no_grad():
        for clip in ld_a:
            s = model(clip.view(-1, clip.size(-1)).to(device))
            scores.append(torch.sigmoid(s).max().item()); labels.append(1)
        for clip in ld_n:
            s = model(clip.view(-1, clip.size(-1)).to(device))
            scores.append(torch.sigmoid(s).max().item()); labels.append(0)
    return np.asarray(labels, np.int32), np.asarray(scores, np.float32)

def train_epoch(model, opt, ld_n, ld_a, device, seg_per_video, k, pos_weight):
    model.train()
    tot, it_n, it_a = 0.0, itertools.cycle(ld_n), itertools.cycle(ld_a)
    n_batches  = max(len(ld_n), len(ld_a))
    pw = torch.tensor(pos_weight, device=device) if pos_weight else None
    for _ in range(n_batches):
        v_n, v_a = next(it_n), next(it_a)
        B        = v_n.size(0)
        x_n = v_n.view(-1, v_n.size(-1)).to(device)
        x_a = v_a.view(-1, v_a.size(-1)).to(device)
        logits = torch.cat([model(x_n), model(x_a)], 0)
        labels = torch.cat([torch.zeros(B, device=device),
                            torch.ones (B, device=device)], 0)
        loss = topk_mil_bce(logits, labels, seg_per_video, k=k, pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    return tot / n_batches

def save_final_plots(y, y_score, m, save_dir):
    """Save ROC, PR, and Confusion Matrix plots for final model."""
    os.makedirs(save_dir, exist_ok=True)

    # ROC and PR curve
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(m["fpr"], m["tpr"], label="ROC")
    plt.plot([0, 1], [0, 1], "--", c="gray")
    plt.xlabel("FPR"); plt.ylabel("TPR")
    plt.title(f"ROC (AUC {m['roc_auc']:.3f})")

    plt.subplot(1, 2, 2)
    plt.plot(m["rec"], m["prec"], label="PR")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(f"PR (AUC {m['pr_auc']:.3f})")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "final_roc_pr.png"))
    plt.close()

    # Confusion matrix at best_thr
    y_pred_best = (y_score >= m["best_thr"]).astype(int)
    cm = skm.confusion_matrix(y, y_pred_best)
    disp = skm.ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anomaly"])
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=120)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".0f")
    ax.set_title(f"Confusion @ thr={m['best_thr']:.3f}")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, "final_confusion.png"))
    plt.close()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root",    required=True)
    ap.add_argument("--feat_type",    choices=["timesformer","i3d","both"], default="both")
    ap.add_argument("--epochs",       type=int, default=40)
    ap.add_argument("--batch_size",   type=int, default=30)
    ap.add_argument("--num_segments", type=int, default=32)
    ap.add_argument("--topk",         type=int, default=16)
    ap.add_argument("--pos_weight",   type=float, default=5.0)
    ap.add_argument("--save_scores",  action="store_true")
    ap.add_argument("--dump_curve",   action="store_true")
    ap.add_argument("--curve_vid",    type=int, default=0)
    args = ap.parse_args()

    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)

    ld_n_tr, ld_a_tr, ld_n_te, ld_a_te = make_loaders(
        args.data_root, args.feat_type, args.batch_size, args.num_segments)

    feat_dim = {"timesformer":768,"i3d":1024,"both":1792}[args.feat_type]
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device =", device)
    model    = Learner(feat_dim).to(device)

    head = [p for n,p in model.named_parameters() if "cls" in n]
    base = [p for n,p in model.named_parameters() if "cls" not in n]
    opt  = torch.optim.Adam(
        [{"params": base, "lr": 6e-4},
         {"params": head, "lr": 9e-3}],
        weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    run_name = f"{args.feat_type}_{args.num_segments}s_k{args.topk}"
    ckpt_dir = os.path.join("checkpoints", run_name); os.makedirs(ckpt_dir,  exist_ok=True)
    plots_dir = os.path.join("plots", run_name, "curves", "avg_best3")
    os.makedirs(plots_dir, exist_ok=True)
    if args.save_scores or args.dump_curve:
        os.makedirs("runs", exist_ok=True)

    csv_path = os.path.join("plots", run_name, "metrics.csv")
    best_auc, keep = 0.0, []

    for ep in range(1, args.epochs+1):
        loss = train_epoch(model, opt, ld_n_tr, ld_a_tr, device,
                           args.num_segments, args.topk, args.pos_weight)
        y, y_score = evaluate(model, ld_a_te, ld_n_te, device)
        m          = compute_all_metrics(y, y_score)
        print(f"Epoch {ep:03d} | loss {loss:.4f} "
              f"| AUC {m['roc_auc']:.4f} | PR-AUC {m['pr_auc']:.4f} "
              f"| F1 {m['best_f1']:.4f} @thr={m['best_thr']:.3f} "
              f"| FAR {m['far']*100:5.2f}%")

        scalar_keys = ["roc_auc","pr_auc","best_f1","best_thr","far"]
        row = {k: m[k] for k in scalar_keys}; row.update(epoch=ep, loss=loss)
        save_epoch_csv(row, csv_path)

        if args.save_scores:
            np.save(f"runs/scores_ep{ep:02d}.npy", y_score)
        if args.dump_curve:
            clip = ld_a_te.dataset[args.curve_vid]
            with torch.no_grad():
                seg_curve = torch.sigmoid(
                    model(clip.view(-1, feat_dim).to(device))
                ).cpu().numpy().squeeze()
            np.save(f"runs/segcurve_ep{ep:02d}.npy", seg_curve)
        if m['roc_auc'] > best_auc:
            best_auc = m['roc_auc']
            path     = os.path.join(ckpt_dir, f"ep{ep:02d}_{best_auc:.4f}.pth")
            torch.save(model.state_dict(), path)
            keep.append((best_auc, path)); keep = sorted(keep)[-3:]

        scheduler.step()

    # average best-3
    avg_state = {}
    for _, pth in keep:
        w = torch.load(pth)
        for k,v in w.items():
            avg_state[k] = avg_state.get(k, 0) + v / len(keep)
    model.load_state_dict(avg_state)

    y, y_score = evaluate(model, ld_a_te, ld_n_te, device)
    m = compute_all_metrics(y, y_score)
    print(f"\nAveraged best-3 → AUC {m['roc_auc']:.4f} | "
          f"PR-AUC {m['pr_auc']:.4f} | FAR {m['far']*100:5.2f}%")

    # Manual confusion and plot save
    y_pred_manual = (y_score >= m["best_thr"]).astype(int)
    np.save("runs/y_true.npy", y)
    np.save("runs/y_score.npy", y_score)

    # Save final plots
    save_final_plots(y, y_score, m, plots_dir)

    # Save CSV and also call original vis plot for full metrics/epoch if needed
    scalar_keys = ["roc_auc","pr_auc","best_f1","best_thr","far"]
    row = {k: m[k] for k in scalar_keys}; row.update(epoch="avg-best-3", loss=np.nan)
    save_epoch_csv(row, csv_path)
    plot_root = os.path.join("plots", run_name, "curves", "avg_best3")
    epoch_summary_plots(plot_root, m)

if __name__ == "__main__":
    main()
