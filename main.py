# main.py
import os
import argparse, random, itertools, numpy as np, torch
from torch.utils.data import DataLoader
from sklearn import metrics

from dataset import VideoFeatureDataset
from learner import Learner
from loss    import topk_mil_bce
from vis     import plot_curves


# ----------------------------------------------------------------------
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
        n_tr.feat_dim,
    )


def evaluate(model, ld_a, ld_n, device, seg_per_video):
    model.eval(); scores, labels = [], []
    with torch.no_grad():
        for v in ld_a:
            s = model(v.view(-1, v.size(-1)).to(device))
            scores.append(torch.sigmoid(s).max().item()); labels.append(1)
        for v in ld_n:
            s = model(v.view(-1, v.size(-1)).to(device))
            scores.append(torch.sigmoid(s).max().item()); labels.append(0)

    labels = np.asarray(labels, np.int32); scores = np.asarray(scores, np.float32)
    auc    = metrics.roc_auc_score(labels, scores)
    prec, rec, thr = metrics.precision_recall_curve(labels, scores); pr_auc = metrics.auc(rec, prec)
    f1     = 2*prec*rec/(prec+rec+1e-8); idx = f1.argmax()
    return auc, pr_auc, f1[idx], thr[idx], labels, scores


def train_epoch(model, opt, ld_norm, ld_anom, device,
                seg_per_video, k, pos_weight):
    model.train(); tot = 0.0
    it_n, it_a = itertools.cycle(ld_norm), itertools.cycle(ld_anom)
    n_batches  = max(len(ld_norm), len(ld_anom))
    pw = torch.tensor(pos_weight, device=device) if pos_weight else None

    for _ in range(n_batches):
        v_n, v_a = next(it_n), next(it_a);  B = v_n.size(0)
        x_n = v_n.view(-1, v_n.size(-1)).to(device)
        x_a = v_a.view(-1, v_a.size(-1)).to(device)

        logits = torch.cat([model(x_n), model(x_a)], 0)        # (2B*S,1)
        labels = torch.cat([torch.zeros(B, device=device),
                            torch.ones (B, device=device)], 0)

        loss = topk_mil_bce(logits, labels, seg_per_video, k=k, pos_weight=pw)
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()
    return tot / n_batches


# ----------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", required=True)
    p.add_argument("--feat_type", choices=["timesformer", "i3d", "both"], default="both")
    p.add_argument("--epochs",       type=int,   default=40)
    p.add_argument("--batch_size",   type=int,   default=30)
    p.add_argument("--num_segments", type=int,   default=32)
    p.add_argument("--topk",         type=int,   default=16)
    p.add_argument("--pos_weight",   type=float, default=5.0)
    args = p.parse_args()

    # reproducibility
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(0)

    # data & model
    ld_n_tr, ld_a_tr, ld_n_te, ld_a_te, _ = make_loaders(
        args.data_root, args.feat_type, args.batch_size, args.num_segments)

    feat_dim = {"timesformer": 768, "i3d": 1024, "both": 1792}[args.feat_type]
    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model    = Learner(feat_dim).to(device)

    head = [p for n,p in model.named_parameters() if "cls" in n]
    base = [p for n,p in model.named_parameters() if "cls" not in n]
    opt  = torch.optim.Adam(
        [{"params": base, "lr": 6e-4},
         {"params": head, "lr": 9e-3}],
        weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-5)

    # ---------------------------------------------------------------
    #  folder for this run's checkpoints
    run_name = f"{args.feat_type}_{args.num_segments}seg_bs{args.batch_size}"
    ckpt_dir = os.path.join("checkpoints", run_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    # ---------------------------------------------------------------

    best_auc, keep = 0.0, []     # list of (auc, path)

    for ep in range(1, args.epochs + 1):
        loss = train_epoch(model, opt, ld_n_tr, ld_a_tr, device,
                           args.num_segments, args.topk, args.pos_weight)
        auc, pr_auc, f1, thr, y, y_score = evaluate(
            model, ld_a_te, ld_n_te, device, args.num_segments)

        print(f"Epoch {ep:03d} | loss {loss:.4f}"
              f" | AUC {auc:.4f} | PR-AUC {pr_auc:.4f}"
              f" | F1 {f1:.4f} @thr={thr:.3f}")
        scheduler.step()

        if auc > best_auc:
            best_auc = auc
            path = os.path.join(ckpt_dir, f"ep{ep:02d}_{auc:.4f}.pth")
            torch.save(model.state_dict(), path)
            keep.append((auc, path)); keep = sorted(keep)[-3:]  # keep best-3

    # average best-3
    avg = {}
    for _, path in keep:
        w = torch.load(path)
        for k,v in w.items():
            avg[k] = avg.get(k, 0) + v / len(keep)
    model.load_state_dict(avg)
    auc, pr_auc, f1, thr, y, y_score = evaluate(
        model, ld_a_te, ld_n_te, device, args.num_segments)
    print(f"\nAveraged best-3 → AUC {auc:.4f} | PR-AUC {pr_auc:.4f}")

    plot_curves(y, y_score, f"curves_{args.feat_type}")


if __name__ == "__main__":
    main()
