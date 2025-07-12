# vis.py  ── put this near the top, after imports
import csv, numbers, os, pathlib, matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
# ‥ everything else in vis.py can stay …

# ----------------------------------------------------------------------
def save_epoch_csv(row: dict, csv_path: str):
    """
    Append one line to metrics.csv, **keeping only scalar values**.
    Creates the file (and header) if it does not exist.
    """
    # 1. keep numbers & short strings only
    row = {k: v for k, v in row.items()
           if isinstance(v, (numbers.Number, str))}
    
    # 2. ensure parent folder
    pathlib.Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    
    # 3. write
    write_header = not os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            w.writeheader()
        w.writerow(row)

# ----------------------------------------------------------------------
def epoch_summary_plots(save_root: str, m: dict):
    """
    Quick ROC / PR / Confusion-matrix plots for one epoch (or avg-best-3).
    Only uses arrays already inside `m`.
    """
    

    os.makedirs(os.path.dirname(save_root), exist_ok=True)

    # ROC + PR
    fig, ax = plt.subplots(1, 2, figsize=(10, 4), dpi=120)
    ax[0].plot(m["fpr"], m["tpr"]);  ax[0].plot([0, 1], [0, 1], "--", c="gray")
    ax[0].set(xlabel="FPR", ylabel="TPR",
              title=f"ROC (AUC {m['roc_auc']:.3f})")

    ax[1].plot(m["rec"], m["prec"])
    ax[1].set(xlabel="Recall", ylabel="Precision",
              title=f"PR (AUC {m['pr_auc']:.3f})")

    fig.tight_layout()
    fig.savefig(save_root + "_roc_pr.png"); plt.close(fig)

    # Confusion at best threshold
    cm = np.array([[m["tn"], m["fp"]],
                   [m["fn"], m["tp"]]])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anomaly"])
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=120)
    disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".0f")
    ax.set_title(f"Confusion @ thr={m['best_thr']:.2f}")
    fig.savefig(save_root + "_conf.png"); plt.close(fig)
