# metrics_utils.py
import numpy as np
from sklearn import metrics


def compute_all_metrics(labels, scores, far_thr=0.05):
    """
    Compute the common detection metrics.

    Parameters
    ----------
    labels : 1-D numpy array (0/1 ground-truth)
    scores : 1-D numpy array (higher = more anomalous)
    far_thr: float ∈ (0,1)  # threshold to report FAR

    Returns
    -------
    dict with keys:
      roc_auc, pr_auc,
      fpr, tpr, prec, rec, thr_pr,
      best_f1, best_thr,
      tn, fp, fn, tp, far
    """
    fpr, tpr, roc_thr = metrics.roc_curve(labels, scores)
    prec, rec, pr_thr = metrics.precision_recall_curve(labels, scores)

    roc_auc = metrics.auc(fpr, tpr)
    pr_auc  = metrics.auc(rec, prec)

    f1_vals  = 2 * prec * rec / (prec + rec + 1e-8)
    best_idx = np.nanargmax(f1_vals)
    best_thr = pr_thr[best_idx] if best_idx < len(pr_thr) else 0.5
    best_f1  = f1_vals[best_idx]

    # FAR @ chosen threshold (quantile on normal scores)
    norm_scores = scores[labels == 0]
    thr_far     = np.quantile(norm_scores, 1 - far_thr)
    preds_far   = (scores > thr_far).astype(int)
    tn, fp, fn, tp = metrics.confusion_matrix(labels, preds_far).ravel()
    far = fp / (fp + tn + 1e-8)

    return dict(
        roc_auc=roc_auc, pr_auc=pr_auc,
        fpr=fpr, tpr=tpr, prec=prec, rec=rec, thr_pr=pr_thr,
        best_f1=best_f1, best_thr=best_thr,
        tn=tn, fp=fp, fn=fn, tp=tp, far=far
    )
