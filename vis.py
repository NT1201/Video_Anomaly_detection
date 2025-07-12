import matplotlib.pyplot as plt
from sklearn import metrics


def plot_curves(labels, scores, save_to):
    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    prec, rec, _ = metrics.precision_recall_curve(labels, scores)

    # ROC
    plt.figure()
    plt.plot(fpr, tpr, linewidth=2)
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("ROC curve")
    plt.grid(True)
    plt.savefig(f"{save_to}_roc.png", dpi=200)
    plt.close()

    # PR
    plt.figure()
    plt.plot(rec, prec, linewidth=2)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall curve")
    plt.grid(True)
    plt.savefig(f"{save_to}_pr.png", dpi=200)
    plt.close()
