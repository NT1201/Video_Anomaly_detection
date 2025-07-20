import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import argparse

# Command-line threshold
parser = argparse.ArgumentParser()
parser.add_argument("--thresh", type=float, default=0.90)
args = parser.parse_args()
THRESHOLD = args.thresh  # Use this everywhere

# Load saved true labels and scores
y = np.load("runs/y_true.npy")
y_score = np.load("runs/y_score.npy")

# Get binary predictions
y_pred = (y_score >= THRESHOLD).astype(int)

print("y_true[:10]:", y[:10])
print("y_score[:10]:", y_score[:10])
print("y_true.shape:", y.shape)
print("y_score.shape:", y_score.shape)
print("Number of positives:", (y == 1).sum())
print("Number of negatives:", (y == 0).sum())

# Compute confusion matrix
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["Normal", "Anomaly"])
fig, ax = plt.subplots(figsize=(3.5, 3.5))
disp.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".0f")
ax.set_title(f"Confusion @ thr={THRESHOLD:.2f}")
plt.tight_layout()
plt.savefig(f"conf_matrix_thr{int(THRESHOLD*100)}.png")
plt.show()
