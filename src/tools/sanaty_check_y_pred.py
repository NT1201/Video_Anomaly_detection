import numpy as np
from sklearn.metrics import confusion_matrix

# Load your npy files
y = np.load("runs/y_true.npy")
y_score = np.load("runs/y_score.npy")

# Use your threshold
best_thr = 0.9936615
y_pred = (y_score >= best_thr).astype(int)

cm = confusion_matrix(y, y_pred)
print("Shape:", y.shape)
print("Positive (Anomaly):", (y==1).sum())
print("Negative (Normal):", (y==0).sum())
print("Confusion matrix at thr=%.7f:" % best_thr)
print(cm)
