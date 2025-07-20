# ---------- copy from here ----------
import glob, os, numpy as np, matplotlib.pyplot as plt
os.makedirs("plots/evolution", exist_ok=True)

scores = sorted(glob.glob("runs/scores_ep*.npy"))
vid    = 0                       # which test-video to visualise
for n, f in enumerate(scores, 1):
    score_vec = np.load(f)       # shape  (N_test ,)
    plt.plot(score_vec[vid], lw=1)
    plt.title(f"Iter {n*1000}")
    plt.axvspan(60,120, color='salmon', alpha=.3)  # <-- GT window
    plt.savefig(f"plots/evolution/iter{n:02d}.png", dpi=120)
    plt.clf()
print("Saved plots/evolution/iterXX.png for all saved epochs")
# ---------- copy up to here ----------
