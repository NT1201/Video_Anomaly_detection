import os, argparse, numpy as np, torch, matplotlib.pyplot as plt
from dataset import VideoFeatureDataset
from learner import Learner
from scipy.ndimage import gaussian_filter1d      # smooth for prettier curves

parser = argparse.ArgumentParser()
parser.add_argument("--data_root"); parser.add_argument("--model")
parser.add_argument("--vid_idx", type=int, default=0)
args = parser.parse_args()

ds = VideoFeatureDataset(args.data_root, "test", "anomaly", "both", num_seg=64)
device = "cuda" if torch.cuda.is_available() else "cpu"
model  = Learner(1792).to(device); model.load_state_dict(torch.load(args.model)); model.eval()

x = ds[args.vid_idx].to(device)
gt1, gt2 = ds.get_gt_window(args.vid_idx)
with torch.no_grad():
    scores = torch.sigmoid(model(x)).cpu().numpy().ravel()
scores = gaussian_filter1d(scores, sigma=2)

plt.figure(figsize=(6,2.4))
plt.plot(scores, lw=2, label="Ours")
plt.axvspan(gt1, gt2, color="red", alpha=.15)
plt.ylim(0,1); plt.xlabel("Segment"); plt.ylabel("Score")
plt.legend(); os.makedirs("plots", exist_ok=True)
out = f"plots/timeline_{args.vid_idx}.png"; plt.savefig(out, bbox_inches="tight")
print("saved →", out)
