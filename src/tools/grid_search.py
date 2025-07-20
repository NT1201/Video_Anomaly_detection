# grid_search.py
import itertools, subprocess, argparse, sys, shlex

parser = argparse.ArgumentParser()
parser.add_argument("--data_root", required=True)
parser.add_argument("--num_segments", type=int, default=64)
parser.add_argument("--batch_size",   type=int, default=15)
parser.add_argument("--k_list",  default="8,16,32")   # comma-separated
parser.add_argument("--w_list",  default="4,5,6")     # comma-separated
parser.add_argument("--epochs",  type=int, default=20)
args = parser.parse_args()

k_vals  = [int(x)   for x in args.k_list.split(",")]
w_vals  = [float(x) for x in args.w_list.split(",")]

for k, w in itertools.product(k_vals, w_vals):
    cmd = [
        sys.executable, "main.py",
        "--data_root", args.data_root,
        "--feat_type", "both",
        "--num_segments", str(args.num_segments),
        "--batch_size",  str(args.batch_size),
        "--topk",        str(k),
        "--pos_weight",  str(w),
        "--epochs",      str(args.epochs)
    ]
    print("\n>>>", " ".join(shlex.quote(c) for c in cmd), "\n")
    subprocess.run(cmd, check=True)
