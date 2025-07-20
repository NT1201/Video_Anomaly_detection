import pandas as pd, os; os.makedirs("plots", exist_ok=True)
df = pd.DataFrame({
    "Method": ["Binary clf","Hasan et al.","Lu et al.","Prop.-w/o-constr.","Prop.-w-constr.","Ours k32"],
    "AUC":    [50.0, 50.6, 65.51, 74.44, 75.41, 90.70]   # <- last one = your best Micro-AUC
})
df.to_csv("plots/table3_auc.csv", index=False)
print(df.to_markdown())
