import pandas as pd
import matplotlib.pyplot as plt
import os

# ================= PATH =================
VAL_DIR = "../yolov5/runs/val/exp"
RESULTS_CSV = os.path.join(VAL_DIR, "results.csv")

# ================= LOAD RESULTS =================
df = pd.read_csv(RESULTS_CSV)

# Last epoch values
last = df.iloc[-1]

metrics = {
    "Precision": last["metrics/precision"],
    "Recall": last["metrics/recall"],
    "mAP@0.5": last["metrics/mAP_0.5"],
    "mAP@0.5:0.95": last["metrics/mAP_0.5:0.95"]
}

print("\n📊 PERFORMANCE METRICS\n")
for k, v in metrics.items():
    print(f"{k}: {v:.4f}")

# ================= PLOT BAR GRAPH =================
plt.figure(figsize=(8,5))
plt.bar(metrics.keys(), metrics.values())
plt.title("Performance Metrics of MCVXAI-VPD")
plt.ylabel("Score")
plt.grid(axis="y")
plt.tight_layout()

plt.savefig("performance_metrics.png")
plt.show()

print("\n✅ performance_metrics.png saved")
