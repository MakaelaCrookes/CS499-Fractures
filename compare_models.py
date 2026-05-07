# Imports
import os
import json
import matplotlib.pyplot as plt
import numpy as np

# Paths
RESNET_PATH = "outputs/metrics/resnet_metrics.json"
VIT_PATH = "outputs/metrics/vit_metrics.json"

# Load metrics
def load_metrics(path, model_name):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{model_name} metrics not found at {path}")
    
    with open(path, "r") as f:
        metrics = json.load(f)
    
    return metrics

print("Loading metrics...")

resnet_metrics = load_metrics(RESNET_PATH, "ResNet")
vit_metrics = load_metrics(VIT_PATH, "ViT")

# Comparison table
print("\n" + "="*50)
print("MODEL COMPARISON")
print("="*50)

print(f"{'Metric':<12} {'ResNet18':<12} {'ViT':<12}")
print("-" * 36)

for key in ["accuracy", "precision", "recall", "f1"]:
    print(f"{key.capitalize():<12} "
          f"{resnet_metrics[key]:<12.4f} "
          f"{vit_metrics[key]:<12.4f}")

# Prep data
metrics_names = ["Accuracy", "Precision", "Recall", "F1-Score"]

resnet_scores = [
    resnet_metrics["accuracy"],
    resnet_metrics["precision"],
    resnet_metrics["recall"],
    resnet_metrics["f1"]
]

vit_scores = [
    vit_metrics["accuracy"],
    vit_metrics["precision"],
    vit_metrics["recall"],
    vit_metrics["f1"]
]

# Bar chart
os.makedirs("outputs/figures", exist_ok=True)

x = np.arange(len(metrics_names))
width = 0.35

plt.figure(figsize=(10, 6))

plt.bar(x - width/2, resnet_scores, width, label="ResNet18")
plt.bar(x + width/2, vit_scores, width, label="ViT")

plt.xlabel("Metrics")
plt.ylabel("Score")
plt.title("ResNet18 vs ViT: Fracture Detection Performance")
plt.xticks(x, metrics_names)
plt.ylim(0, 1)
plt.legend()
plt.grid(True, alpha=0.3)

# Label
for i, (r, v) in enumerate(zip(resnet_scores, vit_scores)):
    plt.text(i - width/2, r + 0.02, f"{r:.3f}", ha='center', fontsize=9)
    plt.text(i + width/2, v + 0.02, f"{v:.3f}", ha='center', fontsize=9)

plt.tight_layout()

output_path = "outputs/figures/model_comparison.png"
plt.savefig(output_path, dpi=150)

print(f"\nSaved comparison chart to {output_path}")
print("\nDone!")