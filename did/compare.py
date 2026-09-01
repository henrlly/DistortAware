"""Combine the per-config robustness.csv files into one grouped bar chart +
markdown table.  Reads results_<recon>_<clf>/robustness.csv, writes
results/did/comparison/{comparison.png,comparison.md}.

    python did/compare.py results/did/sd15_resnet18 results/did/sd15_resnet50 \
        results/did/sana16_resnet18 results/did/sana16_resnet50
"""
import csv, os, sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = "results/did/comparison"
dirs = sys.argv[1:] or [
    "results/did/sd15_resnet18", "results/did/sd15_resnet50",
    "results/did/sana16_resnet18", "results/did/sana16_resnet50",
    "results/did/sid_set_sd15_resnet18", "results/did/pooled_sd15_resnet18",
]


def load(d):
    rows = list(csv.DictReader(open(os.path.join(d, "robustness.csv"))))
    return {r["transform"]: float(r["acc"]) for r in rows}


def label(d):
    n = os.path.basename(d).replace("results_", "")
    n = n.replace("sid_set_", "SID ").replace("pooled_", "Pooled ")
    n = n.replace("sd15", "SD-1.5").replace("sana16", "SANA-1.6B")
    n = n.replace("_resnet", " / RN")
    n = n.replace("SID SD-1.5", "SID / SD-1.5").replace("Pooled SD-1.5", "Pooled / SD-1.5")
    return n


def main():
    os.makedirs(OUT, exist_ok=True)
    data = {label(d): load(d) for d in dirs if os.path.exists(os.path.join(d, "robustness.csv"))}
    order = ["clean", "jpeg90", "jpeg70", "jpeg50", "jpeg30", "blur0.5", "blur1.0",
             "blur2.0", "resize0.5", "resize0.25", "noise0.02", "noise0.05",
             "noise0.10", "jitter", "crop80"]
    transforms = [t for t in order if all(t in v for v in data.values())]

    # markdown table
    lines = ["# Reconstructor × classifier comparison", "",
             "| config | " + " | ".join(transforms) + " | mean(transf) |",
             "|" + "---|" * (len(transforms) + 2)]
    for name, v in data.items():
        tr = [v[t] for t in transforms if t != "clean"]
        cells = " | ".join(f"{v[t]*100:.1f}" for t in transforms)
        lines.append(f"| {name} | {cells} | **{sum(tr)/len(tr)*100:.1f}** |")
    open(os.path.join(OUT, "comparison.md"), "w").write("\n".join(lines) + "\n")

    # grouped bars
    import numpy as np
    x = np.arange(len(transforms))
    w = 0.8 / len(data)
    fig, ax = plt.subplots(figsize=(14, 5))
    for i, (name, v) in enumerate(data.items()):
        ax.bar(x + i * w, [v[t] * 100 for t in transforms], w, label=name)
    ax.axhline(80, color="#e34a33", ls="--", lw=1, label="80% target")
    ax.set_xticks(x + 0.4 - w / 2)
    ax.set_xticklabels(transforms, rotation=45, ha="right")
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(60, 100)
    ax.set_title("DID detector accuracy by reconstructor × classifier, clean vs. 14 transforms")
    ax.legend(ncol=3, fontsize=9); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "comparison.png"), dpi=120)
    print("wrote", OUT + "/comparison.png and comparison.md")


if __name__ == "__main__":
    main()
