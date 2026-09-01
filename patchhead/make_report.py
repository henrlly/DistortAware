"""results/patchhead/results_<tag>/report.md + robustness.png, and -- if a DID
metrics.json is given -- a side-by-side PatchHead-vs-DID per-transform chart.
"""
import argparse
import csv
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--did-metrics", default=None)
    ap.add_argument("--title", default="PatchHead detector")
    args = ap.parse_args()
    R = args.results
    rows = list(csv.DictReader(open(os.path.join(R, "robustness.csv"))))
    m = json.load(open(os.path.join(R, "metrics.json")))
    try:
        err = json.load(open(os.path.join(R, "error_analysis.json")))
    except FileNotFoundError:
        err = None
    did = json.load(open(args.did_metrics)) if args.did_metrics and os.path.exists(args.did_metrics) else None

    L = [f"# {args.title} — robustness summary", ""]
    L.append(f"- **Clean accuracy:** {m['clean_acc']*100:.1f}%  (AUC {m['clean_auc']:.3f})")
    L.append(f"- **Mean over 14 transforms:** {m['mean_transformed_acc']*100:.1f}%")
    L.append(f"- **Worst transform:** {m['min_transformed_acc']*100:.1f}%  ({m.get('worst_transform','?')})")
    if m.get("distortion_mode"):
        L.append(f"- **Distortion conditioning:** `{m['distortion_mode']}`")
    if did:
        L.append("")
        L.append("| | PatchHead | DID (SD-1.5 / RN-18) |")
        L.append("|---|---:|---:|")
        L.append(f"| clean acc | {m['clean_acc']*100:.1f}% | {did['clean_acc']*100:.1f}% |")
        L.append(f"| clean AUC | {m['clean_auc']:.3f} | {did['clean_auc']:.3f} |")
        L.append(f"| mean transformed | {m['mean_transformed_acc']*100:.1f}% | {did['mean_transformed_acc']*100:.1f}% |")
        L.append(f"| worst transformed | {m['min_transformed_acc']*100:.1f}% | {did['min_transformed_acc']*100:.1f}% |")
    has_conditioning = bool(rows and "base_acc" in rows[0])
    if has_conditioning:
        L += ["", "| Transform | n | Adjusted acc | Base acc | AUC | Real acc | Fake acc | Mean dynamic threshold |",
              "|---|---:|---:|---:|---:|---:|---:|---:|"]
    else:
        L += ["", "| Transform | n | Accuracy | AUC | Real acc | Fake acc |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        if has_conditioning:
            L.append(f"| {r['transform']} | {r['n']} | {float(r['acc'])*100:.1f}% | "
                     f"{float(r['base_acc'])*100:.1f}% | {float(r['auc']):.3f} | "
                     f"{float(r['real_acc'])*100:.1f}% | {float(r['fake_acc'])*100:.1f}% | "
                     f"{float(r['mean_dynamic_threshold']):.3f} |")
        else:
            L.append(f"| {r['transform']} | {r['n']} | {float(r['acc'])*100:.1f}% | "
                     f"{float(r['auc']):.3f} | {float(r['real_acc'])*100:.1f}% | "
                     f"{float(r['fake_acc'])*100:.1f}% |")
    if err:
        L += ["", "## Error analysis (clean test set)", "",
              f"- False positives (real flagged AIGC): **{err['n_false_pos']}**",
              f"- False negatives (AIGC missed): **{err['n_false_neg']}**", "",
              "Worst false positives:"]
        L += [f"  - `{e['key']}`  score={e['score']:.3f}" for e in err["worst_false_positives"][:5]]
        L += ["", "Worst false negatives:"]
        L += [f"  - `{e['key']}`  score={e['score']:.3f}" for e in err["worst_false_negatives"][:5]]
    open(os.path.join(R, "report.md"), "w").write("\n".join(L) + "\n")
    print("wrote", os.path.join(R, "report.md"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        names = [r["transform"] for r in rows]
        accs = [float(r["acc"]) * 100 for r in rows]
        x = np.arange(len(names))
        fig, ax = plt.subplots(figsize=(12, 4.5))
        if did:
            dvals = [did["per_transform"].get(n, np.nan) * 100 for n in names]
            ax.bar(x - 0.2, accs, 0.4, label="PatchHead", color="#2b8cbe")
            ax.bar(x + 0.2, dvals, 0.4, label="DID", color="#fdae61")
        else:
            ax.bar(x, accs, 0.6, color="#2b8cbe", label="PatchHead")
        ax.axhline(80, color="#e34a33", ls="--", lw=1)
        ax.set_xticks(x); ax.set_xticklabels(names, rotation=45, ha="right")
        ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
        ax.set_title(f"{args.title}: clean vs transformed test accuracy")
        ax.legend(); plt.tight_layout()
        plt.savefig(os.path.join(R, "robustness.png"), dpi=120)
        print("wrote", os.path.join(R, "robustness.png"))
    except Exception as e:
        print("skipped chart:", e)


if __name__ == "__main__":
    main()
