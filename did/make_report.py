"""Turn results/robustness.csv + error_analysis.json into results/report.md and a
clean-vs-transformed bar chart (results/robustness.png)."""
import argparse, csv, json, os

R = "results"


def main():
    global R
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results/did")
    R = ap.parse_args().results
    rows = list(csv.DictReader(open(os.path.join(R, "robustness.csv"))))
    metrics = json.load(open(os.path.join(R, "metrics.json")))
    try:
        err = json.load(open(os.path.join(R, "error_analysis.json")))
    except FileNotFoundError:
        err = None

    lines = ["# Robustness Evaluation Summary", ""]
    lines.append(f"- **Clean accuracy:** {metrics['clean_acc']*100:.1f}%  "
                 f"(AUC {metrics['clean_auc']:.3f})")
    if metrics.get("mean_transformed_acc") is not None:
        lines.append(f"- **Mean accuracy over transformed test sets:** "
                     f"{metrics['mean_transformed_acc']*100:.1f}%")
        lines.append(f"- **Worst-case transformed accuracy:** "
                     f"{metrics['min_transformed_acc']*100:.1f}%")
    lines += ["", "| Transform | n | Accuracy | AUC | Real acc | Fake acc |",
              "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| {r['transform']} | {r['n']} | {float(r['acc'])*100:.1f}% | "
                     f"{float(r['auc']):.3f} | {float(r['real_acc'])*100:.1f}% | "
                     f"{float(r['fake_acc'])*100:.1f}% |")

    if err:
        lines += ["", "## Error Analysis (clean test set)", "",
                  f"- False positives (real flagged as AIGC): **{err['n_false_pos']}**",
                  f"- False negatives (AIGC missed): **{err['n_false_neg']}**", "",
                  "Representative worst false positives (real, high AIGC score):"]
        for e in err["worst_false_positives"][:5]:
            lines.append(f"  - `{os.path.basename(e['path'])}`  score={e['score']:.3f}")
        lines.append("")
        lines.append("Representative worst false negatives (AIGC, low score):")
        for e in err["worst_false_negatives"][:5]:
            lines.append(f"  - `{os.path.basename(e['path'])}`  score={e['score']:.3f}")

    open(os.path.join(R, "report.md"), "w").write("\n".join(lines) + "\n")
    print("wrote", os.path.join(R, "report.md"))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        names = [r["transform"] for r in rows]
        accs = [float(r["acc"]) * 100 for r in rows]
        fig, ax = plt.subplots(figsize=(11, 4))
        colors = ["#2b8cbe" if n == "clean" else "#a6bddb" for n in names]
        ax.bar(names, accs, color=colors)
        ax.axhline(80, color="#e34a33", ls="--", lw=1, label="80% target")
        ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 100)
        ax.set_title("DID detector: accuracy on clean vs. transformed WildFake test images")
        ax.legend(); plt.xticks(rotation=45, ha="right"); plt.tight_layout()
        plt.savefig(os.path.join(R, "robustness.png"), dpi=120)
        print("wrote", os.path.join(R, "robustness.png"))
    except Exception as e:
        print("skipped chart:", e)


if __name__ == "__main__":
    main()
