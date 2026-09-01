"""Compare the PatchHead detector with the DID detector image-for-image on the
clean test set.

Key question from the brief: for the images a detector gets wrong, is the *other*
detector wrong on the same images (shared hard cases) or does each fail on a
different set (complementary errors -> an ensemble would help)?

Inputs (all produced by the pipeline):
  --patchhead   results/patchhead/current/preds_clean.json     (evaluate.py)
  --did         results/patchhead/did_preds_<tag>.json          (harness/did_predictions.py)
Optionally --patchhead-metrics / --did-metrics (the two metrics.json) for the
headline robustness table.

Writes <out>/comparison.md and <out>/comparison.json.
"""
import argparse
import json
import math
import os


def load_preds(path):
    d = json.load(open(path))
    return d["preds"], float(d.get("threshold", 0.5))


def mcnemar(b, c):
    """Exact-ish McNemar on discordant counts b (only A wrong) and c (only B
    wrong).  Returns (chi2 with continuity correction, two-sided p)."""
    n = b + c
    if n == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / n if n > 0 else 0.0
    # normal approx to the two-sided binomial p-value
    z = (abs(b - c) - 1) / math.sqrt(n) if n > 0 else 0.0
    p = math.erfc(z / math.sqrt(2))
    return float(chi2), float(min(1.0, p))


def phi(n11, n10, n01, n00):
    """phi coefficient between the two 'is-wrong' indicators."""
    num = n11 * n00 - n10 * n01
    den = math.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00))
    return float(num / den) if den else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--patchhead", default="results/patchhead/current/preds_clean.json")
    ap.add_argument("--did", required=True)
    ap.add_argument("--patchhead-metrics", default="results/patchhead/current/metrics.json")
    ap.add_argument("--did-metrics", default=None)
    ap.add_argument("--name-a", default="PatchHead")
    ap.add_argument("--name-b", default="DID")
    ap.add_argument("--out", default="results/patchhead/comparison")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    A, B = args.name_a, args.name_b

    pa, ta = load_preds(args.patchhead)
    pb, tb = load_preds(args.did)
    keys = sorted(set(pa) & set(pb))
    only_a = sorted(set(pa) - set(pb))
    only_b = sorted(set(pb) - set(pa))
    if not keys:
        raise SystemExit("no overlapping image keys -- check --ds namespaces match")

    rows = []
    for k in keys:
        y = pa[k]["label"]
        assert pb[k]["label"] == y, f"label mismatch for {k}"
        a_wrong = pa[k]["pred"] != y
        b_wrong = pb[k]["pred"] != y
        rows.append((k, y, a_wrong, b_wrong, pa[k]["score"], pb[k]["score"]))

    n = len(rows)
    both_ok = [r for r in rows if not r[2] and not r[3]]
    a_only = [r for r in rows if r[2] and not r[3]]      # only PatchHead wrong
    b_only = [r for r in rows if not r[2] and r[3]]      # only DID wrong
    both_bad = [r for r in rows if r[2] and r[3]]

    acc_a = 1 - (len(a_only) + len(both_bad)) / n
    acc_b = 1 - (len(b_only) + len(both_bad)) / n
    err_a = len(a_only) + len(both_bad)
    err_b = len(b_only) + len(both_bad)

    chi2, p = mcnemar(len(a_only), len(b_only))
    ph = phi(len(both_bad), len(a_only), len(b_only), len(both_ok))

    # direction breakdown for the shared errors
    def direction(r):  # r = (k, y, a_wrong, b_wrong, sa, sb)
        return "false_pos" if r[1] == 0 else "false_neg"
    both_bad_fp = [r for r in both_bad if r[1] == 0]
    both_bad_fn = [r for r in both_bad if r[1] == 1]

    # per-class
    def split_class(lst):
        return sum(1 for r in lst if r[1] == 0), sum(1 for r in lst if r[1] == 1)

    overlap_of_a = len(both_bad) / err_a if err_a else 0.0
    overlap_of_b = len(both_bad) / err_b if err_b else 0.0
    union_err = err_a + err_b - len(both_bad)
    oracle_acc = 1 - len(both_bad) / n     # an oracle picking the right detector per image

    summary = dict(
        n_compared=n, keys_only_in_a=len(only_a), keys_only_in_b=len(only_b),
        name_a=A, name_b=B, threshold_a=ta, threshold_b=tb,
        acc_a=acc_a, acc_b=acc_b, err_a=err_a, err_b=err_b,
        both_correct=len(both_ok), both_wrong=len(both_bad),
        only_a_wrong=len(a_only), only_b_wrong=len(b_only),
        both_wrong_false_pos=len(both_bad_fp), both_wrong_false_neg=len(both_bad_fn),
        share_of_a_errors_also_wrong_in_b=overlap_of_a,
        share_of_b_errors_also_wrong_in_a=overlap_of_b,
        union_errors=union_err, oracle_upper_bound_acc=oracle_acc,
        mcnemar_chi2=chi2, mcnemar_p=p, error_phi_coefficient=ph,
        agreement_rate=(len(both_ok) + len(both_bad)) / n,
        both_wrong_keys=[r[0] for r in both_bad],
        only_a_wrong_keys=[r[0] for r in a_only],
        only_b_wrong_keys=[r[0] for r in b_only],
    )
    json.dump(summary, open(os.path.join(args.out, "comparison.json"), "w"), indent=2)

    # ---- markdown ----
    L = []
    L.append(f"# {A} vs {B} — image-for-image comparison (clean test set)\n")
    L.append(f"{n} images scored by both detectors "
             f"(`{A}`-only keys: {len(only_a)}, `{B}`-only keys: {len(only_b)}).\n")

    ma = _try(args.patchhead_metrics)
    mb = _try(args.did_metrics)
    if ma or mb:
        L.append("## Headline metrics\n")
        L.append(f"| | {A} | {B} |")
        L.append("|---|---:|---:|")
        L.append(f"| clean acc (this comparison) | {acc_a:.3f} | {acc_b:.3f} |")
        if ma and mb:
            L.append(f"| clean AUC | {ma.get('clean_auc', float('nan')):.3f} | "
                     f"{mb.get('clean_auc', float('nan')):.3f} |")
            L.append(f"| mean over 14 transforms | {ma.get('mean_transformed_acc', float('nan')):.3f} | "
                     f"{mb.get('mean_transformed_acc', float('nan')):.3f} |")
            L.append(f"| worst transform | {ma.get('min_transformed_acc', float('nan')):.3f} | "
                     f"{mb.get('min_transformed_acc', float('nan')):.3f} |")
        L.append("")

    L.append("## Error agreement\n")
    L.append(f"| | {B} correct | {B} wrong | total |")
    L.append("|---|---:|---:|---:|")
    L.append(f"| **{A} correct** | {len(both_ok)} | {len(b_only)} | {len(both_ok)+len(b_only)} |")
    L.append(f"| **{A} wrong** | {len(a_only)} | {len(both_bad)} | {len(a_only)+len(both_bad)} |")
    L.append(f"| **total** | {len(both_ok)+len(a_only)} | {len(b_only)+len(both_bad)} | {n} |")
    L.append("")
    L.append(f"- **{A} errors: {err_a}** — of which **{len(both_bad)} "
             f"({overlap_of_a*100:.0f}%)** are *also* wrong in {B}, "
             f"**{len(a_only)}** are unique to {A}.")
    L.append(f"- **{B} errors: {err_b}** — of which **{len(both_bad)} "
             f"({overlap_of_b*100:.0f}%)** are *also* wrong in {A}, "
             f"**{len(b_only)}** are unique to {B}.")
    L.append(f"- **Both wrong on the same image: {len(both_bad)}** "
             f"({len(both_bad_fp)} false positives / real images called fake, "
             f"{len(both_bad_fn)} false negatives / fakes called real).")
    L.append(f"- Union of all errors: {union_err}.  An oracle that picked the "
             f"better detector per image would score **{oracle_acc*100:.1f}%** "
             f"(vs {max(acc_a, acc_b)*100:.1f}% for the better single model) — "
             f"the headroom an ensemble could reach.")
    L.append("")
    L.append("## Are the errors correlated?\n")
    corr = ("essentially independent" if abs(ph) < 0.15 else
            "weakly correlated" if abs(ph) < 0.35 else
            "strongly correlated")
    L.append(f"- phi coefficient between the two 'is-wrong' indicators: **{ph:+.3f}** "
             f"({corr}). phi≈0 ⇒ the detectors fail on largely *different* images "
             f"(complementary); phi→1 ⇒ they trip on the *same* hard images.")
    exp_overlap = err_a * err_b / n
    L.append(f"- Shared errors observed: {len(both_bad)}.  If the two error sets "
             f"were independent you'd expect ≈ {exp_overlap:.1f}.")
    L.append(f"- McNemar (do the two disagree asymmetrically?): "
             f"χ²={chi2:.2f}, p={p:.3g} — "
             f"{'a significant' if p < 0.05 else 'no significant'} difference in "
             f"which detector is more accurate.")
    L.append("")
    L.append(f"## The {len(both_bad)} images both detectors get wrong\n")
    L.append("These are the genuinely hard cases — a bigger ensemble won't fix them.\n")
    for r in sorted(both_bad, key=lambda r: r[1]):
        kind = "real→fake" if r[1] == 0 else "fake→real"
        L.append(f"- `{r[0]}` ({kind});  {A} score {r[4]:.3f}, {B} score {r[5]:.3f}")
    L.append("")
    L.append(f"## Images only {A} gets wrong ({len(a_only)})\n")
    for r in sorted(a_only, key=lambda r: r[1])[:40]:
        kind = "real→fake" if r[1] == 0 else "fake→real"
        L.append(f"- `{r[0]}` ({kind});  {A} {r[4]:.3f} vs {B} {r[5]:.3f}")
    L.append("")
    L.append(f"## Images only {B} gets wrong ({len(b_only)})\n")
    for r in sorted(b_only, key=lambda r: r[1])[:40]:
        kind = "real→fake" if r[1] == 0 else "fake→real"
        L.append(f"- `{r[0]}` ({kind});  {A} {r[4]:.3f} vs {B} {r[5]:.3f}")
    L.append("")

    open(os.path.join(args.out, "comparison.md"), "w").write("\n".join(L))
    print("\n".join(L[:40]))
    print(f"\nwrote {args.out}/comparison.md and comparison.json")


def _try(path):
    try:
        return json.load(open(path)) if path else None
    except Exception:
        return None


if __name__ == "__main__":
    main()
