# Independent physics-evidence review protocol

This protocol measures whether shadow and reflection explanations are repeatable enough to present. It does not ask reviewers to predict whether an image is real or generated.

## Assignment

1. Start with records marked `independent_double_review_target: true` in the SID `review_queue.json` (10 per native label in the current 150-image sample).
2. Give both reviewers the same image keys but separate annotation directories.
3. Hide SID labels, primary-detector scores, physics prefill, and the other reviewer's annotations during review.
4. Use distinct, stable reviewer IDs in the browser annotator.
5. Continue beyond the initial target if needed until at least 20% of cases considered shadow/reflection-applicable by either reviewer have two reviews.

The agreement command rejects exports without a non-empty `reviewer.id`; filenames are never treated as reviewer identity.

## Per-image decisions

For each cast-shadow and planar-reflection cue, choose exactly one applicability state:

- `applicable`: the scene satisfies the cue assumptions and at least three comparable correspondences can be marked;
- `not_applicable`: the cue is absent or its physical model clearly does not apply;
- `uncertain`: the cue may be present, but correspondence/model assumptions cannot be defended;
- `unreviewed`: work is incomplete; this is never treated as a negative.

For perspective, reviewers may draw rectangles around defensible structural regions. Avoid people, foliage, food, text, decorative texture, curved objects, and reflection content unless those edges are genuinely structural. Region confidence below 0.5 is excluded by default.

## Cue assumptions

Cast-shadow pairs require comparable object-ground contacts and corresponding shadow tips. Multiple lights, diffuse illumination, uneven terrain, and soft or occluded shadows can make the model inapplicable.

Reflection pairs require a planar reflector and matched visible/reflected object points. Curved mirrors, water, multiple panes, refraction, and ambiguous correspondence can make the model inapplicable.

## Freeze and compare

Do not adjudicate disagreements until both directories are frozen. Then run:

```bash
physics-review-agreement \
  --images path/to/images \
  --reviewer-a path/to/reviewer-a \
  --reviewer-b path/to/reviewer-b \
  --output outputs/reviewer_agreement.json
```

The report includes:

- coverage per reviewer and cue;
- applicability confusion matrices, observed agreement, and Cohen's kappa;
- agreement of the resulting geometric status;
- matched-pair count, pair-count Dice, and mean endpoint distance in normalized image coordinates;
- unresolved image keys and identity provenance.

Do not reduce the report to kappa alone. Report the confusion matrix and prevalence because rare applicable cues can make kappa unstable. Coordinate agreement checks whether reviewers clicked similar correspondences; it does not prove either interpretation is physically correct.

## Adjudication and proposal models

After the frozen report, reviewers may jointly inspect disagreements and create a separate adjudicated set. Never overwrite either original export. Learned shadow/reflection proposals may be evaluated only against the frozen/adjudicated baseline, must preserve editable endpoints and confidence, and must return `not_applicable` or `uncertain` when correspondence ambiguity is high.
