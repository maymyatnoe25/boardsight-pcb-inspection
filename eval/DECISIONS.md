------------------------------------------------------------
# BoardSight: research question, plan and rules

Written on Day 1, before any model was trained or evaluated.

## Research question

With about 1,000 good boards and only 100 defective ones per board type, when does
supervision help? Does a fine-tuned CNN catch defect types it has never seen, and how
many labelled defects does it need before it beats an anomaly detector trained on good
boards only?

## The two models

- Model A, anomaly detector: pretrained CNN patch features from good boards are stored
  in a memory bank. A new image is scored by how far its patches are from the nearest
  stored normal patches.
- Model B, supervised CNN: a pretrained ResNet18 fine-tuned on labelled good and
  defective boards.

## Experiments

- E1: Image-level comparison of both models. Metrics: precision, recall, F1,
  false-negative rate, CPU latency per image.
- E2: Leave-one-defect-type-out for the CNN. This needs defect-type labels, which are
  not yet verified. If they do not exist, E2 is replaced by a simpler experiment.
- E3: Learning curve. The CNN is trained with 5, 10, 25 and 50 defective examples.
- E4: Localisation. The anomaly heatmap and Grad-CAM are each scored against the
  ground-truth masks.
- E5: Ablations of the anomaly detector: backbone layers, image size and memory-bank size,
  against accuracy and speed.

## Rules

- The train, validation and test split is fixed and written in this file before any
  model runs.
- Decision thresholds are chosen on validation data only.
- The test set is scored once.
- Anomaly scores are not probabilities and are not described as probabilities.
- Grad-CAM is an explanation map, not a guaranteed defect localiser.
- Results are reported with confidence intervals, because each defect type has only
  about 14 to 25 images.
- Failures (missed defects and false alarms) are shown as images in the README.

## Decisions made on Day 2 (before any model was run)

Data finding: defect-type labels exist. PCB1 has 4 types (melt, missing, bent, scratch);
91 defective images have one type and 9 have two or three. E2 is therefore possible.
In E2, "holding out a type" removes every image whose label contains that type.

Split (per board, seed 0, written by src/make_split.py to eval/split_<board>.csv):
- Normal images: 200 test, 200 validation, the rest (604 for PCB1) for training.
- Defective images: 30 test (stratified by type), the other 70 are the development pool.
- The anomaly detector uses train_normal only.
- The CNN uses train_normal plus defective images drawn from the development pool.
  Defective images not used for training serve as its validation set.
- The 30 test defects and 200 test normals are scored once, at the end.

Threshold rule: the threshold is the highest score that reaches at least 90% recall on
validation defects. The resulting false-alarm rate on validation normals is reported.
The same threshold is applied unchanged to the test set.

Primary metrics: image-level AUROC (no threshold), and the false-negative rate and
false-alarm rate at the threshold rule. Differences between models are called meaningful
only if a paired bootstrap 95% interval for the difference excludes zero.

Limits stated in advance: the test set has only 30 defective images (about 14 melt and
4 to 5 of each other type), so per-type test results are not interpretable. E2 and E3
are run on the development pool and reported with bootstrap intervals.
## Decisions made on Day 4 (before any CNN was trained)

Comparison design: 5-fold cross-validation over the 70 development defects, stratified
by defect type (any image with several types forms one "multi" group), seed 0.
For each fold k, the CNN is trained on train_normal plus the development defects outside
fold k, and scored on the development defects in fold k plus the 200 validation normals.
The anomaly detector is scored on exactly the same images. Both models' AUROC is
computed per fold, and their difference is summarised as the mean over folds with a
paired bootstrap 95% interval (resampling defects within each fold and the 200 normals).

CNN settings, fixed now and not tuned on these folds: ResNet18 with ImageNet weights,
input 288 x 384, AdamW (learning rate 1e-4, weight decay 1e-4), batch size 16, 6 epochs,
loss weighted by the normal-to-defect ratio, random shifts of up to 8 pixels and
brightness jitter of about 10%. No early stopping and no hyperparameter search.
If training fails outright (for example the loss does not fall), any change is recorded
here with the reason.

Not yet decided: E2 and E3 design (Day 5).
## Decisions made on Day 5 (before E2 and E3 were run)

Already seen when this was written: the 5-fold comparison on the development defects.
CNN mean AUROC 0.992, anomaly detector 0.976, mean difference +0.016 (95% paired
bootstrap interval +0.003 to +0.031). Neither model has been tuned.

E2, leave-one-type-out: for each of melt, missing, bent, scratch, remove every
development defect whose label contains that type from CNN training. The CNN trains on
train_normal plus the remaining development defects, and is scored on the removed
defects plus the 200 validation normals. The anomaly detector is scored on the same
images. Images with several types appear in several test sets. Settings as on Day 4.

E3, learning curve: for each of the 5 folds, the CNN trains on train_normal plus a random
subset of 5, 10 or 25 of the development defects outside the fold, or on all of them
(about 56). One random subset per setting (seed 1000 + 10 x fold + n), so the curve is
noisy. Scored on the fold's defects plus the 200 validation normals, as on Day 4.
The crossover point is the smallest n at which the paired 95% interval for
(CNN minus anomaly) lies above zero. If there is none, that is reported.

Thresholds (planned, applied on Day 6): the CNN's threshold is chosen from its
out-of-fold scores, because its in-sample scores on development defects are meaningless.
The anomaly detector's threshold comes from the validation scores. For both, the highest
score that reaches 90% recall on development defects; the false-alarm rate on the
validation normals is reported. The final CNN is trained once on all 70 development
defects. The test set is scored once, after all of this.

Known risks, stated in advance: possible near-duplicate images across folds; a single
seed per run; the test set has only 30 defects.
## Decisions made on Day 6 (before the ablations were run)

Already seen: E2 and E3 results (CNN needs about 25 labelled defects to match the anomaly
detector, 56 to beat it slightly; only melt shows a clear gap when a type is held out).

E5 grid for the anomaly detector (validation images only, seed 0):
image size 192x256, 288x384, 384x512 and memory-bank fraction 0.01, 0.05, 0.2, except that
384x512 with 0.2 is skipped for speed. Recorded per configuration: image-level AUROC with
bootstrap interval, pixel-level AUROC against the ground-truth masks, a pointing score
(share of defective images whose strongest heatmap peak lies inside the true defect or
within 5% of the image height of it), nearest-neighbour search time, and end-to-end time
per image.

Selection rule: the final anomaly configuration is the one with the highest validation
AUROC. If it is within 0.005 of the default (288x384, 0.05), the default is kept for
simplicity and speed. The choice is then fixed before the test set is scored.

Limits stated in advance: one memory-bank seed per configuration, so differences of a
few thousandths are not interpretable; the default is repeated with seeds 1 and 2 to
show the size of that noise.
## Decisions made on Day 7 (before the head-to-head was recomputed)

Selection rule applied: the highest validation AUROC in the E5 grid is 384x512 with a 5%
memory bank (0.9914); the default scored 0.9756 (0.9756, 0.9794, 0.9759 over three seeds).
The margin of 0.016 exceeds the 0.005 tolerance, so the final anomaly configuration is
384x512, 5% bank, seed 0. Noted but not chosen: 384x512 with a 1% bank (AUROC 0.9841,
295 ms per image) is cheaper.

Limits stated in advance: the configuration was picked as the best of eight on 70 validation
defects, so its validation score is optimistic. The CNN was not tuned at higher resolution
(fixed at 288x384), so the comparison favours the anomaly detector's tuned setting.

Head-to-head: the CNN's out-of-fold scores from the five E3 "all labels" runs are paired with
the final anomaly detector on the same images. Per fold, I report AUROC, the false-alarm
rate at the threshold that reaches 90% recall (highest score reaching at least
ceil(0.9 x defects)), and the share of defects missed at a 5% false-alarm rate.

Thresholds for the test set (fixed now, saved by src.head_to_head):
- anomaly detector: the highest score reaching 90% recall on all 70 development defects;
- CNN: the mean of the five per-fold thresholds at 90% recall, applied to a CNN trained once
  on all 70 development defects. This assumes the five fold models and the final model share
  a score scale; the spread of the five thresholds is reported as a check.
The test set is scored once, on a later day, after the Grad-CAM experiment (E4).
## Amendment on Day 8 (made before any test image was scored)

The Day 7 check failed. The five CNN fold thresholds at 90% recall were 1.31, 1.48, 1.13,
0.98 and 2.95 (about a threefold range), so the planned rule (average them and apply the
mean to a CNN trained on all 70 development defects) is not trustworthy. It is replaced.

New rule for both models: the decision threshold is the 95th percentile of the model's
scores on the 200 validation normals (a 5% false-alarm target). An image is flagged if its
score is strictly above the threshold. The final CNN is trained once on train_normal plus
all 70 development defects (288x384, 6 epochs, seed 0, the Day 4 settings), so the 200
validation normals were not used in its training. The final anomaly detector is
384x512 with a 5% bank, seed 0.

Reported on the test set, scored once: defects caught out of 30 (exact 95% interval),
false-alarm rate on the 200 test normals (exact 95% interval), and AUROC (bootstrap
interval). The earlier anomaly threshold (1.943, 90% recall on development defects) is
kept as a secondary number and is not used for the comparison.

Stated now: with 30 test defects, one defect is 3.3 percentage points of recall, and the
test set has about 14 melt defects and 4 to 5 of each other type, so per-type test results
are not interpretable.
## Decisions made on Day 8 (before the test set was scored)

Seen before writing this: the E3 comparison against the final detector (CNN never beats it;
matches only at 56 labels), and the validation galleries.

E4, localisation: for each of the 70 development defects, the strongest peak of a map is
a "pointing hit" if it lies inside the true defect mask, or within 5% of the image height of
it ("slack"); a strict version uses zero slack. Maps compared: the final anomaly heatmap
(384x512, 5% bank) and Grad-CAM from layer3 and from layer4 of the CNN. Both layers are
reported and neither is selected after the fact. Grad-CAM uses the out-of-fold CNN (5 folds,
same settings and seeds as the E3 "all" runs; CPU training is not bit-exact, so small
differences from those runs are possible). Differences are summarised with a bootstrap over
defects (95% interval). Pixel AUROC is not used, since it was above 0.99 for every setting.

Prediction written in advance: both Grad-CAM maps will point worse than the anomaly heatmap,
because their cells (16 and 32 pixels at 288x384) are as large as or larger than the median
defect (about 15 pixels).

Final models and thresholds (this replaces the amendment's wording only where stated):
the final CNN is trained once on train_normal plus all 70 development defects. The decision
threshold of each model is the 95th percentile of its scores on the 200 validation normals;
an image is flagged if its score is strictly above it. Weights are not committed to Git.