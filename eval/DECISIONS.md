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