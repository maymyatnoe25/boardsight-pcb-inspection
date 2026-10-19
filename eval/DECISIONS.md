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

## Decisions to be recorded here (not yet made)

- The split design.
- The decision threshold rule.
- The success criteria for each experiment.
------------------------------------------------------------