# BoardSight

**Computer-vision inspection of printed circuit boards: anomaly detection compared with a fine-tuned CNN.**

*Work in progress.* No results have been produced yet.

## Research question

With about 1,000 good boards and only 100 defective ones per board type, when does
supervision help? Does a fine-tuned CNN catch defect types it has never seen, and how
many labelled defects does it need before it beats an anomaly detector trained on
good boards only?

The planned experiments and evaluation rules are in [`eval/DECISIONS.md`](eval/DECISIONS.md).

## Data

VisA dataset (Zou et al., 2022): https://github.com/amazon-science/spot-diff

The official repository states CC BY 4.0; some third-party documentation lists
CC BY-NC-SA 4.0. It is used here for non-commercial research and learning only, with
attribution. The images are not included in this repository.