# Failure analysis (anomaly detector, 384x512, 5% bank, validation set)

Missed defects (lowest scores 1.80 to 1.90): tiny, mostly along the bottom edge near the pin
header and on small components. Heatmaps show little or no response. Median defect is 0.2%
of the image.

False alarms (highest normal scores 1.92 to 2.39): a dark mark on the background at the left
edge (worst, 2.39); boards rotated 180 degrees or tilted and shifted; small bright spots near
bottom components (cause not determined).

Overlap: the lowest defect scores and the highest normal scores overlap, so no threshold
separates them cleanly. Read from small screenshots; to be re-checked at full size.