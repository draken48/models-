FNIS QUALITY GATE
=================

Architecture:
MobileNetV3-Small (mobilenetv3_small_100)

Input:
224 x 224 RGB

Classes:
0 = low_quality
1 = high_quality

Weak-label definition:
Plane == "Fetal brain" -> class 1
All other Plane values -> class 0

Important:
This is a weak dataset-derived proxy for suitability for the fetal brain
processing pipeline. It is NOT a radiologist-rated clinical image-quality
assessment.

Best checkpoint:
Epoch 6

Best validation Macro F1:
0.994367

Calibrated threshold:
0.74

Inference:
quality_score = softmax_probability(class_1)

Decision:
quality_score >= 0.74 -> high_quality
quality_score < 0.74 -> low_quality

Locked test evaluation was performed only after model training and threshold
calibration.

Generated:
20260928_131608