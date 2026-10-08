============================================================
FNIS MODEL 6 — FINAL RELEASE
============================================================

Purpose
-------
Uncertainty and reliability estimation for FNIS Model 5.

Scope
-----
STRICTLY fetal brain/head ultrasound.

No whole-body fetal imaging is included.

------------------------------------------------------------
MODEL 5 BASE CLASSIFIER
------------------------------------------------------------

Architecture:
EfficientNetV2-S

Input:
384 x 384 RGB

Classes:
16 fetal brain ultrasound abnormality classes.

Model 5 final reported performance:
Accuracy          = 90.05% with fixed TTA
Macro-F1          = 72.43% with fixed TTA
Weighted-F1       = 88.60% with fixed TTA

------------------------------------------------------------
MODEL 6
------------------------------------------------------------

Method:
Monte-Carlo Dropout

MC samples:
30

Temperature scaling:
Validation-only calibration

Primary uncertainty:
Predictive entropy

Reliability threshold:
Calibrated using validation data only

Final test error-detection AUC:
0.9302

Final test reliability:
84.16% of test samples classified as HIGH reliability

Accuracy among HIGH reliability predictions:
96.24%

------------------------------------------------------------
FILES
------------------------------------------------------------

model5/
    model5_efficientnetv2s_best.pth
    model5_efficientnetv2s_best.onnx
    classes.txt
    config.json
    inference.py

model6/
    model6_config.json
    model6_inference.py
    model6_validation_uncertainty.csv
    model6_final_test_predictions.csv
    model6_reliability_quality.csv
    model6_final_results.json

------------------------------------------------------------
IMPORTANT
------------------------------------------------------------

The .pth checkpoint is required for MC-Dropout inference.

The ONNX model is provided for normal Model 5 inference,
but ONNX alone does NOT reproduce Monte-Carlo Dropout
uncertainty estimation.

Model confidence/reliability is NOT clinical certainty.

This is a research/prototype system and has not undergone
clinical validation or external prospective validation.

============================================================