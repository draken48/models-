FNIS / FetalScan AI
MODEL 4 — V2
================

Purpose
-------
Fetal head/brain ultrasound landmark detection
for biometric measurement.

Scope
-----
Brain/head ultrasound only.

Landmarks
---------
0 = OFD1
1 = OFD2
2 = BPD1
3 = BPD2

Architecture
------------
Backbone: ConvNeXt Tiny
Input: 512 x 512
Heatmaps: 128 x 128
Landmarks: 4
Parameters: 30,170,892

Inference
---------
The model produces:
- 4 landmark heatmaps
- direct coordinate predictions

For final localization inference, use the
heatmap branch with local soft-argmax refinement.

Endpoint identity is treated as permutation-invariant
for OFD and BPD pairs.

Final Test Set
--------------
Samples: 1153

Landmark mean error: 16.024 px
Landmark median: 11.805 px
PCK@20: 82.11%
PCK@30: 91.65%
PCK@50: 96.81%
PCK@100: 99.13%

OFD MAE: 15.707 px
OFD MAPE: 5.488%

BPD MAE: 15.013 px
BPD MAPE: 6.363%

Important
---------
This is a research/development model and is not
clinically validated.

The final FNIS pipeline should perform:
ultrasound image
-> quality/plane processing
-> landmark detection
-> BPD/OFD
-> pixel-to-mm calibration
-> HC calculation
-> confidence/uncertainty
-> clinical output.

Files
-----
checkpoints/
    model4_v2_best_state_dict.pth
    model4_v2_best_full_model.pth

data/
    fnis_model4_final_manifest.csv

results/
    model4_v2_corrected_history.csv
    model4_v2_FINAL_CORRECT_test_results.csv

metadata/
    model4_v2_metadata.json