FNIS Swin-B Final Export
========================

Component:
    Brain Plane Classification

Model:
    swin_base_patch4_window12_384.ms_in22k_ft_in1k

Input:
    384 x 384 RGB

Classes:
    0 - Trans-thalamic
    1 - Trans-ventricular
    2 - Trans-cerebellar
    3 - Non-standard

Patient-level split:
    Train: 2159 images / 756 patients
    Val:    482 images / 163 patients
    Test:   451 images / 163 patients

Patient overlap:
    0

Best validation Macro-F1:
    0.7698343860941845

Best checkpoint epoch:
    19

FINAL TEST RESULTS
------------------
Accuracy:
    0.831486

Macro Precision:
    0.798153

Macro Recall:
    0.740935

Macro F1:
    0.764909

Balanced Accuracy:
    0.740935

Test images:
    451

Test patients:
    163

The test set was not used during training or model selection.

Files
-----
FNIS_SwinB_384_best_model.pt
    Best validation Macro-F1 model checkpoint.

SwinB_final_test_results.json
    Complete final test metrics.

SwinB_final_per_class_metrics.csv
    Precision, recall, F1 and support for every class.

SwinB_final_confusion_matrix.csv
    Four-class confusion matrix.

SwinB_final_test_predictions.csv
    Prediction and probability for every test image.

SwinB_training_history.json
    Training history.

SwinB_model_metadata.json
    Model and split metadata.

SwinB_inference_manifest.json
    Integration/inference contract.
