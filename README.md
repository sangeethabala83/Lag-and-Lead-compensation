## Lag-and-Lead Compensation for Lung Nodule Classification##

This repository contains the source code, preprocessing scripts, trained-model checkpoints, and experimental configuration files used in the manuscript:

“Frequency-Shaping Preprocessing Inspired by Classical Compensator Design for Robust Lung Nodule Classification”

The repository is intended to support reproducibility of the proposed lag–lead compensation framework integrated with EfficientNetB3 for benign and malignant lung nodule classification.
The repository includes:

Complete preprocessing scripts
Lag and lead compensation implementation
Model training code
Model evaluation code
Trained-model checkpoints
Performance metric calculation scripts
Figure-generation scripts used for manuscript analysis
Research Overview

The proposed method applies a lag–lead compensation concept as a frequency-shaping preprocessing module before classification using EfficientNetB3. The lag component is designed to suppress noise-sensitive high-frequency variations, while the lead component enhances diagnostically relevant edge and boundary information. The compensated image representation is then used for lung nodule classification.

The main objective of this repository is to allow independent users to reproduce the preprocessing pipeline, training procedure, evaluation metrics, and visual analyses reported in the manuscript.
Dataset

The experiments were conducted using the IQ-OTH/NCCD lung cancer CT image dataset. The original dataset contains benign, malignant, and normal chest CT images. In this study, only benign and malignant images were used for binary classification, while normal images were excluded.

Because the public dataset does not provide explicit patient-level identifiers, image-level splitting was performed using a 70:15:15 ratio for training, validation, and testing.

Training data augmentation was applied only to the training subset. Validation and test images were not augmented to avoid biased performance evaluation.
External Validation done by LIDC-IDRI and JSRT X-ray dataset.
Preprocessing Pipeline

## The preprocessing steps are implemented in the data_preprocessing/ folder ##

The pipeline includes:
Image loading, Image resizing, Pixel normalization, Train-validation-test splitting Training-only augmentation
Lag–lead compensation-based frequency-shaping preprocessing.
Model Architecture
The classification model is based on EfficientNetB3. The lag–lead compensated image is used as the input representation to the EfficientNetB3 classifier.

The model implementation is available in:

models/IQOTHEFFICIENTNETB3.py
Main Experimental Outputs

The following result files are saved in the results/ folder:

results/confusion_matrix.png
results/classification_report.csv
results/sensitivity_analysis.png
results/bode_plot.png
results/grad_cam_examples.png
results/feature_map_examples.png
Code Availability Statement

The complete source code, trained-model checkpoints, preprocessing scripts, and evaluation scripts are publicly available at:

https://github.com/sangeethabala83/Lag-and-Lead-compensation
The repository contains a README file describing the environment setup, preprocessing pipeline, training procedure, evaluation scripts, and model checkpoint usage.
## Additional Experimental Analyses ##

In addition to the main training and evaluation pipeline, this repository includes scripts for ablation study, comparison with conventional preprocessing techniques, sensitivity analysis of lag–lead parameters, and statistical validation of performance differences. These analyses were added to improve reproducibility and to support the methodological claims reported in the manuscript.

## Ablation Study ##

The ablation study evaluates the contribution of each major component of the proposed framework. The purpose of this analysis is to determine whether the observed performance improvement is caused by the lag–lead compensation module rather than only by the EfficientNetB3 backbone.

## The ablation experiments include the following settings:##
1. EfficientNetB3 without lag–lead compensation
2. EfficientNetB3 with lag compensation only
3. EfficientNetB3 with lead compensation only
4. EfficientNetB3 with combined lag–lead compensation
The ablation scripts are available in:
ablation/
├── efficientnetb3_baseline.py
├── efficientnetb3_lag_only.py
├── efficientnetb3_lead_only.py
└── efficientnetb3_lag_lead.py
The corresponding results are saved in:
results/ablation_results.csv
results/ablation_comparison_plot.png
These experiments report accuracy, precision, sensitivity, specificity, F1-score, and AUC where applicable.
o evaluate the influence of the insertion location of the lag–lead compensation module, an additional architectural ablation study was included. This analysis compares the proposed preprocessing-level lag–lead compensation with an internal MBConv-level compensation strategy.

The following ablation settings are included:

1. Baseline EfficientNetB3 without lag–lead compensation
2. EfficientNetB3 with preprocessing-level lag–lead compensation
3. EfficientNetB3 with lag compensation only
4. EfficientNetB3 with lead compensation only
5. EfficientNetB3 with combined lag–lead compensation
6. EfficientNetB3 with lag–lead compensation inserted inside the MBConv block

For the MBConv-level ablation, the lag–lead compensation module is inserted after the depthwise convolution operation and before the squeeze-and-excitation recalibration stage. This position was selected because depthwise convolution captures spatial feature responses, while the subsequent squeeze-and-excitation module recalibrates channel-wise feature importance. Therefore, inserting the lag–lead module before squeeze-and-excitation allows the compensated spatial features to be recalibrated by the attention mechanism.

The MBConv-level ablation is intended only as a comparative architectural variant. The

## Comparison with Conventional Preprocessing Methods

To demonstrate the advantage of the proposed lag–lead frequency-shaping preprocessing, this repository also includes comparison experiments against commonly used image preprocessing methods.
The compared preprocessing methods include:
1. Original resized and normalized image
2. Histogram equalization
3. CLAHE
4. Gaussian filtering
5. Median filtering
6. Unsharp masking
7. Proposed lag–lead compensation preprocessing
The comparison scripts are available in:
preprocessing_comparison/
├── conventional_preprocessing.py
├── train_with_preprocessing_methods.py
└── compare_preprocessing_results.py

The outputs are saved in:
results/preprocessing_comparison.csv
results/preprocessing_comparison_plot.png

This analysis helps verify whether the proposed compensator provides additional benefit compared with standard contrast-enhancement and denoising techniques.

## Sensitivity Analysis ##

A sensitivity analysis was performed to evaluate the influence of the lag–lead control parameters on model performance. This analysis examines whether small changes in the compensator parameters significantly affect classification accuracy and other evaluation metrics.

The analyzed parameters include:

K  : compensator gain
T1 : lag/lead time constant
T2 : lag/lead time constant
α  : lead compensator parameter
β  : lag compensator parameter

Each parameter is varied within a predefined range while keeping the other parameters fixed. The following metrics are recorded for each setting:

Accuracy, Precision ,Sensitivity/Recall, Specificity, F1-score, AUC

The sensitivity analysis scripts are available in:
sensitivity_analysis/
├── run_parameter_sensitivity.py
├── analyze_K_sensitivity.py
├── analyze_T1_sensitivity.py
├── analyze_T2_sensitivity.py
└── plot_sensitivity_results.py

The results are saved in:
results/sensitivity_analysis.csv
results/sensitivity_accuracy_plot.png
results/sensitivity_f1_score_plot.png

This analysis supports the robustness of the proposed method by showing how performance changes when the compensator parameters deviate from the selected configuration.

## Statistical Analysis ##

Statistical analysis was conducted to determine whether the performance differences between the proposed method and baseline/comparison methods are statistically meaningful.

The statistical analysis includes:

1. 95% confidence intervals for accuracy, precision, recall, specificity, and F1-score
2. McNemar’s test for paired classifier comparison
3. Comparative significance testing between the proposed model and baseline models

The statistical scripts are available in:
statistical_analysis/
├── confidence_interval_metrics.py
├── mcnemar_test.py
├── proportion_test.py
└── statistical_summary.py
The outputs are saved in:
results/statistical_analysis.csv
results/confidence_intervals.csv
results/mcnemar_test_results.csv

The statistical analysis provides quantitative evidence that the proposed lag–lead compensation framework improves classification performance beyond random variation.

## Summary of Included Reproducibility Materials##

This repository therefore contains all scripts required to reproduce the main and supplementary experiments:
Main model training and evaluation
Ablation study
Conventional preprocessing comparison
Lag–lead parameter sensitivity analysis
Statistical significance analysis
Bode-response visualization
Feature-map visualization
Trained-model checkpoints


“The MBConv-level variant was included only as an architectural ablation, whereas the final proposed framework uses preprocessing-level lag–lead compensation to maintain reproducibility and avoid modifying the EfficientNetB3 backbone.


