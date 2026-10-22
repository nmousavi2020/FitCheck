FitCheck — ML Model Fit Diagnostics



Overfitting • Underfitting • Generalization Analysis



FitCheck is a Python project that demonstrates how model complexity and dataset size affect machine learning performance and generalization.



Features

Training and validation accuracy analysis

Generalization gap calculation

Overfitting and underfitting diagnostics

Visual comparison of different fitting scenarios

CSV statistics and JSON summary reports

Diagnostic Examples

1\. Underfitting



Underfitting occurs when a model is too simple to learn the underlying patterns in the dataset. Both training and validation accuracy may be low.



!Underfitting Example



Example: Dataset containing 10 samples.



2\. Overfitting



Overfitting occurs when a model learns the training data too closely and performs less effectively on unseen data. A large gap between training and validation accuracy can indicate overfitting.



!Overfitting Example



Example: Dataset containing 100 samples.



3\. Good Fit



A good fit occurs when a model achieves satisfactory training and validation performance with a relatively small generalization gap.



!Good Fit Example



Example: Dataset containing 1,000 samples.



Methodology



FitCheck uses a Decision Tree classifier with varying maximum tree depths.



For each depth, it calculates:



Training Accuracy: Accuracy on the training dataset.

Validation Accuracy: Accuracy on held-out validation data.

Generalization Gap: Training accuracy minus validation accuracy.



The diagnostic system uses configurable thresholds to flag potential fitting problems.



Installation



Requirements: Python 3.10 or later.



Install the dependencies:



pip install -r requirements.txt



Usage



Run FitCheck from the project directory:



python fitcheck.py





The diagnostic reports and plots are saved in the results/ directory.



Output Files

File	Description

statistics.csv	Metrics for each tested tree depth

summary.json	Diagnosis and selected model statistics

diagnostics.png	Training, validation, and generalization plots



The three example images above illustrate separate diagnostic scenarios.



Limitations

A single train-validation split may produce unstable estimates.

Very small datasets may not represent unseen data reliably.

Accuracy thresholds are heuristic and do not prove overfitting or underfitting.

Cross-validation and a separate test set are recommended for more reliable evaluation.

Author



Naeim Mousavi



GitHub: @nmousavi2020



License



Copyright (c) 2026 Naeim Mousavi. All rights reserved.



No permission is granted to redistribute, modify, or use this project without the author's prior written permission.

