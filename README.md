FitCheck — ML Model Fit Diagnostics



Overfitting • Underfitting • Generalization Analysis



FitCheck is a lightweight Python project that demonstrates how model complexity and dataset size affect machine learning performance and generalization.



Overview



FitCheck evaluates a Decision Tree classifier at different maximum depths and compares training accuracy, validation accuracy, and generalization gap.



Features

Training and validation accuracy analysis

Generalization gap calculation

Overfitting and underfitting diagnostics

Visual comparison of three fitting scenarios

CSV statistics and JSON summary reports





Diagnostic Results

Comparison of All Cases



The following figure compares underfitting, overfitting, and good-fit scenarios.



!FitCheck Comparison



1\. Underfitting



Underfitting occurs when a model is too simple to learn the underlying patterns. Training and validation accuracy may both be low.



!Underfitting Example



Example: Dataset containing 10 samples.



2\. Overfitting



Overfitting occurs when a model performs well on training data but generalizes poorly to unseen data.



!Overfitting Example



Example: Dataset containing 100 samples.



3\. Good Fit



A good fit generally combines satisfactory training and validation performance with a relatively small generalization gap.



!Good Fit Example



Example: Dataset containing 1,000 samples.





Methodology



FitCheck uses a Decision Tree classifier with varying maximum tree depths.



For each depth, it calculates:



Training Accuracy: Accuracy on the training dataset.

Validation Accuracy: Accuracy on the held-out validation dataset.

Generalization Gap: Training accuracy minus validation accuracy.



Configurable thresholds flag potential fitting problems. These diagnoses are heuristic indicators, not definitive proof of overfitting or underfitting.



Workflow

Dataset

&#x20;  |

&#x20;  v

Train / Validation Split

&#x20;  |

&#x20;  v

Decision Tree — Varying Depths

&#x20;  |

&#x20;  v

Training and Validation Accuracy

&#x20;  |

&#x20;  v

Generalization Gap

&#x20;  |

&#x20;  v

Fit Diagnosis

&#x20;  |

&#x20;  v

CSV + JSON + Diagnostic Plots



Project Structure

FitCheck/

├── fitcheck.py

├── README.md

├── requirements.txt

├── LICENSE

└── results/

&#x20;   ├── compare\_cases.png

&#x20;   ├── Goodfit\_n\_samples=1000.png

&#x20;   ├── Overfitting\_n\_samples=100.png

&#x20;   ├── Underfittingn\_samples=10.png

&#x20;   ├── statistics.csv

&#x20;   ├── summary.json

&#x20;   └── diagnostics.png



Installation



Requirements: Python 3.10 or later.



Install the dependencies:



pip install -r requirements.txt



Usage



Run the project from the root directory:



python fitcheck.py





Generated reports and diagnostic plots are saved in the results/ directory.



Output Files

File	Description

compare\_cases.png	Visual comparison of the three fitting scenarios

statistics.csv	Metrics for every tested tree depth

summary.json	Selected model and diagnostic summary

diagnostics.png	Training, validation, and generalization plots



The three example images illustrate separate dataset scenarios. Their results depend on the data, train-validation split, and model configuration used to generate them.



Limitations

Small datasets can produce unreliable validation estimates.

A single train-validation split may be unstable.

Threshold-based diagnoses are heuristic and may not identify every fitting problem.

Cross-validation and a separate test set are recommended for more reliable evaluation.

Author



Naeim Mousavi



GitHub: @nmousavi2020



License



This project is licensed under the MIT License. See the LICENSE file for details.

