# ENG2440 Assignment 3: Molecular Property Prediction with Graph Neural Networks

**Student:** Nasir Adem Degu  
**Course:** ENG 2440 Medical Imaging & AI in Healthcare

## Overview

This repository contains the implementation for Assignment 3: Molecular Property Prediction using the Blood–Brain Barrier Penetration (BBBP) dataset from MoleculeNet.

The project compares two approaches for predicting whether a molecule can penetrate the blood–brain barrier:

1. **Fingerprint baseline:** Morgan fingerprints (ECFP4) with logistic regression.
2. **Graph neural network (GNN):** A bond-aware molecular graph classifier using GINEConv message passing.

Each molecule is represented by a SMILES string and a binary label, where 1 indicates BBB penetration and 0 indicates non-penetration.

The notebook covers dataset inspection, SMILES canonicalization, duplicate and conflicting-label handling, molecular visualization, categorical atom and bond feature encoding, PyTorch Geometric batching, stratified random and Bemis–Murcko scaffold splitting, molecular fingerprint generation, logistic regression hyperparameter selection, GNN architecture and training, early stopping, multi-seed evaluation, threshold selection, test-set comparison, and molecular error analysis.

The analysis also investigates whether Morgan-fingerprint Tanimoto similarity to training molecules helps explain selected false-positive and false-negative predictions.

## Repository contents

```text
.
├── assignment3.ipynb
├── requirements.txt
├── ai_use_declaration.txt
├── README.md
└── .gitignore
```

- `assignment3.ipynb`: fully executed assignment notebook containing the implementation, results, figures, and analysis.
- `requirements.txt`: pinned Python dependencies required to run the notebook.
- `ai_use_declaration.txt`: declaration of generative-AI assistance used during the assignment.
- `README.md`: project overview, setup, execution, and reproducibility instructions.
- `.gitignore`: excludes downloaded datasets, local virtual environments, Jupyter checkpoints, and other generated files from Git.

## Dataset

The project uses the **Blood–Brain Barrier Penetration (BBBP)** dataset from MoleculeNet. It contains molecular structures represented by SMILES strings and binary labels indicating whether molecules penetrate the blood–brain barrier.

The notebook downloads and processes the dataset automatically through PyTorch Geometric:

```python
DATA_ROOT = Path("data/MoleculeNet")

dataset = MoleculeNet(
    root=str(DATA_ROOT),
    name="BBBP"
)
```

The original dataset contains 2,050 records. PyTorch Geometric retains 2,039 molecular graphs after processing.

The notebook then canonicalizes SMILES strings, identifies duplicate molecules, removes all records from groups with conflicting labels, and retains one record per consistently labelled molecule.

After cleaning, the dataset contains **1,965 unique molecules**.

The downloaded dataset is not included in the repository. PyTorch Geometric stores it locally under:

```text
data/MoleculeNet/
```

No separate dataset download or extraction step is required.

## Setup and execution

The submitted notebook was executed using **Python 3.11**.

Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
```

On Windows, activate it using:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Then open `assignment3.ipynb` in Jupyter, JupyterLab, VS Code, or another Jupyter-compatible environment.

Select the Python environment in which the dependencies were installed as the notebook kernel, restart the kernel, and run all cells sequentially from top to bottom.

The notebook is designed to be executed in order. Later sections depend on the cleaned dataset, split indices, models, and evaluation results produced in earlier sections.

The dataset is downloaded automatically on the first run. Subsequent runs reuse the locally cached dataset.

The implementation trains the models on the CPU. A CUDA-capable GPU is not required.

The trained GNN checkpoints are retained in memory during notebook execution rather than saved as separate model files.

## Dependencies

The Python dependencies are pinned in `requirements.txt`:

```text
torch==2.15.0
torch-geometric==2.8.1
rdkit==2026.9.1
scikit-learn==1.9.1
numpy==2.4.6
pandas==3.1.1
matplotlib==3.11.2
networkx==3.6.1
jupyter==1.1.1
ipykernel==7.4.0
```

Python standard-library modules such as `pathlib`, `collections`, and `textwrap` do not require separate installation.

The installed environment was checked using `python -m pip check`, which reported no broken requirements.

## Reproducibility

The submitted notebook is fully executed and contains the outputs used in the assignment analysis.

A fixed random seed of **42** is used to construct two separate train, validation, and test splitting strategies:

1. **Stratified random split:** preserves the overall class proportions across subsets.
2. **Bemis–Murcko scaffold split:** groups molecules by molecular scaffold so that molecules belonging to the same scaffold group remain in one subset. Molecules without a Bemis–Murcko scaffold are assigned to a shared acyclic group.

Both strategies use approximately 70% of the cleaned molecules for training, 15% for validation, and 15% for testing:

```text
Training:    1,375 molecules
Validation:    295 molecules
Test:          295 molecules
```

The split indices are checked for complete coverage, disjointness, and molecular duplication. Scaffold separation is also verified for the scaffold split.

### Fingerprint baseline

The fingerprint baseline uses:

```text
Fingerprint:          Morgan (ECFP4)
Radius:               2
Length:               2,048 bits
Classifier:           Logistic regression
Class weighting:      Balanced
Maximum iterations:   2,000
C candidates:         0.01, 0.1, 1, 10, 100
Selected C:           0.1 for both splits
Selection metric:     Validation ROC-AUC
```

Decision thresholds are selected separately for each split using validation macro-F1, with candidate thresholds from 0.10 to 0.90 in steps of 0.05.

The selected thresholds are 0.30 for the stratified random split and 0.40 for the scaffold split.

### Graph neural network

The GNN uses trainable embeddings for nine categorical atom features and three categorical bond features. Its architecture consists of:

- Two bond-aware GINEConv message-passing layers.
- 32-dimensional hidden representations.
- Global mean pooling to obtain one representation per molecule.
- A linear binary-classification head.

The network contains **10,881 trainable parameters**.

Training uses:

```text
Loss:                 BCEWithLogitsLoss
Class weighting:      Positive-class weight calculated from training labels
Optimizer:            Adam
Learning rate:        0.001
Batch size:           32
Maximum epochs:       100
Early stopping:       15 epochs without validation ROC-AUC improvement
Training seeds:       42, 43, 44
```

Each seed is used to initialize and train a separate model for each splitting strategy. The checkpoint with the highest validation ROC-AUC is retained for each training run.

The notebook reports validation ROC-AUC mean and sample standard deviation across the three seeds.

For final comparison, the best GNN run is selected using validation ROC-AUC. Its classification threshold is selected using validation macro-F1.

The test sets are reserved for final evaluation and are not used for training, hyperparameter selection, checkpoint selection, or threshold selection.

Exact numerical reproduction across different machines, operating systems, and software environments is not guaranteed.

## Evaluation

Both the fingerprint baseline and GNN are evaluated under the stratified random and Bemis–Murcko scaffold splitting strategies.

The notebook reports six test metrics:

- ROC-AUC
- PR-AUC (average precision)
- Accuracy
- Sensitivity
- Specificity
- F1 score

It also produces ROC curves, precision–recall curves, confusion matrices, and tables comparing the models under both splitting strategies.

For the GNN, results from all three training seeds are reported alongside their mean and standard deviation. The separately identified validation-selected GNN is used for the main model comparison.

The final section examines two false-positive and two false-negative molecules from the scaffold test set. For each error, the notebook identifies the most similar molecule in the scaffold training subset using Morgan-fingerprint Tanimoto similarity and displays the molecular structures, true labels, predicted labels, and predicted probabilities.

## Data and clinical-use notice

The BBBP labels describe observed blood–brain barrier penetration outcomes within the source dataset. These labels may contain experimental uncertainty or inconsistencies, and the dataset does not represent the full range of molecular structures encountered in drug discovery.

The models learn statistical associations between molecular structure and BBB penetration labels. Their predictions do not establish causal effects or demonstrate that a molecule is safe, effective, or suitable for treatment.

This project was developed for coursework and is **not a clinical diagnostic, drug approval, or medical decision-support system**. Independent experimental validation would be required before using these predictions in real-world drug development or clinical decisions.