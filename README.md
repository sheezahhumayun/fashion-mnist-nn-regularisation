# Building, Breaking and Fixing a Neural Network
Deep Learning for Perception — Assignment 1 (Fall 2026)

## Overview
This repository contains a single notebook implementing all 7 parts of the assignment on the
Fashion-MNIST dataset: backprop from scratch, activation study, loss function comparison,
optimiser comparison, forced overfitting, a regularisation study, and hyperparameter tuning
with k-fold cross-validation.

**Final test accuracy: 88.83%** (macro F1: 88.66%), a +5.53 percentage point improvement
over the untuned Part 2 baseline (83.30%).

## Reproducibility
- Random seed **42** is set at the start of every part for Python's `random`, NumPy, PyTorch,
  and TensorFlow (where used).
- All DataLoader shuffling uses a seeded `torch.Generator`.
- The test set is loaded once in the environment setup and is not touched or used for any
  decision-making until Part 7's final evaluation.

## How to Reproduce

### 1. Environment
- Platform: Google Colab (or Kaggle, with path adjustment — see note below)
- Accelerator: GPU (T4 recommended)
- Python packages required: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`,
  `torch`, `torchvision`, `tensorflow` (TensorFlow is only used for its seeding utility in
  the environment setup cell; all modeling is done in PyTorch).

### 2. Dataset
Dataset: [Fashion-MNIST on Kaggle](https://www.kaggle.com/datasets/zalando-research/fashionmnist)

Download `fashion-mnist_train.csv` and `fashion-mnist_test.csv` and place them in your
Google Drive (or Kaggle input directory). Update the paths in the environment setup cell:

```python
train_df = pd.read_csv("/content/drive/MyDrive/.../fashion-mnist_train.csv")
test_df  = pd.read_csv("/content/drive/MyDrive/.../fashion-mnist_test.csv")
```

Mount Google Drive first if running in Colab:
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 3. Run Order
Run all cells top to bottom, in this order:
1. **Environment Setup** — load data, normalise, flatten, 80/20 train/val split, class
   distribution report.
2. **Part 1** — NumPy MLP from scratch (784→64→10), gradient check against PyTorch.
3. **Part 2** — 2-hidden-layer network, activation comparison (sigmoid/tanh/ReLU/leaky ReLU).
4. **Part 3** — Cross-entropy vs MSE, plus California Housing regression (loaded via
   `sklearn.datasets.fetch_california_housing`, no download needed).
5. **Part 4** — Optimiser comparison (SGD, SGD+momentum, RMSProp, Adam), shared LR then
   tuned LR per optimiser.
6. **Part 5** — Forced overfitting on a 2,000-sample subset with a 4×512 network.
7. **Part 6** — Regularisation study (L2, L1, dropout, batch norm, early stopping,
   augmentation, more data) starting from the Part 5 setup.
8. **Part 7** — Random search (12 configs) + 5-fold CV, final retrain on the full training
   set with the selected configuration and best Part 6 regularisation choices, single
   evaluation on the held-out test set.

Each part depends on variables created in the environment setup cell (`x_train`, `y_train`,
`x_val`, `y_val`, `x_test`, `y_test`, `SEED`) and, where noted, on the reduced-sample subset
built in Part 5 (`x_train_small`, `y_train_small`). Run cells in order within a single
kernel session.


### 4. Key Results Reference
See `results_summary.docx` for the one-page summary of the final configuration, test score,
and the single change that most improved performance (widening the network to 512 units
combined with dropout regularisation, informed by the Part 6 study).

## Repository Contents
- `DL_ASS01_23F-0665_23F-0701.ipynb` — full notebook, all 7 parts, executed with visible outputs.
- `results_summary.docx` — one-page results summary (final score, configuration, key change).
- `README.md` — this file.
