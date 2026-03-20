# Acoustic Scene Classification
**Deep Learning for Audio Processing**

---

This repository documents a systematic investigation of deep learning architectures for acoustic scene classification (ASC), structured around a single guiding question: *how does performance change as the model's inductive bias is increasingly aligned with the structure of the problem?*

Rather than jumping straight to the strongest model, we take an evolutionary approach, starting from a simple MLP baseline and progressively introducing task-aligned structural assumptions.
Each step constrains the hypothesis space further, but in a direction that matches the data: locality and translation invariance via ConvNets, long-range temporal modelling via a TCN extension, and multi-perspective representations via a dual-stream ensemble.
The result is a controlled comparison across model families that makes the source of each performance gain explicit.

The project is oriented on Task 1 of the [DCASE Challenge](https://dcase.community/) and evaluated on the TUT Acoustic Scenes 2016–2018 development datasets.

[Report](fibonacci_DLAP-report.pdf) · [Slides](dlap_inductive-bias-slides.pdf)

---

## Results

The best model achieved in this work, the **mid/side 2-ConvNet**, reaches **84.8% validation accuracy**, an 18.7 percentage point improvement over the MLP baseline.

| Model | Input | Val. Accuracy |
|---|---|---|
| Random Forest | Mono | 65.7% |
| Baseline MLP (~#54 in DCASE 2017 Challenge, Task 1) | Mono | 66.1% |
| PCA + SVC | Mono | 66.7% |
| Extended MLP | Mono | 71.5% |
| PCA + SVC & Random Forest | Mono | 72.6% |
| VGGNet-style ConvNet | Mono | 79.6% |
| Stereo 2-ConvNet | Stereo | 79.6% |
| ConvNet–TCN Hybrid | Mono | 80.4% |
| HPSS 2-ConvNet | Harmonic/Percussive | 80.9% |
| Multi-stream ConvNet Ensemble | Combined | 83.3% |
| HLP17 1-ConvNet (reference) | Mono | 84.4% |
| **Mid/side 2-ConvNet** | **Mid/Side** | **84.8%** |
| HLP17 Full Ensemble (reference, #2 in DCASE 2017 Challenge, Task 1) | Combined | 91.7% |

Performance improvements followed a consistent pattern: models that introduced stronger task-aligned inductive biases yielded the most substantial gains.
The largest single jump comes from switching to convolutional architectures; representational alignment via mid/side decomposition then pushes further without adding architectural complexity.
The mid/side 2-ConvNet would rank approximately 6th among DCASE 2017 Task 1 submissions; the baseline places around 54th.

---

## Framing

The experiments are structured as a progressive narrowing of the hypothesis space:

```
MLP → ConvNet → ConvNet–TCN → Ensemble 2-ConvNet
 ↓         ↓           ↓                ↓
Universal  Locality +  + Long-range      + Multi-perspective
approx.    transl. inv.  temporal struct.   representations
```

This reflects a core theme of the work: architectural choices are, at their root, hypotheses about the data.
Encoding the right structural assumptions proved to be the primary driver of performance in these experiments.

---

## Setup

```bash
git clone https://github.com/osperaja/dlap_acoustic-scene-classification.git
cd dlap_acoustic-scene-classification
python -m venv venv
source venv/bin/activate
pip install -e .
```

The project uses [PyTorch Lightning](https://lightning.ai/) throughout.
All models are configured via YAML files and share a common training pipeline with shuffled data loaders, min-epoch gated early stopping (after epoch 100), and best-checkpoint saving based on validation accuracy.

---

## Training

### PyTorch Models

```bash
python3 dcase/src/train.py --config dcase/src/config/baseline.yaml
python3 dcase/src/train.py --config dcase/src/config/linseq.yaml
python3 dcase/src/train.py --config dcase/src/config/cnn.yaml
python3 dcase/src/train.py --config dcase/src/config/cnntcn.yaml
python3 dcase/src/train.py --config dcase/src/config/cnn_ensemble.yaml
python3 dcase/src/train.py --config dcase/src/config/dccnn_hpss.yaml
python3 dcase/src/train.py --config dcase/src/config/dccnn_ms.yaml
python3 dcase/src/train.py --config dcase/src/config/dccnn_stereo.yaml
```

### Sklearn Models

```bash
python3 dcase/src/train_sklearn.py --config dcase/src/config/random_forest.yaml
python3 dcase/src/train_sklearn.py --config dcase/src/config/pca_svm.yaml
python3 dcase/src/train_sklearn.py --config dcase/src/config/sklearn_ensemble.yaml
```

---

## Models

| Model Class | Config | Description |
|---|---|---|
| `BaselineModel` | `baseline.yaml` | DCASE baseline MLP, 5-frame context (200-dim input) |
| `LinSeqModel` | `linseq.yaml` | Extended MLP with optional SpecAugment |
| `CNNModel` | `cnn.yaml` | VGGNet-style ConvNet with optional SpecAugment and Mixup |
| `CNNTCNModel` | `cnntcn.yaml` | ConvNet–TCN hybrid with dilated temporal modelling |
| `DualChannelCNNModel` | `dccnn_hpss.yaml` | 2-ConvNet on harmonic/percussive separation |
| `DualChannelCNNModel` | `dccnn_ms.yaml` | 2-ConvNet on mid/side channel decomposition |
| `DualChannelCNNModel` | `dccnn_stereo.yaml` | 2-ConvNet on raw stereo channels |
| `EnsembleCNNModel` | `cnn_ensemble.yaml` | Multi-stream ensemble over pre-trained ConvNets |
| `SklearnAudioClassifier` | `random_forest.yaml` | Random Forest on mel spectrogram statistics |
| `SklearnAudioClassifier` | `pca_svm.yaml` | PCA + SVM on mel spectrogram statistics |
| `SklearnAudioEnsembleClassifier` | `sklearn_ensemble.yaml` | Hard-voting ensemble of Random Forest and PCA+SVM |

---

## Baseline System

The DCASE baseline is a lightweight MLP that processes log-mel spectrogram features with a 5-frame temporal context window.
It functions as the performance floor against which all other architectures are compared; and as a concrete illustration of what universal function approximation looks like without any structural prior.

**Feature pipeline:**
- Short-Time Fourier Transform → log-mel spectrogram (40 mel bands, 0–22,050 Hz)
- 40 ms frames with 50% overlap
- 5-frame context window → 200-dimensional input vector per frame

**Architecture:**
- 2 hidden layers × 50 units, ReLU activations, 20% dropout
- Output: 15-class softmax
- Optimiser: Adam (lr = 0.001), up to 200 epochs, early stopping after epoch 100 (patience 10)
- Inference: frame-level predictions aggregated by majority vote

A correct baseline implementation should achieve **> 60% validation accuracy** on the DCASE 2017 development set.

---

## Dataset

Experiments use the development portion of the TUT Acoustic Scenes 2016–2018 datasets, covering 15 distinct acoustic scenes.

- **Train:** 4,680 clips × 10 s
- **Validation:** 540 clips × 10 s
- **Total:** ~14.5 hours of audio, stereo, 44,100 Hz
- **Classes:** beach, bus, café/restaurant, car, city center, forest path, grocery store, home, library, metro station, office, park, residential area, train, tram

---

## Attribution

The baseline framework and course infrastructure were provided by the **Signal Processing Group, University of Hamburg (UHH)** as part of the course *Deep Learning for Audio Processing*.

All architectural developments beyond the provided baseline—the extended MLP, VGGNet-style ConvNet, ConvNet–TCN hybrid, multi-stream ensemble, mid/side decomposition, and pipeline extensions—were designed and implemented independently.

---

## Reproducibility

All source code, configurations, and preprocessing scripts are included in the repository.
Note that reported validation accuracies are single-run point estimates; future work should include repeated runs across multiple seeds and k-fold cross-validation for rigorous statistical comparison.

---

## Acknowledgement

I would like to express sincere gratitude to a researcher whose detailed and thoughtful feedback, in regard to the spanning overarching narrative structure, scientific storytelling, and granular prose decisions, was instrumental in shaping the final quality of this report.
