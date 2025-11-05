#  Pneumonia Detection using Ensemble Deep Learning  
**DenseNet121 • ResNet50 • MobileNetV2 | Transfer Learning + Segmentation + Explainability**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1y2OqGRk8fk3HS3gVXBni3IIkLg-hJwk7?usp=sharing)  
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)]()  
[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]()  

---

##  Overview
This repository contains my deep learning project for automated pneumonia detection from chest X-ray images using an ensemble of pretrained convolutional neural networks (CNNs).  
The ensemble combines **DenseNet121**, **ResNet50**, and **MobileNetV2** — each trained via transfer learning and integrated through average voting.  

The system achieves high accuracy and interpretability through segmentation-based preprocessing and Grad-CAM heatmaps, offering a reproducible pipeline for academic and practical use.

---

##  Project Motivation
Pneumonia is a major public health concern, especially in regions with limited access to radiologists. X-rays remain the most common diagnostic tool, but expert interpretation is time-consuming.  
This project aims to explore how pretrained deep models can automate pneumonia detection while maintaining transparency and reproducibility.  

It also helped me understand core concepts in transfer learning, ensemble voting, and model explainability.

---

## 📊 Dataset
**Source:** [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

**Structure:**
'''chest_xray/
├── train/
│ ├── NORMAL/
│ └── PNEUMONIA/
├── val/
│ ├── NORMAL/
│ └── PNEUMONIA/
└── test/
├── NORMAL/
└── PNEUMONIA/'''


**Dataset Details:**
- Total images: **5,863**
- Train: **4,187**  
- Validation: **1,046**
- Test: **630**

Images were resized to **224×224** and normalized between 0 and 1.  
A light **lung segmentation mask** (U-Net based) was applied to highlight lung regions and reduce background noise.  

> *Segmentation improved consistency in Grad-CAMs and slightly improved validation accuracy.*

---

## ⚙️ Model Architecture & Training

### 🔹 Base Networks
| Model | Parameters | Description |
|:--|:--:|:--|
| DenseNet121 | ~8M | Deep, feature-rich architecture |
| ResNet50 | ~25M | Residual blocks for gradient stability |
| MobileNetV2 | ~3.5M | Lightweight and mobile-efficient |

Each model was pretrained on **ImageNet** and adapted for binary classification via:

###  Transfer Learning & Unfreezing
Initially, all base convolutional layers were **frozen** to retain ImageNet features.  
I later explored **partial unfreezing** (only a few top layers) to understand how fine-tuning influences learning — although the final submitted version used mostly frozen layers due to limited training time and compute resources.  

This experimentation helped me learn how unfreezing affects convergence speed and overfitting.

###  Training Setup
| Setting | Value |
|:--|:--|
| Optimizer | Adam (`lr=1e-4`) |
| Loss | Binary Crossentropy |
| Batch Size | 16 |
| Epochs | **10 (early stopping applied)** |
| Validation Split | 20% |
| Metrics | Accuracy, Precision, Recall, F1-score, ROC-AUC |
| Framework | TensorFlow 2.12.0 / Keras |

Training used **class weights** to address imbalance between Normal and Pneumonia samples.

---

## 🧮 Ensemble Voting Mechanism
Each model outputs a probability `p` for Pneumonia.  
The ensemble combines them using simple averaging:

\[
p_{ensemble} = \frac{1}{3}(p_{DenseNet} + p_{ResNet} + p_{MobileNet})
\]

Final label:
- Pneumonia if \( p_{ensemble} ≥ 0.55 \)
- Normal otherwise

I also experimented with **weighted averaging** based on validation ROC-AUCs, which slightly improved stability but not enough to replace equal averaging for simplicity.

---

##  Explainability with Grad-CAM
Grad-CAM (Gradient-weighted Class Activation Mapping) was used to visualize which lung regions influenced the model's decision.  
This provided insights into model interpretability and helped verify that predictions were based on medical regions rather than spurious edges.

>  *Insert Grad-CAM overlays here (from DenseNet, ResNet, MobileNet, Ensemble)*  

In most cases, the ensemble heatmaps aligned with pulmonary lobes in pneumonia-positive X-rays.

---

## 📈 Results

| Metric | DenseNet121 | ResNet50 | MobileNetV2 | **Ensemble** |
|:--|:--:|:--:|:--:|:--:|
| Accuracy | 0.874 | 0.861 | 0.845 | **0.888** |
| Precision | 0.95 | 0.93 | 0.91 | **0.95** |
| Recall | 0.86 | 0.84 | 0.81 | **0.87** |
| F1-Score | 0.90 | 0.88 | 0.86 | **0.90** |
| ROC-AUC | 0.941 | 0.937 | 0.918 | **0.948** |

**Observations:**
- DenseNet121 provided the strongest baseline.
- The ensemble consistently improved overall ROC-AUC by ~0.7–1%.
- Segmentation added minor but stable improvements in recall.

---

## 🔍 Analysis and Cross-Validation
I performed **5-Fold Stratified Cross-Validation** for robustness.  
Mean ROC-AUC: **0.951 ± 0.007**

**Ablation insights:**
- Without segmentation → ~2% drop in accuracy.
- Using a single model → slightly less generalization.
- Ensemble improved recall and reduced false negatives.

---

##  Repository Structure
'''
├── weights/
│ ├── DenseNet121_saved.keras
│ ├── MobileNetV2_saved.keras
│ └── ResNet50_saved.keras
├── notebooks/
│ └── Pneumonia_Colab_Demo.ipynb
├── requirements.txt
├── Dockerfile
└── README.md'''
##  How to Run

### 🔹 Run in Colab (Recommended)  
The Colab notebook automatically:
- Installs dependencies  
- Downloads pretrained weights from Google Drive  
- Extracts test data  
- Loads models and runs predictions  
- Displays Grad-CAM visualizations  

No manual setup required.
## Key Learnings
- This project helped me understand:
- How transfer learning works on medical datasets
- How to combine pretrained CNNs through ensemble voting
- The importance of data preprocessing and segmentation
- How unfreezing layers affects model fine-tuning and overfitting
- How to interpret predictions using Grad-CAM

## Future Work
- Evaluate on external datasets (CheXpert, MIMIC-CXR)
- Add uncertainty quantification using Monte-Carlo dropout
- Explore hybrid CNN-Transformer models



