# Comprehensive Project Report: Image Classification with Transfer Learning

**Project Title**: Image Classifier with Transfer Learning  
**Author**: Sadiraja  
**Repository**: [Image-Classifier-with-Transfer-Learning](https://github.com/Sadiraja/Image-Classifier-with-Transfer-Learning)  
**Date**: October 2026  
**Framework**: PyTorch 2.x, Torchvision, Streamlit, Scikit-Learn  

---

## 1. Executive Summary

Deep learning for image recognition typically requires millions of annotated images to train convolutional neural networks from scratch. In real-world domains with limited data, **Transfer Learning** represents the state-of-the-art methodology. By leveraging representations previously learned by deep networks on massive benchmarks such as ImageNet-1K (comprising 1.28 million images across 1,000 object categories), we can transfer foundational spatial features (edges, textures, geometric patterns, motifs) and achieve top-tier classification performance on small domain-specific datasets in fractions of the training time.

In this work, we developed, trained, evaluated, and deployed an end-to-end deep learning classifier for five distinct natural flower species (**Daisy**, **Dandelion**, **Roses**, **Sunflowers**, and **Tulips**). Using an ImageNet-pretrained **MobileNetV2** backbone with a two-stage training strategy (feature extraction followed by selective fine-tuning), our model achieved **92.09% test accuracy** and a **weighted F1-score of 0.9206** across 556 completely unseen test images. We deployed this model with both a command-line interface (`src/predict.py`) and an interactive web application built with **Streamlit** (`app.py`).

---

## 2. Dataset Description & Preprocessing

### 2.1 Source and Composition
The dataset is derived from the **Google / TensorFlow Flower Photos Benchmark** ([Dataset Source](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz)), comprising natural photographs collected across indoor and outdoor environments under diverse illumination, viewpoints, resolutions, and backgrounds.

| Class Name | Total Samples | Train (70%) | Validation (15%) | Test (15%) |
| :--- | :---: | :---: | :---: | :---: |
| **Daisy** | 633 | 443 | 94 | 96 |
| **Dandelion** | 898 | 628 | 134 | 136 |
| **Roses** | 641 | 448 | 96 | 97 |
| **Sunflowers** | 699 | 489 | 104 | 106 |
| **Tulips** | 799 | 559 | 119 | 121 |
| **Total** | **3,670** | **2,567** | **547** | **556** |

### 2.2 Stratified Splitting
To ensure zero data leakage and guarantee that evaluation strictly measures generalizability:
- Data was divided into 70% training, 15% validation, and 15% testing splits using a fixed pseudorandom seed (`seed=42`).
- The validation set was exclusively utilized for hyperparameter tracking, model checkpointing, and early stopping decisions.
- The test set was held out and only evaluated once after the final model was checkpointed.

### 2.3 Data Augmentation
To regularize the model and prevent overfitting on classes with subtle variations, we implemented real-time stochastic data augmentations for the training split:
- **`RandomResizedCrop(224, scale=(0.8, 1.0))`**: Forces spatial invariance against scale variations and partial occlusions.
- **`RandomHorizontalFlip(p=0.5)`**: Enhances left-right mirror symmetry invariance.
- **`RandomRotation(degrees=15)`**: Prevents orientation bias.
- **`ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`**: Simulates variable outdoor lighting and camera color calibration shifts.
- **ImageNet Normalization**: Mean $\mu = [0.485, 0.456, 0.406]$ and standard deviation $\sigma = [0.229, 0.224, 0.225]$.

Validation and test pipelines use deterministic transforms: `Resize(256)`, `CenterCrop(224)`, `ToTensor()`, and ImageNet normalization.

---

## 3. Model Architecture & Transfer Learning Strategy

### 3.1 Network Backbone: MobileNetV2
We selected **MobileNetV2** as the core feature extractor due to its combination of high computational efficiency, small memory footprint (~10.5 MB weights), and strong ImageNet representation. Its architecture utilizes inverted residual blocks with linear bottlenecks and depthwise separable convolutions:
- **Base Feature Dimension**: 1,280 channels output from the final convolutional pooling stage.
- **Total Parameters**: ~3.5 Million parameters.

### 3.2 Custom Classification Head
The original 1,000-class linear projection head was stripped and replaced with a regularized two-stage multilayer perceptron (MLP):
```
MobileNetV2 Features (Frozen / Fine-tuned)
     ↓
Dropout (p = 0.30)
     ↓
Linear (in_features = 1280, out_features = 256)
     ↓
ReLU (inplace = True)
     ↓
Dropout (p = 0.15)
     ↓
Linear (in_features = 256, out_features = 5)
```

### 3.3 Two-Stage Training Regimen

1. **Stage 1: Feature Extraction / Head Warmup (Epochs 1–4)**
   - All parameters in `model.features` were frozen (`requires_grad = False`).
   - Only the custom classification head was updated.
   - Optimizer: `AdamW` ($lr = 1 \times 10^{-3}$, weight decay $= 1 \times 10^{-4}$).
   - Loss Function: Cross-Entropy Loss.
   - Purpose: Establish stable gradient representations for the newly initialized head weights without destroying pretrained feature weights.
   - Outcome: Validation accuracy rapidly climbed from 86.84% in Epoch 1 to 87.93% in Epoch 2.

2. **Stage 2: Differential Fine-Tuning (Epochs 5–7)**
   - The top inverted residual blocks (stages 15–18) were unfrozen (`requires_grad = True`).
   - Optimizer: `AdamW` with a reduced learning rate ($lr = 1 \times 10^{-4}$, weight decay $= 1 \times 10^{-4}$).
   - Scheduler: `CosineAnnealingLR` decaying $lr$ smoothly down to $1 \times 10^{-6}$.
   - Purpose: Adapt high-level semantic feature detectors to specific petal textures, petal count patterns, and botanical structures.
   - Outcome: Validation accuracy reached **90.49%**, and training accuracy reached **94.04%**.

---

## 4. Quantitative Results & Evaluation

### 4.1 Test Split Performance Summary
The model was tested on 556 unseen images from the held-out test split.

| Metric | Score |
| :--- | :---: |
| **Test Accuracy** | **92.09%** |
| **Test Loss (Cross-Entropy)** | **0.2264** |
| **Macro Average Precision** | **91.88%** |
| **Macro Average Recall** | **92.09%** |
| **Macro Average F1-Score** | **91.92%** |
| **Weighted Average F1-Score** | **92.06%** |

### 4.2 Per-Class Classification Report

| Class | Precision | Recall | F1-Score | Support (Test Images) |
| :--- | :---: | :---: | :---: | :---: |
| **Daisy** | 0.8774 | **0.9688** | 0.9208 | 96 |
| **Dandelion** | **0.9485** | 0.9485 | **0.9485** | 136 |
| **Roses** | 0.9043 | 0.8763 | 0.8901 | 97 |
| **Sunflowers** | 0.9346 | 0.9434 | 0.9390 | 106 |
| **Tulips** | 0.9292 | 0.8678 | 0.8974 | 121 |

### 4.3 Confusion Matrix Analysis
The confusion matrix demonstrates strong diagonal dominance across all five categories:
- **Daisy (93/96, 96.9% recall)**: Exceptional classification performance; minor confusion with dandelion (2 samples) when white ray florets were clustered tightly.
- **Dandelion (129/136, 94.9% recall)**: High precision and recall; minor confusion with daisy (4 samples) due to yellow center disks.
- **Roses (85/97, 87.6% recall)**: The primary confusion was with **Tulips** (7 samples). Both species share overlapping crimson/red color profiles and clustered petal geometries from specific camera angles.
- **Sunflowers (100/106, 94.3% recall)**: Strong performance; easily separated from other classes due to characteristic dark central disc florets and broad yellow petals.
- **Tulips (105/121, 86.8% recall)**: 8 samples were predicted as roses, mirroring the rose-tulip visual similarity.

The confusion matrix artifact is archived at `reports/confusion_matrix.png`, and the learning curve artifact is at `reports/training_curves.png`.

---

## 5. Software Architecture & Interfaces

The codebase is engineered modularly for reproducibility, deployment, and testing:

```
├── models/
│   ├── classifier.py         # Factory for MobileNetV2 / ResNet18, checkpoint I/O
│   └── best_model.pth        # Saved PyTorch checkpoint weights (10.45 MB)
├── src/
│   ├── dataset.py            # Automated download, splitting, augmentation & DataLoaders
│   ├── train.py              # Two-stage training pipeline with cosine annealing
│   ├── evaluate.py           # Test evaluation, metrics computation & confusion matrix
│   └── predict.py            # CLI inference script
├── notebooks/
│   └── transfer_learning_classifier.ipynb # Full interactive pipeline notebook
├── reports/
│   ├── report.md             # This comprehensive technical report
│   ├── evaluation_results.json
│   ├── training_history.json
│   ├── confusion_matrix.png
│   └── training_curves.png
├── sample_images/            # 5 curated test samples (daisy, dandelion, roses, sunflowers, tulips)
├── app.py                    # Production Streamlit web application
├── requirements.txt          # Pinned dependency requirements
└── README.md                 # Complete documentation and quickstart guide
```

### 5.1 CLI Prediction Tool (`src/predict.py`)
Predicts class probabilities for any target image:
```bash
python src/predict.py sample_images/roses.jpg
```
Output:
```
Analyzing image: sample_images/roses.jpg

Predictions:
-----------------------------------
 1. Roses        : 100.00%
 2. Tulips       :   0.00%
 3. Daisy        :   0.00%
 4. Dandelion    :   0.00%
 5. Sunflowers   :   0.00%
-----------------------------------
```

### 5.2 Streamlit Web Application (`app.py`)
An interactive dashboard with:
- **Interactive Inference**: Drag-and-drop file uploader or 1-click preset sample selector.
- **Confidence Visualization**: Probability bar charts rendered across all 5 flower classes.
- **Metrics Explorer**: Displays normalized confusion matrix, training curves, and per-class classification report.
- **Methodology Documentation**: In-app documentation covering dataset origin, augmentation, and model limits.

---

## 6. Limitations & Future Directions

1. **Inter-Class Floral Resemblance**:
   - Red roses and red closed tulips share close chromatic and volumetric features, causing mutual misclassifications (~7–8% error between these two classes).
   - *Future Work*: Applying attention mechanisms (such as Vision Transformers - ViT) or fine-grained botanical part localization could differentiate petal edges and stem structures.

2. **Multi-Flower and Dense Scenes**:
   - The dataset consists primarily of single-subject or dominant-subject photos. Images featuring multiple blooming species simultaneously will produce an ambiguous single-label classification.
   - *Future Work*: Extending the model to multi-label classification or integrating bounding box detection (YOLOv8 / Faster-RCNN).

3. **Out-of-Distribution (OOD) Vulnerability**:
   - The softmax activation forces the sum of probabilities to 1.0. If an arbitrary object (e.g., an automobile or household item) is inputted, the network will predict one of the five flowers with false confidence.
   - *Future Work*: Implement temperature scaling and an energy-based OOD rejection threshold (e.g., rejecting predictions if max softmax probability $< 0.65$ or if reconstruction error from an autoencoder exceeds a threshold).

---

## 7. Conclusion

By employing a two-stage transfer learning strategy on MobileNetV2 with data augmentation, we established a fast, lightweight, and robust flower classifier achieving **92.09% test accuracy**. The complete project, trained weights (`10.45 MB`), Jupyter notebook, modular source code, evaluation metrics, and Streamlit frontend are fully packaged and ready for immediate deployment.
