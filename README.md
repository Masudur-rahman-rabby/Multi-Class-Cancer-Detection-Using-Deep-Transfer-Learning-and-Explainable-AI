# Multi-Cancer Detection and Classification using Deep Learning

This repository presents a deep learning–based framework for **automated multi-class cancer detection** from medical images. The system classifies **Blood Cancer, Lung Cancer, and Skin Cancer** using state-of-the-art convolutional neural networks with **transfer learning, rigorous evaluation, and explainable AI (Grad-CAM)**.

The primary goal is to achieve **high classification accuracy** while maintaining **clinical interpretability and reliability**.

---

## Key Features

- Multi-class cancer classification (3 classes)
- Transfer learning with **EfficientNetB0** (main model)
- Comparative analysis with **ResNet50**
- Two-stage training strategy (feature extraction + fine-tuning)
- Extensive evaluation metrics beyond accuracy
- **Grad-CAM explainability** for model interpretability
- Publication-ready visualization and analysis

---

##Models Used

### 1. EfficientNetB0 (Primary Model)
- Pretrained on ImageNet
- Custom classification head
- Fine-tuned last convolutional layers
- Strong generalization with fewer parameters

### 2. ResNet50 (Comparison Model)
- Residual learning architecture
- Used as a performance benchmark

---

## Dataset

The dataset contains medical images from three cancer categories:

- **Blood Cancer**
- **Lung Cancer**
- **Skin Cancer**

Images are resized to **224 × 224** and normalized.  
A **training–validation split (80:20)** is applied using Keras generators.

> Note: Dataset files are not included in this repository due to licensing constraints.

---

## Data Augmentation

To improve generalization and reduce overfitting:

- Rotation (±20°)
- Zoom (±20%)
- Horizontal flipping
- Pixel normalization

---

## Training Strategy

### Stage 1: Feature Extraction
- Backbone frozen
- Learning rate: `1e-3`
- Objective: Learn task-specific representations

### Stage 2: Fine-Tuning
- Last 30 layers unfrozen
- Learning rate reduced to `1e-4`
- Callbacks used:
  - EarlyStopping
  - ModelCheckpoint
  - ReduceLROnPlateau

---

## Evaluation Metrics

The model is evaluated using comprehensive metrics suitable for medical AI:

- Accuracy
- Precision
- Recall (Sensitivity)
- F1-score
- Specificity (per class)
- ROC–AUC (multi-class, OVR)
- Cohen’s Kappa
- Matthews Correlation Coefficient (MCC)
- Confusion Matrix (normalized)

---

## Final Validation Results (EfficientNetB0)

| Metric | Value |
|------|------|
Precision | 0.902  
Recall (Sensitivity) | 0.861  
F1-score | 0.855  
Cohen’s Kappa | 0.792  
MCC | 0.816  
ROC–AUC | 0.9999  
Validation Accuracy | 99.67%  

---

## Explainability with Grad-CAM

Grad-CAM is used to visualize the regions of medical images that influence model predictions.  
This ensures:

- Transparency in decision-making
- Clinical trustworthiness
- Detection based on meaningful anatomical features

Examples include:
- Lesion-focused regions for skin cancer
- Pathological lung areas for lung cancer
- Cellular patterns for blood cancer

---

## Model Comparison

| Model | Max Validation Accuracy |
|------|-------------------------|
EfficientNetB0 | 99.67% |
ResNet50 | 100% |

EfficientNetB0 is preferred due to better generalization and efficiency.

---

## Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- OpenCV
- Scikit-learn
- Matplotlib
- Seaborn

---

## Limitations

- Skin vs Lung cancer confusion due to cross-modality texture similarity
- Single-dataset evaluation
- No external validation dataset
- These limitations are discussed transparently and provide scope for future work.

---

## Future Work

- External dataset validation
- Class-specific preprocessing pipelines
- Vision Transformer (ViT) comparison
- Clinical deployment testing

---

## Disclaimer

This project is intended for research and academic purposes only and not for clinical diagnosis.

---

## License

This project is licensed under the MIT License.

---

## Author

Md. Masudur Rahman Rabby
Department of Computer Science and Engineering
