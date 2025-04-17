## Histopathologic Cancer Detection

This repository contains code and resources for the Kaggle competition:  
[Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection)

The goal is to classify 96×96 histopathologic images as containing metastatic cancer tissue (label = 1) or not (label = 0).

---

## Overview

- **Objective**  
  Classify whether histopathology images show evidence of cancer.

- **Key Approaches**  
  1. **Baseline CNN**: A simple convolutional neural network to validate the overall training pipeline.  
  2. **Transfer Learning (EfficientNetB0)**: Leverage a pretrained ImageNet model to improve performance.

---

## Setup and Environment

- **Kaggle Notebook (GPU)** or equivalent environment (e.g., Google Colab, local GPU).  
- **Libraries**:  
  - Python 3.7+  
  - TensorFlow 2.x  
  - NumPy, pandas, scikit-learn, matplotlib  
  - EfficientNet (under `tf.keras.applications`)

If you are using the Kaggle platform, you can attach the dataset by going to “Add Data” → “Histopathologic Cancer Detection”.

---

## Data

- **Provided by Kaggle**:
  - `train/`: ~220,000 images (96×96, RGB)
  - `test/`: ~57,000 images
  - `train_labels.csv`: mapping of `id` to label (0 or 1)
- Ensure these are placed or mounted at `/kaggle/input/histopathologic-cancer-detection/`.

---

## Methods

### 1. Baseline CNN

1. **Preprocessing**  
   Use `ImageDataGenerator(rescale=1./255)` to scale pixel values to [0,1].

2. **Architecture**  
   A small CNN with multiple Conv+Pool blocks, followed by dense layers.

3. **Training**  
   Fit for several epochs using `EarlyStopping` to avoid overfitting.

4. **Evaluation**  
   Compute ROC AUC on the validation split.

### 2. Transfer Learning (EfficientNetB0)

1. **Data Augmentation**  
   Random rotations, flips, brightness changes, etc.

2. **Pretrained Model**  
   Load `EfficientNetB0(include_top=False)` for feature extraction.

3. **Fine-Tuning**  
   Freeze the base model initially, then unfreeze later with a lower learning rate.

4. **Evaluation**  
   Track validation accuracy and ROC AUC; save the best weights (`best_efficient.keras`).

---

## Submission

1. **Predict on `test/`**  
   Use `model.predict()` on the test images to generate outputs.

2. **Create `submission.csv`**  
   Contain columns `id` and `label`.

3. **Submit on Kaggle**  
   Locate `submission.csv` in the Kaggle Notebook Files/Output section, then click “Submit to Competition” or download and upload it manually on the competition page.

---

## Results

- **Baseline CNN**  
  Often achieves a decent ROC AUC on validation (e.g., ~0.96), depending on parameters.

- **EfficientNetB0**  
  May yield higher accuracy, but can plateau if hyperparameters (e.g., learning rate) are not well-tuned.

---

## Future Improvements

- Fine-tune learning rate and batch size.
- Selectively freeze/unfreeze BatchNormalization layers.
- Explore alternative pretrained networks (ResNet50, InceptionV3, etc.).
- Try ensemble methods or TTA (Test-Time Augmentation).

---

## References

- [Kaggle Competition — Histopathologic Cancer Detection](https://www.kaggle.com/c/histopathologic-cancer-detection/overview)  
- [Keras Documentation](https://keras.io/)  
- Various public Kaggle kernels/notebooks (cited where relevant).

---

### Leaderboard Screenshot Example
<img width="1160" alt="kaggle_v2" src="https://github.com/user-attachments/assets/5c4fd4e0-ed0b-4f44-aad6-8fca767ea4f3" />
