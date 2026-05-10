# Skin-Cancer-MNIST-HAM10000
# Skin Cancer Classification System

A deep learning project that classifies dermatoscopic skin lesion images into **7 diagnostic categories** using the **HAM10000 dataset**, with an interactive **Streamlit web application** for inference and prediction visualization.

---

# Overview

This project implements a complete deep learning pipeline for **skin lesion classification** using a custom **Convolutional Neural Network (CNN)**. The system assists in identifying potentially malignant skin lesions by learning visual patterns from dermatoscopic images.

The project covers:

- Data loading and preprocessing
- Exploratory Data Analysis (EDA)
- Data augmentation
- CNN model development
- Model training and evaluation
- Explainability using Grad-CAM
- Baseline ML model comparison
- Deployment with Streamlit

---

# Dataset

Dataset used: **HAM10000 (Human Against Machine with 10000 training images)**

Kaggle Dataset:  
https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000

The project uses the **28×28 RGB CSV version** of the dataset.

---

# Lesion Classes

| Code | Full Name | Nature |
|---|---|---|
| `akiec` | Actinic Keratoses | Pre-malignant |
| `bcc` | Basal Cell Carcinoma | Malignant |
| `bkl` | Benign Keratosis | Benign |
| `df` | Dermatofibroma | Benign |
| `mel` | Melanoma | Malignant |
| `nv` | Melanocytic Nevi | Benign |
| `vasc` | Vascular Lesions | Benign |

---

# Project Pipeline

## 1. Data Loading & Exploratory Data Analysis (EDA)

Performed detailed exploratory analysis including:

- Class distribution analysis
- Imbalance ratio visualization
- RGB channel distribution plots
- Pixel intensity analysis
- Mean image visualization for each class

---

## 2. Preprocessing

Preprocessing steps include:

- Duplicate removal
- Pixel normalization to `[0, 1]`
- Stratified `80/20` train-test split
- Label encoding
- Class weight computation for imbalance handling

---

## 3. Data Augmentation

To improve generalization and reduce overfitting, the following augmentations are applied using `ImageDataGenerator`:

- Random rotation
- Horizontal flip
- Vertical flip
- Width and height shifts
- Zoom transformations

---

## 4. CNN Model Architecture

The proposed CNN contains **3 convolutional blocks** with progressively increasing depth.

### Architecture

```text
Input Image (28×28×3)
        ↓

Block 1
Conv2D(32) × 2
BatchNormalization
MaxPooling2D
Dropout(0.25)

        ↓

Block 2
Conv2D(64) × 2
BatchNormalization
MaxPooling2D
Dropout(0.30)

        ↓

Block 3
Conv2D(128)
BatchNormalization
MaxPooling2D
Dropout(0.40)

        ↓

Flatten

        ↓

Dense(256)
Dropout

        ↓

Dense(128)

        ↓

Softmax(7)
```

### Compilation

- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Loss Function:** Categorical Cross-Entropy
- **Metric:** Accuracy

---

## 5. Model Training

Training configuration:

- Maximum epochs: `100`
- Early stopping with patience `15`
- Learning rate reduction using `ReduceLROnPlateau`
- Best model saving using `ModelCheckpoint`

---

## 6. Evaluation

The trained model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix
- Normalized Confusion Matrix
- ROC Curves (One-vs-Rest)
- Per-class AUC scores

---

## 7. Explainability with Grad-CAM

Grad-CAM visualizations are generated to interpret CNN predictions by highlighting important image regions responsible for classification decisions.

Features include:

- Heatmap generation
- Overlay visualization
- Class-specific attention mapping

---

## 8. Baseline Machine Learning Comparison

The CNN model is compared against traditional ML algorithms trained on flattened pixel features:

- Logistic Regression
- Random Forest
- K-Nearest Neighbors (KNN)

---

# Streamlit Web Application

An interactive Streamlit application is provided for real-time prediction.

### Features

- Upload dermatoscopic skin lesion images
- Predict lesion category
- Display confidence score
- Probability breakdown for all classes
- User-friendly interface

---

# Installation

Install the required dependencies:

```bash
pip install kaggle tensorflow scikit-learn pandas numpy matplotlib seaborn pillow opencv-python-headless streamlit
```

---

# Usage

## Train the Model

Run all notebook cells sequentially to:

1. Load dataset
2. Train the CNN
3. Evaluate the model
4. Save the best model

---

## Launch the Streamlit App

```bash
streamlit run app.py
```

---

# Requirements

```text
streamlit==1.28.0
tensorflow==2.13.0
numpy==1.24.0
Pillow==10.0.0
opencv-python-headless==4.8.0
```

---

# Technologies Used

- Python
- TensorFlow / Keras
- NumPy
- Pandas
- Scikit-learn
- OpenCV
- Matplotlib
- Seaborn
- Streamlit

---

# Project Features

✔ Deep Learning-based Skin Lesion Classification  
✔ Custom CNN Architecture  
✔ Data Augmentation  
✔ Model Explainability using Grad-CAM  
✔ Baseline ML Comparisons  
✔ Streamlit Deployment  
✔ End-to-End ML Pipeline

---

# Future Improvements

- Use transfer learning models such as ResNet50 or EfficientNet
- Integrate real-time dermatoscopic image capture
- Add dermatologist feedback loop
- Deploy on cloud platforms
- Improve class balancing techniques

---

# Author

Developed as part of a deep learning and medical image analysis project for automated skin cancer classification.
