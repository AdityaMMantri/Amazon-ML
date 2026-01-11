# Amazon-ML

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Platform](https://img.shields.io/badge/platform-Windows%20%7C%20Linux-lightgrey)
![Status](https://img.shields.io/badge/status-active-success)


A structured machine learning repository for **feature-based and image-based predictive modeling** using Amazon product data.
The project emphasizes **clean feature engineering, reproducibility, and modular design** suitable for academic evaluation and extension.

---

## Project Overview

This project builds an ML pipeline that combines:

* **Text-based feature engineering**
* **Brand-level target encoding**
* **Numerical normalization**
* **Optional image-based learning**
* **Consistent evaluation using SMAPE**

The repository is structured to clearly separate:

* feature construction
* preprocessing artifacts
* image pipelines
* metrics
* orchestration logic

This separation makes the project **defensible in faculty reviews** and **easy to extend**.

---

## 📂 Repository Structure

```
Amazon-ML/
├── Dataset/                       # Dataset CSV files (tracked)
│   ├── train_final_features.csv
│   ├── test_final_features.csv
│   └── sample_test_out.csv
│
├── features/                      # Feature engineering & encoding
│   ├── binary_features.py
│   ├── brand_features.py
│   ├── log_quantity.py
│   ├── train_brand_expected_price.py
│   └── test_brand_expected_price.py
│
├── images/                        # Image-based pipelines
│   ├── train_images.py
│   └── test_images.py
│
├── preprocessing/                 # Preprocessing artifacts
│   └── trained_scaler.py
│
├── metrics/                       # Evaluation metrics
│   └── smap.py
│
├── train.py                       # Training orchestrator
├── test.py                        # Testing orchestrator
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🧠 Feature Engineering Pipeline

### Implemented Features

* **Brand extraction** from catalog text
* **Binary indicators**

  * brand present
  * bulk purchase detection
* **Log-normalized quantity** from value/unit parsing
* **Brand expected price encoding**

  * Smoothed target encoding using global mean

These features are designed to capture **semantic, statistical, and behavioral signals** beyond raw text.

---

## 🖼️ Image Pipeline

Image processing is handled separately to avoid mixing modalities.

* `train_images.py` → image feature learning / training
* `test_images.py` → inference or evaluation on image data

This separation allows:

* independent experimentation
* multimodal extensions later

---

## 📏 Preprocessing & Artifacts

* `trained_scaler.py` reconstructs and saves a **StandardScaler**
* The scaler is fit **only on training data**
* Ensures **train–test consistency**

---

## 📊 Evaluation Metric

### SMAPE (Symmetric Mean Absolute Percentage Error)

Used for robust evaluation when prices vary across scales.

Implemented in:

```
metrics/smap.py
```

---

## 🔁 Training & Testing Flow

### High-Level Architecture

```
                ┌────────────────────┐
                │   Raw Dataset CSV  │
                └─────────┬──────────┘
                          │
          ┌───────────────▼────────────────┐
          │        Feature Engineering     │
          │  (brand, binary, quantity, etc)│
          └───────────────┬────────────────┘
                          │
              ┌───────────▼───────────┐
              │   Brand Price Encoding│
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │   Scaling / Normalize │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │   Model Training/Test │
              └───────────┬───────────┘
                          │
              ┌───────────▼───────────┐
              │        Evaluation     │
              │        (SMAPE)        │
              └───────────────────────┘
```

---

## 🚀 How to Run

### 1️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

---

### 2️⃣ Train the model

```bash
python train.py
```

This script:

* builds features
* applies scaling
* runs the training workflow

---

### 3️⃣ Test / Evaluate

```bash
python test.py
```

This script:

* applies learned encodings and scaler
* evaluates predictions using SMAPE

---

## 🧩 Import Conventions

When extending the project, follow these imports:

```python
from features.binary_features import ...
from features.brand_features import ...
from features.log_quantity import ...
from preprocessing.trained_scaler import ...
from metrics.smap import smape
from images.train_images import ...
```

**Always run scripts from the repository root.**

---

## 📌 Notes & Design Decisions

* Dataset CSV files are **intentionally tracked** for reproducibility
* Other CSVs outside `Dataset/` are ignored to prevent clutter
* Absolute paths should be avoided in future refactors
* Feature scripts currently operate as standalone modules (by design)

----

