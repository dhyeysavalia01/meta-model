# Meta-Model: Meta-Learning AutoML Recommender

A meta-learning-based AutoML recommender that analyzes dataset characteristics and recommends suitable machine learning algorithms for **Classification** and **Regression** tasks.

Instead of testing every algorithm blindly, the system learns from previously evaluated datasets and uses a trained meta-model to rank candidate algorithms based on their performance on similar datasets.

---

## Overview

The application provides an interactive workflow where users can:

- Select a Classification or Regression task.
- Upload a dataset in CSV or Excel format.
- Inspect dataset statistics, feature types, and missing values.
- Configure missing-value imputation strategies.
- Select the target variable.
- Receive a ranked list of recommended ML algorithms.
- Validate recommended algorithms using cross-validation.
- Explore evaluation metrics and visualizations.

## System Architecture

The project consists of three main components:

1. **Meta-Dataset Generation:** Collects benchmark datasets and evaluates candidate algorithms to construct meta-datasets.
2. **Meta-Model Training:** Extracts dataset-level characteristics and trains Random Forest-based meta-models.
3. **Recommendation & Evaluation:** Processes uploaded datasets, predicts suitable algorithms, and validates their performance.

## Project Structure

```text
meta-model/
│
├── backend/
│   ├── main.py
│   ├── meta_base.py
│   ├── meta_classifier.py
│   ├── meta_features.py
│   ├── meta_regressor.py
│   └── preprocessing.py
│
├── data/
│   ├── meta_clf.csv
│   └── meta_reg.csv
│
├── frontend/
│   ├── app.js
│   ├── index.html
│   └── style.css
│
├── models/
│   ├── meta_clf.joblib
│   └── meta_reg.joblib
│
├── notebooks/
│   ├── 01_build_meta_dataset.ipynb
│   ├── 02_train_meta_models.ipynb
│   └── 03_evaluate_meta_models.ipynb
│ 
├── requirements.txt
└── README.md
```

## Methodology

### 1. Meta-Dataset Generation

Benchmark datasets are collected from OpenML:

- Classification: OpenML-CC18 (Study ID: 99).
- Regression: AutoML Benchmark (Study ID: 269).

Each dataset is evaluated using seven candidate algorithms. Their cross-validation scores are recorded, and the best-performing algorithm becomes the target label for the corresponding meta-learning instance.

Two meta-datasets are generated:

- `meta_clf.csv` — Classification meta-dataset.
- `meta_reg.csv` — Regression meta-dataset.

### 2. Meta-Feature Extraction

The system extracts dataset-level characteristics to describe the underlying data.

| Category | Meta-features |
|---|---|
| Dataset Size | Number of samples, number of features |
| Feature Composition | Numerical features, categorical proportion |
| Data Quality | Missing-value proportion |
| Statistical Properties | Skewness, kurtosis, feature correlation |
| Classification | Number of classes, class entropy, imbalance ratio |
| Regression | Target skewness, kurtosis, coefficient of variation |
| Landmarking | Decision stump, 1-NN, linear model scores |

Landmarking scores provide additional information about how simple baseline algorithms perform on the dataset.

### 3. Meta-Model Training

Separate Random Forest Classifiers are trained for Classification and Regression recommendation tasks.

**Meta-model configuration:**

- Algorithm: Random Forest Classifier
- Number of estimators: 300
- Minimum samples per leaf: 2
- Class weighting: `balanced_subsample`
- Random state: 0
- Missing meta-feature handling: Median imputation

The trained models are serialized using Joblib:

- `meta_clf.joblib`
- `meta_reg.joblib`

### 4. Candidate Algorithms

**Classification**

1. Logistic Regression
2. K-Nearest Neighbors
3. Decision Tree
4. Random Forest
5. Extra Trees
6. Histogram Gradient Boosting
7. Gaussian Naive Bayes

**Regression**

1. Ridge Regression
2. Lasso Regression
3. K-Nearest Neighbors
4. Decision Tree
5. Random Forest
6. Extra Trees
7. Histogram Gradient Boosting

### 5. Recommendation Pipeline

When a user uploads a dataset:

1. Dataset inspection identifies numerical and categorical columns.
2. Missing values are handled using the selected imputation strategies.
3. The user selects the target variable.
4. The system separates input features and target values.
5. Meta-features are extracted from the dataset.
6. The trained meta-model generates probability scores for candidate algorithms.
7. Algorithms are ranked based on their predicted probabilities.
8. The highest-ranked algorithm is presented as the recommendation.

The recommendation confidence represents the meta-model's predicted probability, **not the expected accuracy of the recommended algorithm**.

### 6. Model Evaluation

Users can validate recommended algorithms through cross-validation.

**Classification evaluation:**
- Confusion Matrix
- Classification Report
- ROC Curve
- ROC-AUC Score
- Permutation Feature Importance

**Regression evaluation:**
- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Actual vs. Predicted Plot
- Residual Distribution
- Residuals vs. Predicted Plot
- Permutation Feature Importance

Classification uses stratified cross-validation, while regression uses 5-fold cross-validation.

## Notebooks

The `notebooks/` directory contains the experimental and model development workflow.

| Notebook | Purpose |
|---|---|
| `01_build_meta_dataset.ipynb` | Collects OpenML datasets, extracts meta-features, evaluates candidate algorithms, and generates meta-datasets. |
| `02_train_meta_models.ipynb` | Trains and saves the classification and regression meta-models. |
| `03_evaluate_meta_models.ipynb` | Evaluates recommendation quality using Leave-One-Out cross-validation and regret analysis. |

The evaluation notebook also compares meta-model recommendations against fixed-algorithm and random-selection baselines.

## Tech Stack

**Backend & Machine Learning**
- Python
- FastAPI
- Pandas
- NumPy
- Scikit-learn
- Joblib

**Frontend**
- HTML5
- CSS3
- JavaScript
- Chart.js

**Data & Experimentation**
- OpenML
- Jupyter Notebook
- CSV / Excel

## Installation & Setup

### 1. Clone the repository

```bash
git clone https://github.com/your-username/meta-model.git
cd meta-model
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it:

**macOS / Linux**
```bash
source venv/bin/activate
```

**Windows**
```bash
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install fastapi uvicorn pandas numpy scikit-learn joblib python-multipart openpyxl
```

For notebook-based experimentation:

```bash
pip install jupyter openml tqdm
```

### 4. Run the application

From the project root directory:

```bash
uvicorn backend.main:app --reload
```

Open the application in your browser:

```text
http://127.0.0.1:8000
```

API documentation is available at:

```text
http://127.0.0.1:8000/docs
```

**Note:** The application expects the trained model artifacts (`meta_clf.joblib` and `meta_reg.joblib`) to be present in the `models/` directory. The CSV meta-datasets are required for reproducing the training workflow, not for normal application execution.

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/upload` | Uploads and inspects a dataset. |
| POST | `/preprocess` | Handles missing-value imputation. |
| POST | `/recommend` | Extracts meta-features and recommends algorithms. |
| POST | `/evaluate` | Evaluates a selected algorithm using cross-validation. |

## Limitations

- Recommendations depend on the coverage and quality of the benchmark meta-datasets.
- Meta-model confidence is not equivalent to actual model accuracy.
- Uploaded datasets must contain sufficient valid samples and suitable target variables.
- Dataset sessions are stored in memory and are not persistent.
- The current implementation is designed primarily as a demonstration rather than a production deployment.

---