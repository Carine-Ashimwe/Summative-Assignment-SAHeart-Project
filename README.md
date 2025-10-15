# Coronary Heart Disease Prediction: ML vs. Deep Learning Comparison

## 1. Project Overview

This project implements a **reproducible pipeline** for comparing **Traditional Machine Learning (ML)** approaches using **Scikit-learn** against **Deep Learning (DL)** approaches using **TensorFlow** (Sequential and Functional APIs).

The primary goal is to predict the **10-year risk of Coronary Heart Disease (CHD)**, framed as a **binary classification task**. Early prediction of CHD risk is critical in healthcare for guiding preventive interventions.

**Key Objectives:**

* Identify the best-performing model for CHD prediction (ML vs. DL).
* Demonstrate advanced **data preprocessing** and **feature engineering** techniques.
* Implement both **TensorFlow Sequential and Functional API** architectures.
* Conduct multiple experiments and clearly compare model performances.

---

## 2. Dataset

| Detail              | Description                                                                           |
| ------------------- | ------------------------------------------------------------------------------------- |
| **Dataset Name**    | South African Heart Disease (SAHeart) Dataset                                         |
| **Source**          | External Open Source                                                                  |
| **Problem Type**    | Binary Classification                                                                 |
| **Target Variable** | `chd` (1 = Yes, 0 = No)                                                               |
| **Key Features**    | `sbp`, `tobacco`, `ldl`, `adiposity`, `famhist`, `typea`, `obesity`, `alcohol`, `age` |

The dataset reflects real-world risk factors for CHD in a South African population sample.

---

## 3. Methodology and Model Comparison

### A. Data Preprocessing & Feature Engineering

* **Custom Column Transformer** for preprocessing:

  * **One-Hot Encoding** for nominal categorical features (`famhist`).
  * **Standard Scaling** for numerical features to normalize inputs.
* **Data Splitting** into training, validation, and test sets with **stratification** to maintain class balance.

### B. Traditional Machine Learning (Scikit-learn)

* **Models Explored:** Logistic Regression, K-Nearest Neighbors, Random Forest, and more.
* **Hyperparameter Tuning:** `GridSearchCV` used to find optimal parameters for top-performing models.

### C. Deep Learning (TensorFlow)

* **Sequential API Model:** Standard feed-forward network for quick prototyping.
* **Functional API Model:** Flexible architecture allowing complex data flow.
* **Data Handling:** `tf.data` API for efficient batching and shuffling.

---

## 4. Project Structure

```
├── Summative_Assignment_Model_Training_and_Evaluation_Carine_Ashimwe.ipynb
├── README.md
├── preprocessor.joblib                  # Saved custom Scikit-learn preprocessor
├── sequential_model.keras               # Saved TensorFlow Sequential model
├── functional_model.keras               # Saved TensorFlow Functional model
└── [TOP_CLASSICAL_MODEL]_best_estimator.joblib  # Saved best classical ML model
```

**Reproducibility:**
All experiments use `RANDOM_SEED = 42` for Python, NumPy, and TensorFlow to ensure consistent results.

---

## 5. Setup and Installation

### Prerequisites

* **Git** (for cloning the repository)
* **Anaconda/Miniconda** (recommended)

### Environment Setup

```bash
# Clone the repository
git clone [YOUR-GITHUB-REPO-URL]
cd [repository-name]

# Install dependencies
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
```

> **Note:** Developed with Python 3.12, Scikit-learn 1.6.1, and TensorFlow 2.19.0.

---

## 6. How to Run the Project

1. Open the notebook in **Jupyter** or **Google Colab**.
2. Run all cells sequentially.
3. Review outputs:

   * Preprocessing steps and engineered features.
   * Learning curves for DL models.
   * Performance metrics (**Accuracy, F1-Score, AUC**) for all models on the test set.
   * Final comparison table of best ML vs. DL models.

---

## 7. Results Highlights

* **Best Model:** [Functional API Deep Learning Model] achieved the highest performance on the test set with [Best Metric Value, e.g., 0.78 AUC].
* **Top Classical Model:** [Logistic Regression] demonstrated strong generalization despite its simpler architecture.
* **Observations:** Performance was limited by [Data Limitation, e.g., Class Imbalance], mitigated partially through preprocessing and loss functions. DL models showed slight overfitting, addressed with [Dropout/Other Techniques].
