# Credit Card Fraud Detection Using Machine Learning

## Project Overview

This project develops a **machine learning-based fraud detection system** to identify fraudulent credit card transactions in a highly imbalanced dataset.

The primary challenge is the extremely low proportion of fraudulent transactions, where the minority class represents only about **0.17% of all transactions**. The project therefore focuses on handling class imbalance effectively while maintaining a strong balance between **precision and recall**.

Multiple classification approaches are trained and evaluated, with **Random Forest** providing the strongest overall performance on the test data.

### Objectives

- Explore and preprocess credit card transaction data
- Analyze the severe class imbalance between legitimate and fraudulent transactions
- Apply **SMOTE (Synthetic Minority Over-sampling Technique)** to improve minority-class representation
- Train and compare multiple machine learning classifiers
- Evaluate models using fraud-focused classification metrics
- Identify the best-performing model for fraud detection
- Analyze model performance on unseen test data

---

## Dataset

The project uses an anonymized credit card transaction dataset containing **284,807 transactions** and **492 fraudulent transactions**.

| Attribute | Description |
|---|---|
| Total Transactions | **284,807** |
| Fraudulent Transactions | **492** |
| Legitimate Transactions | **284,315** |
| Fraud Ratio | **~0.17%** |
| Features | 30 input features |
| Target | `Class` |

The dataset contains:

- `Time` – elapsed time between transactions
- `Amount` – transaction amount
- `V1`–`V28` – anonymized numerical features
- `Class` – target variable (`0` = legitimate, `1` = fraudulent)

The anonymized `V1`–`V28` variables are treated as numerical predictors, while `Class` is used as the classification target.

---

# Methodology

## 1. Data Exploration

Initial exploratory analysis is performed to understand the structure and distribution of the transaction data.

Key steps include:

- Inspecting dataset dimensions and feature types
- Checking for missing values
- Examining class distribution
- Comparing legitimate and fraudulent transactions
- Analyzing transaction amount distributions
- Investigating relationships between features and fraud labels

The analysis highlights the severe imbalance between the majority and minority classes, making conventional accuracy an unreliable standalone metric.

---

## 2. Data Preprocessing

The dataset is prepared for machine learning through:

- Feature-target separation
- Train-test splitting
- Numerical feature preparation
- Scaling where required by the classification algorithm
- Maintaining a clear separation between training and unseen test data

Special attention is given to preventing the imbalance-handling procedure from contaminating the test set.

---

## 3. Handling Class Imbalance with SMOTE

Because fraudulent transactions account for only approximately **0.17%** of the dataset, directly training a classifier on the original distribution can cause the model to favor the majority class.

To address this issue, **SMOTE** is applied to synthetically oversample the minority fraud class in the training data.

### Why SMOTE?

SMOTE generates synthetic minority-class observations based on existing minority samples rather than simply duplicating fraud transactions.

```text
Original Training Data
        │
        ├── Legitimate Transactions
        │       ███████████████████████████
        │
        └── Fraudulent Transactions
                █
                │
                ▼
              SMOTE
                │
                ▼
Balanced Training Data
        │
        ├── Legitimate Transactions
        │       ███████████████████████████
        │
        └── Fraudulent Transactions
        │       ███████████████████████████
        │
        ▼
       Model Training
```

SMOTE is applied only to the training data so that the final test-set evaluation continues to represent unseen transaction behavior.

---

# Machine Learning Models

Multiple classification algorithms are evaluated to determine which approach best identifies fraudulent transactions.

### Models considered

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Decision Tree
- Random Forest

The models are compared using metrics that are particularly relevant to fraud detection rather than relying solely on accuracy.

---

# Model Evaluation

The following metrics are used to assess performance:

### Precision

Measures the proportion of transactions predicted as fraudulent that are actually fraudulent.

\[
Precision = \frac{TP}{TP + FP}
\]

High precision helps reduce the number of legitimate transactions incorrectly flagged as fraud.

### Recall

Measures the proportion of actual fraudulent transactions correctly identified.

\[
Recall = \frac{TP}{TP + FN}
\]

High recall is important because missed fraudulent transactions can directly translate into financial losses.

### F1-Score

The harmonic mean of precision and recall:

\[
F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}
\]

F1-score provides a balanced measure when both false positives and false negatives matter.

### ROC-AUC

ROC-AUC evaluates the model's ability to distinguish between legitimate and fraudulent transactions across different classification thresholds.

### Confusion Matrix

The confusion matrix provides a detailed breakdown of:

- True Positives
- True Negatives
- False Positives
- False Negatives

---

# Results

## Best Model Performance

The **Random Forest classifier** achieved the strongest overall performance on the unseen test data.

| Metric | Random Forest |
|---|---:|
| **Precision** | **94.3%** |
| **F1-Score** | **93.6%** |

The model demonstrates a strong balance between correctly identifying fraudulent transactions and limiting false fraud alerts.

### Key Finding

**Random Forest achieved the highest F1-score of 93.6% on the test data**, making it the best-performing model among the evaluated classifiers.

The model also achieved **94.3% precision**, indicating that the majority of transactions flagged as fraudulent were genuinely fraudulent.

---

# Evaluation Workflow

```text
Credit Card Transactions
          │
          ▼
    Data Exploration
          │
          ▼
   Class Distribution
          │
          ▼
   Data Preprocessing
          │
          ▼
      Train / Test Split
          │
          ▼
   SMOTE on Training Data
          │
          ▼
   Model Training
          │
    ┌─────┼─────┬──────┐
    ▼     ▼     ▼      ▼
   LR    KNN    SVM    DT
                     │
                     ▼
                Random Forest
                     │
          ┌──────────┴──────────┐
          ▼                     ▼
    Model Comparison       Test Evaluation
          │                     │
          └──────────┬──────────┘
                     ▼
              Best Model
             Random Forest
                     │
                     ▼
       Fraud Detection Predictions
```

---

# Error Analysis

Because fraud detection is a high-impact classification problem, the analysis emphasizes the types of errors rather than only the overall score.

### False Positives

Legitimate transactions incorrectly classified as fraudulent.

**Business impact:** unnecessary transaction declines, customer friction, and additional verification.

### False Negatives

Fraudulent transactions incorrectly classified as legitimate.

**Business impact:** direct financial losses and increased fraud exposure.

### True Positives

Fraudulent transactions correctly identified by the model.

### True Negatives

Legitimate transactions correctly classified as legitimate.

The confusion matrix is used to examine these four outcomes and understand the precision-recall trade-off of each model.

---

# Why Accuracy Alone Is Not Enough

With only around **0.17% fraudulent transactions**, a model could achieve extremely high accuracy simply by predicting nearly every transaction as legitimate.

For example, a classifier that ignores the minority class would still appear highly accurate while failing to detect the transactions that matter most.

Therefore, the project prioritizes:

- **Precision**
- **Recall**
- **F1-score**
- **ROC-AUC**
- **Confusion Matrix**

This provides a more meaningful evaluation of fraud-detection performance.

---

# End-to-End Architecture

```text
             Transaction Dataset
                     │
                     ▼
              Data Preprocessing
                     │
                     ▼
             Exploratory Analysis
                     │
                     ▼
              Class Imbalance
                     │
                     ▼
                  SMOTE
                     │
                     ▼
            Balanced Training Set
                     │
                     ▼
             Model Development
                     │
        ┌────────────┼────────────┐
        ▼            ▼            ▼
   Logistic       SVM/KNN      Tree Models
   Regression                    │
                                  ▼
                            Random Forest
                                  │
                                  ▼
                         Model Evaluation
                                  │
              ┌───────────────────┼──────────────────┐
              ▼                   ▼                  ▼
          Precision            Recall            F1-Score
              │                   │                  │
              └───────────────────┼──────────────────┘
                                  ▼
                            Fraud Detection
```

---

# Technologies Used

| Area | Technologies |
|---|---|
| Programming | **Python** |
| Data Manipulation | **Pandas, NumPy** |
| Visualization | **Matplotlib, Seaborn** |
| Machine Learning | **Scikit-learn** |
| Imbalanced Learning | **imbalanced-learn / SMOTE** |
| Models | Logistic Regression, KNN, SVM, Decision Tree, Random Forest |
| Evaluation | ROC-AUC, Confusion Matrix, Precision, Recall, F1-Score |
| Development | Jupyter Notebook / Google Colab |

---

# Project Structure

```text
Credit-Card-Fraud-Detection/
│
├── Code/
│   ├── Credit Card Fraud Detection - Logistic Regression.ipynb
│   ├── Credit Card Fraud Detection - K-Nearest Neighbor.ipynb
│   ├── Credit Card Fraud Detection - Support Vector Machines.ipynb
│   └── Credit Card Fraud Detection - Decision Tree.ipynb
│
├── Model/
│   ├── Model.docx
│   └── Machine_Learning_Project.pdf
│
├── Presentation/
│   └── AI-Report-Presentation.pptx
│
├── Project Proposal/
│   └── Project proposal documents
│
├── Report/
│   └── Project report
│
└── README.md
```

---

# Key Takeaways

- Built a machine learning pipeline for **credit card fraud detection** on a severely imbalanced dataset.
- Addressed the **0.17% fraud-to-total transaction imbalance** using **SMOTE**.
- Compared multiple classification approaches including Logistic Regression, KNN, SVM, Decision Tree, and Random Forest.
- **Random Forest achieved the highest F1-score of 93.6%** on unseen test data.
- Achieved **94.3% precision**, reducing the proportion of legitimate transactions incorrectly flagged as fraud.
- Evaluated model performance using **ROC-AUC, confusion matrix, precision, recall, and F1-score**.
- Focused on the precision-recall trade-off to build a more meaningful fraud detection system than one optimized purely for accuracy.
