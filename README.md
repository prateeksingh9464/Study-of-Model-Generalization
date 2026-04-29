# Model Generalization using Stacked Ensemble Learning & Advanced CV

> Official repository for the paper "Study of Model Generalization Using Advanced Cross-Validation Techniques and Stacked Ensemble Learning." It tackles ML overfitting and dataset bias by combining a Stacked Ensemble framework (RF, SVM, LR) with 10-fold Stratified CV to boost predictive reliability on unseen data.

## 📌 Project Overview
The core objective of this research is to evaluate and improve model generalization by combining a **Stacked Ensemble Learning** framework with sophisticated **Cross-Validation** protocols. The study demonstrates that ensemble strategies reach higher predictive accuracy and greater stability compared to individual classifiers across heterogeneous datasets.

## 🏗️ System Architecture
The proposed architecture follows a two-level hierarchy:

1. **Level 1 (Base Learners):** Utilizes diverse algorithms to learn complementary patterns from the data.
   - **Random Forest (RF):** Minimizes variance through bagging and random feature selection.
   - **Support Vector Machine (SVM):** Identifies optimal hyperplanes for complex classification boundaries.
   - **Logistic Regression (LR):** Provides interpretable, probabilistic predictions.
2. **Level 2 (Meta-Learner):** Employs **Logistic Regression** to aggregate base model predictions, finding the optimal blend of outputs to improve overall accuracy.

## 🧪 Experimental Methodology
The implementation follows a rigorous pipeline:
- **Data Preprocessing:** Cleaning, standard normalization, and stratified sampling to maintain class distribution.
- **Stratified K-Fold Cross-Validation:** A 10-fold approach ensures unbiased performance estimation by testing on multiple data partitions.
- **Out-of-Fold Predictions:** Base models generate predictions used to construct the meta-level dataset for the final classifier.
- **Generalization Stability Score (GSS):** A metric to quantify the consistency of model performance across folds.

## 📊 Performance Results
The framework was tested on three benchmark datasets: **Breast Cancer**, **Wine**, and **Diabetes**.

| Model | Avg. Accuracy (CV) | Generalization Gap | GSS (Stability) |
| :--- | :---: | :---: | :---: |
| Random Forest | 95.60% | 0.0438 | 0.9815 |
| SVM | 97.36% | - | 0.9839 |
| Logistic Regression| 97.14% | - | 0.9900 |
| **Stacked Ensemble** | **97.80%** | **0.0175** | **0.9912** |

*Key Finding: The Stacked Ensemble significantly reduced the Generalization Gap to **0.0175**, proving its effectiveness in preventing overfitting.*

## 🚀 Getting Started

### Dependencies
- Python 3.x
- NumPy & Pandas
- Scikit-Learn
- Matplotlib & Seaborn

### Installation & Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/your-repo-name.git](https://github.com/your-username/your-repo-name.git)
   cd your-repo-name
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook to reproduce the experiments:
   ```bash
   jupyter notebook final_generalization_model.ipynb
   ```

## ✍️ Authors
* **Shikha Bharti**
* **Bharat Dutta**
* **Pranjal Singh**
* **Prateek Singh Kanghuta**
* **Gurleen Kaur**

*Department of CSE (AIML), Chandigarh University, Mohali, India.*
