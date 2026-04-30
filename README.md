# Model Generalization via Stacked Ensemble Learning

A robust machine learning implementation designed to maximize model generalization and minimize overfitting using a two-tier Stacked Ensemble architecture combined with Stratified K-Fold Cross-Validation.

## 🚀 Key Features
- **Stacked Ensemble Architecture:** Combines multiple high-performing base learners to improve predictive reliability.
- **Cross-Validation Integration:** Utilizes 10-fold Stratified CV to ensure stability across diverse data distributions.
- **Bias Reduction:** Specifically engineered to narrow the "Generalization Gap" between training and testing performance.
- **Multi-Dataset Support:** Pre-configured for Breast Cancer, Wine, and Diabetes classification tasks.

## 🏗️ System Architecture
The system employs a hierarchical stacking approach to aggregate the strengths of different classification algorithms:

### Tier 1: Base Learners
- **Random Forest (RF):** Handles non-linear relationships and reduces variance via bagging.
- **Support Vector Machine (SVM):** Optimizes decision boundaries for high-dimensional data.
- **Logistic Regression (LR):** Provides a baseline probabilistic framework.

### Tier 2: Meta-Learner
- **Logistic Regression:** Acts as the final decision-maker, learning the optimal weights for each base learner's predictions to produce the final output.

## 🧪 Experimental Results
The implementation achieves high stability and accuracy across all tested environments.

| Model | Avg. Accuracy | Stability Score (GSS) |
| :--- | :---: | :---: |
| Random Forest | 95.60% | 0.9815 |
| SVM | 97.36% | 0.9839 |
| Logistic Regression| 97.14% | 0.9900 |
| **Stacked Ensemble (Final)** | **97.80%** | **0.9912** |

## 🚀 Getting Started

### Prerequisites
- Python 3.x
- `requirements.txt` dependencies (NumPy, Pandas, Scikit-Learn, Matplotlib, imbalanced-learn)

### Installation
1. **Clone the repository:**
```bash
git clone [https://github.com/prateeksingh9464/Study-of-Model-Generalization.git](https://github.com/prateeksingh9464/Study-of-Model-Generalization.git)
```

2. **Move into the project directory:**
```bash
cd Study-of-Model-Generalization
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

### Usage
Open the Jupyter Notebook in VS Code or your preferred environment to run the model:
- `final_generalization_model.ipynb`

## ✍️ Contributors
* **Shikha Bharti**
* **Bharat Dutta**
* **Pranjal Singh**
* **Prateek Singh Kanghuta**
* **Gurleen Kaur**

*Department of CSE (AIML), Chandigarh University, Mohali, India.*
