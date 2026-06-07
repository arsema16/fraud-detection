# Fraud Detection System

## Project Overview
This project develops a unified fraud detection capability for Adey Innovations Inc.

## Setup Instructions
1. Create virtual environment: `python -m venv venv`
2. Activate: `.\venv\Scripts\activate`
3. Install: `pip install -r requirements.txt`
4. Download data to `data/raw/`
5. Run: `python scripts/run_preprocessing.py`

## Branches
- `main` - Production-ready code
- `task1-data-preprocessing` - Data cleaning, EDA, feature engineering
- `task2-model-building` - Model training and evaluation
- `task3-shap-analysis` - Model explainability with SHAP
# Fraud Detection System for E-commerce and Bank Transactions

[![Unit Tests](https://github.com/yourusername/fraud-detection/actions/workflows/unittests.yml/badge.svg)](https://github.com/yourusername/fraud-detection/actions/workflows/unittests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

## 📋 Project Overview

This project develops a unified fraud detection capability for **Adey Innovations Inc.**, a leading FinTech company serving e-commerce and banking clients. The system handles two distinct transaction types:

- **E-commerce Transactions**: Rich context including user profiles, device information, and behavioral patterns
- **Bank Credit Transactions**: Anonymized PCA-transformed features for privacy compliance

### 🎯 Business Impact

Effective fraud detection has direct financial and reputational consequences:

| Impact Type | Consequence | Business Cost |
|------------|-------------|---------------|
| **False Positives** | Legitimate transactions flagged as fraud | Customer frustration, lost trust, churn |
| **False Negatives** | Actual fraud missed | Direct financial loss, chargebacks |
| **Real-time Detection** | Quick identification and prevention | Reduced risk exposure |

## 📊 Dataset Information

### E-commerce Fraud Data (`Fraud_Data.csv`)
| Field | Description | Type |
|-------|-------------|------|
| user_id | Unique user identifier | Integer |
| signup_time | Timestamp of user signup | DateTime |
| purchase_time | Timestamp of the purchase | DateTime |
| purchase_value | Purchase amount in dollars | Float |
| device_id | Unique device identifier | String |
| source | Traffic source (SEO, Ads, Direct) | Categorical |
| browser | Browser used (Chrome, Safari, etc.) | Categorical |
| sex | Gender of the user (M/F) | Categorical |
| age | Age of the user | Integer |
| ip_address | IP address used for the transaction | String |
| class | Target (1=fraud, 0=legitimate) | Binary |

### IP Geolocation Data (`IpAddress_to_Country.csv`)
| Field | Description |
|-------|-------------|
| lower_bound_ip_address | Lower bound of IP range |
| upper_bound_ip_address | Upper bound of IP range |
| country | Country corresponding to IP range |

### Credit Card Fraud Data (`creditcard.csv`)
| Field | Description |
|-------|-------------|
| Time | Seconds elapsed since first transaction |
| V1-V28 | Anonymized PCA-transformed features |
| Amount | Transaction amount in dollars |
| Class | Target (1=fraud, 0=legitimate) |

## 🏗️ Project Structure
```
fraud-detection/
├── .github/ # GitHub Actions workflows
│ └── workflows/
│ └── unittests.yml # CI/CD pipeline
├── .vscode/ # VS Code configuration
│ └── settings.json
├── data/
│ ├── raw/ # Original datasets (gitignored)
│ │ ├── Fraud_Data.csv
│ │ ├── IpAddress_to_Country.csv
│ │ └── creditcard.csv
│ └── processed/ # Cleaned and engineered data/
├── raw/ # Place original datasets here (gitignored)
│ ├── Fraud_Data.csv
│ ├── IpAddress_to_Country.csv
│ └── creditcard.csv
└── processed/ # Output from preprocessing pipeline
├── fraud_clean.csv
├── fraud_data_engineered.csv
├── creditcard_clean.csv
└── country_fraud_analysis.csv
├── notebooks/ # Jupyter notebooks
│ ├── 01_eda_fraud_data.ipynb
│ ├── 02_eda_creditcard.ipynb
│ ├── 03_feature_engineering.ipynb
│ ├── 04_modeling.ipynb
│ └── 05_shap_explainability.ipynb
├── src/ # Source code modules
│ ├── init.py
│ ├── data_cleaning.py # Data cleaning utilities
│ ├── geolocation.py # IP to country mapping
│ ├── feature_engineering.py # Feature creation
│ ├── preprocessing.py # Scaling and encoding
│ └── imbalance_handling.py # SMOTE and resampling
├── tests/ # Unit tests
│ ├── init.py
│ ├── conftest.py # Test fixtures
│ ├── test_data_cleaning.py
│ ├── test_geolocation.py
│ ├── test_feature_engineering.py
│ └── test_imbalance_handling.py
├── models/ # Saved model artifacts
│ ├── fraud_preprocessor.pkl
│ └── credit_scaler.pkl
├── scripts/ # Execution scripts
│ └── run_preprocessing.py # Main preprocessing pipeline
├── reports/ # Visualizations and reports
│ ├── class_distribution.png
│ ├── fraud_by_country.png
│ └── task1_summary.csv
├── requirements.txt # Python dependencies
├── README.md # This file
└── .gitignore # Git ignore rules
  ```

## 🚀 Getting Started

### Prerequisites

- Python 3.9 or higher
- Git
- Virtual environment (recommended)

### Installation

#### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/fraud-detection.git
cd fraud-detection
```
#### 2. Create and Activate Virtual Environment
Windows:

```bash
python -m venv venv
.\venv\Scripts\activate
```
Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```
#### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
#### 4. Download Dataset
***Place the following files in data/raw/:***

- Fraud_Data.csv - [Download from source]

- IpAddress_to_Country.csv - [Download from source]

- creditcard.csv - Download from Kaggle
#### 5. Run Preprocessing Pipeline
```bash
python scripts/run_preprocessing.py
```
***This will:***

- Clean and validate all datasets

- Add geolocation information to e-commerce transactions

- Engineer 20+ features for fraud detection

- Analyze class imbalance and recommend SMOTE strategy

- Save processed data to data/processed/

#### 6. Launch Jupyter Notebooks
```
jupyter notebook notebooks/
```
***Recommended order:***

- 01_eda_fraud_data.ipynb - Explore e-commerce data

- 02_eda_creditcard.ipynb - Explore credit card data

- 03_feature_engineering.ipynb - Understand feature creation

- 04_modeling.ipynb - Build and evaluate models

- 05_shap_explainability.ipynb - Interpret model predictions

### Notebooks (Ordered by Task)

| Notebook | Purpose | Task |
|----------|---------|------|
| `01_eda_fraud_data.ipynb` | Univariate/bivariate analysis of e-commerce data | Task 1 |
| `02_eda_creditcard.ipynb` | EDA and imbalance quantification for credit card data | Task 1 |
| `03_feature_engineering.ipynb` | Detailed feature engineering walkthrough | Task 1 |
| `04_modeling.ipynb` | Model training, tuning, and evaluation | Task 2 |
| `05_shap_explainability.ipynb` | SHAP analysis and business recommendations | Task 3 |

### Source Code Modules

| Module | Purpose |
|--------|---------|
| `data_cleaning.py` | Handle missing values, duplicates, invalid data |
| `geolocation.py` | IP to integer conversion and country mapping |
| `feature_engineering.py` | Create temporal, behavioral, derived features |
| `preprocessing.py` | Scaling, encoding, train-test split |
| `preprocessing_pipeline.py` | **Complete integrated pipeline (all steps)** |
| `imbalance_handling.py` | SMOTE application with justification |

### Scripts

| Script | Purpose |
|--------|---------|
| `run_preprocessing.py` | Execute complete preprocessing pipeline |
| `run_training.py` | Execute model training (Task 2) |

### How They Relate

1. **EDA Notebooks** (`01_*`, `02_*`) explore raw data and inform feature engineering
2. **Feature Engineering Notebook** (`03_*`) demonstrates feature creation logic
3. **Preprocessing Script** (`run_preprocessing.py`) calls the integrated pipeline
4. **Source Modules** contain reusable functions used by notebooks and scripts
5. **Tests** validate each module independently

### Complete Pipeline Execution

```bash
# Run entire preprocessing pipeline
python scripts/run_preprocessing.py

# Or use the integrated pipeline class
python -c "from src.preprocessing_pipeline import FraudPreprocessingPipeline; pipeline = FraudPreprocessingPipeline(); pipeline.run_complete_pipeline('data/raw/Fraud_Data.csv', 'data/raw/IpAddress_to_Country.csv')"
## 🔧 Key Features Engineered

### Temporal Features

| Feature | Description | Fraud Signal |
|---------|-------------|--------------|
| `time_since_signup` | Hours between signup and purchase | Short time = higher risk |
| `transactions_24h` | Transaction count in last 24 hours | High velocity = suspicious |
| `purchase_hour` | Hour of day (0-23) | Off-hour transactions riskier |
| `hours_since_prev_purchase` | Time since last transaction | Unusually rapid = fraud |

### Behavioral Features

| Feature | Description | Fraud Signal |
|---------|-------------|--------------|
| `user_avg_amount` | User's historical average | Deviation from baseline |
| `user_fraud_rate` | User's historical fraud rate | Past fraud indicates risk |
| `device_fraud_rate` | Device reputation score | Shared devices = higher risk |
| `country_risk_score` | Geographic fraud rate | High-risk countries flagged |

### Derived Features

- `value_per_hour` - Purchase value normalized by time since signup
- `age_group` - Binned age categories for demographic analysis
- `is_high_value` / `is_low_value` - Amount percentile flags

## 📈 Class Imbalance Handling

Both datasets have severe class imbalance:

| Dataset | Fraud Rate | Imbalance Ratio |
|---------|------------|-----------------|
| E-commerce | ~0.1-1% | >100:1 |
| Credit Card | 0.17% | 578:1 |

**Selected Strategy: SMOTE (Synthetic Minority Over-sampling)**

**Why SMOTE?**

- ✅ Preserves all legitimate transaction data
- ✅ Creates synthetic fraud examples (not just duplication)
- ✅ Prevents overfitting compared to undersampling
- ✅ Industry standard for fraud detection

**Implementation:**

```python
from imblearn.over_sampling import SMOTE

smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```
## 🧪 Testing

Run the test suite to verify functionality:

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ -v --cov=src --cov-report=html

# Run specific test file
python -m pytest tests/test_geolocation.py -v
```
## 🧪 Testing

Test coverage includes:
- Data cleaning validation (7 tests)
- IP geolocation mapping (7 tests)
- Feature engineering (4 tests)
- Imbalance handling (4 tests)

## 📊 Evaluation Metrics

| Metric | Why It's Used |
|--------|----------------|
| AUC-PR | Better than ROC for imbalanced data |
| F1-Score | Harmonic mean of precision and recall |
| Precision | How many flagged transactions are actually fraud |
| Recall | How many actual frauds were caught |
| Confusion Matrix | Visualizes false positives vs false negatives |

> ⚠️ Never use Accuracy - on imbalanced data, a model that predicts "legitimate" for every transaction would have 99.8% accuracy but catch zero fraud!

## 📝 Task Completion Status

✅ **Task 1: Data Analysis and Preprocessing (Interim-1)**
- Data cleaning and validation
- Exploratory Data Analysis
- IP to country geolocation mapping
- Feature engineering (20+ features)
- Class imbalance analysis
- SMOTE strategy selection
- Unit tests implementation

🔄 **Task 2: Model Building and Training (Interim-2 - In Progress)**
- Baseline Logistic Regression
- XGBoost / Random Forest models
- Hyperparameter tuning
- Cross-validation
- Model comparison and selection

📅 **Task 3: Model Explainability (Final)**
- SHAP analysis
- Feature importance visualization
- Business recommendations

## 🤝 Contributing

```bash
git checkout -b task1-data-preprocessing
git add .
git commit -m "feat: add data cleaning module"
git push origin task1-data-preprocessing
```
## Branch Structure
```text
main (production-ready)
├── task1-data-preprocessing (✅ Interim-1)
├── task2-model-building (🚧 In Progress)
└── task3-shap-analysis (📅 Planned)
```
## Troubleshooting
ModuleNotFoundError: No module named 'src'

```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/fraud-detection"
cd fraud-detection
python scripts/run_preprocessing.py
Memory errors with large datasets
```
```python
df_sample = fraud_df.sample(frac=0.1, random_state=42)
IP conversion tests failing
```
### IP conversion tests failing

Ensure using Python 3.9+

Run python test_ip_conversion.py to debug

### SMOTE requires more memory

```python
smote = SMOTE(sampling_strategy=0.1)  # Instead of 0.3
```