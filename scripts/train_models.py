import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_recall_curve, auc, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os

print('='*50)
print('FRAUD DETECTION MODEL TRAINING')
print('='*50)

# Load data
print('\n1. Loading data...')
df = pd.read_csv('data/processed/fraud_data_engineered.csv')
print(f'   Shape: {df.shape}')
fraud_rate = df['class'].mean() * 100
print(f'   Fraud rate: {fraud_rate:.2f}%')

# Prepare features
X = df.drop('class', axis=1)
y = df['class']

# Identify column types
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print(f'   Numeric features: {len(numeric_cols)}')
print(f'   Categorical features: {len(categorical_cols)}')
if categorical_cols:
    print(f'   Categories: {categorical_cols}')

# Preprocessor
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols)
])

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Transform
print('\n2. Preprocessing data...')
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)
print(f'   Training shape: {X_train_processed.shape}')
print(f'   Test shape: {X_test_processed.shape}')

# Logistic Regression
print('\n3. Training Logistic Regression...')
lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr.fit(X_train_processed, y_train)

y_pred_lr = lr.predict(X_test_processed)
y_pred_proba_lr = lr.predict_proba(X_test_processed)[:, 1]

precision_lr, recall_lr, _ = precision_recall_curve(y_test, y_pred_proba_lr)
auc_pr_lr = auc(recall_lr, precision_lr)
f1_lr = f1_score(y_test, y_pred_lr)

print(f'   AUC-PR: {auc_pr_lr:.4f}')
print(f'   F1-Score: {f1_lr:.4f}')

# Random Forest
print('\n4. Training Random Forest...')
rf = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10, class_weight='balanced', n_jobs=-1)
rf.fit(X_train_processed, y_train)

y_pred_rf = rf.predict(X_test_processed)
y_pred_proba_rf = rf.predict_proba(X_test_processed)[:, 1]

precision_rf, recall_rf, _ = precision_recall_curve(y_test, y_pred_proba_rf)
auc_pr_rf = auc(recall_rf, precision_rf)
f1_rf = f1_score(y_test, y_pred_rf)

print(f'   AUC-PR: {auc_pr_rf:.4f}')
print(f'   F1-Score: {f1_rf:.4f}')

# Model Comparison
print('\n5. Model Comparison')
print('='*50)
print(f'{"Model":<20} {"AUC-PR":<12} {"F1-Score":<12}')
print('-'*50)
print(f'{"Logistic Regression":<20} {auc_pr_lr:.4f}        {f1_lr:.4f}')
print(f'{"Random Forest":<20} {auc_pr_rf:.4f}        {f1_rf:.4f}')

# Save models
print('\n6. Saving models...')
os.makedirs('models', exist_ok=True)
joblib.dump(lr, 'models/lr_fraud_model.pkl')
joblib.dump(rf, 'models/rf_fraud_model.pkl')
joblib.dump(preprocessor, 'models/preprocessor.pkl')
print('   ✓ Models saved to models/ directory')

# Print confusion matrix for best model
print('\n7. Confusion Matrix - Random Forest (Best Model)')
print('='*50)
cm = confusion_matrix(y_test, y_pred_rf)
print(cm)
print(f'\nTrue Negatives: {cm[0,0]:,}')
print(f'False Positives: {cm[0,1]:,}')
print(f'False Negatives: {cm[1,0]:,}')
print(f'True Positives: {cm[1,1]:,}')

print('\n' + '='*50)
print('TRAINING COMPLETE!')
print('='*50)