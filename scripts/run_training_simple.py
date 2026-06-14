import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE

print("Loading data...")
df = pd.read_csv('data/processed/fraud_data_engineered.csv')

X = df.drop('class', axis=1)
y = df['class']

print(f"Data shape: {X.shape}")
print(f"Fraud rate: {y.mean()*100:.2f}%")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# SMOTE
smote = SMOTE(sampling_strategy=0.3, random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print(f"After SMOTE - Training fraud rate: {y_train_res.mean()*100:.2f}%")

# Logistic Regression
print("\n" + "="*50)
print("LOGISTIC REGRESSION")
print("="*50)

lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_res, y_train_res)

y_pred = lr.predict(X_test)
y_pred_proba = lr.predict_proba(X_test)[:, 1]

precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
auc_pr = auc(recall, precision)
f1 = f1_score(y_test, y_pred)

print(f"AUC-PR: {auc_pr:.4f}")
print(f"F1-Score: {f1:.4f}")
print(f"\nClassification Report:")
print(classification_report(y_test, y_pred))
print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Save model
joblib.dump(lr, 'models/lr_fraud_model.pkl')
print("\n✓ Model saved to models/lr_fraud_model.pkl")

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)