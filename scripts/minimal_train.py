import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import pickle
import os

os.makedirs('models', exist_ok=True)

print("="*50)
print("MINIMAL FRAUD DETECTION TRAINING")
print("="*50)

print("\n1. Loading data...")
df = pd.read_csv('data/processed/fraud_data_engineered.csv')
print(f"   Shape: {df.shape}")
print(f"   Fraud rate: {df['class'].mean()*100:.2f}%")

print("\n2. Encoding categorical variables...")
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].astype('category').cat.codes
    print(f"   Encoded: {col}")

X = df.drop('class', axis=1)
y = df['class']
print(f"   Final features: {X.shape[1]}")

print("\n3. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

print("\n4. Training Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
f1 = f1_score(y_test, y_pred)
print(f"\n   F1-Score: {f1:.4f}")

print("\n5. Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f"\n   True Negatives: {cm[0,0]:,}")
print(f"   False Positives: {cm[0,1]:,}")
print(f"   False Negatives: {cm[1,0]:,}")
print(f"   True Positives: {cm[1,1]:,}")

print("\n6. Classification Report:")
print(classification_report(y_test, y_pred))

print("\n7. Saving model...")
with open('models/lr_fraud_model.pkl', 'wb') as f:
    pickle.dump(lr, f)
print("   Model saved to models/lr_fraud_model.pkl")

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
