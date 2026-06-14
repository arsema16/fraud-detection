import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import pickle
import os

os.makedirs('models', exist_ok=True)

print("="*50)
print("FRAUD DETECTION TRAINING (NO PANDAS)")
print("="*50)

print("\n1. Loading data from CSV...")
data = []
with open('data/processed/fraud_data_engineered.csv', 'r') as f:
    reader = csv.reader(f)
    header = next(reader)
    for row in reader:
        data.append(row)

print(f"   Loaded {len(data)} rows")
print(f"   Columns: {header[:5]}...")

# Convert to numpy arrays
print("\n2. Converting to numpy arrays...")
X_list = []
y_list = []

# Find class column index
class_idx = header.index('class')

for row in data:
    y_list.append(float(row[class_idx]))
    # Convert features to float, skip class column
    features = []
    for i, val in enumerate(row):
        if i != class_idx:
            try:
                features.append(float(val))
            except:
                features.append(0)  # Handle non-numeric as 0
    X_list.append(features)

X = np.array(X_list)
y = np.array(y_list)

print(f"   X shape: {X.shape}")
print(f"   Fraud rate: {y.mean()*100:.2f}%")

# Split
print("\n3. Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print(f"   Train: {X_train.shape}, Test: {X_test.shape}")

# Train
print("\n4. Training Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr.fit(X_train, y_train)

# Evaluate
y_pred = lr.predict(X_test)
f1 = f1_score(y_test, y_pred)
print(f"\n   F1-Score: {f1:.4f}")

print("\n5. Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n6. Classification Report:")
print(classification_report(y_test, y_pred))

# Save
print("\n7. Saving model...")
with open('models/lr_fraud_model.pkl', 'wb') as f:
    pickle.dump(lr, f)
print("   Model saved!")

print("\n" + "="*50)
print("TRAINING COMPLETE!")
print("="*50)
