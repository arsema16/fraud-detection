import numpy as np
from imblearn.over_sampling import SMOTE

def analyze_imbalance(y_train):
    """Analyze class imbalance in training data"""
    fraud_count = sum(y_train)
    legitimate_count = len(y_train) - fraud_count
    fraud_rate = fraud_count / len(y_train)
    
    print(f"Legitimate: {legitimate_count:,}")
    print(f"Fraud: {fraud_count:,}")
    print(f"Fraud rate: {fraud_rate:.4f} ({fraud_rate*100:.4f}%)")
    print(f"Imbalance ratio: {legitimate_count/fraud_count:.2f}:1")
    
    return {
        'legitimate_count': legitimate_count,
        'fraud_count': fraud_count,
        'fraud_rate': fraud_rate
    }

def apply_smote(X_train, y_train, sampling_strategy=0.3, random_state=42):
    """Apply SMOTE to handle class imbalance"""
    print("\nBefore SMOTE:")
    analyze_imbalance(y_train)
    
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print("\nAfter SMOTE:")
    analyze_imbalance(y_resampled)
    
    return X_resampled, y_resampled

def get_sampling_recommendation(y_train):
    """Provide recommendation for sampling strategy"""
    fraud_rate = sum(y_train) / len(y_train)
    
    if fraud_rate < 0.01:
        strategy = 0.2
        justification = "Very severe imbalance - conservative SMOTE recommended"
    elif fraud_rate < 0.05:
        strategy = 0.3
        justification = "Moderate imbalance - standard SMOTE recommended"
    else:
        strategy = 0.5
        justification = "Mild imbalance - higher SMOTE ratio possible"
    
    return {
        'method': 'SMOTE',
        'sampling_strategy': strategy,
        'justification': justification
    }

print("Imbalance handling module loaded successfully")
