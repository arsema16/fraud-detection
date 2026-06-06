import pandas as pd
import numpy as np

def clean_fraud_data(df):
    """Clean and preprocess the fraud transaction dataset"""
    df_clean = df.copy()
    
    df_clean['signup_time'] = pd.to_datetime(df_clean['signup_time'])
    df_clean['purchase_time'] = pd.to_datetime(df_clean['purchase_time'])
    
    critical_fields = ['user_id', 'purchase_value', 'class']
    df_clean = df_clean.dropna(subset=critical_fields)
    
    categorical_cols = ['source', 'browser', 'sex']
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna('Unknown')
    
    df_clean = df_clean.drop_duplicates()
    df_clean['age'] = df_clean['age'].astype(int).clip(0, 120)
    df_clean['purchase_value'] = df_clean['purchase_value'].astype(float).clip(lower=0)
    
    return df_clean

def clean_credit_data(df):
    """Clean and preprocess the credit card dataset"""
    df_clean = df.copy()
    df_clean = df_clean.drop_duplicates()
    df_clean['Amount'] = df_clean['Amount'].clip(lower=0)
    return df_clean

if __name__ == "__main__":
    print("Data cleaning module loaded successfully")
