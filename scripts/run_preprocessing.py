import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from src.data_cleaning import clean_fraud_data, clean_credit_data
from src.geolocation import add_country_info, analyze_fraud_by_country
from src.feature_engineering import engineer_fraud_features
from src.imbalance_handling import get_sampling_recommendation

def main():
    print("=" * 60)
    print("TASK 1: DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    os.makedirs('data/processed', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    print("\n[1/5] Loading raw data...")
    fraud_raw = pd.read_csv('data/raw/Fraud_Data.csv')
    ip_mapping = pd.read_csv('data/raw/IpAddress_to_Country.csv')
    credit_raw = pd.read_csv('data/raw/creditcard.csv')
    
    print("\n[2/5] Cleaning data...")
    fraud_clean = clean_fraud_data(fraud_raw)
    credit_clean = clean_credit_data(credit_raw)
    
    fraud_clean.to_csv('data/processed/fraud_clean.csv', index=False)
    credit_clean.to_csv('data/processed/creditcard_clean.csv', index=False)
    
    print("\n[3/5] Adding geolocation...")
    fraud_with_country = add_country_info(fraud_clean, ip_mapping)
    country_analysis = analyze_fraud_by_country(fraud_with_country)
    country_analysis.to_csv('data/processed/country_fraud_analysis.csv')
    
    print("\n[4/5] Engineering features...")
    fraud_features = engineer_fraud_features(fraud_with_country)
    fraud_features.to_csv('data/processed/fraud_data_engineered.csv', index=False)
    
    print("\n[5/5] Analyzing class imbalance...")
    recommendation = get_sampling_recommendation(fraud_features['class'])
    
    print(f"\nRecommendation: {recommendation['method']}")
    print(f"Sampling strategy: {recommendation['sampling_strategy']}")
    print(f"Justification: {recommendation['justification']}")
    
    summary = {
        'fraud_data_shape': fraud_features.shape,
        'credit_data_shape': credit_clean.shape,
        'fraud_rate': fraud_features['class'].mean(),
        'credit_fraud_rate': credit_clean['Class'].mean(),
        'sampling_strategy': recommendation['sampling_strategy']
    }
    pd.DataFrame([summary]).to_csv('reports/task1_summary.csv', index=False)
    
    print("\n" + "=" * 60)
    print("TASK 1 COMPLETE!")
    print("=" * 60)

if __name__ == "__main__":
    main()
