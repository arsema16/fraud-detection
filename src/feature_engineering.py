import pandas as pd
import numpy as np
from datetime import timedelta

def engineer_fraud_features(df):
    """Engineer features for fraud detection"""
    df_fe = df.copy()
    
    df_fe['purchase_hour'] = df_fe['purchase_time'].dt.hour
    df_fe['purchase_day_of_week'] = df_fe['purchase_time'].dt.dayofweek
    df_fe['purchase_weekend'] = (df_fe['purchase_day_of_week'] >= 5).astype(int)
    
    df_fe['time_since_signup'] = (
        df_fe['purchase_time'] - df_fe['signup_time']
    ).dt.total_seconds() / 3600
    
    df_fe = df_fe.sort_values(['user_id', 'purchase_time'])
    df_fe['hours_since_prev_purchase'] = df_fe.groupby('user_id')['purchase_time'].diff().dt.total_seconds() / 3600
    df_fe['hours_since_prev_purchase'] = df_fe['hours_since_prev_purchase'].fillna(-1)
    
    df_fe['transactions_24h'] = df_fe.groupby('user_id')['purchase_time'].transform(
        lambda x: [
            sum(1 for t in x[:i] if t > x.iloc[i] - pd.Timedelta(hours=24)) 
            for i in range(len(x))
        ]
    )
    
    user_stats = df_fe.groupby('user_id').agg({
        'purchase_value': ['mean', 'std', 'count', 'sum'],
        'class': 'mean'
    })
    user_stats.columns = ['user_avg_amount', 'user_std_amount', 'user_transaction_count', 
                          'user_total_amount', 'user_fraud_rate']
    user_stats = user_stats.fillna(0)
    df_fe = df_fe.merge(user_stats, on='user_id', how='left')
    df_fe['user_std_amount'] = df_fe['user_std_amount'].replace([np.inf, -np.inf], 0)
    
    device_stats = df_fe.groupby('device_id').agg({
        'user_id': 'nunique',
        'class': 'mean'
    })
    device_stats.columns = ['users_per_device', 'device_fraud_rate']
    df_fe = df_fe.merge(device_stats, on='device_id', how='left')
    
    country_risk = df_fe.groupby('country')['class'].mean().to_dict()
    df_fe['country_risk_score'] = df_fe['country'].map(country_risk).fillna(0)
    
    df_fe['age_group'] = pd.cut(
        df_fe['age'], 
        bins=[0, 18, 25, 35, 50, 65, 120],
        labels=['<18', '18-25', '26-35', '36-50', '51-65', '65+']
    )
    
    drop_cols = ['signup_time', 'purchase_time', 'ip_address']
    df_fe = df_fe.drop([col for col in drop_cols if col in df_fe.columns], axis=1)
    
    return df_fe

print("Feature engineering module loaded successfully")
