import pandas as pd
import os

# Create directories
os.makedirs('data/processed', exist_ok=True)

# Load raw data
print("Loading raw data...")
fraud_df = pd.read_csv('data/raw/Fraud_Data.csv')
credit_df = pd.read_csv('data/raw/creditcard.csv')
ip_df = pd.read_csv('data/raw/IpAddress_to_Country.csv')

print(f"Fraud data shape: {fraud_df.shape}")
print(f"Credit data shape: {credit_df.shape}")
print(f"IP data shape: {ip_df.shape}")

# Simple cleaning for fraud data
print("\nCleaning fraud data...")
fraud_clean = fraud_df.copy()
fraud_clean['signup_time'] = pd.to_datetime(fraud_clean['signup_time'])
fraud_clean['purchase_time'] = pd.to_datetime(fraud_clean['purchase_time'])
fraud_clean = fraud_clean.drop_duplicates()
fraud_clean['age'] = fraud_clean['age'].clip(0, 120)
fraud_clean['purchase_value'] = fraud_clean['purchase_value'].clip(lower=0)

# Fill missing
for col in ['source', 'browser', 'sex']:
    if col in fraud_clean.columns:
        fraud_clean[col] = fraud_clean[col].fillna('Unknown')

# Simple cleaning for credit data
print("Cleaning credit data...")
credit_clean = credit_df.copy()
credit_clean = credit_clean.drop_duplicates()
credit_clean['Amount'] = credit_clean['Amount'].clip(lower=0)

# Save cleaned data
fraud_clean.to_csv('data/processed/fraud_clean.csv', index=False)
credit_clean.to_csv('data/processed/creditcard_clean.csv', index=False)

print(f"\nSaved cleaned data:")
print(f"  - fraud_clean.csv: {fraud_clean.shape}")
print(f"  - creditcard_clean.csv: {credit_clean.shape}")

# Simple feature engineering
print("\nEngineering features...")
df = fraud_clean.copy()

# Time features
df['purchase_hour'] = df['purchase_time'].dt.hour
df['purchase_day'] = df['purchase_time'].dt.dayofweek
df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600

# User stats
user_stats = df.groupby('user_id').agg({
    'purchase_value': ['mean', 'count'],
    'class': 'mean'
})
user_stats.columns = ['user_avg_amount', 'user_transaction_count', 'user_fraud_rate']
user_stats = user_stats.fillna(0)
df = df.merge(user_stats, on='user_id', how='left')

# Drop original time columns
df = df.drop(['signup_time', 'purchase_time', 'ip_address'], axis=1)

df.to_csv('data/processed/fraud_data_engineered.csv', index=False)
print(f"Saved fraud_data_engineered.csv: {df.shape}")

# Print statistics
print("\n" + "="*50)
print("DATA STATISTICS")
print("="*50)
print(f"Fraud data shape: {df.shape}")
print(f"Fraud rate: {df['class'].mean()*100:.4f}%")
print(f"Legit: {(df['class']==0).sum():,}")
print(f"Fraud: {(df['class']==1).sum():,}")
print(f"Ratio: {(df['class']==0).sum()/(df['class']==1).sum():.2f}:1")

print("\nFraud rate by source:")
source_fraud = df.groupby('source')['class'].mean().sort_values(ascending=False)
for source, rate in source_fraud.head(5).items():
    print(f"  {source}: {rate*100:.2f}%")

print("\nPreprocessing complete!")