"""
Complete integrated preprocessing pipeline for fraud detection.
This script sequences all preprocessing steps in the correct order.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
import joblib

class FraudPreprocessingPipeline:
    """
    Integrated preprocessing pipeline for e-commerce fraud data.
    
    Steps in order:
    1. Load raw data
    2. Clean data (missing values, duplicates, invalid values)
    3. Add geolocation (IP to country mapping)
    4. Engineer features (temporal, behavioral, derived)
    5. Split data (stratified, 80-20)
    6. Transform features (scaling for numeric, encoding for categorical)
    7. Handle imbalance (SMOTE on training set only)
    """
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.preprocessor = None
        self.feature_names = None
        
    def load_data(self, fraud_path, ip_path, credit_path=None):
        """Load raw data files"""
        print("Step 1: Loading raw data...")
        self.fraud_raw = pd.read_csv(fraud_path)
        self.ip_mapping = pd.read_csv(ip_path)
        if credit_path:
            self.credit_raw = pd.read_csv(credit_path)
        print(f"  - Fraud data: {self.fraud_raw.shape}")
        print(f"  - IP mapping: {self.ip_mapping.shape}")
        return self
    
    def clean_data(self):
        """Step 2: Clean data - handle missing, duplicates, invalid values"""
        print("\nStep 2: Cleaning data...")
        df = self.fraud_raw.copy()
        
        # Convert timestamps
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        
        # Handle missing values - fill categorical with 'Unknown'
        categorical_cols = ['source', 'browser', 'sex']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna('Unknown')
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Clip invalid values
        df['age'] = df['age'].clip(0, 120)
        df['purchase_value'] = df['purchase_value'].clip(lower=0)
        
        self.fraud_clean = df
        print(f"  - Cleaned shape: {self.fraud_clean.shape}")
        return self
    
    def add_geolocation(self):
        """Step 3: Add geolocation from IP to country mapping"""
        print("\nStep 3: Adding geolocation...")
        df = self.fraud_clean.copy()
        
        def ip_to_int(ip):
            if pd.isna(ip):
                return None
            try:
                parts = str(ip).split('.')
                if len(parts) != 4:
                    return None
                octets = [int(p) for p in parts]
                if any(o < 0 or o > 255 for o in octets):
                    return None
                return (octets[0] << 24) | (octets[1] << 16) | (octets[2] << 8) | octets[3]
            except:
                return None
        
        # Convert IPs to integers
        df['ip_int'] = df['ip_address'].apply(ip_to_int)
        
        # Prepare IP mapping
        ip_map = self.ip_mapping.copy()
        ip_map['lower_int'] = ip_map['lower_bound_ip_address'].apply(ip_to_int)
        ip_map['upper_int'] = ip_map['upper_bound_ip_address'].apply(ip_to_int)
        ip_map = ip_map.dropna(subset=['lower_int', 'upper_int'])
        
        # Find country for each IP
        def find_country(ip_int_val):
            if pd.isna(ip_int_val):
                return 'Unknown'
            for _, row in ip_map.iterrows():
                if row['lower_int'] <= ip_int_val <= row['upper_int']:
                    return row['country']
            return 'Unknown'
        
        df['country'] = df['ip_int'].apply(find_country)
        df = df.drop('ip_int', axis=1)
        
        self.fraud_with_country = df
        print(f"  - Countries mapped: {df['country'].nunique()}")
        return self
    
    def engineer_features(self):
        """Step 4: Engineer features for fraud detection"""
        print("\nStep 4: Engineering features...")
        df = self.fraud_with_country.copy()
        
        # Temporal features
        df['purchase_hour'] = df['purchase_time'].dt.hour
        df['purchase_day_of_week'] = df['purchase_time'].dt.dayofweek
        df['purchase_weekend'] = (df['purchase_day_of_week'] >= 5).astype(int)
        df['time_since_signup'] = (df['purchase_time'] - df['signup_time']).dt.total_seconds() / 3600
        
        # Sort for velocity features
        df = df.sort_values(['user_id', 'purchase_time'])
        df['hours_since_prev_purchase'] = df.groupby('user_id')['purchase_time'].diff().dt.total_seconds() / 3600
        df['hours_since_prev_purchase'] = df['hours_since_prev_purchase'].fillna(-1)
        
        # Transaction velocity (last 24 hours)
        df['transactions_24h'] = df.groupby('user_id')['purchase_time'].transform(
            lambda x: [sum(1 for t in x[:i] if t > x.iloc[i] - pd.Timedelta(hours=24)) for i in range(len(x))]
        )
        
        # User statistics
        user_stats = df.groupby('user_id').agg({
            'purchase_value': ['mean', 'std', 'count', 'sum'],
            'class': 'mean'
        })
        user_stats.columns = ['user_avg_amount', 'user_std_amount', 'user_transaction_count',
                              'user_total_amount', 'user_fraud_rate']
        user_stats = user_stats.fillna(0)
        df = df.merge(user_stats, on='user_id', how='left')
        df['user_std_amount'] = df['user_std_amount'].replace([np.inf, -np.inf], 0)
        
        # Device statistics
        device_stats = df.groupby('device_id').agg({
            'user_id': 'nunique',
            'class': 'mean'
        })
        device_stats.columns = ['users_per_device', 'device_fraud_rate']
        df = df.merge(device_stats, on='device_id', how='left')
        
        # Country risk score
        country_risk = df.groupby('country')['class'].mean().to_dict()
        df['country_risk_score'] = df['country'].map(country_risk).fillna(0)
        
        # Age groups
        df['age_group'] = pd.cut(df['age'], bins=[0, 18, 25, 35, 50, 65, 120],
                                  labels=['<18', '18-25', '26-35', '36-50', '51-65', '65+'])
        
        # Derived features
        df['value_per_hour'] = df['purchase_value'] / (df['time_since_signup'] + 1)
        df['is_common_browser'] = df['browser'].isin(['Chrome', 'Safari', 'Firefox']).astype(int)
        
        # Time category
        def time_cat(hour):
            if hour < 6: return 'night'
            elif hour < 12: return 'morning'
            elif hour < 18: return 'afternoon'
            else: return 'evening'
        df['time_category'] = df['purchase_hour'].apply(time_cat)
        
        # Drop intermediate columns
        drop_cols = ['signup_time', 'purchase_time', 'ip_address']
        df = df.drop([c for c in drop_cols if c in df.columns], axis=1)
        
        self.fraud_features = df
        print(f"  - Features engineered: {len(df.columns)}")
        return self
    
    def split_data(self, test_size=0.2):
        """Step 5: Split data with stratification to preserve class distribution"""
        print("\nStep 5: Splitting data...")
        X = self.fraud_features.drop('class', axis=1)
        y = self.fraud_features['class']
        
        # Stratified split - CRITICAL for imbalanced data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        
        print(f"  - Train shape: {X_train.shape}")
        print(f"  - Test shape: {X_test.shape}")
        print(f"  - Train fraud rate: {y_train.mean():.4f}")
        print(f"  - Test fraud rate: {y_test.mean():.4f}")
        print(f"  - Stratified split preserves class distribution")
        return self
    
    def transform_features(self):
        """Step 6: Scale numerical features and encode categorical features"""
        print("\nStep 6: Transforming features...")
        
        # Define feature types
        numeric_features = [
            'purchase_value', 'age', 'time_since_signup', 'hours_since_prev_purchase',
            'transactions_24h', 'user_avg_amount', 'user_std_amount', 'user_transaction_count',
            'user_total_amount', 'user_fraud_rate', 'users_per_device', 'device_fraud_rate',
            'country_risk_score', 'purchase_hour', 'purchase_day_of_week', 'value_per_hour'
        ]
        
        categorical_features = ['source', 'browser', 'sex', 'country', 'time_category', 'age_group']
        
        # Filter to existing columns
        numeric_features = [f for f in numeric_features if f in self.X_train.columns]
        categorical_features = [c for c in categorical_features if c in self.X_train.columns]
        
        # Create preprocessor
        numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Fit on training, transform both
        self.X_train_processed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_processed = self.preprocessor.transform(self.X_test)
        
        # Get feature names
        cat_feature_names = []
        for col in categorical_features:
            unique_vals = self.X_train[col].dropna().unique()
            cat_feature_names.extend([f"{col}_{v}" for v in unique_vals])
        
        self.feature_names = numeric_features + cat_feature_names
        
        print(f"  - Training features shape: {self.X_train_processed.shape}")
        print(f"  - Test features shape: {self.X_test_processed.shape}")
        return self
    
    def handle_imbalance(self, sampling_strategy=0.3):
        """Step 7: Apply SMOTE on training set ONLY - NEVER on test set"""
        print("\nStep 7: Handling class imbalance with SMOTE...")
        
        # Justification for SMOTE over undersampling:
        # 1. Preserves all legitimate transaction data (no information loss)
        # 2. Creates synthetic fraud examples using k-nearest neighbors
        # 3. Prevents overfitting compared to random undersampling
        # 4. Standard approach in fraud detection literature
        
        original_fraud = self.y_train.sum()
        original_legit = len(self.y_train) - original_fraud
        
        # Apply SMOTE - ONLY on training data
        smote = SMOTE(sampling_strategy=sampling_strategy, random_state=self.random_state)
        self.X_train_balanced, self.y_train_balanced = smote.fit_resample(
            self.X_train_processed, self.y_train
        )
        
        new_fraud = self.y_train_balanced.sum()
        new_legit = len(self.y_train_balanced) - new_fraud
        
        print(f"  - Before SMOTE: Legit={original_legit:,}, Fraud={original_fraud:,}")
        print(f"  - After SMOTE: Legit={new_legit:,}, Fraud={new_fraud:,}")
        print(f"  - Sampling strategy: {sampling_strategy} (target ratio)")
        print(f"  - SMOTE applied ONLY to training set")
        print(f"  - Test set remains imbalanced for realistic evaluation")
        return self
    
    def save_artifacts(self, model_path='../models/fraud_preprocessor.pkl'):
        """Save preprocessor and feature names for later use"""
        print("\nStep 8: Saving artifacts...")
        artifacts = {
            'preprocessor': self.preprocessor,
            'feature_names': self.feature_names,
            'random_state': self.random_state
        }
        joblib.dump(artifacts, model_path)
        print(f"  - Preprocessor saved to {model_path}")
        return self
    
    def run_complete_pipeline(self, fraud_path, ip_path, test_size=0.2, sampling_strategy=0.3):
        """Run all steps in sequence"""
        print("="*60)
        print("COMPLETE PREPROCESSING PIPELINE")
        print("="*60)
        
        self.load_data(fraud_path, ip_path)
        self.clean_data()
        self.add_geolocation()
        self.engineer_features()
        self.split_data(test_size)
        self.transform_features()
        self.handle_imbalance(sampling_strategy)
        self.save_artifacts()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE")
        print("="*60)
        
        return self


# Usage
if __name__ == "__main__":
    pipeline = FraudPreprocessingPipeline(random_state=42)
    pipeline.run_complete_pipeline(
        fraud_path='../data/raw/Fraud_Data.csv',
        ip_path='../data/raw/IpAddress_to_Country.csv',
        test_size=0.2,
        sampling_strategy=0.3
    )