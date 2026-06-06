import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.feature_engineering import engineer_fraud_features

class TestFeatureEngineering:
    """Test cases for feature engineering"""
    
    def test_engineer_fraud_features_basic(self):
        """Test basic feature engineering"""
        # Create sample data
        base_time = datetime.now()
        data = {
            'user_id': [1, 1, 2],
            'signup_time': [base_time - timedelta(days=1)] * 3,
            'purchase_time': [
                base_time - timedelta(hours=2),
                base_time - timedelta(hours=1),
                base_time
            ],
            'purchase_value': [100, 150, 200],
            'device_id': ['dev1', 'dev1', 'dev2'],
            'source': ['SEO', 'SEO', 'Ads'],
            'browser': ['Chrome', 'Chrome', 'Safari'],
            'sex': ['M', 'M', 'F'],
            'age': [25, 25, 30],
            'ip_address': ['192.168.1.1', '192.168.1.1', '10.0.0.1'],
            'class': [0, 0, 1],
            'country': ['USA', 'USA', 'UK']
        }
        df = pd.DataFrame(data)
        
        result = engineer_fraud_features(df)
        
        # Check that engineered features exist
        expected_features = [
            'purchase_hour', 'purchase_day_of_week', 'purchase_weekend',
            'time_since_signup', 'hours_since_prev_purchase', 'transactions_24h',
            'user_avg_amount', 'user_std_amount', 'user_transaction_count',
            'user_total_amount', 'user_fraud_rate', 'users_per_device',
            'device_fraud_rate', 'country_risk_score', 'age_group'
        ]
        
        for feature in expected_features:
            assert feature in result.columns, f"Missing feature: {feature}"
    
    def test_time_since_signup_calculation(self):
        """Test time since signup calculation"""
        signup = datetime(2024, 1, 1, 10, 0, 0)
        purchase = datetime(2024, 1, 1, 15, 0, 0)  # 5 hours later
        
        data = {
            'user_id': [1],
            'signup_time': [signup],
            'purchase_time': [purchase],
            'purchase_value': [100],
            'device_id': ['dev1'],
            'source': ['SEO'],
            'browser': ['Chrome'],
            'sex': ['M'],
            'age': [25],
            'ip_address': ['192.168.1.1'],
            'class': [0],
            'country': ['USA']
        }
        df = pd.DataFrame(data)
        
        result = engineer_fraud_features(df)
        
        # Time difference should be approximately 5 hours
        assert abs(result['time_since_signup'].iloc[0] - 5) < 0.1
    
    def test_transaction_velocity_features(self):
        """Test transaction velocity calculations"""
        base_time = datetime.now()
        data = {
            'user_id': [1, 1, 1],
            'signup_time': [base_time - timedelta(days=1)] * 3,
            'purchase_time': [
                base_time - timedelta(hours=24),
                base_time - timedelta(hours=12),
                base_time
            ],
            'purchase_value': [100, 150, 200],
            'device_id': ['dev1', 'dev1', 'dev1'],
            'source': ['SEO', 'SEO', 'SEO'],
            'browser': ['Chrome', 'Chrome', 'Chrome'],
            'sex': ['M', 'M', 'M'],
            'age': [25, 25, 25],
            'ip_address': ['192.168.1.1', '192.168.1.1', '192.168.1.1'],
            'class': [0, 0, 0],
            'country': ['USA', 'USA', 'USA']
        }
        df = pd.DataFrame(data)
        
        result = engineer_fraud_features(df)
        
        # Should have 3 transactions
        assert len(result) == 3
        # Check that hours_since_prev_purchase is calculated
        assert 'hours_since_prev_purchase' in result.columns
        # First transaction should have -1 (no previous)
        assert result['hours_since_prev_purchase'].iloc[0] == -1
    
    def test_user_statistics_calculation(self):
        """Test user statistics aggregation"""
        data = {
            'user_id': [1, 1, 2],
            'signup_time': [datetime.now()] * 3,
            'purchase_time': [datetime.now()] * 3,
            'purchase_value': [100, 200, 300],
            'device_id': ['dev1', 'dev1', 'dev2'],
            'source': ['SEO', 'SEO', 'Ads'],
            'browser': ['Chrome', 'Chrome', 'Safari'],
            'sex': ['M', 'M', 'F'],
            'age': [25, 25, 30],
            'ip_address': ['192.168.1.1', '192.168.1.1', '10.0.0.1'],
            'class': [0, 1, 0],
            'country': ['USA', 'USA', 'UK']
        }
        df = pd.DataFrame(data)
        
        result = engineer_fraud_features(df)
        
        # User 1 should have avg amount of 150
        user1_avg = result[result['user_id'] == 1]['user_avg_amount'].iloc[0]
        assert user1_avg == 150
        
        # User 1 should have transaction count of 2
        user1_count = result[result['user_id'] == 1]['user_transaction_count'].iloc[0]
        assert user1_count == 2
        
        # User 1 fraud rate should be 0.5 (1 out of 2)
        user1_fraud_rate = result[result['user_id'] == 1]['user_fraud_rate'].iloc[0]
        assert user1_fraud_rate == 0.5


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
