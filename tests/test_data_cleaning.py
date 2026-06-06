import pytest
import pandas as pd
import numpy as np
from src.data_cleaning import clean_fraud_data, clean_credit_data

class TestFraudDataCleaning:
    """Test cases for fraud data cleaning functions"""
    
    def test_clean_fraud_data_basic(self):
        """Test basic cleaning functionality"""
        # Create sample data
        data = {
            'user_id': [1, 2, 3],
            'signup_time': ['2024-01-01 10:00:00', '2024-01-02 11:00:00', '2024-01-03 12:00:00'],
            'purchase_time': ['2024-01-01 10:30:00', '2024-01-02 11:30:00', '2024-01-03 12:30:00'],
            'purchase_value': [100, 200, 300],
            'device_id': ['dev1', 'dev2', 'dev3'],
            'source': ['SEO', 'Ads', 'Direct'],
            'browser': ['Chrome', 'Safari', 'Firefox'],
            'sex': ['M', 'F', 'M'],
            'age': [25, 30, 35],
            'ip_address': ['192.168.1.1', '10.0.0.1', '172.16.0.1'],
            'class': [0, 0, 1]
        }
        df = pd.DataFrame(data)
        
        # Convert to datetime
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        
        result = clean_fraud_data(df)
        
        assert result is not None
        assert len(result) == 3
        assert 'signup_time' in result.columns
        assert 'purchase_time' in result.columns
    
    def test_clean_fraud_data_handles_missing_values(self):
        """Test that missing values are handled correctly"""
        data = {
            'user_id': [1, 2],
            'signup_time': ['2024-01-01 10:00:00', '2024-01-02 11:00:00'],
            'purchase_time': ['2024-01-01 10:30:00', '2024-01-02 11:30:00'],
            'purchase_value': [100, 200],
            'device_id': ['dev1', 'dev2'],
            'source': ['SEO', None],  # Missing value
            'browser': ['Chrome', 'Safari'],
            'sex': ['M', None],  # Missing value
            'age': [25, 30],
            'ip_address': ['192.168.1.1', '10.0.0.1'],
            'class': [0, 0]
        }
        df = pd.DataFrame(data)
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        
        result = clean_fraud_data(df)
        
        # Missing values should be filled with 'Unknown'
        assert result['source'].iloc[1] == 'Unknown'
        assert result['sex'].iloc[1] == 'Unknown'
    
    def test_clean_fraud_data_removes_duplicates(self):
        """Test duplicate removal"""
        data = {
            'user_id': [1, 1, 2],
            'signup_time': ['2024-01-01 10:00:00', '2024-01-01 10:00:00', '2024-01-02 11:00:00'],
            'purchase_time': ['2024-01-01 10:30:00', '2024-01-01 10:30:00', '2024-01-02 11:30:00'],
            'purchase_value': [100, 100, 200],
            'device_id': ['dev1', 'dev1', 'dev2'],
            'source': ['SEO', 'SEO', 'Ads'],
            'browser': ['Chrome', 'Chrome', 'Safari'],
            'sex': ['M', 'M', 'F'],
            'age': [25, 25, 30],
            'ip_address': ['192.168.1.1', '192.168.1.1', '10.0.0.1'],
            'class': [0, 0, 0]
        }
        df = pd.DataFrame(data)
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        
        result = clean_fraud_data(df)
        
        # Should have 2 rows after removing duplicate
        assert len(result) == 2
    
    def test_clean_fraud_data_clips_invalid_values(self):
        """Test that invalid values are clipped"""
        data = {
            'user_id': [1, 2],
            'signup_time': ['2024-01-01 10:00:00', '2024-01-02 11:00:00'],
            'purchase_time': ['2024-01-01 10:30:00', '2024-01-02 11:30:00'],
            'purchase_value': [-100, 200],  # Negative value
            'device_id': ['dev1', 'dev2'],
            'source': ['SEO', 'Ads'],
            'browser': ['Chrome', 'Safari'],
            'sex': ['M', 'F'],
            'age': [-5, 150],  # Invalid ages
            'ip_address': ['192.168.1.1', '10.0.0.1'],
            'class': [0, 0]
        }
        df = pd.DataFrame(data)
        df['signup_time'] = pd.to_datetime(df['signup_time'])
        df['purchase_time'] = pd.to_datetime(df['purchase_time'])
        
        result = clean_fraud_data(df)
        
        # Negative values should be clipped to 0
        assert result['purchase_value'].iloc[0] == 0
        # Age should be clipped to 0-120 range
        assert result['age'].iloc[0] == 0
        assert result['age'].iloc[1] == 120


class TestCreditDataCleaning:
    """Test cases for credit data cleaning functions"""
    
    def test_clean_credit_data_basic(self):
        """Test basic credit data cleaning"""
        data = {
            'Time': [0, 1, 2],
            'V1': [-1.5, 0.5, 1.2],
            'V2': [-0.5, 1.2, -0.8],
            'Amount': [100, 200, 300],
            'Class': [0, 0, 1]
        }
        df = pd.DataFrame(data)
        
        result = clean_credit_data(df)
        
        assert result is not None
        assert len(result) == 3
        assert 'Class' in result.columns
    
    def test_clean_credit_data_removes_duplicates(self):
        """Test duplicate removal in credit data"""
        data = {
            'Time': [0, 0, 1],
            'V1': [-1.5, -1.5, 0.5],
            'V2': [-0.5, -0.5, 1.2],
            'Amount': [100, 100, 200],
            'Class': [0, 0, 0]
        }
        df = pd.DataFrame(data)
        
        result = clean_credit_data(df)
        
        assert len(result) == 2
    
    def test_clean_credit_data_clips_negative_amounts(self):
        """Test that negative amounts are clipped"""
        data = {
            'Time': [0, 1],
            'V1': [-1.5, 0.5],
            'V2': [-0.5, 1.2],
            'Amount': [-100, 200],
            'Class': [0, 0]
        }
        df = pd.DataFrame(data)
        
        result = clean_credit_data(df)
        
        assert result['Amount'].iloc[0] == 0
        assert result['Amount'].iloc[1] == 200


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
