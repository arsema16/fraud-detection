import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

@pytest.fixture
def sample_fraud_data():
    """Provide sample fraud data for testing"""
    base_time = datetime.now()
    data = {
        'user_id': [1, 1, 2, 3],
        'signup_time': [base_time - timedelta(days=1)] * 4,
        'purchase_time': [
            base_time - timedelta(hours=24),
            base_time - timedelta(hours=12),
            base_time,
            base_time - timedelta(hours=1)
        ],
        'purchase_value': [100, 150, 200, 50],
        'device_id': ['dev1', 'dev1', 'dev2', 'dev3'],
        'source': ['SEO', 'SEO', 'Ads', 'Direct'],
        'browser': ['Chrome', 'Chrome', 'Safari', 'Firefox'],
        'sex': ['M', 'M', 'F', 'M'],
        'age': [25, 25, 30, 45],
        'ip_address': ['192.168.1.1', '192.168.1.1', '10.0.0.1', '172.16.0.1'],
        'class': [0, 0, 1, 0],
        'country': ['USA', 'USA', 'UK', 'Canada']
    }
    df = pd.DataFrame(data)
    df['signup_time'] = pd.to_datetime(df['signup_time'])
    df['purchase_time'] = pd.to_datetime(df['purchase_time'])
    return df

@pytest.fixture
def sample_ip_mapping():
    """Provide sample IP mapping data"""
    return pd.DataFrame({
        'lower_bound_ip_address': ['0.0.0.0', '10.0.0.0', '172.16.0.0', '192.168.0.0'],
        'upper_bound_ip_address': ['127.255.255.255', '10.255.255.255', '172.31.255.255', '192.168.255.255'],
        'country': ['Reserved', 'Private', 'Private', 'Private']
    })

@pytest.fixture
def sample_credit_data():
    """Provide sample credit card data for testing"""
    data = {
        'Time': [0, 1, 2, 3],
        'V1': [-1.5, 0.5, 1.2, -0.3],
        'V2': [-0.5, 1.2, -0.8, 0.1],
        'V3': [0.2, -0.1, 0.5, -0.2],
        'Amount': [100, 200, 300, 50],
        'Class': [0, 0, 1, 0]
    }
    return pd.DataFrame(data)

@pytest.fixture
def imbalanced_labels():
    """Provide imbalanced labels for testing"""
    return np.array([0] * 950 + [1] * 50)
