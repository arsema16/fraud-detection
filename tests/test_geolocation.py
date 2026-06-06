import pytest
import pandas as pd
from src.geolocation import ip_to_int, add_country_info, analyze_fraud_by_country

class TestIPConversion:
    """Test cases for IP to integer conversion"""
    
    def test_ip_to_int_valid_ipv4(self):
        """Test valid IPv4 address conversion"""
        assert ip_to_int('192.168.1.1') == 3232235521
        assert ip_to_int('0.0.0.0') == 0
        assert ip_to_int('255.255.255.255') == 4294967295
        assert ip_to_int('10.0.0.1') == 167772161
    
    def test_ip_to_int_invalid_ip(self):
        """Test invalid IP addresses"""
        assert ip_to_int('invalid') is None
        assert ip_to_int('192.168.1') is None  # Only 3 parts
        assert ip_to_int('256.1.1.1') is None  # Invalid octet
        assert ip_to_int('') is None
        assert ip_to_int(None) is None
    
    def test_ip_to_int_with_dataframe(self):
        """Test IP conversion with pandas Series"""
        ip_series = pd.Series(['192.168.1.1', '10.0.0.1', 'invalid'])
        results = ip_series.apply(ip_to_int)
        
        assert results.iloc[0] == 3232235521
        assert results.iloc[1] == 167772161
        assert results.iloc[2] is None


class TestGeolocation:
    """Test cases for geolocation functionality"""
    
    def test_add_country_info_basic(self):
        """Test basic country addition"""
        # Create sample fraud data
        fraud_data = {
            'user_id': [1, 2],
            'purchase_time': ['2024-01-01 10:00:00', '2024-01-02 11:00:00'],
            'purchase_value': [100, 200],
            'ip_address': ['192.168.1.1', '10.0.0.1'],
            'class': [0, 1],
            'device_id': ['dev1', 'dev2'],
            'source': ['SEO', 'Ads'],
            'browser': ['Chrome', 'Safari'],
            'sex': ['M', 'F'],
            'age': [25, 30],
            'signup_time': ['2024-01-01 09:00:00', '2024-01-02 10:00:00']
        }
        fraud_df = pd.DataFrame(fraud_data)
        fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])
        fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])
        
        # Create sample IP mapping
        ip_mapping = pd.DataFrame({
            'lower_bound_ip_address': ['0.0.0.0', '10.0.0.0'],
            'upper_bound_ip_address': ['127.255.255.255', '10.255.255.255'],
            'country': ['Reserved', 'Private Network']
        })
        
        result = add_country_info(fraud_df, ip_mapping)
        
        assert result is not None
        assert 'country' in result.columns
        assert len(result) == 2
    
    def test_analyze_fraud_by_country(self):
        """Test country fraud analysis"""
        data = {
            'country': ['USA', 'USA', 'UK', 'UK', 'Canada'],
            'class': [0, 1, 0, 1, 0],
            'purchase_value': [100, 200, 150, 250, 300]
        }
        df = pd.DataFrame(data)
        
        result = analyze_fraud_by_country(df)
        
        assert result is not None
        assert 'transaction_count' in result.columns
        assert 'fraud_rate' in result.columns
        assert 'avg_purchase_value' in result.columns
        assert len(result) == 3  # 3 unique countries


if __name__ == "__main__":
    pytest.main([__file__, '-v'])
