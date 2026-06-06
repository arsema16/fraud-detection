import pandas as pd
import numpy as np

def ip_to_int(ip_address):
    """
    Convert IP address string to integer format using Python's ipaddress module.
    This is the most reliable method.
    
    Example: '192.168.1.1' -> 3232235521
    """
    if pd.isna(ip_address) or ip_address == '':
        return None
    
    try:
        import ipaddress
        # Convert to integer using IPv4Address
        return int(ipaddress.IPv4Address(str(ip_address)))
    except (ValueError, TypeError, ImportError):
        # Fallback manual calculation if ipaddress not available
        try:
            parts = str(ip_address).split('.')
            if len(parts) != 4:
                return None
            
            octets = []
            for part in parts:
                octet = int(part)
                if octet < 0 or octet > 255:
                    return None
                octets.append(octet)
            
            # Manual calculation: (a << 24) | (b << 16) | (c << 8) | d
            # This is the correct bitwise operation
            return (octets[0] << 24) | (octets[1] << 16) | (octets[2] << 8) | octets[3]
        except (ValueError, TypeError):
            return None


def add_country_info(fraud_df, ip_mapping_df):
    """
    Add country information to fraud transactions based on IP address range lookup
    """
    df_with_country = fraud_df.copy()
    
    print(f"Processing {len(df_with_country):,} transactions...")
    
    # Convert IP to integer
    df_with_country['ip_int'] = df_with_country['ip_address'].apply(ip_to_int)
    
    # Count successful conversions
    successful_conversions = df_with_country['ip_int'].notna().sum()
    print(f"Successfully converted {successful_conversions:,}/{len(df_with_country):,} IPs")
    
    # Prepare IP mapping
    ip_mapping = ip_mapping_df.copy()
    ip_mapping['lower_bound_int'] = ip_mapping['lower_bound_ip_address'].apply(ip_to_int)
    ip_mapping['upper_bound_int'] = ip_mapping['upper_bound_ip_address'].apply(ip_to_int)
    
    # Remove rows with conversion failures
    ip_mapping = ip_mapping.dropna(subset=['lower_bound_int', 'upper_bound_int'])
    
    # Create a function to find country for each IP using binary search for efficiency
    def find_country(ip_int_value):
        if pd.isna(ip_int_value):
            return 'Unknown'
        
        # Binary search through sorted ranges
        for _, row in ip_mapping.iterrows():
            if row['lower_bound_int'] <= ip_int_value <= row['upper_bound_int']:
                return row['country']
        return 'Unknown'
    
    # Apply the lookup
    df_with_country['country'] = df_with_country['ip_int'].apply(find_country)
    
    # Drop intermediate column
    df_with_country = df_with_country.drop('ip_int', axis=1)
    
    # Print country distribution
    country_counts = df_with_country['country'].value_counts()
    print("\nTop 10 countries by transaction volume:")
    for country, count in country_counts.head(10).items():
        print(f"  {country}: {count:,}")
    
    return df_with_country


def analyze_fraud_by_country(df):
    """
    Analyze fraud rates by country
    """
    if 'country' not in df.columns:
        print("Warning: 'country' column not found. Run add_country_info first.")
        return pd.DataFrame()
    
    summary = df.groupby('country').agg({
        'class': ['count', 'mean'],
        'purchase_value': 'mean'
    }).round(4)
    
    summary.columns = ['transaction_count', 'fraud_rate', 'avg_purchase_value']
    summary = summary.sort_values('fraud_rate', ascending=False)
    
    print("\nTop 10 countries by fraud rate:")
    for idx, row in summary.head(10).iterrows():
        print(f"  {idx}: {row['fraud_rate']*100:.2f}% fraud rate ({row['transaction_count']:,} transactions)")
    
    return summary


# Quick test function
if __name__ == "__main__":
    # Test IP conversion
    test_ips = [
        ('192.168.1.1', 3232235521),
        ('0.0.0.0', 0),
        ('255.255.255.255', 4294967295),
        ('10.0.0.1', 167772161),
        ('8.8.8.8', 134744072),
        ('127.0.0.1', 2130706433),
    ]
    
    print("Testing IP conversion:")
    all_passed = True
    for ip, expected in test_ips:
        result = ip_to_int(ip)
        passed = result == expected
        all_passed = all_passed and passed
        status = "✓" if passed else "✗"
        print(f"  {status} {ip} -> {result} (expected: {expected})")
    
    if all_passed:
        print("\n✓ All IP conversion tests passed!")
    else:
        print("\n✗ Some tests failed!")
