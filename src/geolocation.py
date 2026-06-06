import pandas as pd

def ip_to_int(ip_address):
    """
    Convert IP address string to integer format
    Example: '192.168.1.1' -> 3232235521
    """
    if pd.isna(ip_address) or ip_address == '':
        return None
    
    try:
        parts = str(ip_address).split('.')
        if len(parts) != 4:
            return None
        
        # Validate each octet is between 0 and 255
        octets = []
        for part in parts:
            octet = int(part)
            if octet < 0 or octet > 255:
                return None
            octets.append(octet)
        
        # Correct conversion: (first << 24) | (second << 16) | (third << 8) | fourth
        ip_int = (octets[0] << 24) | (octets[1] << 16) | (octets[2] << 8) | octets[3]
        return ip_int
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
    
    # Sort for merge_asof
    ip_mapping = ip_mapping.sort_values('lower_bound_int')
    df_with_country = df_with_country.sort_values('ip_int')
    
    # Perform range-based lookup
    df_with_country = pd.merge_asof(
        df_with_country,
        ip_mapping[['lower_bound_int', 'upper_bound_int', 'country']],
        left_on='ip_int',
        right_on='lower_bound_int',
        direction='backward'
    )
    
    # Filter where ip_int falls within range
    mask = (df_with_country['ip_int'] >= df_with_country['lower_bound_int']) & \
           (df_with_country['ip_int'] <= df_with_country['upper_bound_int'])
    
    df_with_country.loc[~mask, 'country'] = 'Unknown'
    
    # Drop intermediate columns
    df_with_country = df_with_country.drop(['ip_int', 'lower_bound_int', 'upper_bound_int'], axis=1)
    
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
    ]
    
    print("Testing IP conversion:")
    for ip, expected in test_ips:
        result = ip_to_int(ip)
        status = "✓" if result == expected else "✗"
        print(f"  {status} {ip} -> {result} (expected: {expected})")
