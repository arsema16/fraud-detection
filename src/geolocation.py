import pandas as pd

def ip_to_int(ip_address):
    """Convert IP address string to integer format"""
    if pd.isna(ip_address) or ip_address == '':
        return None
    
    try:
        parts = str(ip_address).split('.')
        if len(parts) != 4:
            return None
        
        ip_int = 0
        for i, part in enumerate(parts):
            ip_int += int(part) << (24 - 8 * i)
        return ip_int
    except (ValueError, TypeError):
        return None

def add_country_info(fraud_df, ip_mapping_df):
    """Add country information to fraud transactions"""
    df_with_country = fraud_df.copy()
    
    df_with_country['ip_int'] = df_with_country['ip_address'].apply(ip_to_int)
    
    ip_mapping = ip_mapping_df.copy()
    ip_mapping['lower_bound_int'] = ip_mapping['lower_bound_ip_address'].apply(ip_to_int)
    ip_mapping['upper_bound_int'] = ip_mapping['upper_bound_ip_address'].apply(ip_to_int)
    ip_mapping = ip_mapping.dropna(subset=['lower_bound_int', 'upper_bound_int'])
    
    ip_mapping = ip_mapping.sort_values('lower_bound_int')
    df_with_country = df_with_country.sort_values('ip_int')
    
    df_with_country = pd.merge_asof(
        df_with_country,
        ip_mapping[['lower_bound_int', 'upper_bound_int', 'country']],
        left_on='ip_int',
        right_on='lower_bound_int',
        direction='backward'
    )
    
    mask = (df_with_country['ip_int'] >= df_with_country['lower_bound_int']) & \
           (df_with_country['ip_int'] <= df_with_country['upper_bound_int'])
    df_with_country.loc[~mask, 'country'] = 'Unknown'
    
    df_with_country = df_with_country.drop(['ip_int', 'lower_bound_int', 'upper_bound_int'], axis=1)
    
    return df_with_country

def analyze_fraud_by_country(df):
    """Analyze fraud rates by country"""
    summary = df.groupby('country').agg({
        'class': ['count', 'mean'],
        'purchase_value': 'mean'
    }).round(4)
    summary.columns = ['transaction_count', 'fraud_rate', 'avg_purchase_value']
    return summary.sort_values('fraud_rate', ascending=False)

print("Geolocation module loaded successfully")
