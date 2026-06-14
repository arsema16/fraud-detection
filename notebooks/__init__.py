import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create reports directory if it doesn't exist
os.makedirs('reports', exist_ok=True)

# Load your actual data
df = pd.read_csv('data/processed/fraud_data_engineered.csv')

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette('Set2')

print("="*50)
print("GENERATING VISUALIZATIONS WITH ACTUAL DATA")
print("="*50)

# 1. Class Distribution Plot
print("\n1. Creating class_distribution.png...")
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Bar chart
class_counts = df['class'].value_counts()
class_counts.plot(kind='bar', ax=axes[0], color=['green', 'red'])
axes[0].set_title(f'Class Distribution (Fraud Rate: {df["class"].mean()*100:.2f}%)', fontsize=14)
axes[0].set_xlabel('Class (0=Legitimate, 1=Fraud)')
axes[0].set_ylabel('Count')
axes[0].set_xticklabels(['Legitimate', 'Fraud'], rotation=0)
axes[0].ticklabel_format(style='plain', axis='y')

# Pie chart
class_counts.plot(kind='pie', autopct='%1.2f%%', ax=axes[1], 
                  colors=['green', 'red'], explode=[0, 0.1])
axes[1].set_title('Class Proportion', fontsize=14)
axes[1].set_ylabel('')

plt.tight_layout()
plt.savefig('reports/class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved to reports/class_distribution.png")

# 2. Purchase Value by Class
print("\n2. Creating purchase_value_by_class.png...")
fig, ax = plt.subplots(figsize=(10, 6))

# Box plot
df.boxplot(column='purchase_value', by='class', ax=ax)
ax.set_title('Purchase Value Distribution by Class', fontsize=14)
ax.set_xlabel('Class (0=Legitimate, 1=Fraud)')
ax.set_ylabel('Purchase Value ($)')
ax.set_yscale('log')
ax.set_xticklabels(['Legitimate', 'Fraud'])
plt.suptitle('')
plt.savefig('reports/purchase_value_by_class.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved to reports/purchase_value_by_class.png")

# 3. Purchase Value Histogram by Class
print("\n3. Creating purchase_value_histogram.png...")
fig, ax = plt.subplots(figsize=(10, 6))

legit_vals = df[df['class'] == 0]['purchase_value']
fraud_vals = df[df['class'] == 1]['purchase_value']

legit_vals.hist(bins=50, alpha=0.5, label=f'Legitimate (n={len(legit_vals):,})', edgecolor='black')
fraud_vals.hist(bins=50, alpha=0.5, label=f'Fraud (n={len(fraud_vals):,})', edgecolor='black')
ax.set_title('Purchase Value Distribution by Class', fontsize=14)
ax.set_xlabel('Purchase Value ($)')
ax.set_ylabel('Frequency')
ax.set_xscale('log')
ax.legend()
plt.savefig('reports/purchase_value_histogram.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved to reports/purchase_value_histogram.png")

# 4. Fraud Rate by Hour
print("\n4. Creating fraud_by_hour.png...")
hour_fraud = df.groupby('purchase_hour')['class'].mean()

fig, ax = plt.subplots(figsize=(12, 5))
hour_fraud.plot(kind='line', marker='o', linewidth=2, markersize=8, ax=ax)
ax.set_title('Fraud Rate by Hour of Day', fontsize=14)
ax.set_xlabel('Hour of Day (0-23)')
ax.set_ylabel('Fraud Rate')
ax.set_ylim(0, hour_fraud.max() * 1.2)
ax.axhline(y=df['class'].mean(), color='red', linestyle='--', 
           label=f'Average Fraud Rate: {df["class"].mean()*100:.2f}%')
ax.legend()
ax.grid(True, alpha=0.3)

# Highlight peak hour
peak_hour = hour_fraud.idxmax()
peak_rate = hour_fraud.max()
ax.axvline(x=peak_hour, color='orange', linestyle=':', alpha=0.7)
ax.text(peak_hour + 0.5, peak_rate, f'Peak: {peak_hour}:00 ({peak_rate*100:.2f}%)', 
        fontsize=10, color='orange')

plt.savefig('reports/fraud_by_hour.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓ Saved to reports/fraud_by_hour.png (Peak fraud hour: {peak_hour}:00)")

# 5. Time Since Signup Analysis
print("\n5. Creating time_since_signup_analysis.png...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Histogram of time since signup
df[df['class'] == 0]['time_since_signup'].hist(bins=50, alpha=0.5, 
                                                 label='Legitimate', edgecolor='black', ax=axes[0])
df[df['class'] == 1]['time_since_signup'].hist(bins=50, alpha=0.5, 
                                                 label='Fraud', edgecolor='black', ax=axes[0])
axes[0].set_title('Time Since Signup Distribution', fontsize=14)
axes[0].set_xlabel('Hours Since Signup')
axes[0].set_ylabel('Frequency')
axes[0].legend()
axes[0].set_xlim(0, 168)  # Focus on first week

# Fraud rate by time since signup bins
bins = [0, 1, 6, 12, 24, 48, 72, 168, 720, float('inf')]
labels = ['<1h', '1-6h', '6-12h', '12-24h', '1-2d', '2-3d', '3-7d', '1-4w', '>4w']
df['time_bucket'] = pd.cut(df['time_since_signup'], bins=bins, labels=labels)
time_fraud = df.groupby('time_bucket')['class'].mean()

colors = ['red' if x > df['class'].mean() else 'green' for x in time_fraud]
time_fraud.plot(kind='bar', ax=axes[1], color=colors)
axes[1].set_title('Fraud Rate by Time Since Signup', fontsize=14)
axes[1].set_xlabel('Time Between Signup and Purchase')
axes[1].set_ylabel('Fraud Rate')
axes[1].axhline(y=df['class'].mean(), color='blue', linestyle='--', 
                label=f'Average: {df["class"].mean()*100:.2f}%')
axes[1].legend()
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('reports/time_since_signup_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved to reports/time_since_signup_analysis.png")

# 6. Correlation Matrix
print("\n6. Creating correlation_matrix.png...")
numeric_cols = ['purchase_value', 'age', 'time_since_signup', 'transactions_24h', 
                'user_avg_amount', 'user_transaction_count', 'user_fraud_rate', 'class']
available_cols = [col for col in numeric_cols if col in df.columns]
corr_matrix = df[available_cols].corr()

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', square=True, ax=ax)
ax.set_title('Correlation Matrix of Numerical Features', fontsize=14)
plt.tight_layout()
plt.savefig('reports/correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("  ✓ Saved to reports/correlation_matrix.png")

# 7. Fraud Rate by Source
print("\n7. Creating fraud_by_source.png...")
if 'source' in df.columns:
    source_fraud = df.groupby('source')['class'].mean().sort_values(ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    source_fraud.plot(kind='bar', ax=ax, color='coral')
    ax.set_title('Fraud Rate by Source Channel', fontsize=14)
    ax.set_xlabel('Source')
    ax.set_ylabel('Fraud Rate')
    ax.set_ylim(0, source_fraud.max() * 1.2)
    ax.axhline(y=df['class'].mean(), color='red', linestyle='--', 
               label=f'Average: {df["class"].mean()*100:.2f}%')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('reports/fraud_by_source.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved to reports/fraud_by_source.png")
    
    # Print actual values
    print("\n  Actual fraud rates by source:")
    for source, rate in source_fraud.items():
        print(f"    {source}: {rate*100:.2f}%")

# 8. Fraud Rate by Browser
print("\n8. Creating fraud_by_browser.png...")
if 'browser' in df.columns:
    browser_fraud = df.groupby('browser')['class'].mean().sort_values(ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    browser_fraud.plot(kind='bar', ax=ax, color='teal')
    ax.set_title('Fraud Rate by Browser (Top 10)', fontsize=14)
    ax.set_xlabel('Browser')
    ax.set_ylabel('Fraud Rate')
    ax.set_ylim(0, browser_fraud.max() * 1.2)
    ax.axhline(y=df['class'].mean(), color='red', linestyle='--', 
               label=f'Average: {df["class"].mean()*100:.2f}%')
    ax.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('reports/fraud_by_browser.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Saved to reports/fraud_by_browser.png")
    
    # Print actual values
    print("\n  Actual fraud rates by browser:")
    for browser, rate in browser_fraud.items():
        print(f"    {browser}: {rate*100:.2f}%")

print("\n" + "="*50)
print("ALL VISUALIZATIONS COMPLETE!")
print("="*50)
print("\nFiles saved in 'reports/' folder:")
print("  1. class_distribution.png")
print("  2. purchase_value_by_class.png")
print("  3. purchase_value_histogram.png")
print("  4. fraud_by_hour.png")
print("  5. time_since_signup_analysis.png")
print("  6. correlation_matrix.png")
print("  7. fraud_by_source.png")
print("  8. fraud_by_browser.png")