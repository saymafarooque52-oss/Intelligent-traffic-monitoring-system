import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 11

def perform_eda():
    print("=== Starting Exploratory Data Analysis ===")
    
    # Load dataset
    csv_path = 'Traffic_Flow_Dataset.csv'
    if not os.path.exists(csv_path):
        print(f"Error: Dataset not found at {csv_path}")
        return
        
    df = pd.read_csv(csv_path)
    
    print("\n--- Dataset Basic Info ---")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nColumns and Data Types:")
    print(df.dtypes)
    
    print("\nMissing values:")
    print(df.isnull().sum())
    
    print("\nSummary Statistics for Numeric Columns:")
    print(df.describe())
    
    # Ensure plots directory exists
    plots_dir = 'plots'
    os.makedirs(plots_dir, exist_ok=True)
    print(f"\nSaving visualizations to: {os.path.abspath(plots_dir)}")
    
    # 1. Traffic Condition Distribution
    plt.figure(figsize=(8, 5))
    order = ['Low', 'Medium', 'High']
    sns.countplot(x='Traffic_Condition', data=df, order=order, hue='Traffic_Condition', palette='viridis', legend=False)
    plt.title('Distribution of Traffic Conditions', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Traffic Condition', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'traffic_condition_dist.png'), dpi=300)
    plt.close()
    print("Saved traffic_condition_dist.png")
    
    # 2. Correlation Matrix for Numeric Fields
    plt.figure(figsize=(10, 8))
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    # Drop Sin_Time, Cos_Time, Hour_of_Day for cleaner correlation or keep them
    corr = df[numeric_cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5, cbar=True)
    plt.title('Correlation Matrix of Traffic Metrics', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'), dpi=300)
    plt.close()
    print("Saved correlation_matrix.png")
    
    # 3. Speed vs. Density colored by Traffic Condition
    plt.figure(figsize=(9, 6))
    sns.scatterplot(x='Vehicle_Density', y='Average_Speed', hue='Traffic_Condition', 
                    hue_order=order, palette={'Low': '#2ecc71', 'Medium': '#f1c40f', 'High': '#e74c3c'},
                    alpha=0.6, data=df)
    plt.title('Vehicle Speed vs. Density by Traffic Condition', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Vehicle Density (vehicles/km)', fontsize=12)
    plt.ylabel('Average Speed (km/h)', fontsize=12)
    plt.legend(title='Traffic Condition')
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'speed_vs_density.png'), dpi=300)
    plt.close()
    print("Saved speed_vs_density.png")
    
    # 4. Hourly Average Traffic Flow Rate
    plt.figure(figsize=(10, 5))
    hourly_flow = df.groupby('Hour_of_Day')['Traffic_Flow_Rate'].mean().reset_index()
    sns.lineplot(x='Hour_of_Day', y='Traffic_Flow_Rate', data=hourly_flow, marker='o', color='#3498db', linewidth=2.5)
    # Highlight peak hours
    peak_flow = df[df['Is_Peak_Hour'] == 1].groupby('Hour_of_Day')['Traffic_Flow_Rate'].mean().reset_index()
    plt.title('Average Traffic Flow Rate by Hour of Day', fontsize=14, fontweight='bold', pad=15)
    plt.xlabel('Hour of Day (0-23)', fontsize=12)
    plt.ylabel('Average Flow Rate (vehicles/hour)', fontsize=12)
    plt.xticks(range(0, 24))
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'hourly_traffic.png'), dpi=300)
    plt.close()
    print("Saved hourly_traffic.png")
    
    print("\n=== EDA Completed Successfully ===")

if __name__ == '__main__':
    perform_eda()
