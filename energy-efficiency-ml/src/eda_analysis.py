
import pandas as pd
import seaborn as sns
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
from data_loader import download_data

def run_eda():
    df = download_data()
    if df is None:
        print("No data found.")
        return

    # Create images directory
    img_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'notebooks', 'images')
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    print("Generating Correlation Matrix...")
    plt.figure(figsize=(10, 8))
    # Rename columns for better readability if needed, or use default
    # The dataset typically has: X1, X2... Y1, Y2. 
    # Let's rename them based on UCI info if execution worked, but for safety we use existing names.
    # If the excel file has headers, we are good.
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Matrix")
    plt.savefig(os.path.join(img_dir, 'correlation_matrix.png'))
    plt.close()

    print("Generating Distributions...")
    df.hist(figsize=(12, 10), bins=20)
    plt.suptitle("Feature Distributions")
    plt.savefig(os.path.join(img_dir, 'distributions.png'))
    plt.close()
    
    print("EDA completed. Images saved to notebooks/images/.")

if __name__ == "__main__":
    run_eda()
