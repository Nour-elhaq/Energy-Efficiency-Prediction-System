
import os
import pandas as pd
import requests
from ucimlrepo import fetch_ucirepo

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
FILE_PATH = os.path.join(DATA_DIR, 'energy_efficiency.csv')

def download_data():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
        
    # Check for Excel file first
    excel_path = os.path.join(DATA_DIR, 'ENB2012_data.xlsx')
    if os.path.exists(excel_path):
        print(f"Loading data from {excel_path}...")
        df = pd.read_excel(excel_path)
        # Ensure column names are standard if needed, or just return
        # The dataset usually has X1..X8, Y1, Y2
        return df

    if os.path.exists(FILE_PATH):
        print(f"Data already exists at {FILE_PATH}")
        return pd.read_csv(FILE_PATH)

    print("Attempting to download data via ucimlrepo...")
    try:
        energy_efficiency = fetch_ucirepo(id=242)
        
        X = energy_efficiency.data.features
        y = energy_efficiency.data.targets
        
        df = pd.concat([X, y], axis=1)
        df.to_csv(FILE_PATH, index=False)
        print(f"Data downloaded and saved to {FILE_PATH}")
        return df
    except Exception as e:
        print(f"ucimlrepo failed: {e}")
        print("Attempting direct download...")
        
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                with open(excel_path, 'wb') as f:
                    f.write(response.content)
                
                df = pd.read_excel(excel_path)
                return df
            else:
                print(f"Failed to download from {url}")
        except Exception as e2:
             print(f"Direct download failed: {e2}")
             
    return None

if __name__ == "__main__":
    download_data()
