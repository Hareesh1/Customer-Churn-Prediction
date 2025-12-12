import requests
import zipfile
import io
import os

def download_data():
    # URL for Telco Customer Churn (using a direct github raw link if available or the kaggle one if it works without auth)
    # The kaggle API link needs auth. I will use a reliable GitHub mirror for this standard dataset to avoid blocks.
    url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
    output_path = "data/Telco-Customer-Churn.csv"
    
    if os.path.exists(output_path):
        print(f"File already exists at {output_path}")
        return

    print(f"Downloading {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    except Exception as e:
        print(f"Error downloading: {e}")
        # Fallback to the user provided link approach if the above fails (though user link was a zip)
        print("Please ensure internet access is available.")

if __name__ == "__main__":
    if not os.path.exists('data'):
        os.makedirs('data')
    download_data()
