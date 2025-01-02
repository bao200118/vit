# download_cifar.py
import os
import requests
import tarfile

def download_cifar100():
    url = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
    filename = "cifar-100-python.tar.gz"
    
    if not os.path.exists('data'):
        os.makedirs('data')
        
    if not os.path.exists(f'data/{filename}'):
        print("Downloading CIFAR-100...")
        response = requests.get(url, stream=True)
        with open(f'data/{filename}', 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
    
    if not os.path.exists('data/cifar-100-python'):
        print("Extracting files...")
        with tarfile.open(f'data/{filename}', 'r:gz') as tar:
            tar.extractall('data/')
    
    print("CIFAR-100 ready!")

if __name__ == "__main__":
    download_cifar100()