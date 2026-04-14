import os
import gzip
import urllib.request
import numpy as np

def fetch_mnist(data_dir="data"):
    """
    Downloads and parses the MNIST dataset into NumPy arrays.
    Returns: X_train, Y_train, X_test, Y_test
    """
    # 1. The official Google Cloud mirror for MNIST
    base_url = "https://storage.googleapis.com/cvdf-datasets/mnist/"
    files = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz",
    }

    os.makedirs(data_dir, exist_ok=True)
    paths = {}

    # 2. Download the files if they don't exist
    for name, filename in files.items():
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(base_url + filename, filepath)
        paths[name] = filepath

    # 3. Parse the binary files into NumPy arrays
    def parse_images(filepath):
        with gzip.open(filepath, 'rb') as f:
            # offset=16 skips the binary header
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        # Reshape to (N, Channels, Height, Width) and normalize to 0.0 - 1.0
        return data.reshape(-1, 1, 28, 28).astype(np.float32) / 255.0

    def parse_labels(filepath):
        with gzip.open(filepath, 'rb') as f:
            # offset=8 skips the label header
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data.astype(np.int32)

    print("Parsing MNIST binary files...")
    X_train = parse_images(paths["train_images"])
    Y_train = parse_labels(paths["train_labels"])
    X_test = parse_images(paths["test_images"])
    Y_test = parse_labels(paths["test_labels"])

    return X_train, Y_train, X_test, Y_test


class DataLoader:
    """
    A generator that shuffles and chunks the dataset into mini-batches 
    so we don't blow up our MacBook's RAM.
    """
    def __init__(self, X, Y, batch_size=64, shuffle=True):
        self.X = X
        self.Y = Y
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_samples = X.shape[0]

    def __iter__(self):
        # Create an array of indices [0, 1, 2, ..., 59999]
        indices = np.arange(self.num_samples)
        
        # Shuffle the indices so the network doesn't memorize the order
        if self.shuffle:
            np.random.shuffle(indices)

        # Yield chunks of indices based on the batch_size
        for start_idx in range(0, self.num_samples, self.batch_size):
            batch_idx = indices[start_idx : start_idx + self.batch_size]
            
            # Extract and yield the actual data
            yield self.X[batch_idx], self.Y[batch_idx]