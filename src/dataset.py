import h5py
import numpy as np
from huggingface_hub import hf_hub_download

class TSRDDataLoader:
    """DataLoader for Turing Synthetic Radar Dataset (TSRD) HDF5 files."""
    
    def __init__(self, repo_id="alan-turing-institute/turing-synthetic-radar-dataset", filename="archive/train/train_0.h5"):
        self.repo_id = repo_id
        self.filename = filename
        self.features = None
        self.labels = None
        self.feature_names = []
        
    def load_data(self):
        """Downloads and loads features and labels into memory."""
        local_path = hf_hub_download(
            repo_id=self.repo_id,
            filename=self.filename,
            repo_type="dataset"
        )
        
        with h5py.File(local_path, "r") as f:
            self.features = f["data"][:]
            self.labels = f["labels"][:]
            
            raw_names = f["metadata/feature_names"][:]
            self.feature_names = [
                name.decode("utf-8") if isinstance(name, bytes) else str(name)
                for name in raw_names
            ]
            
        return self.features, self.labels, self.feature_names

if __name__ == "__main__":
    loader = TSRDDataLoader()
    X, y, names = loader.load_data()
    print(f"Loaded {X.shape[0]} pulse samples with features: {names[:4]}")