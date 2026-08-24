import h5py
from huggingface_hub import hf_hub_download

REPO_ID = "alan-turing-institute/turing-synthetic-radar-dataset"
FILE_PATH = "archive/train/train_0.h5"

# Download the HDF5 file
local_file = hf_hub_download(
    repo_id=REPO_ID,
    filename=FILE_PATH,
    repo_type="dataset"
)

# Read datasets into NumPy arrays
with h5py.File(local_file, "r") as f:
    X = f["data"][:]
    y = f["labels"][:]
    
    # Extract metadata feature names safely
    raw_features = f["metadata/feature_names"][:]
    feature_names = [
        name.decode("utf-8") if isinstance(name, bytes) else str(name)
        for name in raw_features
    ]

print("--- Loaded Data Summary ---")
print(f"Features array shape : {X.shape}")
print(f"Labels array shape   : {y.shape}")
print(f"Feature Names        : {feature_names}")