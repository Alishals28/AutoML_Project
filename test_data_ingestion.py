"""Test STEP 5: Data Ingestion"""
import os
import pandas as pd

from src.components.data_ingestion import DataIngestion


print("Loading datasets/heart.csv ...")
df = pd.read_csv("datasets/heart.csv")
print(f"Dataset shape: {df.shape}")

ingestion = DataIngestion()

# Use a non-default test_size to verify parameterization
train_path, test_path, eda_path = ingestion.initiate_data_ingestion(
    df=df,
    test_size=0.3,
    random_state=7,
)

print("\nArtifacts written:")
print(f"  Raw : {ingestion.ingestion_config.raw_data_path}")
print(f"  Train: {train_path}")
print(f"  Test : {test_path}")
print(f"  EDA  : {eda_path}")

# Validate files exist
assert os.path.exists(ingestion.ingestion_config.raw_data_path), "Raw data file missing"
assert os.path.exists(train_path), "Train split missing"
assert os.path.exists(test_path), "Test split missing"
assert os.path.exists(eda_path), "EDA report missing"

# Validate split sizes respect test_size (within one row tolerance)
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

expected_test = int(round(len(df) * 0.3))
print(f"\nSplit sizes: train={len(train_df)}, test={len(test_df)}, expected_test≈{expected_test}")
assert abs(len(test_df) - expected_test) <= 1, "Test split size not aligned with test_size"
assert len(train_df) + len(test_df) == len(df), "Train+Test counts do not sum to original"

print("\nSTEP 5 TEST PASSED: Data ingestion saves raw data, parametrized split, and EDA report.")
