"""Test STEP 6: Model Trainer"""
import os
import pandas as pd

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


print("Loading datasets/heart.csv ...")
df = pd.read_csv("datasets/heart.csv")
print(f"Dataset shape: {df.shape}")

# Minimal transformation to get train/test arrays and class weights
transformer = DataTransformation()
train_arr, test_arr, preproc_path, preprocessing_log, class_weights = transformer.initiate_data_transformation(
    df=df,
    target_column_name="target",
    test_size=0.25,
    random_state=42,
)

print("Running model trainer (grid search, slim grids)...")
trainer = ModelTrainer()
report, best_name, model_path = trainer.initiate_model_trainer(
    train_array=train_arr,
    test_array=test_arr,
    problem_type="classification",
    search_type="grid",
    class_weights=class_weights,
)

print(f"Best model: {best_name}")
print(f"Saved to: {model_path}")

# Basic validations
assert best_name in report, "Best model not present in report"
for required in ["accuracy", "precision", "recall", "f1_score", "confusion_matrix"]:
    assert required in report[best_name], f"Missing metric {required} in report"

assert os.path.exists(model_path), "Trained model artifact missing"

print("\nSTEP 6 TEST PASSED: Model trainer returns metrics and saves best model.")
