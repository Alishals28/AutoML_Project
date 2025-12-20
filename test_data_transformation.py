"""Test STEP 4: Data Transformation pipeline"""
import os
import pandas as pd

from src.components.data_transformation import DataTransformation


def test_data_transformation_pipeline():
    """Integration-style check for the step 4 data transformation pipeline."""

    print("Loading datasets/heart.csv ...")
    df = pd.read_csv("datasets/heart.csv")
    print(f"Original shape: {df.shape}")

    df_test = df.copy()
    df_test["mock_constant"] = 1  # Ensure constant feature removal path is hit
    df_test.loc[0:4, "age"] = None  # Inject missing values for imputation

    issues = [
        {"issue_id": "issue_missing_age", "type": "Missing Values", "column": "age"},
        {
            "issue_id": "issue_outlier_chol",
            "type": "Outliers",
            "column": "chol",
            "bounds": {"lower": 130.0, "upper": 310.0},
            "feature_type": "continuous_numeric",
        },
        {"issue_id": "issue_constant_mock", "type": "Constant Feature", "column": "mock_constant"},
        {"issue_id": "issue_imbalance", "type": "Class Imbalance", "column": "target"},
    ]

    user_decisions = {
        "issue_missing_age": "median",
        "issue_outlier_chol": "cap (IQR)",
        "issue_constant_mock": "drop feature",
        "issue_imbalance": "class_weights",
    }

    print("\nRunning DataTransformation.initiate_data_transformation ...")
    transformer = DataTransformation()
    (
        train_arr,
        test_arr,
        preprocessor_path,
        preprocessing_log,
        class_weights,
    ) = transformer.initiate_data_transformation(
        df=df_test,
        target_column_name="target",
        issues=issues,
        user_decisions=user_decisions,
        test_size=0.25,
        random_state=42,
    )

    print("\nResults:")
    print(f"  Train array shape: {train_arr.shape}")
    print(f"  Test array shape : {test_arr.shape}")
    print(f"  Preprocessor path: {preprocessor_path}")
    print(f"  Class weights    : {class_weights}")
    print(f"  Preprocessing log steps: {len(preprocessing_log)}")

    assert train_arr.shape[1] == test_arr.shape[1], "Train/Test feature size mismatch"
    assert train_arr.shape[0] > 0 and test_arr.shape[0] > 0, "Empty train/test splits"
    assert os.path.exists(preprocessor_path), "Preprocessor file was not saved"
    assert any(step.get("action") == "missing_value_imputation" for step in preprocessing_log), "Imputation step missing"
    assert class_weights is not None and len(class_weights) > 0, "Class weights not computed"

    print("\nSTEP 4 TEST PASSED: Data transformation pipeline is working as expected.")
