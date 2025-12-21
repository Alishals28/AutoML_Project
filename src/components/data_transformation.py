import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler, OrdinalEncoder, LabelEncoder
from scipy import sparse
from sklearn.utils.class_weight import compute_class_weight

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.utils.feature_type_inference import FeatureTypeInference
from src.utils.preprocessing_applicator import PreprocessingApplicator


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
    target_encoder_file_path = os.path.join("artifacts", "target_encoder.pkl")


class DataTransformation:
    """Preprocess data before splitting, then build the sklearn transformer."""

    def __init__(
        self,
        imputation_strategy: str = "median",
        scaling_strategy: str = "StandardScaler",
        encoding_strategy: str = "OneHotEncoder",
    ):
        self.data_transformation_config = DataTransformationConfig()
        self.imputation_strategy = imputation_strategy
        self.scaling_strategy = scaling_strategy
        self.encoding_strategy = encoding_strategy

    def _build_preprocessor(self, numerical_columns, categorical_columns):
        try:
            num_imputer_strategy = (
                self.imputation_strategy
                if self.imputation_strategy in ["mean", "median", "most_frequent"]
                else "median"
            )
            num_imputer = SimpleImputer(strategy=num_imputer_strategy)
            scaler = MinMaxScaler() if self.scaling_strategy == "MinMaxScaler" else StandardScaler()

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", num_imputer),
                    ("scaler", scaler),
                ]
            )

            if self.encoding_strategy == "OrdinalEncoder":
                encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                cat_scaler = "passthrough"
            else:
                encoder = OneHotEncoder(handle_unknown="ignore")
                cat_scaler = StandardScaler(with_mean=False)

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", encoder),
                    ("scaler", cat_scaler),
                ]
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)

    def _build_configs_from_issues(self, issues, user_decisions):
        missing_config = {}
        outlier_config = {}
        constant_config = {}
        imbalance_action = "no action"

        if not issues:
            return missing_config, outlier_config, constant_config, imbalance_action

        for issue in issues:
            issue_id = issue.get("issue_id")
            issue_type = issue.get("type")
            col = issue.get("column")
            bounds = issue.get("bounds", {})

            choice = None
            if user_decisions and issue_id in user_decisions:
                choice = user_decisions[issue_id]

            if issue_type == "Missing Values":
                missing_config[col] = choice or "median"
            elif issue_type == "Outliers":
                outlier_config[col] = {
                    "action": choice or "cap (IQR)",
                    "bounds": bounds,
                }
            elif issue_type in ["Constant Feature", "Near-Constant Feature"]:
                constant_config[col] = choice or "drop feature"
            elif issue_type == "Class Imbalance":
                imbalance_action = choice or "class_weights"

        return missing_config, outlier_config, constant_config, imbalance_action

    def _encode_target_if_categorical(self, y_series):
        """
        Encode categorical target column to numeric labels.
        
        Args:
            y_series: Target column as pandas Series
            
        Returns:
            tuple: (encoded_array, label_encoder or None)
        """
        if y_series.dtype == 'object' or y_series.dtype.name == 'category':
            logging.info(f"Target column is categorical with values: {y_series.unique()}")
            logging.info("Encoding categorical target using LabelEncoder")
            
            label_encoder = LabelEncoder()
            encoded_target = label_encoder.fit_transform(y_series)
            
            logging.info(f"Target encoded: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}")
            return encoded_target, label_encoder
        else:
            # Target is already numeric
            return y_series.values, None

    def _compute_class_weights(self, y_train):
        classes = np.unique(y_train)
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
        return dict(zip(classes, weights))

    def initiate_data_transformation(
        self,
        df: pd.DataFrame = None,
        raw_path: str = None,
        train_path: str = None,
        test_path: str = None,
        problem_type: str = "classification",
        target_column_name: str = None,
        issues: list = None,
        user_decisions: dict = None,
        test_size: float = 0.2,
        random_state: int = 42,
        imbalance_action: str = None,
    ):
        try:
            data_df = None
            if df is not None:
                data_df = df.copy()
            elif raw_path is not None:
                data_df = pd.read_csv(raw_path)
            elif train_path and test_path:
                logging.warning(
                    "Using legacy mode (train/test already split). "
                    "Pre-split preprocessing will be skipped. Pass df/raw_path for full pipeline."
                )
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)
                return self._transform_after_split(train_df, test_df, problem_type, target_column_name)
            else:
                raise CustomException("No data provided for transformation (df/raw_path/train_path/test_path).")

            if target_column_name is None or target_column_name not in data_df.columns:
                raise CustomException("Target column name must be provided and exist in the dataframe.")

            ft_infer = FeatureTypeInference(data_df, target_column=target_column_name)
            feature_types = ft_infer.infer_types()

            missing_cfg, outlier_cfg, constant_cfg, imbalance_choice = self._build_configs_from_issues(
                issues, user_decisions
            )
            if imbalance_action:
                imbalance_choice = imbalance_action

            applicator = PreprocessingApplicator(data_df, feature_types, target_column=target_column_name)
            if missing_cfg:
                applicator.apply_missing_value_imputation(missing_cfg)
            if outlier_cfg:
                applicator.apply_outlier_action(outlier_cfg)
            if constant_cfg:
                applicator.apply_constant_feature_removal(constant_cfg)
            applicator.apply_class_imbalance_handling(imbalance_choice, target_col=target_column_name)

            processed_df = applicator.get_processed_dataframe()
            preprocessing_log = applicator.get_preprocessing_log()

            stratify = processed_df[target_column_name] if problem_type == "classification" else None
            train_df, test_df = train_test_split(
                processed_df,
                test_size=test_size,
                random_state=random_state,
                stratify=stratify,
            )

            numerical_columns = train_df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = train_df.select_dtypes(exclude=[np.number]).columns.tolist()
            if target_column_name in numerical_columns:
                numerical_columns.remove(target_column_name)
            if target_column_name in categorical_columns:
                categorical_columns.remove(target_column_name)

            preprocessor = self._build_preprocessor(numerical_columns, categorical_columns)

            input_feature_train_df = train_df.drop(columns=[target_column_name])
            target_feature_train_df = train_df[target_column_name]
            input_feature_test_df = test_df.drop(columns=[target_column_name])
            target_feature_test_df = test_df[target_column_name]

            # Encode categorical target if needed
            y_train_encoded, target_encoder = self._encode_target_if_categorical(target_feature_train_df)
            if target_encoder is not None:
                # Use the same encoder for test data
                y_test_encoded = target_encoder.transform(target_feature_test_df)
            else:
                y_test_encoded = target_feature_test_df.values
            
            # Save target encoder if categorical
            if target_encoder is not None:
                save_object(
                    file_path=self.data_transformation_config.target_encoder_file_path,
                    obj=target_encoder,
                )
                logging.info(f"Target encoder saved to {self.data_transformation_config.target_encoder_file_path}")

            input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessor.transform(input_feature_test_df)

            # Ensure dense arrays for downstream numpy concatenation
            if sparse.issparse(input_feature_train_arr):
                input_feature_train_arr = input_feature_train_arr.toarray()
            if sparse.issparse(input_feature_test_arr):
                input_feature_test_arr = input_feature_test_arr.toarray()

            train_arr = np.c_[input_feature_train_arr, y_train_encoded]
            test_arr = np.c_[input_feature_test_arr, y_test_encoded]

            class_weights = None
            if problem_type == "classification" and imbalance_choice == "class_weights":
                class_weights = self._compute_class_weights(y_train_encoded)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                preprocessing_log,
                class_weights,
            )

        except Exception as e:
            raise CustomException(e, sys)

    def _transform_after_split(self, train_df, test_df, problem_type, target_column_name):
        try:
            numerical_columns = train_df.select_dtypes(include=[np.number]).columns.tolist()
            categorical_columns = train_df.select_dtypes(exclude=[np.number]).columns.tolist()

            if target_column_name:
                numerical_columns = [c for c in numerical_columns if c != target_column_name]
                categorical_columns = [c for c in categorical_columns if c != target_column_name]

            preprocessor = self._build_preprocessor(numerical_columns, categorical_columns)

            if problem_type in ["regression", "classification"]:
                input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
                target_feature_train_df = train_df[target_column_name]

                input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
                target_feature_test_df = test_df[target_column_name]

                # Encode categorical target if needed
                y_train_encoded, target_encoder = self._encode_target_if_categorical(target_feature_train_df)
                if target_encoder is not None:
                    # Use the same encoder for test data
                    y_test_encoded = target_encoder.transform(target_feature_test_df)
                else:
                    y_test_encoded = target_feature_test_df.values
                
                # Save target encoder if categorical
                if target_encoder is not None:
                    save_object(
                        file_path=self.data_transformation_config.target_encoder_file_path,
                        obj=target_encoder,
                    )
                    logging.info(f"Target encoder saved to {self.data_transformation_config.target_encoder_file_path}")

                input_feature_train_arr = preprocessor.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessor.transform(input_feature_test_df)

                # Ensure dense arrays for downstream numpy concatenation
                if sparse.issparse(input_feature_train_arr):
                    input_feature_train_arr = input_feature_train_arr.toarray()
                if sparse.issparse(input_feature_test_arr):
                    input_feature_test_arr = input_feature_test_arr.toarray()

                train_arr = np.c_[input_feature_train_arr, y_train_encoded]
                test_arr = np.c_[input_feature_test_arr, y_test_encoded]

            elif problem_type == "clustering":
                input_feature_train_arr = preprocessor.fit_transform(train_df)
                input_feature_test_arr = preprocessor.transform(test_df)
                train_arr = input_feature_train_arr
                test_arr = input_feature_test_arr
            else:
                raise CustomException("Unsupported problem type")

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
                [],
                None,
            )

        except Exception as e:
            raise CustomException(e, sys)