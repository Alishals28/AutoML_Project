import os
import sys
from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split
from ydata_profiling import ProfileReport

from src.exception import CustomException
from src.logger import logging


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42):
        """Save raw data, optionally create a train/test split, and generate EDA report.

        Preprocessing is intentionally not handled here; it belongs in data_transformation.
        """
        logging.info("Entered the data ingestion component")
        try:
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Persist raw dataset
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Saved raw dataset to artifacts")

            # Train/test split (parameterized test_size)
            train_set, test_set = train_test_split(df, test_size=test_size, random_state=random_state)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Saved train/test splits to artifacts")

            # Generate EDA report
            profile = ProfileReport(df, title="Exploratory Data Analysis Report")
            profile_path = os.path.join("artifacts", "eda_report.html")
            profile.to_file(profile_path)
            logging.info("Generated EDA report")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
                profile_path,
            )

        except Exception as e:
            raise CustomException(e, sys)