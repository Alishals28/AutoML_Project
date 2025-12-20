import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from mlxtend.classifier import OneRClassifier

from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "best_model_info.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(
        self,
        train_array,
        test_array,
        problem_type: str = "classification",
        search_type: str = "grid",
        class_weights: dict | None = None,
    ):
        """Train and optimize classification models, returning metrics and best model info."""
        try:
            logging.info("Split training and test input data")

            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]

            if problem_type != "classification":
                raise CustomException("Only classification problem type is supported")

            # Coerce targets to int for classifiers
            y_train = y_train.astype(int)
            y_test = y_test.astype(int)

            # Base models (inject class weights where supported)
            models = {
                "Random Forest": RandomForestClassifier(class_weight=class_weights),
                "Decision Tree": DecisionTreeClassifier(class_weight=class_weights),
                "Logistic Regression": LogisticRegression(class_weight=class_weights, max_iter=500),
                "AdaBoost Classifier": AdaBoostClassifier(),
                "K-Nearest Neighbors": KNeighborsClassifier(),
                "Naive Bayes": GaussianNB(),
                "Support Vector Machine": SVC(probability=True, class_weight=class_weights),
                "OneR Classifier": OneRClassifier(),
            }

            # Slim grids for speed; can be expanded later
            params = {
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 5, 10],
                },
                "Random Forest": {
                    "n_estimators": [32, 64, 128],
                    "max_depth": [None, 5, 10],
                },
                "Logistic Regression": {
                    "C": [0.5, 1.0, 2.0],
                    "penalty": ["l2"],
                    "solver": ["lbfgs"],
                },
                "AdaBoost Classifier": {
                    "learning_rate": [0.1, 0.5, 1.0],
                    "n_estimators": [32, 64, 128],
                },
                "K-Nearest Neighbors": {
                    "n_neighbors": [3, 5, 9],
                    "metric": ["euclidean", "manhattan"],
                },
                "Naive Bayes": {},
                "Support Vector Machine": {
                    "C": [0.5, 1.0, 2.0],
                    "kernel": ["linear", "rbf"],
                },
                "OneR Classifier": {},
            }

            metric = accuracy_score

            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
                metric=metric,
                search_type=search_type,
            )

            best_model_name = max(model_report.items(), key=lambda x: x[1].get("f1_score", -np.inf))[0]
            best_model = model_report[best_model_name]["model"]

            logging.info(f"Best found model: {best_model_name}")

            save_payload = {
                "best_model_name": best_model_name,
                "best_model": best_model,
                "model_report": model_report,
            }
            save_object(self.model_trainer_config.trained_model_file_path, save_payload)

            return model_report, best_model_name, self.model_trainer_config.trained_model_file_path

        except Exception as e:
            logging.error(f"Exception occurred in model training: {e}")
            raise CustomException(e, sys)