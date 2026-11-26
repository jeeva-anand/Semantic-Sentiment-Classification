
import os
import sys

from dataclasses import dataclass

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from xgboost import XGBClassifier

from src.utils import save_object
from src.exception import CustomException
from src.logger import logging


@dataclass
class ModelTrainingConfig:

    model_training_path: str = os.path.join(
        "artifacts",
        "model.pkl"
    )


class ModelTraining:
    def __init__(self):

        self.model_training_config = ModelTrainingConfig()

        # Classification threshold used in your notebook
        self.threshold = 0.3

    # -----------------------------------------
    # Custom F1 scorer using threshold 0.3
    # -----------------------------------------

    def f1_threshold_scorer(self, estimator, X, y):

        y_prob = estimator.predict_proba(X)[:, 1]

        y_pred = (y_prob >= self.threshold).astype(int)

        return f1_score(y, y_pred, zero_division=0)

    # -----------------------------------------
    # Model training
    # -----------------------------------------

    def initiate_model_training(self, train_arr, y_train):

        try:
            y_train_original = y_train.copy()            
            X_train, X_valid, y_train_split, y_valid = train_test_split(
                train_arr,
                y_train_original,
                test_size=0.3,
                random_state=42,
            )
            models = {

                "Logistic Regression": LogisticRegression(),
                "Support Vector Classifier": SVC(),
                "Random Forest Classifier": RandomForestClassifier(),                               
                "XGBoost Classifier": XGBClassifier()
            }

            params = {

                "Logistic Regression": {
                    "C": [0.1, 1, 10],
                    "solver": ["liblinear"],
                    "class_weight": [None, "balanced"]
                },
                "Support Vector Classifier": {
                    "C": [0.1, 1, 10],
                    "kernel": ["linear", "rbf"],
                    "class_weight": [None, "balanced"]
                },
                "Random Forest Classifier": {
                    "n_estimators": [100, 200],
                    "max_depth": [None, 10, 20],
                    "min_samples_split": [2, 5],
                    "min_samples_leaf": [1, 2],
                    "class_weight": [None, "balanced"]
                },
                "XGBoost Classifier": {
                    "n_estimators": [200, 400],
                    "learning_rate": [0.03, 0.1],
                    "max_depth": [4, 6],
                    "min_child_weight": [1, 5],
                    "subsample": [0.8, 1.0],
                    "colsample_bytree": [0.8, 1.0]
                }
            }

            model_report = {}
            trained_models = {}

            for model_name, model in models.items():

                grid_search = GridSearchCV(
                    estimator=model,
                    param_grid=params[model_name],
                    scoring=self.f1_threshold_scorer,
                    cv=3,
                    n_jobs=-1,
                    verbose=1
                )

                grid_search.fit( X_train,y_train_split)

                best_model = grid_search.best_estimator_
                
                y_prob = best_model.predict_proba(X_valid)[:, 1]

                y_pred = ( y_prob >= self.threshold).astype(int)

                model_report[model_name] = f1

                trained_models[model_name] = best_model

            best_model_name = max(
                model_report,
                key=model_report.get
            )

            best_model_score = model_report[best_model_name]

            best_model = trained_models[best_model_name]

            best_model.fit(
                train_arr,
                y_train_original
            )

            save_object(
                self.model_training_config.model_training_path,
                best_model
            )

            return (
                best_model_name,
                best_model_score
            )

        except Exception as e:
            raise CustomException(e, sys)
