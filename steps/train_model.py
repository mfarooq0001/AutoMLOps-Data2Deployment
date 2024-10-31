import logging
import pandas as pd
import mlflow

from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin

from .model_config import ModelConfig

from zenml import step
from zenml.client import Client

# Experiment tracker
experiment_tracker = Client().active_stack.experiment_tracker


@step(experiment_tracker=experiment_tracker.name)
def model_training(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
) -> RegressorMixin:
    
    # Create an instance of ModelConfig
    config = ModelConfig()
    try:
        model = None
        if config.model_name == "LinearRegressionModel":
            # Enable auto-logging
            mlflow.sklearn.autolog()

            # Training the model
            model = LinearRegressionModel()
            trained_model = model.Train(X_train, y_train)
            return trained_model
        else:
            raise ValueError("Model {} not supoorted.".format(config.model_name))
    except Exception as e:
        logging.error("Error training the model".format(e))
        raise e
    
    