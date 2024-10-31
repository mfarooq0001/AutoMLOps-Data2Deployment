import logging
import mlflow

from src.evaluation import RMSE
from sklearn.base import RegressorMixin

import pandas as pd
from zenml import step
from zenml.client import Client

# Experiment tracker
experiment_tracker = Client().active_stack.experiment_tracker

@step(experiment_tracker=experiment_tracker.name)
def model_evaluation(
    model: RegressorMixin,
    X_test: pd.DataFrame,
    y_test: pd.DataFrame
) -> float:
    """
    Evaluates the model

    Args:
        model: Takes the model and evaluates it
    """
    
    try:
        prediction = model.predict(X_test)
        rmse_class = RMSE()
        calculated_rmse = rmse_class.evaluate_model(y_test, prediction)

        # Mlflow log metrics
        mlflow.log_metric("rmse", calculated_rmse)

        logging.info("RMSE: {}".format(calculated_rmse))
        return calculated_rmse
    except Exception as e:
        logging.error("Error while calculting RMSE: {}".format(e))
        raise e