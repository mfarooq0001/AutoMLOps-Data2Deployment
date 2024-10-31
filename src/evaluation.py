import logging

from abc import ABC, abstractmethod

from sklearn.metrics import root_mean_squared_error

import numpy as np

class EvaluateModel:
    """
    Abstract base class for model evaluation, requiring an `evaluate_model` 
    method for implementing evaluation metrics.
    """

    @abstractmethod
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray):
        """
        Evaluates the model's performance based on actual and predicted values.

        Args:
            y_true (np.ndarray): Array of true target values.
            y_pred (np.ndarray): Array of predicted values.
        """
        pass

class RMSE(EvaluateModel):
    """
    Calculates the Root Mean Squared Error (RMSE) of a model's predictions.
    """

    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating RMSE")
            rmse = root_mean_squared_error(y_true, y_pred)
            logging.info("RMSE: {}".format(rmse))
            return rmse
        except Exception as e:
            logging.error("Error evaluating the model. {}".format(e))
            raise e

            
            

