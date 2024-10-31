import logging

from abc import ABC, abstractmethod

from sklearn.linear_model import LinearRegression


class Model:
    """
    Abstract base class for all models, requiring a Train method.
    """

    @abstractmethod 
    def Train(self, X_train, y_train):
        """Trains the model on provided data."""
        pass

class LinearRegressionModel(Model):
    """
    Linear regression model for training on input data.
    """

    def Train(self, X_train, y_train, **kwargs):
        """
        Trains a linear regression model.

        Args:
            X_train: Training feature data.
            y_train: Training target labels.
        """
        try:
            reg = LinearRegression(**kwargs)
            reg.fit(X_train, y_train)
            logging.info("Model training completed!")
            return reg
        except Exception as e:
            logging.error("Error while training the model: {}".format(e))
            raise e
