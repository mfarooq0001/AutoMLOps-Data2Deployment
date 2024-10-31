import logging

import numpy as np
import pandas as pd

from typing import Union

from abc import ABC, abstractmethod
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):
    """
    Abstract base class for data handling strategies.
    """

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """Abstract method for data processing tasks."""
        pass


class DataPreProcessingStrategy(DataStrategy):
    """
    Strategy for cleaning and preprocessing data.
    """

    def handle_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Cleans the dataset by dropping unnecessary columns, filling missing values,
        and selecting numeric columns.

        Args:
            data (pd.DataFrame): Raw input data.

        Returns:
            pd.DataFrame: Cleaned and processed data.
        """
        logging.info("Cleaning data")
        try:
            # Drop specified columns and fill missing values
            data = data.drop(
                [
                    "order_approved_at",
                    "order_delivered_carrier_date",
                    "order_delivered_customer_date",
                    "order_estimated_delivery_date",
                    "order_purchase_timestamp",
                ],
                axis=1
            )

            # Fill missing values with median or default text
            data["product_weight_g"].fillna(data["product_weight_g"].median(), inplace=True)
            data["product_length_cm"].fillna(data["product_length_cm"].median(), inplace=True)
            data["product_height_cm"].fillna(data["product_height_cm"].median(), inplace=True)
            data["product_width_cm"].fillna(data["product_width_cm"].median(), inplace=True)
            data["review_comment_message"].fillna("No review", inplace=True)

            # Select numeric columns and drop specified columns
            data = data.select_dtypes(include=[np.number])
            cols_to_drop = ["customer_zip_code_prefix", "order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            return data
        except Exception as e:
            logging.error("Error cleaning the data: {}".format(e))
            raise e


class DataSplittingStrategy(DataStrategy):
    """
    Strategy for splitting data into training and testing sets.
    """

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        """
        Splits the dataset into training and test sets.

        Args:
            data (pd.DataFrame): Cleaned dataset.

        Returns:
            Tuple: X_train, X_test, y_train, y_test.
        """
        try:
            X = data.drop(["review_score"], axis=1)
            y = data["review_score"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error splitting the dataset: {}".format(e))
            raise e


class DataCleaner:
    """
    Applies a data handling strategy to clean or split the dataset.
    """

    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        """
        Executes the strategy's data handling method on the dataset.

        Returns:
            Union[pd.DataFrame, pd.Series]: Processed data as defined by the strategy.
        """
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error handling the data: {}".format(e))
            raise e
