import logging

import pandas as pd
from zenml import step

from src.data_cleaning import DataCleaner, DataPreProcessingStrategy, DataSplittingStrategy

from typing_extensions import Annotated
from typing import Tuple

@step
def clean_df(data: pd.DataFrame) -> Tuple [
    Annotated[pd.DataFrame, "X_train"],
    Annotated[pd.DataFrame, "X_test"],
    Annotated[pd.Series, "y_train"],
    Annotated[pd.Series, "y_test"],
]:

    """
    Cleans and splits the data

    Args:
        df: data
    Returns:
        X_train: Training dataset
        X_test: Test dataset
        y_train: Train labels
        y_test: Test labels
    """

    try:
        data_clean_strategy = DataPreProcessingStrategy()
        data_cleaner = DataCleaner(data, data_clean_strategy)
        cleaned_data = data_cleaner.handle_data()

        data_split_strategy = DataSplittingStrategy()
        data_split = DataCleaner(cleaned_data, data_split_strategy)
        X_train, X_test, y_train, y_test = data_split.handle_data()
        logging.info("Data cleaning and splitting is complete")
        return X_train, X_test, y_train, y_test
    except Exception as e:
        logging.info("Error cleaning the data: {}".format(e))
        raise e



        