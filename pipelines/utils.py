import logging

import pandas as pd
from src.data_cleaning import DataCleaner, DataPreProcessingStrategy


def get_data_for_test():
    """
    Loads and preprocesses a sample of customer data for testing.

    Reads data from a CSV, samples 100 records, applies cleaning, removes 
    the "review_score" column, and converts the result to JSON.

    Returns:
        str: JSON-formatted string of processed data.

    Raises:
        Exception: Logs and re-raises any processing errors.
    """
    try:
        df = pd.read_csv("dataset/olist_customers_dataset_updated.csv")
        df = df.sample(n=100)
        preprocess_strategy = DataPreProcessingStrategy()
        data_cleaning = DataCleaner(df, preprocess_strategy)
        df = data_cleaning.handle_data()
        df.drop(["review_score"], axis=1, inplace=True)
        result = df.to_json(orient="split")
        return result
    except Exception as e:
        logging.error(e)
        raise e
