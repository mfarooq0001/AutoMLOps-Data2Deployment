import logging

from zenml import pipeline
import pandas as pd

from steps.ingest_data import ingest_data
from steps.clean_data import clean_df
from steps.train_model import model_training
from steps.evaluate_model import model_evaluation

@pipeline
def train_pipeline(data_path: str) -> None:
    """
    Training pipeline that executes all steps for data ingestion, cleaning, 
    model training, and evaluation.

    Args:
        data_path (str): Path to the dataset.

    Steps:
        1. ingest_data: Reads the dataset from the provided path.
        2. clean_df: Processes and splits the dataset into training and test sets.
        3. model_training: Trains a model on the training data.
        4. model_evaluation: Evaluates the trained model on the test data and 
           computes the RMSE.

    Logs:
        Outputs the RMSE after model evaluation.
    """
    ingested_data = ingest_data(data_path)
    X_train, X_test, y_train,  y_test = clean_df(ingested_data)
    model = model_training(X_train, y_train)
    rmse = model_evaluation(model, X_test, y_test)
    logging.info("Model Training and Evaluation Completed: RMSE {}".format(rmse))

    