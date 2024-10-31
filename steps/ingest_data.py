import logging 

import pandas as pd
from zenml import step


class IngestData():

    def __init__(self, data_path: str):
        self.data_path = data_path

    def run(self):
        logging.info(f"Ingesting data from: {self.data_path}")
        return pd.read_csv(self.data_path, index_col=0, parse_dates=False)
    

@step
def ingest_data(data_path: str) -> pd.DataFrame:
    """
    Ingests all the data

    Args:
        data_path: Path of the dataset
    """
    try:
        ingest_data = IngestData(data_path)
        data = ingest_data.run()
        return data
    except Exception as e:
        logging.error("Error during Data Ingestion: {}".format(e))
        raise e

