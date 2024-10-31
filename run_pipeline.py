from pipelines.training_pipeline import train_pipeline
# Uncomment the next line to access the ZenML client if needed
# from zenml.client import Client

def main():
    """
    Main function to initiate the training pipeline with the specified data path.
    """
    data_path = "dataset/olist_customers_dataset_updated.csv"
    
    # Uncomment the following line to print the tracking URI of the active experiment tracker
    # print(Client().active_stack.experiment_tracker.get_tracking_uri())
    
    train_pipeline(data_path)

    # To launch MLflow UI, run this command in your terminal:
    # mlflow ui --backend-store-uri "file:/Users/farooq/Library/Application Support/zenml/local_stores/eb0d1e76-ca2f-4b14-9fb9-6de77f3d9951/mlruns"

if __name__ == "__main__":
    main()
