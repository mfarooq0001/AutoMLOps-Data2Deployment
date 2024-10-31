import json
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
from pipelines.deployment_pipeline import prediction_service_loader
from run_deployment import main

def main():
    st.set_page_config(page_title="Customer Satisfaction Prediction", layout="wide")
    st.title("End-to-End Customer Satisfaction Prediction Pipeline with MlOps (ZenML, MlFlow)")
    
    whole_pipeline_image = Image.open("_assets/training_and_deployment_pipeline_updated.png")
    st.image(whole_pipeline_image, caption="Whole Pipeline", use_column_width=True)

    # Problem Statement Section
    st.markdown(
        """ 
        ### Problem Statement 
        The objective here is to predict the customer satisfaction score for a given order based on features like order status, price, payment, etc. 
        Using [ZenML](https://zenml.io/), we will build a production-ready pipeline to predict customer satisfaction scores for future orders.
        """
    )

    # Pipeline Description
    st.markdown(
        """ 
        The pipeline includes data ingestion, cleaning, model training, and evaluation. If there are changes in the data source or hyperparameters, 
        deployment is triggered, retraining the model. If the model meets accuracy requirements, it will be deployed.
        """
    )

    # Feature Description Section
    st.markdown(
        """ 
        ### Description of Features 
        This app predicts customer satisfaction scores. Input the following product features to get the predicted score. 
        """
    )
    
    feature_table = pd.DataFrame({
        "Models": ["Payment Sequential", "Payment Installments", "Payment Value", "Price", 
                   "Freight Value", "Product Name Length", "Product Description Length", 
                   "Product Photos Quantity", "Product Weight (g)", "Product Length (cm)", 
                   "Product Height (cm)", "Product Width (cm)"],
        "Description": [
            "Sequence of payments made by the customer.",
            "Number of installments chosen by the customer.",
            "Total amount paid by the customer.",
            "Price of the product.",
            "Freight value of the product.",
            "Length of the product name.",
            "Length of the product description.",
            "Number of published product photos.",
            "Weight of the product in grams.",
            "Length of the product in centimeters.",
            "Height of the product in centimeters.",
            "Width of the product in centimeters."
        ]
    })

    st.table(feature_table)

    # Sidebar Inputs
    st.sidebar.header("Input Features")
    payment_sequential = st.sidebar.slider("Payment Sequential", min_value=0, max_value=10)
    payment_installments = st.sidebar.slider("Payment Installments", min_value=1, max_value=12)
    payment_value = st.sidebar.number_input("Payment Value", min_value=0.0, step=0.01)
    price = st.sidebar.number_input("Price", min_value=0.0, step=0.01)
    freight_value = st.sidebar.number_input("Freight Value", min_value=0.0, step=0.01)
    product_name_length = st.sidebar.number_input("Product Name Length", min_value=1)
    product_description_length = st.sidebar.number_input("Product Description Length", min_value=1)
    product_photos_qty = st.sidebar.number_input("Product Photos Quantity", min_value=0, step=1)
    product_weight_g = st.sidebar.number_input("Product Weight (g)", min_value=0)
    product_length_cm = st.sidebar.number_input("Product Length (cm)", min_value=0)
    product_height_cm = st.sidebar.number_input("Product Height (cm)", min_value=0)
    product_width_cm = st.sidebar.number_input("Product Width (cm)", min_value=0)

    # Prediction Button
    if st.button("Predict"):
        service = prediction_service_loader(
            pipeline_name="continuous_deployment_pipeline",
            pipeline_step_name="mlflow_model_deployer_step",
            running=False,
        )
        if service is None:
            st.warning("No service found. The pipeline will run first to create a service.")
            main()

        # Prepare DataFrame for prediction
        df = pd.DataFrame({
            "payment_sequential": [payment_sequential],
            "payment_installments": [payment_installments],
            "payment_value": [payment_value],
            "price": [price],
            "freight_value": [freight_value],
            "product_name_length": [product_name_length],
            "product_description_length": [product_description_length],
            "product_photos_qty": [product_photos_qty],
            "product_weight_g": [product_weight_g],
            "product_length_cm": [product_length_cm],
            "product_height_cm": [product_height_cm],
            "product_width_cm": [product_width_cm],
        })
        
        json_list = json.loads(json.dumps(list(df.T.to_dict().values())))
        data = np.array(json_list)
        pred = service.predict(data)

        st.success(f"Your Customer Satisfaction Score (range between 0 - 5) is: {pred[0]:.2f}")

    # Results Section
    if st.button("Show Results"):
        st.markdown(
            "We have experimented with two ensemble and tree-based models and compared their performances."
        )

        results_df = pd.DataFrame({
            "Models": ["LightGBM", "XGBoost"],
            "MSE": [1.804, 1.781],
            "RMSE": [1.343, 1.335],
        })
        st.dataframe(results_df)

        st.markdown(
            "The following figure shows the importance of each feature in predicting customer satisfaction."
        )
        feature_importance_image = Image.open("_assets/feature_importance_gain.png")
        st.image(feature_importance_image, caption="Feature Importance Gain", use_column_width=True)

if __name__ == "__main__":
    main()
