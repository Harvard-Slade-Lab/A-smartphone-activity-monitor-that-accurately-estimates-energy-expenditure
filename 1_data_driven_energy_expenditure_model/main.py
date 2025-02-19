"""
Copyright (c) 2025 Harvard Ability lab
Title: "A smartphone activity monitor that accurately estimates energy expenditure"
"""

import os
import ml_models
from utils import load_dataset

def main():
    # Define the path to the training data directory
    data_path = './training_data'

    # List all files in the data directory
    data_list = os.listdir(data_path)

    # Load the dataset
    dataset = load_dataset(data_path, data_list)

    # Initialize and use the XGBoost regressor model
    xgb_regressor = ml_models.XGBregressor()
    xgb_regressor.predict(dataset)

if __name__ == "__main__":
    main()
