"""
Copyright (c) 2025 Harvard Ability lab
Title: "A smartphone activity monitor that accurately estimates energy expenditure"
"""

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import RandomizedSearchCV, KFold
from xgboost import XGBRegressor

# Define a class for XGBoost regressor
class XGBregressor:
    def __init__(self):
        pass

    def predict(self, dataset):
        x_train = dataset['x_train']
        y_train = dataset['y_train']
        
        print('XGBoost - x_train:', x_train.shape, ' y_train:', y_train.shape)

        # Define parameter grid for hyperparameter tuning
        param_grid = {
            'n_estimators': [300, 400, 500, 1000],
            'max_depth': [3, 4, 5, 8, 10],
            'eta': [0.01, 0.05, 0.1, 0.3],
            'subsample': [0.6, 0.7, 0.8],
            'colsample_bytree': [0.6, 0.7, 0.8]
        }

        xgb_model = XGBRegressor()

        # Perform random search with cross-validation
        kfold = KFold(n_splits=5)
        grid_search = RandomizedSearchCV(
            estimator=xgb_model,
            param_distributions=param_grid,
            scoring='neg_mean_squared_error',
            cv=kfold,
            n_iter=10,  # Number of iterations for RandomizedSearchCV
            random_state=42,
            return_train_score=True,
            n_jobs=-1
        )

        grid_search.fit(x_train, y_train[:, 0])
        
        # Get the best parameters and model
        best_params = grid_search.best_params_
        print("Best Parameters:", best_params)

        best_model = grid_search.best_estimator_

        # Calculate overall average validation score
        cv_results = grid_search.cv_results_
        val_scores = -cv_results['mean_test_score']
        print(f"Overall Average Validation Score: {np.mean(val_scores):.4f}")

        # Save feature importance to a CSV file
        feature_importance_path = './feature_importance'
        os.makedirs(feature_importance_path, exist_ok=True)
        df_feature = pd.DataFrame(best_model.feature_importances_)
        df_feature.to_csv(os.path.join(feature_importance_path, 'xgboost_model_features.csv'), index=False)

        # Save the trained model
        model_save_path = './saved_model_weight'
        os.makedirs(model_save_path, exist_ok=True)
        model_filename = 'data_driven_ee_model.pkl'
        pickle.dump(best_model, open(os.path.join(model_save_path, model_filename), 'wb'))
