#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Model training module for sales forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import joblib
import time
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor


class ModelTrainer:
    """Class for training sales forecast models."""
    
    def __init__(self, model_dir="models/saved"):
        """
        Initialize the model trainer.
        
        Args:
            model_dir: Directory to save trained models
        """
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
    
    def prepare_training_data(self, df, target_col='quantity', test_size=0.2):
        """
        Prepare data for model training.
        
        Args:
            df: Feature DataFrame
            target_col: Target column name
            test_size: Proportion of data to use for testing
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        
        # Drop rows with missing target values
        df = df.dropna(subset=[target_col])
        
        # Sort by time for proper train/test split
        df = df.sort_values(['year', 'week'])
        
        # Remove features not useful for training
        drop_cols = ['transaction_date', 'reference_date'] + \
                    [col for col in df.columns if col.endswith('_encoded')] + \
                    [col for col in df.columns if col in ['marca', 'fabricante', 'descricao']]
        
        drop_cols = [col for col in drop_cols if col in df.columns]
        
        # Create feature matrix and target vector
        X = df.drop([target_col] + drop_cols, axis=1, errors='ignore')
        y = df[target_col]
        
        # Handle remaining categorical features
        cat_cols = X.select_dtypes(include=['category', 'object']).columns
        for col in cat_cols:
            X[col] = X[col].astype('category')
        
        # Split into training and testing sets by time
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        return X_train, X_test, y_train, y_test
    
    def initialize_model(self, model_type='lightgbm'):
        """
        Initialize the specified model type.
        
        Args:
            model_type: Type of model ('lightgbm', 'xgboost', 'catboost', 'random_forest')
            
        Returns:
            Initialized model
        """
        
        if model_type == 'lightgbm':
            model = lgb.LGBMRegressor(
                objective='regression',
                n_estimators=1000,
                learning_rate=0.05,
                num_leaves=31,
                feature_fraction=0.8,
                subsample=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42
            )
        elif model_type == 'xgboost':
            model = xgb.XGBRegressor(
                objective='reg:squarederror',
                n_estimators=1000,
                learning_rate=0.05,
                max_depth=7,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42
            )
        elif model_type == 'catboost':
            model = CatBoostRegressor(
                iterations=1000,
                learning_rate=0.05,
                depth=7,
                loss_function='RMSE',
                eval_metric='RMSE',
                random_seed=42,
                verbose=False
            )
        elif model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=500,
                max_depth=12,
                min_samples_split=5,
                min_samples_leaf=2,
                n_jobs=-1,
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        return model
    
    def evaluate_model(self, model, X, y, cv=5):
        """
        Evaluate model with cross-validation.
        
        Args:
            model: Model to evaluate
            X: Features
            y: Target
            cv: Number of cross-validation folds
            
        Returns:
            Cross-validation results
        """
        
        # Use TimeSeriesSplit for temporal data
        tscv = TimeSeriesSplit(n_splits=cv)
        
        # Calculate cross-validation scores
        cv_mae = cross_val_score(
            model, X, y, 
            scoring='neg_mean_absolute_error',
            cv=tscv,
            n_jobs=-1
        )
        
        cv_rmse = cross_val_score(
            model, X, y, 
            scoring='neg_root_mean_squared_error',
            cv=tscv,
            n_jobs=-1
        )
        
        # Calculate average metrics
        avg_mae = -cv_mae.mean()
        avg_rmse = -cv_rmse.mean()
        
        
        return {
            'mae': avg_mae,
            'rmse': avg_rmse,
            'mae_std': cv_mae.std(),
            'rmse_std': cv_rmse.std()
        }
    
    def train_model(self, df, target_col='quantity', model_type='lightgbm', cv=5):
        """
        Train a sales forecast model.
        
        Args:
            df: Feature DataFrame
            target_col: Target column name
            model_type: Type of model to train
            cv: Number of cross-validation folds
            
        Returns:
            Trained model
        """
        
        # Prepare data
        X_train, X_test, y_train, y_test = self.prepare_training_data(df, target_col)
        
        # Initialize model
        model = self.initialize_model(model_type)
        
        # Evaluate with cross-validation
        cv_results = self.evaluate_model(model, X_train, y_train, cv)
        
        # Train on full training set
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Evaluate on test set
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        
        # Train final model on all data
        X = pd.concat([X_train, X_test])
        y = pd.concat([y_train, y_test])
        model.fit(X, y)
        
        # Save model
        model_path = self.model_dir / f"{model_type}_sales_forecast_{int(time.time())}.joblib"
        joblib.dump(model, model_path)
        
        return model
    
    def get_feature_importance(self, model):
        """
        Get feature importance from trained model.
        
        Args:
            model: Trained model with feature_importances_
            
        Returns:
            DataFrame with feature importances
        """
        try:
            # Extract feature importances based on model type
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = model.feature_name_ if hasattr(model, 'feature_name_') else None
            elif hasattr(model, 'feature_name_'):
                importances = model.feature_importance()
                feature_names = model.feature_name_
            elif hasattr(model, 'get_feature_importance'):
                importances = model.get_feature_importance()
                feature_names = model.feature_names_
            else:
                return pd.DataFrame()
            
            # If feature names not found in model, try to get from model's booster
            if feature_names is None and hasattr(model, 'get_booster'):
                try:
                    feature_names = model.get_booster().feature_names
                except:
                    feature_names = [f'feature_{i}' for i in range(len(importances))]
            
            # Create DataFrame with importances
            importance_df = pd.DataFrame({
                'feature': feature_names if feature_names else [f'feature_{i}' for i in range(len(importances))],
                'importance': importances
            })
            
            # Sort by importance
            importance_df = importance_df.sort_values('importance', ascending=False)
            
            return importance_df
            
        except Exception as e:
            return pd.DataFrame()