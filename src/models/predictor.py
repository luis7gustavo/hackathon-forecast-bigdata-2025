#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Prediction module for sales forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import joblib


class SalesPredictor:
    """Class for generating sales forecasts."""
    
    def __init__(self, output_dir="results/predictions"):
        """
        Initialize the predictor.
        
        Args:
            output_dir: Directory to save predictions
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def generate_forecast_features(self, feature_data, forecast_weeks=5):
        """
        Generate features for forecast period.
        
        Args:
            feature_data: Feature DataFrame from 2022
            forecast_weeks: Number of weeks to forecast
            
        Returns:
            DataFrame with features for forecast period
        """
        
        # Get unique PDV-product combinations
        pdv_product_pairs = feature_data[['internal_store_id', 'internal_product_id']].drop_duplicates()
        
        # Get the last week of data
        max_year = feature_data['year'].max()
        max_week = feature_data[feature_data['year'] == max_year]['week'].max()
        
        # Get the last data point for each PDV-product combination
        last_entries = feature_data.loc[
            feature_data.groupby(['internal_store_id', 'internal_product_id'])['week'].idxmax()
        ]
        
        # Create forecast period data
        forecast_data = []
        
        # For each PDV-product pair, generate features for forecast weeks
        for _, row in pdv_product_pairs.iterrows():
            pdv = row['internal_store_id']
            product = row['internal_product_id']
            
            # Get the last entry for this PDV-product pair
            last_entry = last_entries[
                (last_entries['internal_store_id'] == pdv) &
                (last_entries['internal_product_id'] == product)
            ]
            
            if len(last_entry) == 0:
                continue  # Skip if no historical data for this pair
            
            # Extract values from the last entry
            last_entry = last_entry.iloc[0]
            
            # Generate entries for each forecast week
            for i in range(1, forecast_weeks + 1):
                # Create new week
                week = (max_week + i) % 52
                week = 52 if week == 0 else week
                
                year = max_year if week > max_week else max_year + 1
                
                # Create new entry based on the last known entry
                new_entry = last_entry.copy()
                new_entry['year'] = year
                new_entry['week'] = week
                
                # Set the forecasted week's features
                new_entry['week_sin'] = np.sin(2 * np.pi * week / 52)
                new_entry['week_cos'] = np.cos(2 * np.pi * week / 52)
                
                # Use the last values for lag features
                for lag in range(1, 13):  # Assuming we have lags 1-12
                    lag_col = f"quantity_lag_{lag}"
                    if lag_col in feature_data.columns:
                        if lag == 1:
                            # For the first forecast week, lag 1 is the last actual week
                            new_entry[lag_col] = last_entry['quantity']
                        elif i >= lag:
                            # Use previously forecasted values
                            prev_forecast = forecast_data[-(lag-1)]['quantity']
                            new_entry[lag_col] = prev_forecast
                        # Otherwise, keep the lag from the last entry
                
                # Update rolling features based on recent values
                for window in [2, 4, 8, 12]:
                    if f"quantity_roll_mean_{window}" in feature_data.columns:
                        # Use the last entry's rolling features
                        # In a more sophisticated implementation, we would update these
                        # based on the forecasted values as we go
                        pass
                
                forecast_data.append(new_entry)
        
        # Convert to DataFrame
        forecast_df = pd.DataFrame(forecast_data)
        
        # Add semana column (1-5) for January 2023
        forecast_df['semana'] = forecast_df.index % forecast_weeks + 1
        
        return forecast_df
    
    def prepare_forecast_features(self, forecast_df, drop_cols=None):
        """
        Prepare features for prediction.
        
        Args:
            forecast_df: DataFrame with forecast period features
            drop_cols: Columns to drop before prediction
            
        Returns:
            X_forecast for prediction
        """
        # Make a copy to avoid modifying the original
        X_forecast = forecast_df.copy()
        
        # Default columns to drop
        if drop_cols is None:
            drop_cols = ['quantity', 'transaction_date', 'reference_date', 'semana']
        
        # Drop unnecessary columns
        for col in drop_cols:
            if col in X_forecast.columns:
                X_forecast.drop(col, axis=1, inplace=True)
        
        return X_forecast
    
    def generate_forecast(self, model, feature_data, forecast_weeks=5, pdv_data=None, product_data=None):
        """
        Generate sales forecast.
        
        Args:
            model: Trained model
            feature_data: Feature DataFrame from 2022
            forecast_weeks: Number of weeks to forecast
            pdv_data: PDV metadata
            product_data: Product metadata
            
        Returns:
            DataFrame with forecasted sales
        """
        
        # Generate features for forecast period
        forecast_df = self.generate_forecast_features(feature_data, forecast_weeks)
        
        # Prepare features for prediction
        X_forecast = self.prepare_forecast_features(forecast_df)
        
        # Make predictions
        forecast_df['quantidade'] = np.round(model.predict(X_forecast))
        
        # Ensure predictions are non-negative
        forecast_df['quantidade'] = forecast_df['quantidade'].clip(lower=0)
        
        # Map to PDV and product codes
        forecast_df['pdv'] = forecast_df['internal_store_id']
        forecast_df['produto'] = forecast_df['internal_product_id']
        
        # Select required columns
        result_df = forecast_df[['semana', 'pdv', 'produto', 'quantidade']]
        
        # Save forecast
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = self.output_dir / f"forecast_{timestamp}.csv"
        result_df.to_csv(output_path, index=False, sep=';', encoding='utf-8')
        
        return forecast_df
    
    def load_model(self, model_path):
        """
        Load a saved model.
        
        Args:
            model_path: Path to saved model
            
        Returns:
            Loaded model
        """
        model = joblib.load(model_path)
        return model