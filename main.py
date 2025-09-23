#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Main script to run the sales forecasting pipeline.
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import argparse
import warnings

from src.data.preprocessing import DataPreprocessor
from src.features.feature_engineering import FeatureEngineer
from src.models.trainer import ModelTrainer
from src.models.predictor import SalesPredictor
from src.visualization.visualize import SalesVisualizer


# Suppress specific warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def create_dirs():
    """Create necessary directories if they don't exist"""
    directories = ["data/processed", "models/saved", "results/predictions", "results/visualizations"]
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)


def load_data():
    """Load raw data from parquet files"""
    
    # File paths
    transaction_file = "data/part-00000-tid-5196563791502273604-c90d3a24-52f2-4955-b4ec-fb143aae74d8-4-1-c000.snappy.parquet"
    pdv_file = "data/part-00000-tid-7173294866425216458-eae53fbf-d19e-4130-ba74-78f96b9675f1-4-1-c000.snappy.parquet"
    product_file = "data/part-00000-tid-2779033056155408584-f6316110-4c9a-4061-ae48-69b77c7c8c36-4-1-c000.snappy.parquet"
    
    # Load data
    transactions_df = pd.read_parquet(transaction_file)
    pdv_df = pd.read_parquet(pdv_file)
    product_df = pd.read_parquet(product_file)
    
    
    return transactions_df, pdv_df, product_df


def run_pipeline(args):
    """Run the full sales forecasting pipeline"""
    start_time = datetime.now()
    
    # Create necessary directories
    create_dirs()
    
    # Load data
    transactions_df, pdv_df, product_df = load_data()
    
    # Initialize components
    preprocessor = DataPreprocessor()
    feature_engineer = FeatureEngineer()
    trainer = ModelTrainer()
    predictor = SalesPredictor()
    visualizer = SalesVisualizer()
    
    # 1. Preprocess data
    clean_transactions = preprocessor.preprocess_transactions(transactions_df)
    clean_pdv = preprocessor.preprocess_pdv(pdv_df)
    clean_products = preprocessor.preprocess_products(product_df)
    
    # Save processed data
    preprocessor.save_processed_data(clean_transactions, clean_pdv, clean_products)
    
    # 2. Generate features
    feature_data = feature_engineer.create_features(clean_transactions, clean_pdv, clean_products)
    
    # 3. Train model
    model = trainer.train_model(
        feature_data,
        target_col='quantity',
        model_type=args.model_type,
        cv=args.cv_folds
    )
    
    # 4. Generate predictions for January 2023 (weeks 1-5)
    forecast_data = predictor.generate_forecast(
        model=model,
        feature_data=feature_data,
        forecast_weeks=5,
        pdv_data=clean_pdv,
        product_data=clean_products
    )
    
    # 5. Save predictions
    output_file = f"results/predictions/sales_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    forecast_data[['semana', 'pdv', 'produto', 'quantidade']].to_csv(
        output_file, index=False, sep=';', encoding='utf-8'
    )
    
    # 6. Create visualizations if requested
    if args.visualize:
        # Sales trends
        visualizer.plot_sales_trend(clean_transactions, save=True)
        
        # Category distributions
        visualizer.plot_category_distribution(
            pd.merge(clean_transactions, clean_products, left_on='internal_product_id', right_on='produto'),
            cat_col='categoria', 
            value_col='quantity',
            save=True
        )
        
        # PDV category distribution
        visualizer.plot_category_distribution(
            pd.merge(clean_transactions, clean_pdv, left_on='internal_store_id', right_on='pdv'),
            cat_col='categoria_pdv', 
            value_col='quantity',
            save=True
        )
        
        # Seasonal patterns
        visualizer.plot_seasonal_patterns(clean_transactions, save=True)
        
        # Feature importance
        feature_importance = trainer.get_feature_importance(model)
        visualizer.plot_feature_importance(feature_importance, save=True)
    
    end_time = datetime.now()
    
    return forecast_data


if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description="Sales Forecasting Pipeline")
    
    parser.add_argument("--model-type", type=str, default="lightgbm",
                       choices=["lightgbm", "xgboost", "catboost", "random_forest"],
                       help="Model type to use for forecasting")
    
    parser.add_argument("--cv-folds", type=int, default=5,
                       help="Number of cross-validation folds")
    
    parser.add_argument("--visualize", action="store_true",
                       help="Generate visualizations")
    
    args = parser.parse_args()
    
    # Run pipeline
    forecast_data = run_pipeline(args)
    
    print("\nForecast Summary:")
    print(f"Total predicted sales: {forecast_data['quantidade'].sum():.0f} units")
    print(f"Unique PDVs: {forecast_data['pdv'].nunique()}")
    print(f"Unique Products: {forecast_data['produto'].nunique()}")