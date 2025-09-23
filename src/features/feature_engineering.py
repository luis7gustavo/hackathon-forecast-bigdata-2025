#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Feature engineering module for sales forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import LabelEncoder


class FeatureEngineer:
    """Class for creating features for sales forecasting."""
    
    def __init__(self, output_dir="data/processed"):
        """
        Initialize the feature engineer.
        
        Args:
            output_dir: Directory to save feature data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.label_encoders = {}
    
    def create_time_features(self, df, date_col='transaction_date'):
        """
        Create time-based features.
        
        Args:
            df: DataFrame with date column
            date_col: Name of date column
            
        Returns:
            DataFrame with time features added
        """
        
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Extract basic date components
        df['year'] = df[date_col].dt.year
        df['month'] = df[date_col].dt.month
        df['week'] = df[date_col].dt.isocalendar().week
        df['day'] = df[date_col].dt.day
        df['dayofweek'] = df[date_col].dt.dayofweek
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_month_start'] = df[date_col].dt.is_month_start.astype(int)
        df['is_month_end'] = df[date_col].dt.is_month_end.astype(int)
        df['quarter'] = df[date_col].dt.quarter
        
        # Create cyclical features for day, month, and week
        # This preserves the cyclical nature of these features
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
        df['day_sin'] = np.sin(2 * np.pi * df['day'] / 31)
        df['day_cos'] = np.cos(2 * np.pi * df['day'] / 31)
        df['dayofweek_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['dayofweek_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        return df
    
    def create_lag_features(self, df, groupby_cols, target_col='quantity', lags=[1, 2, 3, 4]):
        """
        Create lag features for time series.
        
        Args:
            df: DataFrame with time series data
            groupby_cols: List of columns to group by
            target_col: Target column to create lags for
            lags: List of lag periods to create
            
        Returns:
            DataFrame with lag features added
        """
        
        df = df.copy()
        
        # Sort by date within groups
        sort_cols = groupby_cols + ['transaction_date']
        df = df.sort_values(sort_cols)
        
        # Create group key for lag creation
        if len(groupby_cols) > 1:
            df['group_key'] = df[groupby_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
            group_col = 'group_key'
        else:
            group_col = groupby_cols[0]
        
        # Create lag features
        for lag in lags:
            lag_col = f"{target_col}_lag_{lag}"
            df[lag_col] = df.groupby(group_col)[target_col].shift(lag)
        
        # Drop the temporary group key if created
        if len(groupby_cols) > 1:
            df.drop('group_key', axis=1, inplace=True)
        
        return df
    
    def create_rolling_features(self, df, groupby_cols, target_col='quantity', windows=[2, 4, 8, 12]):
        """
        Create rolling window features.
        
        Args:
            df: DataFrame with time series data
            groupby_cols: List of columns to group by
            target_col: Target column to create rolling features for
            windows: List of window sizes
            
        Returns:
            DataFrame with rolling features added
        """
        
        df = df.copy()
        
        # Sort by date within groups
        sort_cols = groupby_cols + ['transaction_date']
        df = df.sort_values(sort_cols)
        
        # Create group key for rolling feature creation
        if len(groupby_cols) > 1:
            df['group_key'] = df[groupby_cols].apply(lambda x: '_'.join(x.astype(str)), axis=1)
            group_col = 'group_key'
        else:
            group_col = groupby_cols[0]
        
        # Create rolling features
        for window in windows:
            # Mean
            mean_col = f"{target_col}_roll_mean_{window}"
            df[mean_col] = df.groupby(group_col)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).mean()
            )
            
            # Standard deviation
            std_col = f"{target_col}_roll_std_{window}"
            df[std_col] = df.groupby(group_col)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).std()
            )
            
            # Min and Max
            min_col = f"{target_col}_roll_min_{window}"
            df[min_col] = df.groupby(group_col)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).min()
            )
            
            max_col = f"{target_col}_roll_max_{window}"
            df[max_col] = df.groupby(group_col)[target_col].transform(
                lambda x: x.rolling(window=window, min_periods=1).max()
            )
        
        # Drop the temporary group key if created
        if len(groupby_cols) > 1:
            df.drop('group_key', axis=1, inplace=True)
        
        # Fill NaN values
        rolling_cols = [col for col in df.columns if 'roll_' in col]
        df[rolling_cols] = df[rolling_cols].fillna(0)
        
        return df
    
    def encode_categorical_features(self, df, cat_cols):
        """
        Encode categorical features using Label Encoding.
        
        Args:
            df: DataFrame with categorical features
            cat_cols: List of categorical columns to encode
            
        Returns:
            DataFrame with encoded categorical features
        """
        
        df = df.copy()
        
        for col in cat_cols:
            if col in df.columns:
                # Create or reuse label encoder
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    df[f"{col}_encoded"] = self.label_encoders[col].fit_transform(df[col].fillna('Unknown'))
                else:
                    # Handle unknown categories
                    df[col] = df[col].fillna('Unknown')
                    known_categories = set(self.label_encoders[col].classes_)
                    df.loc[~df[col].isin(known_categories), col] = 'Unknown'
                    df[f"{col}_encoded"] = self.label_encoders[col].transform(df[col])
        
        return df
    
    def merge_datasets(self, transactions, pdvs, products):
        """
        Merge transaction data with PDV and product information.
        
        Args:
            transactions: Transaction DataFrame
            pdvs: PDV DataFrame
            products: Product DataFrame
            
        Returns:
            Merged DataFrame
        """
        
        # Merge with PDVs
        merged_data = transactions.merge(
            pdvs,
            left_on='internal_store_id',
            right_on='pdv',
            how='left'
        )
        
        # Merge with products
        merged_data = merged_data.merge(
            products,
            left_on='internal_product_id',
            right_on='produto',
            how='left'
        )
        
        return merged_data
    
    def create_features(self, transactions, pdvs, products):
        """
        Create all features for sales forecasting.
        
        Args:
            transactions: Transaction DataFrame
            pdvs: PDV DataFrame
            products: Product DataFrame
            
        Returns:
            Feature DataFrame ready for modeling
        """
        
        # 1. Merge datasets
        merged_data = self.merge_datasets(transactions, pdvs, products)
        
        # 2. Create time features
        feature_data = self.create_time_features(merged_data)
        
        # 3. Aggregate data to weekly level by PDV and product
        groupby_cols = ['internal_store_id', 'internal_product_id', 'year', 'week']
        weekly_data = feature_data.groupby(groupby_cols).agg({
            'quantity': 'sum',
            'gross_value': 'sum',
            'net_value': 'sum',
            'gross_profit': 'sum',
            'discount': 'sum',
            'taxes': 'sum',
            'is_weekend': 'mean',
            'zipcode': 'first',
            'categoria': 'first',
            'tipos': 'first',
            'label': 'first',
            'subcategoria': 'first',
            'marca': 'first',
            'fabricante': 'first',
            'premise_type': 'first',
            'categoria_pdv': 'first'
        }).reset_index()
        
        # 4. Create unit price and other ratio features
        weekly_data['unit_price'] = weekly_data['gross_value'] / weekly_data['quantity']
        weekly_data['profit_margin'] = weekly_data['gross_profit'] / weekly_data['gross_value']
        weekly_data['discount_ratio'] = weekly_data['discount'] / weekly_data['gross_value']
        
        # Replace infinite values and NaNs
        weekly_data.replace([np.inf, -np.inf], np.nan, inplace=True)
        weekly_data.fillna({
            'unit_price': weekly_data['unit_price'].median(),
            'profit_margin': weekly_data['profit_margin'].median(),
            'discount_ratio': 0
        }, inplace=True)
        
        # 5. Create lag features
        group_cols = ['internal_store_id', 'internal_product_id']
        weekly_data = self.create_lag_features(
            weekly_data, 
            groupby_cols=group_cols,
            target_col='quantity',
            lags=[1, 2, 3, 4, 8, 12]  # Previous weeks
        )
        
        # 6. Create rolling features
        weekly_data = self.create_rolling_features(
            weekly_data,
            groupby_cols=group_cols,
            target_col='quantity',
            windows=[2, 4, 8, 12]
        )
        
        # 7. Encode categorical features
        cat_cols = [
            'categoria', 'tipos', 'label', 'subcategoria', 'marca', 'fabricante',
            'premise_type', 'categoria_pdv'
        ]
        weekly_data = self.encode_categorical_features(weekly_data, cat_cols)
        
        # 8. Create week of year cyclical features
        weekly_data['week_sin'] = np.sin(2 * np.pi * weekly_data['week'] / 52)
        weekly_data['week_cos'] = np.cos(2 * np.pi * weekly_data['week'] / 52)
        
        # 9. Save feature data
        weekly_data.to_parquet(self.output_dir / "feature_data.parquet", index=False)
        
        return weekly_data