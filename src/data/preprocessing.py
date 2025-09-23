#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Data preprocessing module for sales forecasting.
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DataPreprocessor:
    """Class for preprocessing raw sales data."""
    
    def __init__(self, output_dir="data/processed"):
        """
        Initialize the preprocessor.
        
        Args:
            output_dir: Directory to save processed data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def preprocess_transactions(self, df):
        """
        Preprocess transaction data.
        
        Args:
            df: Raw transaction dataframe
        
        Returns:
            Cleaned transaction dataframe
        """
        
        # Make a copy to avoid modifying the original
        transactions = df.copy()
        
        # Convert date columns to datetime
        transactions['transaction_date'] = pd.to_datetime(transactions['transaction_date'])
        transactions['reference_date'] = pd.to_datetime(transactions['reference_date'])
        
        # Extract datetime features
        transactions['year'] = transactions['transaction_date'].dt.year
        transactions['month'] = transactions['transaction_date'].dt.month
        transactions['week'] = transactions['transaction_date'].dt.isocalendar().week
        transactions['day'] = transactions['transaction_date'].dt.day
        transactions['dayofweek'] = transactions['transaction_date'].dt.dayofweek
        transactions['is_weekend'] = transactions['dayofweek'].isin([5, 6]).astype(int)
        
        # Filter data for 2022
        transactions = transactions[transactions['year'] == 2022]
        
        # Handle negative quantities (returns)
        
        # Option 1: Remove negative quantities
        # transactions = transactions[transactions['quantity'] >= 0]
        
        # Option 2: Keep track of returns as a separate feature
        transactions['is_return'] = (transactions['quantity'] < 0).astype(int)
        
        # Check for missing values
        missing_values = transactions.isnull().sum()
        
        # Handle outliers in quantity
        q_low = transactions['quantity'].quantile(0.001)
        q_high = transactions['quantity'].quantile(0.999)
        transactions_filtered = transactions[
            (transactions['quantity'] >= q_low) & 
            (transactions['quantity'] <= q_high)
        ]
        
        transactions = transactions_filtered
        
        # Calculate transaction metrics
        transactions['unit_price'] = transactions['gross_value'] / transactions['quantity']
        transactions['profit_margin'] = transactions['gross_profit'] / transactions['gross_value']
        transactions['discount_ratio'] = transactions['discount'] / transactions['gross_value']
        
        # Replace infinite values with NaN and then fill with 0
        transactions.replace([np.inf, -np.inf], np.nan, inplace=True)
        transactions.fillna({
            'unit_price': transactions['unit_price'].median(),
            'profit_margin': transactions['profit_margin'].median(),
            'discount_ratio': 0
        }, inplace=True)
        
        return transactions
    
    def preprocess_pdv(self, df):
        """
        Preprocess PDV (point of sale) data.
        
        Args:
            df: Raw PDV dataframe
        
        Returns:
            Cleaned PDV dataframe
        """
        
        # Make a copy to avoid modifying the original
        pdvs = df.copy()
        
        # Rename columns for consistency
        pdvs.rename(columns={'premise': 'premise_type'}, inplace=True, errors='ignore')
        
        
        # Convert categorical columns to category type
        for col in ['premise_type', 'categoria_pdv']:
            if col in pdvs.columns:
                pdvs[col] = pdvs[col].astype('category')
        
        # Create binary indicator for premise type
        if 'premise_type' in pdvs.columns:
            pdvs['is_on_premise'] = (pdvs['premise_type'] == 'On Premise').astype(int)
        
        return pdvs
    
    def preprocess_products(self, df):
        """
        Preprocess product data.
        
        Args:
            df: Raw product dataframe
        
        Returns:
            Cleaned product dataframe
        """
        
        # Make a copy to avoid modifying the original
        products = df.copy()
        
        # Check for missing values
        missing_values = products.isnull().sum()
        if missing_values.sum() > 0:
            # Fill missing categorical values with 'Unknown'
            for col in products.columns[missing_values > 0]:
                if products[col].dtype == 'object':
                    products[col].fillna('Unknown', inplace=True)
        
        # Convert categorical columns to category type
        for col in ['categoria', 'tipos', 'label', 'subcategoria']:
            if col in products.columns:
                products[col] = products[col].astype('category')
        
        return products
    
    def save_processed_data(self, transactions=None, pdvs=None, products=None):
        """
        Save processed data to disk.
        
        Args:
            transactions: Processed transactions dataframe
            pdvs: Processed PDVs dataframe
            products: Processed products dataframe
        """
        if transactions is not None:
            transactions.to_parquet(self.output_dir / "transactions_processed.parquet", index=False)
        
        if pdvs is not None:
            pdvs.to_parquet(self.output_dir / "pdvs_processed.parquet", index=False)
        
        if products is not None:
            products.to_parquet(self.output_dir / "products_processed.parquet", index=False)
