import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from statsmodels.tsa.seasonal import seasonal_decompose
import calendar

class SalesVisualizer:
    """Class for visualizing sales data and forecasting results"""
    
    def __init__(self, output_dir=None):
        """Initialize visualizer with output directory for saving visualizations"""
        self.output_dir = Path(output_dir) if output_dir else Path("results/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set styling for matplotlib
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("viridis")
        
    def plot_sales_trend(self, df, time_col='transaction_date', value_col='quantity', 
                         title='Sales Trend Over Time', freq='W', save=False):
        """
        Plot time series of sales aggregated by specified frequency
        
        Args:
            df: DataFrame with sales data
            time_col: Column name for date
            value_col: Column name for sales value to plot
            title: Plot title
            freq: Frequency for aggregation ('D', 'W', 'M')
            save: Whether to save the plot
        """
        # Ensure date column is datetime
        df[time_col] = pd.to_datetime(df[time_col])
        
        # Aggregate sales by time period
        sales_agg = df.groupby(pd.Grouper(key=time_col, freq=freq))[value_col].sum().reset_index()
        
        # Create plot
        fig = px.line(sales_agg, x=time_col, y=value_col, 
                     title=f"{title} ({freq})",
                     labels={time_col: 'Date', value_col: 'Sales'})
        fig.update_layout(template='plotly_white')
        
        # Display or save
        fig.show()
        if save:
            fig.write_image(self.output_dir / f"sales_trend_{freq}.png")
            
    def plot_category_distribution(self, df, cat_col, value_col='quantity', 
                                 title=None, top_n=10, save=False):
        """
        Plot sales distribution by category
        
        Args:
            df: DataFrame with sales data
            cat_col: Column name for category
            value_col: Column to aggregate (e.g., quantity, revenue)
            title: Plot title
            top_n: Number of top categories to show
            save: Whether to save the plot
        """
        # Aggregate by category
        cat_agg = df.groupby(cat_col)[value_col].sum().reset_index()
        
        # Get top N categories
        cat_agg = cat_agg.sort_values(value_col, ascending=False).head(top_n)
        
        # Create plot
        fig = px.bar(cat_agg, x=cat_col, y=value_col, 
                    title=title or f"Top {top_n} {cat_col} by {value_col}",
                    labels={cat_col: cat_col.replace('_', ' ').title(), 
                           value_col: value_col.replace('_', ' ').title()})
        
        fig.update_layout(template='plotly_white')
        
        # Display or save
        fig.show()
        if save:
            fig.write_image(self.output_dir / f"{cat_col}_distribution.png")
    
    def plot_seasonal_patterns(self, df, date_col='transaction_date', value_col='quantity', 
                             period='weekly', save=False):
        """
        Plot seasonal patterns in sales data
        
        Args:
            df: DataFrame with sales data
            date_col: Column name with date
            value_col: Column with sales value
            period: 'daily', 'weekly', or 'monthly'
            save: Whether to save the plot
        """
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        if period == 'daily':
            df['period'] = df[date_col].dt.day_name()
            order = list(calendar.day_name)
        elif period == 'weekly':
            df['period'] = df[date_col].dt.isocalendar().week
            # Group by week number
            period_data = df.groupby('period')[value_col].mean().reset_index()
            order = None
        elif period == 'monthly':
            df['period'] = df[date_col].dt.month_name()
            order = list(calendar.month_name)[1:]
        
        # If we're doing weekly analysis, we already have period_data
        if period != 'weekly':
            period_data = df.groupby('period')[value_col].mean().reset_index()
            if order:
                # Create categorical with proper order
                period_data['period'] = pd.Categorical(period_data['period'], categories=order, ordered=True)
                period_data = period_data.sort_values('period')
        
        # Plot
        fig = px.bar(period_data, x='period', y=value_col, 
                    title=f"{period.capitalize()} Seasonal Pattern",
                    labels={'period': period.capitalize(), value_col: f'Average {value_col}'})
        
        fig.update_layout(template='plotly_white')
        
        # Display or save
        fig.show()
        if save:
            fig.write_image(self.output_dir / f"{period}_seasonal_pattern.png")
    
    def plot_forecast_vs_actual(self, actual, forecast, date_col='week', 
                              value_col='quantity', groupby=None, save=False):
        """
        Plot forecast vs actual values
        
        Args:
            actual: DataFrame with actual values
            forecast: DataFrame with forecasted values
            date_col: Column representing the time period
            value_col: Column with the target value
            groupby: Optional column to group by before plotting
            save: Whether to save the plot
        """
        if groupby:
            # Aggregate by the specified groupby column
            actual_agg = actual.groupby([date_col, groupby])[value_col].sum().reset_index()
            forecast_agg = forecast.groupby([date_col, groupby])[value_col].sum().reset_index()
            
            # Create line plot with multiple lines per group
            fig = px.line(actual_agg, x=date_col, y=value_col, color=groupby, 
                         line_dash_sequence=['solid'],
                         labels={date_col: 'Time Period', value_col: 'Sales'},
                         title="Forecast vs Actual Sales")
            
            # Add forecast lines
            for group in forecast_agg[groupby].unique():
                group_data = forecast_agg[forecast_agg[groupby] == group]
                fig.add_scatter(x=group_data[date_col], y=group_data[value_col], 
                              mode='lines', line=dict(dash='dash'), 
                              name=f"{group} (Forecast)", showlegend=True)
        else:
            # Aggregate total values by date
            actual_agg = actual.groupby(date_col)[value_col].sum().reset_index()
            forecast_agg = forecast.groupby(date_col)[value_col].sum().reset_index()
            
            # Create plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(x=actual_agg[date_col], y=actual_agg[value_col],
                                  mode='lines+markers', name='Actual',
                                  line=dict(color='blue')))
            
            fig.add_trace(go.Scatter(x=forecast_agg[date_col], y=forecast_agg[value_col],
                                  mode='lines+markers', name='Forecast',
                                  line=dict(color='red', dash='dash')))
            
            fig.update_layout(title="Forecast vs Actual Sales",
                            xaxis_title='Time Period',
                            yaxis_title='Sales',
                            legend_title='Type',
                            template='plotly_white')
        
        # Display or save
        fig.show()
        if save:
            fig.write_image(self.output_dir / "forecast_vs_actual.png")
    
    def plot_feature_importance(self, importance_df, top_n=20, save=False):
        """
        Plot feature importance from a trained model
        
        Args:
            importance_df: DataFrame with feature names and importance values
            top_n: Number of top features to display
            save: Whether to save the plot
        """
        # Sort and get top N features
        df_plot = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        # Create plot
        fig = px.bar(df_plot, x='importance', y='feature', orientation='h',
                   title=f'Top {top_n} Feature Importance',
                   labels={'importance': 'Importance', 'feature': 'Feature'},
                   color='importance')
        
        fig.update_layout(template='plotly_white', yaxis={'categoryorder':'total ascending'})
        
        # Display or save
        fig.show()
        if save:
            fig.write_image(self.output_dir / "feature_importance.png")
    
    def plot_correlation_matrix(self, df, cols=None, threshold=0.1, figsize=(12, 10), save=False):
        """
        Plot correlation matrix of features
        
        Args:
            df: DataFrame with features
            cols: List of columns to include (None for all numeric columns)
            threshold: Minimum absolute correlation to display
            figsize: Figure size
            save: Whether to save the plot
        """
        # Select columns
        if cols is None:
            # Get numeric columns
            cols = df.select_dtypes(include=np.number).columns.tolist()
        
        # Calculate correlation matrix
        corr_matrix = df[cols].corr()
        
        # Apply threshold filter
        corr_matrix = corr_matrix.where((abs(corr_matrix) >= threshold) | (corr_matrix.values == 1), 0)
        
        # Create plot
        plt.figure(figsize=figsize)
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, annot=True, mask=mask, cmap='coolwarm', 
                   vmin=-1, vmax=1, center=0, fmt='.2f', linewidths=0.5)
        plt.title(f'Feature Correlation Matrix (|corr| >= {threshold})')
        plt.tight_layout()
        
        # Display or save
        plt.show()
        if save:
            plt.savefig(self.output_dir / "correlation_matrix.png", dpi=300, bbox_inches='tight')
    
    def plot_error_distribution(self, actual, predicted, metric_name='Error', bins=30, save=False):
        """
        Plot distribution of prediction errors
        
        Args:
            actual: Array of actual values
            predicted: Array of predicted values
            metric_name: Name for the error metric
            bins: Number of histogram bins
            save: Whether to save the plot
        """
        # Calculate error
        error = actual - predicted
        
        # Create histogram with KDE
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(error, kde=True, bins=bins, ax=ax)
        
        # Add mean and median lines
        mean_error = np.mean(error)
        median_error = np.median(error)
        ax.axvline(mean_error, color='red', linestyle='--', label=f'Mean: {mean_error:.2f}')
        ax.axvline(median_error, color='green', linestyle='-.', label=f'Median: {median_error:.2f}')
        ax.axvline(0, color='black', linestyle='-', label='Zero Error')
        
        # Add labels and legend
        ax.set_title(f'Distribution of {metric_name}')
        ax.set_xlabel(metric_name)
        ax.set_ylabel('Frequency')
        ax.legend()
        
        # Display or save
        plt.tight_layout()
        plt.show()
        if save:
            plt.savefig(self.output_dir / "error_distribution.png", dpi=300, bbox_inches='tight')

    def plot_time_series_decomposition(self, df, date_col='transaction_date', value_col='quantity',
                                     period=52, model='additive', figsize=(14, 10), save=False):
        """
        Plot time series decomposition (trend, seasonal, residual)
        
        Args:
            df: DataFrame with time series data
            date_col: Column with dates
            value_col: Column with values to decompose
            period: Period for seasonal component (e.g., 7 for daily with weekly seasonality)
            model: 'additive' or 'multiplicative'
            figsize: Figure size
            save: Whether to save the plot
        """
        # Prepare time series
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Aggregate by date
        ts = df.groupby(date_col)[value_col].sum()
        
        # Decompose
        try:
            result = seasonal_decompose(ts, model=model, period=period)
            
            # Plot
            fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
            
            result.observed.plot(ax=axes[0], title='Observed')
            result.trend.plot(ax=axes[1], title='Trend')
            result.seasonal.plot(ax=axes[2], title='Seasonal')
            result.resid.plot(ax=axes[3], title='Residual')
            
            plt.tight_layout()
            plt.show()
            
            if save:
                plt.savefig(self.output_dir / "time_series_decomposition.png", dpi=300, bbox_inches='tight')
        except Exception as e:
            print(f"Error in time series decomposition: {e}")
            print("Make sure you have enough data points for the specified period.")