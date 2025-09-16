import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import datetime as dt
import warnings
warnings.filterwarnings('ignore')

def load_data(file_path='wrangler.csv'):
    """Load and perform initial inspection of the data"""
    print("Loading and preprocessing data...")
    df = pd.read_csv(file_path)
    print(f"Original data shape: {df.shape}")
    return df

def clean_data(df):
    """Clean the dataset and handle problematic values"""
    print("Cleaning data...")
    
    # Drop the index column if it's unnecessary
    if 'Unnamed: 0' in df.columns:
        df = df.drop('Unnamed: 0', axis=1)
    
    # Convert date columns to datetime
    df['data_transacao'] = pd.to_datetime(df['data_transacao'])
    df['data_referencia'] = pd.to_datetime(df['data_referencia'])
    
    # Check for and handle infinite or extremely large values
    numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_columns:
        # Replace infinity with NaN
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        
        # Replace values that are too large with NaN
        # Define a reasonable threshold based on column stats
        threshold = df[col].quantile(0.999) * 5  # 5 times the 99.9th percentile
        df.loc[df[col] > threshold, col] = np.nan
        
        # Replace values that are too small (negative when they shouldn't be)
        if col not in ['desconto']:  # desconto can be negative
            df.loc[df[col] < 0, col] = np.nan
    
    # Handle missing values
    for col in numeric_columns:
        if df[col].isna().sum() > 0:
            # Fill missing values with median for numeric columns
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
    
    # Check for duplicate records
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        print(f"Found {duplicates} duplicate records, removing them.")
        df = df.drop_duplicates()
    
    return df

def feature_engineering(df):
    """Create new features that might be useful for modeling"""
    print("Performing feature engineering...")
    
    # Extract date features
    df['ano'] = df['data_transacao'].dt.year
    df['mes'] = df['data_transacao'].dt.month
    df['dia'] = df['data_transacao'].dt.day
    df['dia_da_semana'] = df['data_transacao'].dt.dayofweek
    df['semana_do_ano'] = df['data_transacao'].dt.isocalendar().week
    df['dia_do_ano'] = df['data_transacao'].dt.dayofyear
    
    # Time between reference date and transaction date
    df['dias_desde_referencia'] = (df['data_transacao'] - df['data_referencia']).dt.days
    
    # Derived business metrics
    df['margem'] = np.where(df['valor_bruto'] != 0, df['lucro_bruto'] / df['valor_bruto'], 0)
    df['pct_desconto'] = np.where(df['valor_bruto'] != 0, df['desconto'] / df['valor_bruto'], 0)
    df['pct_imposto'] = np.where(df['valor_bruto'] != 0, df['impostos'] / df['valor_bruto'], 0)
    df['valor_medio_item'] = np.where(df['quantidade'] != 0, df['valor_bruto'] / df['quantidade'], df['valor_bruto'])
    
    # Cap the derived metrics to reasonable values (between 0 and 1 for percentages)
    df['margem'] = np.clip(df['margem'], 0, 1)
    df['pct_desconto'] = np.clip(df['pct_desconto'], 0, 1)
    df['pct_imposto'] = np.clip(df['pct_imposto'], 0, 1)
    
    # Create distributor dummy variables
    df_encoded = pd.get_dummies(df['id_distribuidor'], prefix='distribuidor')
    df = pd.concat([df, df_encoded], axis=1)
    
    # Save the engineered data for future use
    df.to_csv('wrangler_engineered.csv', index=False)
    print("Saved engineered data to 'wrangler_engineered.csv'")
    
    return df

def normalize_data(df):
    """Normalize numerical features"""
    print("Normalizing data...")
    
    # Select only numerical columns for normalization, excluding the ones we don't want to scale
    exclude_cols = ['id_loja_interna', 'id_produto_interno', 'id_distribuidor',
                    'ano', 'mes', 'dia', 'dia_da_semana', 'semana_do_ano', 'dia_do_ano',
                    'outlier_flag'] + [col for col in df.columns if col.startswith('distribuidor_')]
    
    numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
    cols_to_scale = [col for col in numeric_cols if col not in exclude_cols]
    
    # Check for any remaining infinite values
    df_selected = df[cols_to_scale].replace([np.inf, -np.inf], np.nan)
    
    # Fill any remaining NaNs with the median
    for col in cols_to_scale:
        median_val = df_selected[col].median()
        df_selected[col] = df_selected[col].fillna(median_val)
    
    # Double-check no infinite or NaN values remain
    if df_selected.isna().sum().sum() > 0 or np.isinf(df_selected.values).sum() > 0:
        print("Warning: NaN or infinite values found after cleaning. Replacing with zeros.")
        df_selected = df_selected.replace([np.inf, -np.inf], 0).fillna(0)
    
    # Use StandardScaler for normalization
    scaler = StandardScaler()
    normalized_data = scaler.fit_transform(df_selected)
    
    # Replace original values with normalized ones
    df_normalized = df.copy()
    df_normalized[cols_to_scale] = normalized_data
    
    return df_normalized, scaler

def prepare_for_model(df, target_col='valor_liquido', sequence_length=7):
    """Prepare data for time series model"""
    print("Preparing data for modeling...")
    
    # Sort by date
    df = df.sort_values('data_transacao')
    
    # Features to use in the model
    feature_cols = ['quantidade', 'valor_bruto', 'valor_liquido', 'lucro_bruto', 
                    'desconto', 'impostos', 'margem', 'pct_desconto', 'pct_imposto',
                    'valor_medio_item', 'dia_da_semana', 'dia_do_ano']
    
    feature_cols += [col for col in df.columns if col.startswith('distribuidor_')]
    
    # Create X and y
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Split into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False  # Keep time ordering
    )
    
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")
    
    # Save train/test data for use in other scripts
    np.save('X_train.npy', X_train)
    np.save('X_test.npy', X_test)
    np.save('y_train.npy', y_train)
    np.save('y_test.npy', y_test)
    
    # Save feature column names for reference
    with open('feature_columns.txt', 'w') as f:
        for col in feature_cols:
            f.write(f"{col}\n")
    
    return X_train, X_test, y_train, y_test

def main():
    # Load data
    df = load_data()
    
    # Clean data
    df = clean_data(df)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Normalize data
    df, scaler = normalize_data(df)
    
    # Prepare for model
    X_train, X_test, y_train, y_test = prepare_for_model(df)
    
    return df, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    df, X_train, X_test, y_train, y_test = main()
    print("Data preparation completed successfully!")