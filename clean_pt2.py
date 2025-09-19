# clean_pt2.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

print("Loading the cleaned dataset...")
# Load the cleaned dataset
df_tratado = pd.read_parquet('./Datasets/dados_limpos.parquet')

# ====================
# STEP 1: DATA VERIFICATION
# ====================
print("\nVerifying data completeness and integrity...")

# Verify date range
print(f"Date range in dataset: {df_tratado['data_transacao'].min()} to {df_tratado['data_transacao'].max()}")

# Check for data consistency - making sure we have data for all of 2022
months_2022 = df_tratado[df_tratado['data_transacao'].dt.year == 2022]['data_transacao'].dt.month.unique()
print(f"Months available in 2022: {sorted(months_2022)}")

# Check unique combinations of PDV/SKU
pdv_sku_combinations = df_tratado.groupby(['id_loja', 'id_produto']).size().reset_index(name='count')
print(f"Total PDV/SKU combinations: {len(pdv_sku_combinations)}")

# ====================
# STEP 2: WEEKLY AGGREGATION
# ====================
print("\nAggregating data to weekly level...")

# Create year-week identifier for aggregation
df_tratado['year'] = df_tratado['data_transacao'].dt.year
df_tratado['week'] = df_tratado['data_transacao'].dt.isocalendar().week
df_tratado['year_week'] = df_tratado['year'].astype(str) + "-" + df_tratado['week'].astype(str).str.zfill(2)

# Aggregate to weekly level by PDV/SKU
weekly_data = df_tratado.groupby(['id_loja', 'id_produto', 'year_week']).agg(
    quantidade_semanal=('quantidade', 'sum'),
    valor_bruto_semanal=('valor_bruto', 'sum'),
    lucro_bruto_semanal=('lucro_bruto', 'sum'),
    transacoes_count=('data_transacao', 'count'),
    year=('year', 'first'),
    week=('week', 'first'),
    categoria=('categoria', 'first'),
    categoria_pdv=('categoria_pdv', 'first'),
    tipo_local=('tipo_local', 'first')
).reset_index()

# Sort by PDV, SKU, and time
weekly_data = weekly_data.sort_values(['id_loja', 'id_produto', 'year', 'week'])

# ====================
# STEP 3: FEATURE ENGINEERING
# ====================
print("\nCreating advanced features for time series forecasting...")

# Label encode categorical variables
label_encoders = {}
categorical_cols = ['categoria', 'categoria_pdv', 'tipo_local']

for col in categorical_cols:
    le = LabelEncoder()
    weekly_data[f'{col}_encoded'] = le.fit_transform(weekly_data[col])
    label_encoders[col] = le

# Create lag features for each PDV/SKU combination
lag_periods = [1, 2, 3, 4, 8, 12]  # Previous weeks
rolling_windows = [4, 8, 12]       # Moving averages

# Initialize progress counter
total_combos = len(weekly_data[['id_loja', 'id_produto']].drop_duplicates())
print(f"Creating lag features for {total_combos} PDV/SKU combinations...")

# Group by PDV/SKU and apply lag features
for (store, product), group in weekly_data.groupby(['id_loja', 'id_produto']):
    # Sort by time
    group = group.sort_values(['year', 'week'])
    
    # Create lag features
    for lag in lag_periods:
        weekly_data.loc[(weekly_data['id_loja'] == store) & 
                        (weekly_data['id_produto'] == product), 
                        f'lag_{lag}w_qty'] = group['quantidade_semanal'].shift(lag)
    
    # Create rolling mean features
    for window in rolling_windows:
        weekly_data.loc[(weekly_data['id_loja'] == store) & 
                        (weekly_data['id_produto'] == product), 
                        f'rolling_{window}w_mean'] = group['quantidade_semanal'].shift(1).rolling(window=window, min_periods=1).mean()
        
        # Rolling standard deviation (volatility)
        weekly_data.loc[(weekly_data['id_loja'] == store) & 
                        (weekly_data['id_produto'] == product), 
                        f'rolling_{window}w_std'] = group['quantidade_semanal'].shift(1).rolling(window=window, min_periods=1).std()

# Create week of year cyclical features (sine and cosine transformation)
weekly_data['week_sin'] = np.sin(2 * np.pi * weekly_data['week'] / 52)
weekly_data['week_cos'] = np.cos(2 * np.pi * weekly_data['week'] / 52)

# Create indicators for special periods
weekly_data['is_yearend'] = ((weekly_data['week'] >= 50) & (weekly_data['week'] <= 52)).astype(int)
weekly_data['is_yearstart'] = ((weekly_data['week'] >= 1) & (weekly_data['week'] <= 3)).astype(int)

# ====================
# STEP 4: PREPARE FOR FORECASTING
# ====================
print("\nPreparing train/test split for time series forecasting...")

# Identify data for 2022 (training) and January 2023 (testing)
train_data = weekly_data[(weekly_data['year'] == 2022)]
test_weeks = [f"2023-{i:02d}" for i in range(1, 6)]  # First 5 weeks of 2023

# Create a template for all PDV/SKU combinations for the 5 weeks of January 2023
pdv_sku_combinations = weekly_data[['id_loja', 'id_produto', 'categoria', 'categoria_pdv', 'tipo_local']].drop_duplicates()

# For each combination, create 5 rows (one for each week of January 2023)
forecast_rows = []
for idx, row in pdv_sku_combinations.iterrows():
    for week_idx, year_week in enumerate(test_weeks, 1):
        new_row = row.copy()
        new_row['year_week'] = year_week
        new_row['year'] = 2023
        new_row['week'] = week_idx
        forecast_rows.append(new_row)

# Create the forecast template dataframe
forecast_template = pd.DataFrame(forecast_rows)

# Apply the same encoding for categorical variables
for col in categorical_cols:
    forecast_template[f'{col}_encoded'] = label_encoders[col].transform(forecast_template[col])

# Create cyclical week features for forecast period
forecast_template['week_sin'] = np.sin(2 * np.pi * forecast_template['week'] / 52)
forecast_template['week_cos'] = np.cos(2 * np.pi * forecast_template['week'] / 52)
forecast_template['is_yearstart'] = 1  # All are in January
forecast_template['is_yearend'] = 0

# Prepare the final datasets
# Fill missing values in lag features
train_data = train_data.fillna(0)

print("\nSaving prepared datasets...")
train_data.to_parquet('./Datasets/train_weekly_data.parquet')
forecast_template.to_parquet('./Datasets/forecast_template.parquet')

print("\nData preparation completed!")
print(f"Training data shape: {train_data.shape}")
print(f"Forecast template shape: {forecast_template.shape}")

# Show a sample of the prepared data
print("\nSample of training data:")
print(train_data.head())

print("\nSample of forecast template:")
print(forecast_template.head())