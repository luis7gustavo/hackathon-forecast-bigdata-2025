#
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Input, Attention, Layer
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import warnings
warnings.filterwarnings('ignore')
import os
import time

# Load the prepared dataset
print("\n==== LOADING PREPARED DATASET ====")
weekly_sales = pd.read_parquet("./Prepared_Data/weekly_sales_features.parquet")
print(f"Loaded dataset with {weekly_sales.shape[0]} rows and {weekly_sales.shape[1]} columns")

# Prepare data for modeling
print("\n==== PREPARING DATA FOR MODELING ====")

# Filter out rows with missing target values
weekly_sales = weekly_sales.dropna(subset=['quantidade_total'])

# Sort by date to ensure proper time series handling
weekly_sales = weekly_sales.sort_values(by=['data_semana'])

# Select features for modeling
feature_cols = [col for col in weekly_sales.columns if col.endswith('_encoded') 
                or 'lag_' in col or 'rolling_' in col or 'week_' in col]

# Handle any remaining missing values
for col in feature_cols:
    if weekly_sales[col].isnull().sum() > 0:
        if col in ['week_sin', 'week_cos']:
            # These should not have missing values, but just in case
            weekly_sales[col] = weekly_sales[col].fillna(0)
        else:
            # For lag features that might have NaNs
            weekly_sales[col] = weekly_sales[col].fillna(0)

# Scale features
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(weekly_sales[feature_cols])
y = weekly_sales['quantidade_total'].values
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Prepare data for time series modeling
# For RNN models we need to reshape data to [samples, timesteps, features]
def create_sequences(X, y, time_steps):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

# Define time steps for sequence models
TIME_STEPS = 12  # Using 12 weeks (approximately one quarter)

# Create sequences for LSTM models
X_seq, y_seq = create_sequences(X, y_scaled, TIME_STEPS)
print(f"Sequence data shape: {X_seq.shape}")

# Time Series Cross-Validation Split
print("\n==== TIME SERIES CROSS-VALIDATION ====")

# For non-sequence models
tscv = TimeSeriesSplit(n_splits=5)

# For sequence models, we need to handle this differently
# We'll use about 80% of the data for training and 20% for testing
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]
y_train_scaled, y_test_scaled = y_scaled[:train_size], y_scaled[train_size:]

# For sequence models
train_size_seq = int(len(X_seq) * 0.8)
X_seq_train, X_seq_test = X_seq[:train_size_seq], X_seq[train_size_seq:]
y_seq_train, y_seq_test = y_seq[:train_size_seq], y_seq[train_size_seq:]

print(f"Training set size: {len(X_train)} samples")
print(f"Test set size: {len(X_test)} samples")
print(f"Sequence training set size: {len(X_seq_train)} samples")
print(f"Sequence test set size: {len(X_seq_test)} samples")

# Define evaluation function
def evaluate_model(y_true, y_pred, model_name):
    """Calculate and print evaluation metrics for the model."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    print(f"\n----- {model_name} Performance -----")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    
    return {
        'model': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

# Model 1: Linear Regression
print("\n==== MODEL 1: LINEAR REGRESSION ====")

start_time = time.time()

# Initialize and train the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions
y_pred_lr = lr_model.predict(X_test)

# Evaluate
lr_metrics = evaluate_model(y_test, y_pred_lr, "Linear Regression")
lr_metrics['training_time'] = time.time() - start_time

# Model 2: Random Forest
print("\n==== MODEL 2: RANDOM FOREST ====")

start_time = time.time()

# Initialize and train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred_rf = rf_model.predict(X_test)

# Evaluate
rf_metrics = evaluate_model(y_test, y_pred_rf, "Random Forest")
rf_metrics['training_time'] = time.time() - start_time

# Model 3: XGBoost
print("\n==== MODEL 3: XGBOOST ====")

start_time = time.time()

# Initialize and train the model
xgb_model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    random_state=42,
    n_jobs=-1
)
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred_xgb = xgb_model.predict(X_test)

# Evaluate
xgb_metrics = evaluate_model(y_test, y_pred_xgb, "XGBoost")
xgb_metrics['training_time'] = time.time() - start_time

# Model 4: Simple Neural Network
print("\n==== MODEL 4: SIMPLE NEURAL NETWORK ====")

start_time = time.time()

# Set random seed for reproducibility
tf.random.set_seed(42)

# Define the model architecture
nn_model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])

# Compile the model
nn_model.compile(optimizer='adam', loss='mse')

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = nn_model.fit(
    X_train, y_train_scaled,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0
)

# Make predictions
y_pred_nn_scaled = nn_model.predict(X_test).flatten()
y_pred_nn = scaler_y.inverse_transform(y_pred_nn_scaled.reshape(-1, 1)).flatten()

# Evaluate
nn_metrics = evaluate_model(y_test, y_pred_nn, "Simple Neural Network")
nn_metrics['training_time'] = time.time() - start_time

# Model 5: Simple LSTM
print("\n==== MODEL 5: SIMPLE LSTM ====")

start_time = time.time()

# Set random seed for reproducibility
tf.random.set_seed(42)

# Define the model architecture
lstm_model = Sequential([
    LSTM(50, activation='relu', input_shape=(X_seq_train.shape[1], X_seq_train.shape[2])),
    Dense(25, activation='relu'),
    Dense(1)
])

# Compile the model
lstm_model.compile(optimizer='adam', loss='mse')

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = lstm_model.fit(
    X_seq_train, y_seq_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0
)

# Make predictions
y_pred_lstm_scaled = lstm_model.predict(X_seq_test).flatten()
y_pred_lstm = scaler_y.inverse_transform(y_pred_lstm_scaled.reshape(-1, 1)).flatten()
y_test_lstm = scaler_y.inverse_transform(y_seq_test.reshape(-1, 1)).flatten()

# Evaluate
lstm_metrics = evaluate_model(y_test_lstm, y_pred_lstm, "Simple LSTM")
lstm_metrics['training_time'] = time.time() - start_time

# Model 6: Bidirectional LSTM with Attention
print("\n==== MODEL 6: BI-LSTM WITH ATTENTION ====")

start_time = time.time()

# Set random seed for reproducibility
tf.random.set_seed(42)

# Create a custom attention layer
class AttentionLayer(Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(name='attention_weight', 
                                 shape=(input_shape[-1], 1), 
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(name='attention_bias', 
                                 shape=(input_shape[1], 1), 
                                 initializer='zeros',
                                 trainable=True)
        super(AttentionLayer, self).build(input_shape)
        
    def call(self, x):
        # x shape: (batch_size, time_steps, input_dim)
        # e shape: (batch_size, time_steps, 1)
        e = tf.tanh(tf.matmul(x, self.W) + self.b)
        
        # a shape: (batch_size, time_steps, 1)
        a = tf.nn.softmax(e, axis=1)
        
        # context shape: (batch_size, input_dim)
        context = x * a
        context = tf.reduce_sum(context, axis=1)
        
        return context

# Define the model architecture
bilstm_model = Sequential([
    Bidirectional(LSTM(50, activation='relu', return_sequences=True, 
                       input_shape=(X_seq_train.shape[1], X_seq_train.shape[2]))),
    AttentionLayer(),
    Dense(25, activation='relu'),
    Dense(1)
])

# Compile the model
bilstm_model.compile(optimizer='adam', loss='mse')

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = bilstm_model.fit(
    X_seq_train, y_seq_train,
    epochs=100,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0
)

# Make predictions
y_pred_bilstm_scaled = bilstm_model.predict(X_seq_test).flatten()
y_pred_bilstm = scaler_y.inverse_transform(y_pred_bilstm_scaled.reshape(-1, 1)).flatten()

# Evaluate
bilstm_metrics = evaluate_model(y_test_lstm, y_pred_bilstm, "Bi-LSTM with Attention")
bilstm_metrics['training_time'] = time.time() - start_time

# Model 7: Encoder-Decoder LSTM
print("\n==== MODEL 7: ENCODER-DECODER LSTM ====")

start_time = time.time()

# Set random seed for reproducibility
tf.random.set_seed(42)

# Define Encoder-Decoder model architecture
def create_encoder_decoder_model(input_shape):
    # Encoder
    encoder_inputs = Input(shape=input_shape)
    encoder = LSTM(100, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    
    # Decoder
    decoder_inputs = Input(shape=(1, input_shape[1]))
    decoder_lstm = LSTM(100, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(1)
    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Define the model
    model = tf.keras.models.Model([encoder_inputs, decoder_inputs], decoder_outputs)
    
    return model

# Prepare decoder input data (just zeros for simplicity in this case)
decoder_input_data = np.zeros((X_seq_train.shape[0], 1, X_seq_train.shape[2]))
decoder_input_data_test = np.zeros((X_seq_test.shape[0], 1, X_seq_test.shape[2]))

# Create and compile the model
encoder_decoder_model = create_encoder_decoder_model((X_seq_train.shape[1], X_seq_train.shape[2]))
encoder_decoder_model.compile(optimizer='adam', loss='mse')

# Train the model
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
history = encoder_decoder_model.fit(
    [X_seq_train, decoder_input_data], 
    y_seq_train.reshape(-1, 1, 1),
    epochs=100, 
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping],
    verbose=0
)

# Make predictions
y_pred_encdec_scaled = encoder_decoder_model.predict([X_seq_test, decoder_input_data_test]).flatten()
y_pred_encdec = scaler_y.inverse_transform(y_pred_encdec_scaled.reshape(-1, 1)).flatten()

# Evaluate
encdec_metrics = evaluate_model(y_test_lstm, y_pred_encdec, "Encoder-Decoder LSTM")
encdec_metrics['training_time'] = time.time() - start_time

# Compare all models
print("\n==== MODEL COMPARISON ====")

# Collect metrics from all models
all_metrics = pd.DataFrame([
    lr_metrics,
    rf_metrics,
    xgb_metrics,
    nn_metrics,
    lstm_metrics,
    bilstm_metrics,
    encdec_metrics
])

# Sort by RMSE (lower is better)
all_metrics = all_metrics.sort_values('rmse')

# Display metrics table
print("\nPerformance metrics sorted by RMSE (lower is better):")
print(all_metrics[['model', 'rmse', 'mae', 'r2', 'training_time']].to_string(index=False))

# Visualize model comparison
plt.figure(figsize=(12, 8))

# RMSE comparison
plt.subplot(2, 2, 1)
sns.barplot(x='model', y='rmse', data=all_metrics)
plt.title('RMSE by Model (lower is better)')
plt.xticks(rotation=45)
plt.tight_layout()

# R² comparison
plt.subplot(2, 2, 2)
sns.barplot(x='model', y='r2', data=all_metrics)
plt.title('R² by Model (higher is better)')
plt.xticks(rotation=45)
plt.tight_layout()

# MAE comparison
plt.subplot(2, 2, 3)
sns.barplot(x='model', y='mae', data=all_metrics)
plt.title('MAE by Model (lower is better)')
plt.xticks(rotation=45)
plt.tight_layout()

# Training time comparison
plt.subplot(2, 2, 4)
sns.barplot(x='model', y='training_time', data=all_metrics)
plt.title('Training Time by Model (seconds)')
plt.xticks(rotation=45)
plt.tight_layout()

plt.tight_layout()
plt.savefig('./model_comparison.png')
plt.show()

# Visualize predictions for the best model
# Find the best model based on RMSE
best_model_name = all_metrics.iloc[0]['model']
print(f"\nVisualizing predictions for the best model: {best_model_name}")

# Get actual and predicted values for the best model
if best_model_name == "Linear Regression":
    y_actual, y_pred = y_test, y_pred_lr
elif best_model_name == "Random Forest":
    y_actual, y_pred = y_test, y_pred_rf
elif best_model_name == "XGBoost":
    y_actual, y_pred = y_test, y_pred_xgb
elif best_model_name == "Simple Neural Network":
    y_actual, y_pred = y_test, y_pred_nn
elif best_model_name == "Simple LSTM":
    y_actual, y_pred = y_test_lstm, y_pred_lstm
elif best_model_name == "Bi-LSTM with Attention":
    y_actual, y_pred = y_test_lstm, y_pred_bilstm
else:  # Encoder-Decoder LSTM
    y_actual, y_pred = y_test_lstm, y_pred_encdec

# Plot actual vs predicted values
plt.figure(figsize=(12, 6))
plt.plot(y_actual[:100], label='Actual', color='blue', alpha=0.7)
plt.plot(y_pred[:100], label='Predicted', color='red', alpha=0.7)
plt.title(f'Actual vs Predicted Values ({best_model_name})')
plt.xlabel('Time Step')
plt.ylabel('Quantity')
plt.legend()
plt.tight_layout()
plt.savefig('./best_model_predictions.png')
plt.show()

# Scatter plot of actual vs predicted
plt.figure(figsize=(8, 8))
plt.scatter(y_actual, y_pred, alpha=0.5)
plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], 'r--')
plt.title(f'Actual vs Predicted Scatter Plot ({best_model_name})')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.tight_layout()
plt.savefig('./best_model_scatter.png')
plt.show()

print("\nModel evaluation complete! The best model based on RMSE is:", best_model_name)

# Save the best model
print("\n==== SAVING BEST MODEL ====")
if best_model_name == "Linear Regression":
    import joblib
    joblib.dump(lr_model, './best_model.joblib')
elif best_model_name == "Random Forest":
    import joblib
    joblib.dump(rf_model, './best_model.joblib')
elif best_model_name == "XGBoost":
    xgb_model.save_model('./best_model.json')
elif best_model_name == "Simple Neural Network":
    nn_model.save('./best_model')
elif best_model_name == "Simple LSTM":
    lstm_model.save('./best_model')
elif best_model_name == "Bi-LSTM with Attention":
    bilstm_model.save('./best_model')
else:  # Encoder-Decoder LSTM
    encoder_decoder_model.save('./best_model')
    
print(f"Best model saved as './best_model'")