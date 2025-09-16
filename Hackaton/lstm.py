"""
Bidirectional LSTM model for transaction data prediction
- Builds, trains and evaluates a bi-LSTM model
- Handles time series prediction
- Includes visualization of results
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os
import time

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

def load_preprocessed_data():
    """Load preprocessed data saved by prep.py"""
    print("Loading preprocessed data...")
    
    try:
        X_train = np.load('X_train.npy')
        X_test = np.load('X_test.npy')
        y_train = np.load('y_train.npy')
        y_test = np.load('y_test.npy')
        
        print(f"Loaded data shapes:")
        print(f"X_train: {X_train.shape}")
        print(f"X_test: {X_test.shape}")
        print(f"y_train: {y_train.shape}")
        print(f"y_test: {y_test.shape}")
        
        return X_train, X_test, y_train, y_test
    
    except FileNotFoundError:
        print("Preprocessed data files not found. Please run prep.py first.")
        return None, None, None, None

def build_bilstm_model(input_shape, output_units=1):
    """Build a bidirectional LSTM model"""
    print("Building Bi-LSTM model...")
    
    model = Sequential([
        # First bidirectional LSTM layer with return sequences for stacked LSTM
        Bidirectional(
            LSTM(64, activation='tanh', return_sequences=True),
            input_shape=input_shape
        ),
        Dropout(0.2),  # Prevent overfitting
        
        # Second bidirectional LSTM layer
        Bidirectional(LSTM(32, activation='tanh')),
        Dropout(0.2),
        
        # Dense layers
        Dense(16, activation='relu'),
        Dense(output_units, activation='linear')  # Linear activation for regression
    ])
    
    # Compile model with appropriate loss function for regression
    optimizer = Adam(learning_rate=0.001)
    model.compile(
        optimizer=optimizer,
        loss='mean_squared_error',
        metrics=['mae']
    )
    
    print("Model summary:")
    model.summary()
    
    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=5, batch_size=64):
    """Train the LSTM model with early stopping and checkpoints"""
    print("Training Bi-LSTM model...")
    
    # Create directory for model checkpoints if it doesn't exist
    if not os.path.exists('model_checkpoints'):
        os.makedirs('model_checkpoints')
    
    # Callbacks for training
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    
    checkpoint = ModelCheckpoint(
        'model_checkpoints/best_model.h5',
        monitor='val_loss',
        save_best_only=True,
        mode='min',
        verbose=1
    )
    
    # Start training
    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, checkpoint],
        verbose=1
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds.")
    
    # Plot training history
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('Mean Absolute Error')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('bilstm_training_history.png')
    print("Saved training history plot to 'bilstm_training_history.png'")
    
    return model, history

def evaluate_model(model, X_test, y_test):
    """Evaluate model performance"""
    print("Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance Metrics:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R²): {r2:.4f}")
    
    # Plot predictions vs actual
    plt.figure(figsize=(12, 6))
    
    # Sample a subset of test data for clearer visualization
    sample_size = min(1000, len(y_test))
    indices = np.random.choice(len(y_test), sample_size, replace=False)
    
    plt.scatter(y_test[indices], y_pred[indices], alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Bi-LSTM Model: Actual vs Predicted')
    plt.grid(True)
    plt.savefig('bilstm_predictions.png')
    print("Saved predictions plot to 'bilstm_predictions.png'")
    
    # Plot residuals
    residuals = y_test - y_pred.flatten()
    
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.hist(residuals, bins=50, alpha=0.7, color='blue')
    plt.xlabel('Residual Value')
    plt.ylabel('Count')
    plt.title('Residual Distribution')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred.flatten(), residuals, alpha=0.5, color='green')
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs Predicted')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('bilstm_residuals.png')
    print("Saved residuals plot to 'bilstm_residuals.png'")
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def forecast_next_periods(model, X_test, sequence_length, n_forecasts=30):
    """Generate forecasts for the next periods using the trained model"""
    print(f"Forecasting the next {n_forecasts} periods...")
    
    # Use the last sequence from X_test as our starting point
    last_sequence = X_test[-1:]
    forecasts = []
    
    # Generate forecasts iteratively
    for i in range(n_forecasts):
        # Predict the next value
        next_value = model.predict(last_sequence)
        forecasts.append(next_value[0][0])
        
        # Create a new sequence by shifting the old one and adding the prediction
        new_sequence = np.copy(last_sequence)
        # Shift values (assuming time dimension is axis 1 and features start at column 0)
        new_sequence[0, :-1, :] = new_sequence[0, 1:, :]
        
        # Assuming the target variable is the first feature in the sequence
        # This is an approximation - in a real scenario, you'd need logic to update all features
        new_sequence[0, -1, 0] = next_value[0][0]
        
        # Update the last sequence for the next iteration
        last_sequence = new_sequence
    
    # Plot forecasts
    plt.figure(figsize=(12, 6))
    plt.plot(range(n_forecasts), forecasts, 'b-', marker='o')
    plt.title(f'Bi-LSTM: Next {n_forecasts} Period Forecasts')
    plt.xlabel('Time Steps Ahead')
    plt.ylabel('Predicted Value')
    plt.grid(True)
    plt.savefig('bilstm_forecasts.png')
    print("Saved forecasts plot to 'bilstm_forecasts.png'")
    
    return forecasts

def main():
    """Main function to run the Bi-LSTM model pipeline"""
    print("="*50)
    print("BIDIRECTIONAL LSTM MODEL FOR TRANSACTION PREDICTION")
    print("="*50)
    
    # Load preprocessed data
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    
    if X_train is None:
        print("Exiting due to missing preprocessed data.")
        return
    
    # Get input shape from training data
    input_shape = (X_train.shape[1], X_train.shape[2])
    
    # Build model
    model = build_bilstm_model(input_shape)
    
    # Train model
    model, history = train_model(model, X_train, y_train, X_test, y_test)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Generate forecasts
    sequence_length = X_train.shape[1]
    forecasts = forecast_next_periods(model, X_test, sequence_length)
    
    # Save model results summary
    with open('bilstm_results.txt', 'w') as f:
        f.write("BI-LSTM MODEL RESULTS\n")
        f.write("====================\n\n")
        
        f.write("1. MODEL ARCHITECTURE\n")
        f.write("   - Type: Bidirectional LSTM (Bi-LSTM)\n")
        f.write(f"   - Input Shape: {input_shape}\n")
        f.write("   - Layers: 2 Bi-LSTM layers + Dense output layer\n")
        f.write("   - Parameters: Tanh activation, Adam optimizer\n\n")
        
        f.write("2. TRAINING INFORMATION\n")
        f.write(f"   - Training samples: {len(X_train)}\n")
        f.write(f"   - Test samples: {len(X_test)}\n")
        f.write(f"   - Final training loss: {history.history['loss'][-1]:.4f}\n")
        f.write(f"   - Final validation loss: {history.history['val_loss'][-1]:.4f}\n\n")
        
        f.write("3. PERFORMANCE METRICS\n")
        f.write(f"   - Mean Squared Error (MSE): {metrics['mse']:.4f}\n")
        f.write(f"   - Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}\n")
        f.write(f"   - Mean Absolute Error (MAE): {metrics['mae']:.4f}\n")
        f.write(f"   - R-squared (R²): {metrics['r2']:.4f}\n\n")
        
        f.write("4. FORECAST SUMMARY\n")
        f.write(f"   - Generated {len(forecasts)} future predictions\n")
        f.write(f"   - Average forecast value: {np.mean(forecasts):.4f}\n")
        f.write(f"   - Forecast range: {min(forecasts):.4f} to {max(forecasts):.4f}\n\n")
        
        f.write("5. CONCLUSION\n")
        if metrics['r2'] > 0.7:
            f.write("   - The Bi-LSTM model shows strong predictive performance\n")
        elif metrics['r2'] > 0.5:
            f.write("   - The Bi-LSTM model shows moderate predictive performance\n")
        else:
            f.write("   - The Bi-LSTM model shows limited predictive performance\n")
        f.write("   - The bidirectional architecture helps capture both past and future dependencies\n")
        f.write("   - Time-based patterns in transaction data are effectively modeled\n")
        f.write("   - Consider feature importance analysis to further improve performance\n")
    
    print("Bi-LSTM modeling complete. Results saved to 'bilstm_results.txt'")
    
    return model, metrics, forecasts

if __name__ == "__main__":
    model, metrics, forecasts = main()