# Retail Sales Forecast

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- pip or conda package manager

### Installation
Clone this repository:
```bash
git clone https://github.com/luis7gustavo/hackathon-forecast-bigdata-2025.git
cd hackathon-forecast-bigdata-2025
```

Install required dependencies:
```bash
pip install -r requirements.txt
```

If there are catboost instalation errors:
```bash
python -m pip install --upgrade pip setuptools wheel
python -m pip install "catboost>=1.2.7" --only-binary=:all:
```

### Data Setup
- Ensure the parquet files are placed in the `data/` directory.
- Run the preprocessing script to prepare the data:
```bash
python -m src.data.preprocessing
```

### Running the Model
To train the model and generate predictions:
```bash
python main.py
```

This will:
- Load and preprocess the data
- Engineer relevant features
- Train the forecasting model
- Generate predictions for January 2023
- Save results to the `results/` directory

---

## 📈 Forecasting Approach

### Data Preprocessing
- Handling missing values and outliers
- Data type conversion
- Joining transaction, product, and PDV data

### Feature Engineering
- Time-based features (day of week, month, holidays)
- Lag features (previous weeks/months sales)
- Rolling statistics (moving averages, trends)
- PDV and product characteristics

### Model Training
- Time series forecasting with gradient boosting (XGBoost/LightGBM)
- Cross-validation with time-based splitting
- Hyperparameter optimization

### Prediction Generation
- Weekly forecasts by PDV/SKU for January 2023
- Output in required format (week, pdv, product, quantity)

---

## 📊 Results Visualization
The project includes visualization tools to evaluate model performance and understand sales patterns:
- Actual vs. predicted sales
- Seasonal patterns analysis
- PDV and product performance comparisons
- Feature importance analysis

---

## 📄 Output Format
The final forecast is output as a CSV file in the format:
```csv
semana;pdv;produto;quantidade
1;1023;123;120
2;1045;234;85
...
```

---

## 🧪 Evaluation Metrics
Model performance is evaluated using:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Mean Absolute Percentage Error (MAPE)

---

## 👥 Contributors
- Pedro Rebello
- Luis Gustavo
- Sávio Nery
- Pedro vargas
- Eduardo Diniz

---

## 📝 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
