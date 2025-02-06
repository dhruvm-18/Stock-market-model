import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
import arch
from arch import arch_model
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

df = pd.read_csv('S&P500HistoricalData.csv')
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values('Date')
df['Price'] = df['Price'].str.replace(',', '').astype(float)
data = df['Price'].values.reshape(-1, 1)

def prepare_data(data, sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    X, y = [], []
    for i in range(len(scaled_data) - sequence_length):
        X.append(scaled_data[i:i + sequence_length])
        y.append(scaled_data[i + sequence_length])
    X, y = np.array(X), np.array(y)
    train_size = int(len(X) * 0.7)
    val_size = int(len(X) * 0.2)
    X_train, X_val, X_test = X[:train_size], X[train_size:train_size + val_size], X[train_size + val_size:]
    y_train, y_val, y_test = y[:train_size], y[train_size:train_size + val_size], y[train_size + val_size:]
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

def generate_volatility(data, sequence_length=60):
    log_returns = np.log(data[1:] / data[:-1])
    garch = arch_model(log_returns, vol='Garch', p=1, q=1)
    garch_fit = garch.fit(disp='off')
    volatility = garch_fit.conditional_volatility
    volatility = np.insert(volatility, 0, 0)
    return volatility.reshape(-1, 1)

sequence_length = 60
X_train, X_val, X_test, y_train, y_val, y_test, scaler = prepare_data(data, sequence_length)

volatility = generate_volatility(data.flatten(), sequence_length)
volatility = volatility[-len(data) + sequence_length:]

volatility_train = volatility[:len(X_train)]
volatility_val = volatility[len(X_train):len(X_train) + len(X_val)]
volatility_test = volatility[len(X_train) + len(X_val):len(X_train) + len(X_val) + len(X_test)]

volatility_train = np.repeat(volatility_train.reshape(-1, 1, 1), X_train.shape[1], axis=1)
volatility_val = np.repeat(volatility_val.reshape(-1, 1, 1), X_val.shape[1], axis=1)
volatility_test = np.repeat(volatility_test.reshape(-1, 1, 1), X_test.shape[1], axis=1)

X_train = np.concatenate([X_train, volatility_train], axis=2)
X_val = np.concatenate([X_val, volatility_val], axis=2)
X_test = np.concatenate([X_test, volatility_test], axis=2)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(25))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

model = build_lstm_model((sequence_length, 2))
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_val, y_val), verbose=1)

predictions = model.predict(X_test)
train_predictions = model.predict(X_train)
val_predictions = model.predict(X_val)

predictions_inv = scaler.inverse_transform(predictions)
train_predictions_inv = scaler.inverse_transform(train_predictions)
val_predictions_inv = scaler.inverse_transform(val_predictions)

y_test_inv = scaler.inverse_transform(y_test)
y_train_inv = scaler.inverse_transform(y_train)
y_val_inv = scaler.inverse_transform(y_val)

rmse_test = np.sqrt(mean_squared_error(y_test_inv, predictions_inv))
mae_test = mean_absolute_error(y_test_inv, predictions_inv)
mape_test = np.mean(np.abs((y_test_inv - predictions_inv) / y_test_inv)) * 100
r2_test = r2_score(y_test_inv, predictions_inv)

rmse_val = np.sqrt(mean_squared_error(y_val_inv, val_predictions_inv))
mae_val = mean_absolute_error(y_val_inv, val_predictions_inv)
mape_val = np.mean(np.abs((y_val_inv - val_predictions_inv) / y_val_inv)) * 100
r2_val = r2_score(y_val_inv, val_predictions_inv)

plt.figure(figsize=(15, 7))
plt.plot(df['Date'].iloc[-len(y_val_inv) - len(y_test_inv):-len(y_test_inv)], y_val_inv, label='Validation Actual', marker='o')
plt.plot(df['Date'].iloc[-len(y_val_inv) - len(y_test_inv):-len(y_test_inv)], val_predictions_inv, label='Validation Predicted', marker='o')
plt.plot(df['Date'].iloc[-len(y_test_inv):], y_test_inv, label='Test Actual', marker='o')
plt.plot(df['Date'].iloc[-len(y_test_inv):], predictions_inv, label='Test Predicted', marker='o')
plt.title('Validation and Test Predictions vs Actual (LSTM+GARCH)', fontsize=16, fontweight='bold')
plt.xlabel('Date')
plt.ylabel('Price')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 8))
plt.scatter(y_val_inv, val_predictions_inv, alpha=0.7, c='blue', edgecolor='black', label='Validation Predictions')
plt.scatter(y_test_inv, predictions_inv, alpha=0.7, c='purple', edgecolor='black', label='Test Predictions')
plt.plot([min(y_val_inv.min(), y_test_inv.min()), max(y_val_inv.max(), y_test_inv.max())],
         [min(y_val_inv.min(), y_test_inv.min()), max(y_val_inv.max(), y_test_inv.max())],
         'r--', label='Ideal Fit')
plt.title('Regression Graph for Validation and Test Sets (LSTM+GARCH)')
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Convergence (LSTM+GARCH)')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.show()

print("\nValidation Set Performance Metrics:")
print(f"RMSE: ${rmse_val:.2f}")
print(f"MAE: ${mae_val:.2f}")
print(f"MAPE: {mape_val:.2f}%")
print(f"R-squared: {r2_val:.4f}")

print("\nTest Set Performance Metrics:")
print(f"RMSE: ${rmse_test:.2f}")
print(f"MAE: ${mae_test:.2f}")
print(f"MAPE: {mape_test:.2f}%")
print(f"R-squared: {r2_test:.4f}")
