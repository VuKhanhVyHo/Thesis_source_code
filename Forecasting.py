import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from datetime import timedelta
from sklearn.metrics import mean_squared_error
from pymodbus.client import ModbusTcpClient

# List of files to load (e.g., monthly data files)
file_paths = ["./unused/082022.csv", "./unused/082023.csv"]

# Load and concatenate all data files
dfs = []
for file_path in file_paths:
    df = pd.read_csv(file_path)
    df['SETTLEMENTDATE'] = pd.to_datetime(df['SETTLEMENTDATE'], format='%Y/%m/%d %H:%M:%S')
    dfs.append(df)

# Concatenate all files and sort by 'SETTLEMENTDATE' to maintain temporal order
df_combined = pd.concat(dfs).sort_values(by='SETTLEMENTDATE').reset_index(drop=True)
print(df_combined[:3])

# Select only the 'RRP' column for training and forecasting
data = df_combined[['RRP']].values

# Apply MinMax scaling to the 'RRP' data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Sequence creation function for 24-hour windows (288 time steps)
def create_sequences(data, time_steps):
    X = []
    y = []
    for i in range(time_steps, len(data)):
        X.append(data[i - time_steps:i])
        y.append(data[i, 0])  # Predicting 'RRP' (spot price)
    return np.array(X), np.array(y)

# Define time steps for 24-hour window
time_steps = 24 * 12
X, y = create_sequences(scaled_data, time_steps)
X = X.reshape((X.shape[0], X.shape[1], 1))  # Adjust shape for single feature

# Define LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], X.shape[2])))
model.add(Dropout(0.2))

# Second LSTM layer
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))

# Third LSTM layer
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))

# Dense layers for final output
model.add(Dense(units=25))
model.add(Dense(units=1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Implement Early Stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model with early stopping
model.fit(X, y, batch_size=64, epochs=100, validation_split=0.05, callbacks=[early_stopping])

# Make predictions and rescale back to original values
predictions = model.predict(X[0:288])
predicted_rrp = scaler.inverse_transform(np.concatenate([predictions], axis=1))

# Define time labels for 24-hour prediction plot
last_time = pd.to_datetime(df_combined['SETTLEMENTDATE'].iloc[-1]) 
time_labels = [(last_time + timedelta(minutes=5 * i)).strftime('%H:%M') for i in range(288)]

# Load actual data for comparison (e.g., August data for 2024)
df_august = pd.read_csv('./unused/082024.csv')
df_august['SETTLEMENTDATE'] = pd.to_datetime(df_august['SETTLEMENTDATE'], format='%Y/%m/%d %H:%M:%S')
df_august_filtered = df_august[df_august['SETTLEMENTDATE'].dt.date == pd.to_datetime('2024-08-01').date()]
actual_rrp = df_august_filtered['RRP'].values[:288]
time_labels_august = df_august_filtered['SETTLEMENTDATE'].dt.strftime('%H:%M').values[0:288]

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(actual_rrp[0:287], predicted_rrp[0:287]))
print(f'RMSE: {rmse}')
residuals = actual_rrp[:287] - predicted_rrp[:287].flatten()

# Create a histogram of the residuals
hist_counts, bin_edges = np.histogram(residuals, bins=50)  # Adjust bins as needed

# Find the bin with the highest count
max_bin_index = np.argmax(hist_counts)
most_frequent_bin_center = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2
bidding_price = np.full(len(predictions), 0)
for i in range(len(predicted_rrp)):
    predicted_rrp[i] += most_frequent_bin_center
    if (predicted_rrp[i] < 0):
        bidding_price[i] = 500

# Ensure `predicted_rrp` is one-dimensional
predicted_rrp = predicted_rrp.flatten()  # Only if `predicted_rrp` has extra dimensions

def detect_and_switch_off(predicted_rrp, times, min_duration=24):
    switch_off_periods = []
    i = 0
    start_time = None

    while i < len(predicted_rrp):
        # Start a new shutdown period if a negative price is detected
        if predicted_rrp[i] < 0:
            if start_time is None:  # Begin tracking a new period
                start_time = times[i]

            # Extend the shutdown period if still below threshold
            end_time = times[i]
        
        # Finalize the shutdown period if the threshold has been met and a positive price is detected
        if start_time and (predicted_rrp[i] >= 0 or i == len(predicted_rrp) - 1):
            # Check if the shutdown period meets the minimum duration
            if (pd.to_datetime(end_time) - pd.to_datetime(start_time)).seconds >= min_duration * 5 * 60:
                switch_off_periods.append((start_time, end_time))
            start_time = None  # Reset to detect new periods
        
        i += 1

    return switch_off_periods


def send_switch_off_signal(periods):
    client = ModbusTcpClient('localhost', port=5020)
    if not client.connect():
        print("Failed to connect to Modbus server.")
        return

    for start_time, end_time in periods:
        print(f"Switching OFF from {start_time} to {end_time}")
        client.write_coil(0x01, False)  # Turn off
        # In practice, you would add a delay or scheduling to turn it back on at `end_time`
        print(f"Switching ON after {end_time}")
        client.write_coil(0x01, True)   # Turn on after the period

    client.close()


# Detect switch-off periods for negative prices over 2 hours
times = df_august_filtered['SETTLEMENTDATE'].values[:288]  # Times for the prediction period
switch_off_periods = detect_and_switch_off(predicted_rrp, times, min_duration=24)

# Send switch-off signal for detected periods
send_switch_off_signal(switch_off_periods)


# Plot predicted vs actual prices
plt.figure(figsize=(12, 6))
plt.plot(predicted_rrp, label='Predicted Prices')
plt.plot(actual_rrp, label='Actual Prices (01/08/2024)')
plt.plot(bidding_price, label= 'Bidding price')
plt.title('Predicted vs Actual Spot Prices for 02/08/2024 with bidding price')
plt.xlabel('Time (HH:mm)')
plt.ylabel('Price')
plt.xticks(ticks=np.arange(0, 288, 6), labels=time_labels_august[::6], rotation=45)
plt.legend()
plt.tight_layout()
plt.show()
