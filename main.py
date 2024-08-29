import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split

# Function to load and process CSV data
def load_data(filename):
    data = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            float_row = [np.float32(value.strip()) for value in row]
            data.append(float_row[1:])  # Assuming the first column is a timestamp or non-numeric identifier
    return np.array(data)

def scale_data(data: np.array):
    # Convert the numpy array to a DataFrame
    df = pd.DataFrame(data=data)

    # Print the data ranges before scaling (for debugging)
    print("Data ranges before scaling:")
    print(df.describe())

    # Initialize the MinMaxScaler
    scaler = MinMaxScaler(feature_range=(-1, 1))

    # Apply the scaler to each column
    scaled_data = scaler.fit_transform(df)

    # Print the data ranges after scaling (for debugging)
    print("Data ranges after scaling:")
    df = pd.DataFrame(scaled_data)

    return df

def generateLabelsPrice(data, window_size, future_window):
    inputLabels = []
    outputLabels = []

    for i in range(window_size, len(data) - future_window):
        window = data[i-window_size:i]
        nextPrices = data[i:i+future_window][:, 1]
        
        averageFuturePrice = np.mean(nextPrices)
        
        inputLabels.append(window)
        outputLabels.append(averageFuturePrice)
    
    return np.array(inputLabels), np.array(outputLabels)

# Load and process the data
data = load_data('data.csv')
data = np.array(scale_data(data=data))

# Generate labels and input data
window_size = 30 # Example window size
future_window = 3  # Example future window
X, Y = generateLabelsPrice(data, window_size, future_window)

# Reshape the input data for LSTM (samples, time steps, features)
X = X.reshape((X.shape[0], X.shape[1], X.shape[2]))

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, shuffle=False)

# Building the LSTM model
model = Sequential()
model.add(LSTM(1024, input_shape=(X.shape[1], X.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(512))
model.add(Dense(2048, activation='relu'))
model.add(Dense(2048, activation='relu'))
model.add(Dense(1))  # Single output for the average future price

# Compile the model
model.compile(optimizer='adam', loss='mse')

model.summary()

# Train the model
history = model.fit(X_train, Y_train, epochs=3, batch_size=32, validation_data=(X_val, Y_val), verbose=1)

# Evaluate the model
loss = model.evaluate(X_val, Y_val)
print(f"Validation Loss: {loss}")

# Plot training & validation loss
import matplotlib.pyplot as plt

# Make predictions
Y_pred = model.predict(X_val)

# Calculate profits
# For simplicity, we'll use a constant transaction cost (e.g., 0.02 or 2%)
transaction_cost = 0.02
profits = (Y_val - Y_pred.squeeze()) - transaction_cost * abs(Y_val - Y_pred.squeeze())

# Calculate cumulative profits
cumulative_profits = np.cumsum(profits)

# Plot predictions vs reality
plt.figure(figsize=(14, 7))

# Plot actual prices
plt.plot(range(len(Y_val)), Y_val, label='True Values', color='blue', linestyle='--')

# Plot predicted prices
plt.plot(range(len(Y_pred)), Y_pred, label='Predicted Values', color='red', linestyle='-')

plt.title('Predictions vs Reality')
plt.xlabel('Index')
plt.ylabel('Average Future Price')
plt.legend()
plt.grid(True)
plt.show()

# Plot cumulative profits
plt.figure(figsize=(14, 7))

# Plot actual prices
plt.plot(range(len(Y_val)), Y_val, label='Actual Prices', color='blue', linestyle='--')

# Plot cumulative profits
plt.plot(range(len(cumulative_profits)), cumulative_profits, label='Cumulative Profit', color='green', linestyle='-')

plt.title('Cumulative Profits vs Actual Prices')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)
plt.show()