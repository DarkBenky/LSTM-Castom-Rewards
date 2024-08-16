import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization, Flatten
import wandb

# Initialize WandB
wandb.init(project="LSTM_CNN_Model_Comparison", config={
    "epochs": 10,
    "batch_size": 128,
    "num_values": 128,
    "future_window": 30,
    "dropout": 0.2,
    "neurons": 128
})
config = wandb.config

# Function to load and process CSV data
def load_data(filename):
    data = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            # Convert each value in the row to float32 and trim any whitespace
            float_row = [np.float32(value.strip()) for value in row]
            # Append everything except the first column (index 0)
            data.append(float_row[1:])
    return np.array(data)

# Function to generate labels and rewards
def generateLabelsRewards(data, window_size, future_window, num_of_records=10_000):
    inputLabels = []
    actions = []
    
    if num_of_records > len(data):
        num_of_records = len(data)

    for i in range(window_size, num_of_records - future_window):
        currentPrice = data[i][1]  # Current price
        window = data[i-window_size:i]  # Window of past prices
        nextPrices = data[i:i+future_window][:, 1]  # Future prices
        
        averageFuturePrice = np.mean(nextPrices)  # Average future price
        
        # Record the input window (features)
        inputLabels.append(window)
        
        # Calculate reward based on price differences
        if currentPrice < averageFuturePrice:
            r = averageFuturePrice - currentPrice
            action_data = {
                "BUY": 0,
                "Sell": -r/100,
            }
        elif currentPrice > averageFuturePrice:
            r = currentPrice - averageFuturePrice
            action_data = {
                "BUY": -r/100,
                "Sell": 0,
            }
        else:
            action_data = {
                "BUY": 0,
                "Sell": 0,
            }
        
        # Append action data
        actions.append(action_data)

    return np.array(inputLabels), np.array(actions)

# Function to check for NaN and infinite values in the data
def check_data_sanity(data, name="Data"):
    if np.isnan(data).any():
        print(f"Warning: {name} contains NaN values.")
    if np.isinf(data).any():
        print(f"Warning: {name} contains infinite values.")
    print(f"{name} summary:")
    print(f" - Mean: {np.mean(data)}")
    print(f" - Std Dev: {np.std(data)}")
    print(f" - Min: {np.min(data)}")
    print(f" - Max: {np.max(data)}")
    print(f" - Number of NaNs: {np.isnan(data).sum()}")
    print(f" - Number of Infs: {np.isinf(data).sum()}")
    print("\n")

# Load the data
data = load_data('data.csv')
data_len = len(data[0])

# Parameters
NUM_VALUES = config.num_values
FUTURE_WINDOW = config.future_window
EPOCHS = config.epochs
BATCH_SIZE = config.batch_size
NEURONS = config.neurons
DROPOUT = config.dropout
NUM_OF_RECORDS = 15_000

# Generate input labels and action dictionaries
inputLabels, actions = generateLabelsRewards(data, NUM_VALUES, FUTURE_WINDOW, NUM_OF_RECORDS)

# Check data sanity
check_data_sanity(inputLabels, "Input Labels")

# Get percentages of BUY, Sell, and Hold actions
buy_percentage = np.mean([action["BUY"] == 0 for action in actions])
sell_percentage = np.mean([action["Sell"] == 0 for action in actions])

print(f"BUY percentage: {buy_percentage * 100:.2f}%")
print(f"Sell percentage: {sell_percentage * 100:.2f}%")

# Log data to WandB
wandb.log({
    "Sell" : sell_percentage,
    "Buy" : buy_percentage})

# Define the LSTM model
def create_lstm_model():
    model = Sequential()
    model.add(LSTM(NEURONS, activation='relu', input_shape=(NUM_VALUES, data_len), return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT))
    model.add(LSTM(NEURONS, activation='relu', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT))
    model.add(LSTM(NEURONS, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT))
    model.add(Dense(2, activation='softmax'))  # 2 classes for BUY, Sell
    return model

# Define the CNN+LSTM model
def create_cnn_lstm_model():
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(NUM_VALUES, data_len)))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    
    # Directly add LSTM layers without flattening
    model.add(LSTM(NEURONS, activation='relu', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT))
    model.add(LSTM(NEURONS, activation='relu', return_sequences=True))
    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT))
    model.add(LSTM(NEURONS, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(DROPOUT))
    model.add(Dense(2, activation='softmax'))  # 2 classes for BUY, Sell
    return model


# Instantiate both models
lstm_model = create_lstm_model()
cnn_lstm_model = create_cnn_lstm_model()

lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn_lstm_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

lstm_model.summary()
cnn_lstm_model.summary()

wandb.log({"lstm_model": lstm_model.summary()})
wandb.log({"cnn_lstm_model": cnn_lstm_model.summary()})

# Custom training loop for both models
for epoch in range(EPOCHS):
    for batch_index in range(0, len(inputLabels), BATCH_SIZE):
        # Extract the current batch of data
        state_batch = inputLabels[batch_index:batch_index+BATCH_SIZE]
        action_batch = actions[batch_index:batch_index+BATCH_SIZE]
        
        # Convert action dictionary to separate target arrays
        buy_rewards = np.array([action["BUY"] for action in action_batch])
        sell_rewards = np.array([action["Sell"] for action in action_batch])
        
        reward_batch = np.stack([buy_rewards, sell_rewards], axis=1)
        # reward_batch = np.clip(reward_batch, -1.0, 1.0)  # Clipping rewards to a manageable range
        
        # Train LSTM model
        lstm_loss, lstm_accuracy = lstm_model.train_on_batch(state_batch, np.argmax(reward_batch, axis=1))
        # Train CNN+LSTM model
        cnn_lstm_loss, cnn_lstm_accuracy = cnn_lstm_model.train_on_batch(state_batch, np.argmax(reward_batch, axis=1))
        
        # Log losses and accuracies to WandB
        wandb.log({
            "epoch": epoch,
            "batch": batch_index // BATCH_SIZE,
            "lstm_loss": lstm_loss,
            "lstm_accuracy": lstm_accuracy,
            "cnn_lstm_loss": cnn_lstm_loss,
            "cnn_lstm_accuracy": cnn_lstm_accuracy
        })
        
        # Logging
        print(f"Epoch {epoch}, Batch {batch_index//BATCH_SIZE}: LSTM Loss: {lstm_loss:.4f}, LSTM Accuracy: {lstm_accuracy:.4f}")
        print(f"Epoch {epoch}, Batch {batch_index//BATCH_SIZE}: CNN+LSTM Loss: {cnn_lstm_loss:.4f}, CNN+LSTM Accuracy: {cnn_lstm_accuracy:.4f}")

# Test Mode: Calculate profits based on the trained models
def calculate_profits(data, model, window_size, future_window, test_size=1000):
    profits = []
    for i in range(window_size, len(data) - future_window):
        state = np.array(data[i-window_size:i]).reshape(1, window_size, data.shape[1])
        prediction = np.argmax(model.predict(state), axis=1)[0]
        
        currentPrice = data[i][1]
        futurePrice = data[i+future_window][1]
        
        # Simulate trading
        if prediction == 0:  # BUY
            profit = futurePrice - currentPrice
        else:  # Sell
            profit = currentPrice - futurePrice
        
        profits.append(profit)
        if len(profits) >= test_size:
            break
    
    return np.sum(profits)

# Calculate and log profits for both models
lstm_profit = calculate_profits(data, lstm_model, NUM_VALUES, FUTURE_WINDOW)
cnn_lstm_profit = calculate_profits(data, cnn_lstm_model, NUM_VALUES, FUTURE_WINDOW)

print(f"Total simulated profit (LSTM): {lstm_profit:.2f}")
print(f"Total simulated profit (CNN+LSTM): {cnn_lstm_profit:.2f}")

# Log profits to WandB
wandb.log({
    "total_profit_lstm": lstm_profit,
    "total_profit_cnn_lstm": cnn_lstm_profit
})

# Finish WandB logging
wandb.finish()
