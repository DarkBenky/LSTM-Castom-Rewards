import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization

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
def generateLabelsRewards(data, window_size, future_window, penalty_factor=2.0, num_of_records=10_000):
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
                "Sell": -r,
                "Hold": r/PENALTY_FACTOR
            }
        elif currentPrice > averageFuturePrice:
            r = currentPrice - averageFuturePrice
            action_data = {
                "BUY": -r,
                "Sell": 0,
                "Hold": r/PENALTY_FACTOR
            }
        else:
            action_data = {
                "BUY": -0.1,
                "Sell": -0.1,
                "Hold": 0
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
NUM_VALUES = 128
FUTURE_WINDOW = 30
PENALTY_FACTOR = 2.5
EPOCHS = 10
NUM_OF_RECORDS = 15_000

# Generate input labels and action dictionaries
inputLabels, actions = generateLabelsRewards(data, NUM_VALUES, FUTURE_WINDOW, PENALTY_FACTOR, NUM_OF_RECORDS)

# Check data sanity
check_data_sanity(inputLabels, "Input Labels")
#TODO: Check if sanity of rewards


# Get percentages of BUY, Sell, and Hold actions
buy_percentage = np.mean([action["BUY"] == 0 for action in actions])
sell_percentage = np.mean([action["Sell"] == 0 for action in actions])
hold_percentage = np.mean([action["Hold"] == 0 for action in actions])

print(f"BUY percentage: {buy_percentage * 100:.2f}%")
print(f"Sell percentage: {sell_percentage * 100:.2f}%")
print(f"Hold percentage: {hold_percentage * 100:.2f}%")

# Prepare LSTM model
NEURONS = 128
DROPOUT = 0.2
BATCH_SIZE = 16

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
# Output layer with softmax activation for multi-class classification
model.add(Dense(3, activation='softmax'))  # 3 classes for BUY, Sell, Hold

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

END = False
# Custom training loop
for epoch in range(EPOCHS):
    for batch_index in range(0, len(inputLabels), BATCH_SIZE):
        # Extract the current batch of data
        state_batch = inputLabels[batch_index:batch_index+BATCH_SIZE]
        action_batch = actions[batch_index:batch_index+BATCH_SIZE]
        
        # Convert action dictionary to separate target arrays
        buy_rewards = np.array([action["BUY"] for action in action_batch])
        sell_rewards = np.array([action["Sell"] for action in action_batch])
        hold_rewards = np.array([action["Hold"] for action in action_batch])
        
        reward_batch = np.stack([buy_rewards, sell_rewards, hold_rewards], axis=1)  # Stack into shape (batch_size, 3)
        
        # Forward pass
        with tf.GradientTape() as tape:
            predictions = model(state_batch, training=True)
            base_loss = tf.keras.losses.sparse_categorical_crossentropy(np.argmax(reward_batch, axis=1), predictions)

            # Modify the loss by incorporating rewards
            weighted_loss = base_loss * np.max(reward_batch, axis=1)
            loss = tf.reduce_mean(weighted_loss)
        
        # Compute gradients and update weights
        gradients = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        # Logging
        print(f"Epoch {epoch}, Batch {batch_index//BATCH_SIZE}: Loss: {loss.numpy():.4f}")
        # Get accuracy
        print(f"Accuracy: {np.mean(np.argmax(reward_batch, axis=1) == np.argmax(predictions, axis=1)):.4f}")
        # if training accuracy is 100% end trading and test model
        if np.mean(np.argmax(reward_batch, axis=1) == np.argmax(predictions, axis=1)) == 1.00:
            END = True
            break 
    if END:
        break 

# Test Mode: Calculate profits based on the trained model
def calculate_profits(data, model, window_size, future_window):
    profits = []
    for i in range(window_size, len(data) - future_window):
        state = np.array(data[i-window_size:i]).reshape(1, window_size, data.shape[1])
        prediction = np.argmax(model.predict(state), axis=1)[0]
        
        currentPrice = data[i][1]
        futurePrice = data[i+future_window][1]
        
        # Simulate trading
        if prediction == 0:  # BUY
            profit = futurePrice - currentPrice
        elif prediction == 1:  # Sell
            profit = currentPrice - futurePrice
        else:  # Hold
            profit = 0
        
        profits.append(profit)
    
    return np.sum(profits)

# Calculate profits
total_profit = calculate_profits(data, model, NUM_VALUES, FUTURE_WINDOW)
print(f"Total simulated profit: {total_profit:.2f}")
