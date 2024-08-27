import csv
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, BatchNormalization
import wandb

# Initialize WandB
wandb.init(project="LSTM_CNN_Comparison", config={
    "epochs": 5,
    "batch_size": 128,
    "num_values": 64,
    "future_window": 32,
    "dropout": 0.2,
    "neurons": 256
})
config = wandb.config

# Function to load and process CSV data
def load_data(filename):
    data = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            float_row = [np.float32(value.strip()) for value in row]
            data.append(float_row[1:])
    return np.array(data)

# Function to generate labels and rewards
def generateLabelsRewards(data, window_size, future_window, num_of_records=10_000):
    inputLabels = []
    actions = []
    
    if num_of_records > len(data):
        num_of_records = len(data)

    for i in range(window_size, num_of_records - future_window):
        currentPrice = data[i][1]
        window = data[i-window_size:i]
        nextPrices = data[i:i+future_window][:, 1]
        
        averageFuturePrice = np.mean(nextPrices)
        
        inputLabels.append(window)
        
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
        
        actions.append(action_data)

    return np.array(inputLabels), np.array(actions)

# Function to generate labels
def generateLabels(data, window_size, future_window, num_of_records=10_000):
    inputLabels = []
    outputLabels = []
    
    if num_of_records > len(data):
        num_of_records = len(data)

    for i in range(window_size, num_of_records - future_window):
        currentPrice = data[i][1]
        window = data[i-window_size:i]
        nextPrices = data[i:i+future_window][:, 1]
        
        averageFuturePrice = np.mean(nextPrices)
        
        inputLabels.append(window)
        
        if currentPrice < averageFuturePrice:
           outputLabels.append([1, 0])
        elif currentPrice > averageFuturePrice:
            outputLabels.append([0, 1])
        else:
            outputLabels.append([0.5, 0.5])
    
    return np.array(inputLabels), np.array(outputLabels)

def generateLabelsPrice(data, window_size, future_window, num_of_records=10_000):
    inputLabels = []
    outputLabels = []
    
    if num_of_records > len(data):
        num_of_records = len(data)

    for i in range(window_size, num_of_records - future_window):
        window = data[i-window_size:i]
        nextPrices = data[i:i+future_window][:, 1]
        
        averageFuturePrice = np.mean(nextPrices) / 100_000
        
        inputLabels.append(window / 100_000)
        
        outputLabels.append(averageFuturePrice)
    
    return np.array(inputLabels), np.array(outputLabels)

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

# Function to calculate profits based on the trained models
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
    return np.mean(profits), np.std(profits), len(profits)

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
X, y = generateLabels(data, NUM_VALUES, FUTURE_WINDOW, NUM_OF_RECORDS)
X_price, y_price = generateLabelsPrice(data, NUM_VALUES, FUTURE_WINDOW, NUM_OF_RECORDS)

# Check data sanity
check_data_sanity(inputLabels, "Input Labels")

# Get percentages of BUY, Sell actions
buy_percentage = np.mean([action["BUY"] == 0 for action in actions])
sell_percentage = np.mean([action["Sell"] == 0 for action in actions])

print(f"BUY percentage: {buy_percentage * 100:.2f}%")
print(f"Sell percentage: {sell_percentage * 100:.2f}%")

# Log data to WandB
wandb.log({
    "Sell" : sell_percentage,
    "Buy" : buy_percentage
})

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
# def create_cnn_lstm_model():
#     model = Sequential()
#     model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(NUM_VALUES, data_len)))
#     model.add(MaxPooling1D(pool_size=2))
#     model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
#     model.add(MaxPooling1D(pool_size=2))
    
#     # Directly add LSTM layers without flattening
#     model.add(LSTM(NEURONS, activation='relu', return_sequences=True))
#     model.add(BatchNormalization())
#     model.add(Dropout(DROPOUT))
#     model.add(LSTM(NEURONS, activation='relu', return_sequences=True))
#     model.add(BatchNormalization())
#     model.add(Dropout(DROPOUT))
#     model.add(LSTM(NEURONS, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(DROPOUT))
#     model.add(Dense(2, activation='softmax'))  # 2 classes for BUY, Sell
#     return model

def create_lstm_model_Price():
    model = Sequential()
    model.add(LSTM(NEURONS, activation='tanh', input_shape=(NUM_VALUES, data_len), return_sequences=True))
    model.add(Dropout(DROPOUT))
    model.add(LSTM(NEURONS, activation='tanh', return_sequences=True))
    model.add(Dropout(DROPOUT))
    model.add(LSTM(NEURONS, activation='tanh'))
    model.add(Dropout(DROPOUT))
    model.add(Dense(1))  # Predict the future price
    
    return model

# Define the CNN+LSTM model
# def create_cnn_lstm_model_Price():
#     model = Sequential()
#     model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(NUM_VALUES, data_len)))
#     model.add(MaxPooling1D(pool_size=2))
#     model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
#     model.add(MaxPooling1D(pool_size=2))
    
#     # Directly add LSTM layers without flattening
#     model.add(LSTM(NEURONS, activation='relu', return_sequences=True))
#     model.add(BatchNormalization())
#     model.add(Dropout(DROPOUT))
#     model.add(LSTM(NEURONS, activation='relu', return_sequences=True))
#     model.add(BatchNormalization())
#     model.add(Dropout(DROPOUT))
#     model.add(LSTM(NEURONS, activation='relu'))
#     model.add(BatchNormalization())
#     model.add(Dropout(DROPOUT))
#     model.add(Dense(1))  # Predict the future price
#     return model

# Function to train and evaluate a given model
def train_and_evaluate_model(model, inputLabels, actions, model_name):
    print(f"\nTraining {model_name}...")
    
    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Training loop
    for epoch in range(EPOCHS):
        for batch_index in range(0, len(inputLabels), BATCH_SIZE):
            # Extract the current batch of data
            state_batch = inputLabels[batch_index:batch_index+BATCH_SIZE]
            action_batch = actions[batch_index:batch_index+BATCH_SIZE]
            
            # Convert action dictionary to separate target arrays
            buy_rewards = np.array([action["BUY"] for action in action_batch])
            sell_rewards = np.array([action["Sell"] for action in action_batch])
            
            reward_batch = np.stack([buy_rewards, sell_rewards], axis=1)
            
            # Train the model
            loss, accuracy = model.train_on_batch(state_batch, np.argmax(reward_batch, axis=1))
            
            # Log metrics to WandB
            wandb.log({
                f"{model_name}_epoch": epoch,
                f"{model_name}_batch_index": batch_index,
                f"{model_name}_loss": loss,
                f"{model_name}_accuracy": accuracy,
            })

        print(f"Epoch: {epoch}, Batch Index: {batch_index}, Loss: {loss}, Accuracy: {accuracy}")
    
    # Evaluate and log profits
    mean_profit, std_profit, num_tests = calculate_profits(data, model, NUM_VALUES, FUTURE_WINDOW)
    print(f"{model_name} - Mean Profit: {mean_profit:.2f}, Std Dev: {std_profit:.2f}, Number of Tests: {num_tests}")
    
    # Log profit metrics to WandB
    wandb.log({
        f"{model_name}_profit_mean": mean_profit,
        f"{model_name}_profit_std": std_profit,
        f"{model_name}_num_tests": num_tests
    })

# Function to train and evaluate a given model with X data
def train_and_evaluate_model_X(model, X, y, model_name):
    print(f"\nTraining {model_name}...")
    
    # Compile model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Training loop
    for epoch in range(EPOCHS):
        for batch_index in range(0, len(inputLabels), BATCH_SIZE):
            # Extract the current batch of data
            state_batch = X[batch_index:batch_index+BATCH_SIZE]
            action_batch = y[batch_index:batch_index+BATCH_SIZE]

            # Train the model
            loss, accuracy = model.train_on_batch(state_batch, np.argmax(action_batch, axis=1))

            # Log metrics to WandB
            wandb.log({
                f"{model_name}_epoch": epoch,
                f"{model_name}_batch_index": batch_index,
                f"{model_name}_loss": loss,
                f"{model_name}_accuracy": accuracy,
            })

        print(f"Epoch: {epoch}, Batch Index: {batch_index}, Loss: {loss}, Accuracy: {accuracy}")

    # Evaluate and log profits
    mean_profit, std_profit, num_tests = calculate_profits(data, model, NUM_VALUES, FUTURE_WINDOW)
    print(f"{model_name} - Mean Profit: {mean_profit:.2f}, Std Dev: {std_profit:.2f}, Number of Tests: {num_tests}")
    
    # Log profit metrics to WandB
    wandb.log({
        f"{model_name}_profit_mean": mean_profit,
        f"{model_name}_profit_std": std_profit,
        f"{model_name}_num_tests": num_tests
    })

# Instantiate and train LSTM model for price prediction
# Function to train and evaluate a given price model
def train_and_evaluate_price_model(model, X_price, y_price, model_name):
    print(f"\nTraining {model_name}...")
    
    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4), loss='mean_squared_error', metrics=['mae'])
    
    # Training loop
    for epoch in range(EPOCHS):
        for batch_index in range(0, len(X_price), BATCH_SIZE):
            # Extract the current batch of data
            state_batch = X_price[batch_index:batch_index+BATCH_SIZE]
            price_batch = y_price[batch_index:batch_index+BATCH_SIZE]
            
            # Train the model
            loss, mae = model.train_on_batch(state_batch, price_batch)

            # Log metrics to WandB
            wandb.log({
                f"{model_name}_loss": loss,
                f"{model_name}_mae": mae,
            })
            print(f"Loss: {loss * 100_000}, MAE: {mae}")
        print(f"Epoch: {epoch}, Batch Index: {batch_index}, Loss: {loss}, MAE: {mae}")
    
    # Evaluate the model
    predictions = model.predict(X_price)
    mean_absolute_error = np.mean(np.abs(predictions - y_price))
    
    print(f"{model_name} - Mean Absolute Error: {mean_absolute_error:.2f}")
    
    # Log performance metrics to WandB
    wandb.log({
        f"{model_name}_mae": mean_absolute_error
    })
    
    # Optional: Calculate and log potential profits
    # Here’s an example of how you might calculate potential profits based on predictions
    def calculate_profits(predictions, actual_prices):
        # Assume a simple profit model where you buy at the predicted price and sell at the actual price
        # This is a simplified example and might need adjustments based on your specific use case
        profits = actual_prices - predictions
        for i in range(len(profits)):
            wandb.log({f"{model_name}_profit_{i}": profits[i]})
        return np.mean(profits), np.std(profits)
    
    mean_profit, std_profit = calculate_profits(predictions, y_price)
    print(f"{model_name} - Mean Profit: {mean_profit:.2f}, Std Dev of Profit: {std_profit:.2f}")
    
    # Log profit metrics to WandB
    wandb.log({
        f"{model_name}_profit_mean": mean_profit,
        f"{model_name}_profit_std": std_profit
    })

# Instantiate and train LSTM model
# lstm_model = create_lstm_model()
# train_and_evaluate_model(lstm_model, inputLabels, actions, "lstm_model")

# # Instantiate and train CNN+LSTM model
# cnn_lstm_model = create_cnn_lstm_model()
# train_and_evaluate_model(cnn_lstm_model, inputLabels, actions, "cnn_lstm_model")

# Instantiate and train LSTM_X model
# lstm_model_X = create_lstm_model()
# train_and_evaluate_model_X(lstm_model_X, X, y, "lstm_model_X")

# Instantiate and train CNN+LSTM_X model
# cnn_lstm_model_X = create_cnn_lstm_model()
# train_and_evaluate_model_X(cnn_lstm_model_X, X, y, "cnn_lstm_model_X")

# Instantiate and train LSTM model for price prediction
lstm_model_price = create_lstm_model_Price()
train_and_evaluate_price_model(lstm_model_price, X_price, y_price, "lstm_model_price")
