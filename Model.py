import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import csv

import wandb

wandb.init(project="btc")

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
    scaler = MinMaxScaler(feature_range=(1, 2))

    # Apply the scaler to each column
    scaled_data = scaler.fit_transform(df)

    # Print the data ranges after scaling (for debugging)
    print("Data ranges after scaling:")
    print(pd.DataFrame(scaled_data).describe())

    return np.array(scaled_data)

class MyEnv:
    def __init__(self, data, balance_usd=1000, balance_btc=0):
        self.data = data
        self.balance_usd = balance_usd
        self.balance_btc = balance_btc
        self.done = False
        self.window_size = 30
        self.future_window = 30
        self.reset()

    def next_price(self):
        return self.data[self.step + self.window_size + self.future_window][1]
    
    def next_prices_window(self, window_size):
        return self.data[self.step + self.window_size:self.step + self.window_size + window_size][1]
    
    def calculate_portfolio_value(self):
        return self.balance_usd + self.balance_btc * self.data[self.step + self.window_size][1]
    
    def calculate_portfolio_value_at_step(self, stepOffset):
        return self.balance_usd + self.balance_btc * self.data[self.step + self.window_size + stepOffset][1]
    
    def calculate_next_average_portfolio_value(self):
        average_portfolio_value = 0
        for i in range(self.future_window):
            average_portfolio_value += self.calculate_portfolio_value_at_step(i)
        return average_portfolio_value / self.future_window

    def reset(self):
        self.step = 0
        self.done = False
        self.balance_usd = 1000
        self.balance_btc = 0.1
        self.current_observation = self.data[self.step:self.step+self.window_size]
        self.current_overall_balance = self.calculate_portfolio_value()
        self.starting_overall_balance = self.calculate_portfolio_value()

        # Pad the new row with zeros to match the observation dimensions (2 -> 3)
        portfolio_info = [self.calculate_portfolio_value(), self.balance_btc, self.balance_usd]
        padded_info = np.zeros((self.window_size, len(portfolio_info)))
        padded_info[-1] = portfolio_info  # Add the portfolio info as the last row
        self.current_observation = np.hstack((self.current_observation, padded_info))

        return self.current_observation

    def __step__(self, action, nextStep=True):
        if self.done:
            raise ValueError("Environment is done, reset it before calling step.")

        current_price = self.data[self.step + self.window_size][1]
        # Extract action and amount
        action_type = np.argmax(action[:2])  # buy, sell (hold removed)
        amount = action[2]  # amount in range 0-1

        current_portfolio_value = self.calculate_portfolio_value()

        # Perform the action
        if action_type == 0:  # Buy
            usd_to_spend = self.balance_usd * amount
            btc_to_buy = usd_to_spend / current_price
            self.balance_btc += btc_to_buy
            self.balance_usd -= usd_to_spend
        elif action_type == 1:  # Sell
            btc_to_sell = self.balance_btc * amount
            self.balance_usd += btc_to_sell * current_price
            self.balance_btc -= btc_to_sell

        if nextStep:
            self.step += 1
            # portfolio_value_change = self.calculate_portfolio_value() - current_portfolio_value
            portfolio_value_change = self.calculate_next_average_portfolio_value() - current_portfolio_value
        else:
            # portfolio_value_change = self.calculate_portfolio_value_at_step(1) - current_portfolio_value
            portfolio_value_change = self.calculate_next_average_portfolio_value() - current_portfolio_value
            # remove the action from the portfolio value change
            if action_type == 0:
                self.balance_btc -= btc_to_buy
                self.balance_usd += usd_to_spend
            elif action_type == 1:
                self.balance_usd -= btc_to_sell * current_price
                self.balance_btc += btc_to_sell


        if self.step + self.window_size + self.future_window >= len(self.data):
            self.done = True
        
        self.current_observation = self.data[self.step:self.step+self.window_size]
        
        # Pad the new row with zeros to match the observation dimensions (2 -> 3)
        portfolio_info = [self.calculate_portfolio_value(), self.balance_btc, self.balance_usd]
        padded_info = np.zeros((self.window_size, len(portfolio_info)))
        padded_info[-1] = portfolio_info  # Add the portfolio info as the last row
        self.current_observation = np.hstack((self.current_observation, padded_info))

        return self.current_observation, portfolio_value_change, self.done

    def render(self):
        print(f"Step: {self.step}, USD: {self.balance_usd:.2f}, BTC: {self.balance_btc:.4f}, Portfolio Value: {self.calculate_portfolio_value():.2f}")

# Define a simple policy network
class PolicyNetwork(tf.keras.Model):
    def __init__(self, lstm_units=128, dense_units=256, num_lstm_layers=2, num_dense_layers=2, dropout_rate=0.2):
        super(PolicyNetwork, self).__init__()
        
        self.lstm_layers = [tf.keras.layers.LSTM(lstm_units, return_sequences=True) for _ in range(num_lstm_layers - 1)]
        self.lstm_layers.append(tf.keras.layers.LSTM(lstm_units))  # Last LSTM layer doesn't return sequences
        
        self.dense_layers = [tf.keras.layers.Dense(dense_units, activation='relu') for _ in range(num_dense_layers)]
        
        self.dropout = tf.keras.layers.Dropout(dropout_rate)
        self.out = tf.keras.layers.Dense(3)  # 2 actions + amount

    def call(self, state, training=False):
        x = tf.convert_to_tensor(state, dtype=tf.float32)
        
        # Apply LSTM layers
        for lstm_layer in self.lstm_layers:
            x = lstm_layer(x)
            x = self.dropout(x, training=training)
        
        # Apply dense layers
        for dense_layer in self.dense_layers:
            x = dense_layer(x)
            x = self.dropout(x, training=training)

        output = self.out(x)

        # Split into action probabilities and amount
        action_probs, amount = tf.split(output, [2, 1], axis=-1)
        action_probs = tf.nn.softmax(action_probs)  # Normalize action probabilities
        amount = tf.nn.sigmoid(amount)  # Amount in range 0-1

        return tf.concat([action_probs, amount], axis=-1)
    
    def save(self, path):
        self.save_weights(path)


def train(env, policy_network, optimizer, episodes=1000, name: str = "model"):
    best_portfolio_value = 0

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        portfolio_values = []
        losses = []

        while not env.done:
            env.render()
            state_input = tf.convert_to_tensor(state, dtype=tf.float32)
            state_input = tf.expand_dims(state_input, axis=-1)  # Expand the shape to 30 * 96 * 1
            
            with tf.GradientTape() as tape:
                # Get action probabilities and amount from the policy network
                action_and_amount = policy_network(state_input, training=True)  # Ensure the model is in training mode
                action_probs = action_and_amount[:, :2]
                amount = action_and_amount[:, 2]

                # Sample an action based on the probabilities
                action_probs_np = action_probs.numpy().squeeze()
                action_probs_np = np.mean(action_probs_np, axis=0)  # Average the probabilities across the window
                action_type = np.random.choice(2, p=action_probs_np)
                chosen_action = np.zeros(3)
                chosen_action[action_type] =  1
                amount = amount.numpy().squeeze()
                chosen_action[2] =  np.mean(amount, axis=0)  # Average the amount across the window

                # Evaluate portfolio change for buy and sell
                _, buy_advantage, _ = env.__step__([1, 0, 1], nextStep=False)  # 100% buy
                _, sell_advantage, _ = env.__step__([0, 1, 1], nextStep=False)  # 100% sell
                
                # Calculate the chosen action's advantage
                _, chosen_advantage, _ = env.__step__(chosen_action)

                # Calculate the optimal advantage
                optimal_advantage = max(buy_advantage, sell_advantage)

                # Define the optimal response based on the maximum advantage
                optimal_response = [1, 0, 1] if buy_advantage > sell_advantage else [0, 1, 1]

                # Convert optimal response to a tensor for loss calculation
                optimal_response_tensor = tf.convert_to_tensor(optimal_response, dtype=tf.float32)
                chosen_action_tensor = tf.convert_to_tensor(chosen_action, dtype=tf.float32)

                # Compute the loss based on the difference between chosen and optimal response
                loss = tf.reduce_mean(tf.square(chosen_action_tensor - optimal_response_tensor))
                # print(f"Chosen: {action_and_amount}, Optimal: {optimal_response}, Loss: {loss.numpy()}")

                # Log metrics
                wandb.log({
                    "Loss": loss.numpy(),
                    "Advantage Error": chosen_advantage - optimal_advantage,
                    "Profit": env.calculate_portfolio_value() - env.starting_overall_balance,
                })

            # Compute gradients and update the network
            grads = tape.gradient(loss, policy_network.trainable_variables)
            if any(g is not None for g in grads):
                optimizer.apply_gradients(zip(grads, policy_network.trainable_variables))
            else:
                print("No gradients provided")

            total_reward += chosen_advantage
            state = env.current_observation
            portfolio_values.append(env.calculate_portfolio_value())
            losses.append(loss.numpy())

            if env.done:
                break

        # Log metrics to W&B
        wandb.log({
            "Episode": episode + 1,
            "Total Reward": total_reward,
            "Portfolio Values": portfolio_values,
            "Average Loss": np.mean(losses),
            "Final Portfolio Value": env.calculate_portfolio_value()
        })

        if env.calculate_portfolio_value() > best_portfolio_value:
            
            print(f"New best portfolio value: {env.calculate_portfolio_value():.2f} is better than {best_portfolio_value / env.calculate_portfolio_value() * 100:.2f}")
            best_portfolio_value = env.calculate_portfolio_value()
            policy_network.save(name)

        print(f"Episode {episode + 1}: Total Reward: {total_reward:.2f}, Final Portfolio Value: {env.calculate_portfolio_value():.2f}")


if __name__ == "__main__":
    # Example data: load real data from file
    data = load_data("bybit.csv")
    # Optionally scale data
    # data = scale_data(data)

    env = MyEnv(data)

    import random
     
    network_layers = {
        "lstm_units":512,
        "dense_units":256,
        "num_lstm_layers":3,
        "num_dense_layers":3,
        "dropout_rate":0.2
    }

    modelName = f"model-{random.randint(0, 10000)} + {network_layers['lstm_units']} + {network_layers['dense_units']} + {network_layers['num_lstm_layers']} + {network_layers['num_dense_layers']} + {network_layers['dropout_rate']}"

    policy_network = PolicyNetwork(**network_layers)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

    example_state = env.reset()
    example_state_input = tf.convert_to_tensor(example_state, dtype=tf.float32)

    print(f"Example state shape: {example_state_input.shape}")
    # Expand the shape to 30 * 96 * 1 
    example_state_input = tf.expand_dims(example_state_input, axis=-1)
    print(f"Expanded state shape: {example_state_input.shape}")

    policy_network.build(input_shape=example_state_input.shape)
    policy_network.summary()

    wandb.log({"model_summary": network_layers})

    train(env, policy_network, optimizer, name=modelName)