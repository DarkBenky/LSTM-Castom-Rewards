import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import csv

import wandb

wandb.init(project="btc-trading")

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
    print(pd.DataFrame(scaled_data).describe())

    return np.array(scaled_data)

class MyEnv:
    def __init__(self, data, balance_usd=1000, balance_btc=0):
        self.data = data
        self.balance_usd = balance_usd
        self.balance_btc = balance_btc
        self.done = False
        self.window_size = 30
        self.future_window = 1
        self.reset()

    def next_price(self):
        return self.data[self.step + self.window_size + self.future_window][1]
    
    def next_prices_window(self, window_size):
        return self.data[self.step + self.window_size:self.step + self.window_size + window_size][1]
    
    def calculate_portfolio_value(self):
        return self.balance_usd + self.balance_btc * self.data[self.step + self.window_size][1]

    def reset(self):
        self.step = 0
        self.done = False
        self.balance_usd = 1000
        self.balance_btc = 0.1
        self.current_observation = self.data[self.step:self.step+self.window_size]
        self.current_overall_balance = self.calculate_portfolio_value()

        # Pad the new row with zeros to match the observation dimensions (2 -> 3)
        portfolio_info = [self.calculate_portfolio_value(), self.balance_btc, self.balance_usd]
        padded_info = np.zeros((self.window_size, len(portfolio_info)))
        padded_info[-1] = portfolio_info  # Add the portfolio info as the last row
        self.current_observation = np.hstack((self.current_observation, padded_info))

        return self.current_observation

    def __step__(self, action):
        if self.done:
            raise ValueError("Environment is done, reset it before calling step.")
        
        penalty = 0

        current_price = self.data[self.step + self.window_size][1]
        # current_value = self.calculate_portfolio_value()

        # Extract action and amount
        action_type = np.argmax(action[:2])  # buy, sell (hold removed)
        amount = action[2]  # amount in range 0-1


        # TODO: fix the amount calculation
        if action_type == 0:  # Buy
            usd_to_spend = self.balance_usd * amount
            if usd_to_spend > self.balance_usd:
                usd_to_spend = self.balance_usd
                penalty = 10
            btc_to_buy = usd_to_spend / current_price
            self.balance_btc += btc_to_buy
            self.balance_usd -= usd_to_spend
        elif action_type == 1:  # Sell
            btc_to_sell = self.balance_btc * amount
            if btc_to_sell > self.balance_btc:
                btc_to_sell = self.balance_btc
                penalty = 10

            self.balance_usd += btc_to_sell * current_price
            self.balance_btc -= btc_to_sell

        self.step += 1

        if self.step + self.window_size + self.future_window >= len(self.data):
            self.done = True
        
        # calculate reward as the change in portfolio value
        # new_value = self.calculate_portfolio_value()

        next_price = self.next_price()
        change_1 = (next_price - current_price) / current_price
        change_10 = (np.mean(self.next_prices_window(10)) - current_price) / current_price
        change_30 = (np.mean(self.next_prices_window(30)) - current_price) / current_price

        reward = 0.5 * change_1 + 0.3 * change_10 + 0.2 * change_30

        if action_type == 1:  # Sell
            price_reward = -price_reward

        # Penalize actions that exceed the balance
        reward -= penalty

        if self.calculate_portfolio_value() < 0:
            self.done = True
            reward -= 1000

        self.current_observation = self.data[self.step:self.step+self.window_size]
        
        # Pad the new row with zeros to match the observation dimensions (2 -> 3)
        portfolio_info = [self.calculate_portfolio_value(), self.balance_btc, self.balance_usd]
        padded_info = np.zeros((self.window_size, len(portfolio_info)))
        padded_info[-1] = portfolio_info  # Add the portfolio info as the last row
        self.current_observation = np.hstack((self.current_observation, padded_info))

        return self.current_observation, reward, self.done

    def render(self):
        print(f"Step: {self.step}, USD: {self.balance_usd:.2f}, BTC: {self.balance_btc:.4f}, Portfolio Value: {self.calculate_portfolio_value():.2f}")

# Define a simple policy network
class PolicyNetwork(tf.keras.Model):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        self.lstm = tf.keras.layers.LSTM(128)
        self.dense1 = tf.keras.layers.Dense(64, activation='relu')
        self.dense2 = tf.keras.layers.Dense(64, activation='relu')
        self.out = tf.keras.layers.Dense(3)  # 2 actions + amount

    def call(self, state):
        x = tf.convert_to_tensor(state, dtype=tf.float32)
        x = self.lstm(x)
        x = self.dense1(x)
        x = self.dense2(x)
        output = self.out(x)

        # Split into action probabilities and amount
        action_probs, amount = tf.split(output, [2, 1], axis=-1)
        action_probs = tf.nn.softmax(action_probs)  # Normalize action probabilities
        amount = tf.nn.sigmoid(amount)  # Amount in range 0-1

        return tf.concat([action_probs, amount], axis=-1)

def safe_log(x, eps=1e-10):
    return tf.math.log(tf.maximum(x, eps))

def train(env, policy_network, optimizer, episodes=1000):
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        portfolio_values = []
        losses = []

        while True:
            env.render()
            state_input = np.expand_dims(state, axis=0)
            action_and_amount = policy_network(state_input).numpy().squeeze()

            next_state, reward, done = env.__step__(action_and_amount)
            total_reward += reward
            portfolio_values.append(env.calculate_portfolio_value())

            with tf.GradientTape() as tape:
                predicted_action_and_amount = policy_network(np.expand_dims(state, axis=0))
                action_prob = predicted_action_and_amount[:, :2]

                action_type = np.argmax(action_and_amount[:2])
                action_prob_taken = action_prob[0, action_type]

                loss = -safe_log(action_prob_taken) * reward
                losses.append(loss.numpy())
                print(f"Loss: {loss.numpy()} , Reward: {reward}, action_prob_taken: {action_prob_taken.numpy()}, amount: {action_and_amount[2]}")

            grads = tape.gradient(loss, policy_network.trainable_variables)
            optimizer.apply_gradients(zip(grads, policy_network.trainable_variables))

            if done:
                break
            state = next_state

        # Log metrics to W&B
        wandb.log({
            "Episode": episode + 1,
            "Total Reward": total_reward,
            "Portfolio Value": env.calculate_portfolio_value(),
            "Average Loss": np.mean(losses)
        })

        print(f"Episode {episode + 1}: Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    # Example data: random price data for demonstration
    data = load_data("data.csv")
    data = scale_data(data)

    print(data)

    env = MyEnv(data)
    policy_network = PolicyNetwork()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    train(env, policy_network, optimizer)

# TODO: Plot training & validation loss
# TODO: Plot portfolio value over time