import tensorflow as tf
import numpy as np

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
        current_value = self.calculate_portfolio_value()

        # Extract action and amount
        action_type = np.argmax(action[:2])  # buy, sell (hold removed)
        amount = action[2]  # amount in range 0-1

        if action_type == 0:  # Buy
            usd_to_spend = self.balance_usd * amount
            if usd_to_spend > self.balance_usd:
                usd_to_spend = self.balance_usd
                penalty = 100
            btc_to_buy = usd_to_spend / current_price
            self.balance_btc += btc_to_buy
            self.balance_usd -= usd_to_spend
        elif action_type == 1:  # Sell
            btc_to_sell = self.balance_btc * amount
            if btc_to_sell > self.balance_btc:
                btc_to_sell = self.balance_btc
                penalty = 100

            self.balance_usd += btc_to_sell * current_price
            self.balance_btc -= btc_to_sell

        self.step += 1

        if self.step + self.window_size + self.future_window >= len(self.data):
            self.done = True
        
        # calculate reward as the change in portfolio value
        new_value = self.calculate_portfolio_value()
        reward = new_value - current_value

        # Penalize actions that exceed the balance
        reward -= penalty

        if new_value <= 0:
            self.done = True

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
        while True:
            env.render()
            state_input = np.expand_dims(state, axis=0)  # Batch dimension
            action_and_amount = policy_network(state_input).numpy().squeeze()

            next_state, reward, done = env.__step__(action_and_amount)
            total_reward += reward

            with tf.GradientTape() as tape:
                predicted_action_and_amount = policy_network(np.expand_dims(state, axis=0))
                action_prob = predicted_action_and_amount[:, :2]  # Only the action probabilities

                # Select the action based on the policy network's output
                action_type = np.argmax(action_and_amount[:2])

                # Get the probability of the action that was taken
                action_prob_taken = action_prob[0, action_type]

                # Compute the loss as the negative log-probability scaled by the reward
                loss = -safe_log(action_prob_taken) * reward
                print(f"Loss: {loss.numpy()} , Reward: {reward}, action_prob_taken: {action_prob_taken.numpy()}, amount: {action_and_amount[2]}")

            grads = tape.gradient(loss, policy_network.trainable_variables)
            optimizer.apply_gradients(zip(grads, policy_network.trainable_variables))

            if done:
                break
            state = next_state

        print(f"Episode {episode + 1}: Total Reward: {total_reward:.2f}")

if __name__ == "__main__":
    # Example data: random price data for demonstration
    np.random.seed(72)
    data = np.cumsum(np.random.randn(1000, 2), axis=0)

    print(data)

    env = MyEnv(data)
    policy_network = PolicyNetwork()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

    train(env, policy_network, optimizer)

# TODO: Plot training & validation loss
# TODO: Plot portfolio value over time