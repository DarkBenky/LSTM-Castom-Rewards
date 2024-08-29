import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import gym
from sklearn.preprocessing import MinMaxScaler
import csv
import pandas as pd

# ... (keep the load_data and scale_data functions as they are)
def load_data(filename):
    data = []
    with open(filename, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            float_row = [np.float32(value.strip()) for value in row]
            data.append(float_row[1:])
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


class TradingEnv(gym.Env):
    def __init__(self, data):
        super(TradingEnv, self).__init__()
        self.data = data.values  # Convert DataFrame to numpy array
        self.current_step = 0
        self.current_position = 0  # 1: long position, -1: short position

        # Define action and observation space
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,))  # Continuous action space
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(data.shape[1],))

    def reset(self):
        self.current_step = 0
        self.current_position = 0
        return self.data[self.current_step]

    def step(self, action):
        # Convert continuous action to discrete
        self.current_position = 1 if action[0] > 0 else -1

        # Move to the next time step
        self.current_step += 1
        done = self.current_step >= len(self.data) - 1
        next_state = self.data[self.current_step] if not done else self.data[-1]

        # Calculate reward
        reward = self._calculate_reward()

        return next_state, reward, done, {}

    def _calculate_reward(self):
        if self.current_step > 0:
            price_diff = self.data[self.current_step][1] - self.data[self.current_step - 1][1]
            return self.current_position * price_diff
        return 0

# Actor network
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))

# Critic network (unchanged)
class Critic(nn.Module):
    def __init__(self, state_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# PPO agent
class PPO:
    def __init__(self, state_dim, action_dim, lr, gamma, clip_epsilon, epochs):
        self.actor = Actor(state_dim, action_dim)
        self.critic = Critic(state_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.epochs = epochs

    def get_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)  # Add batch dimension
        action_mean = self.actor(state)
        cov_mat = torch.diag(torch.full(action_mean.shape, 0.1))
        dist = Normal(action_mean, cov_mat)
        action = dist.sample()
        return action.detach().numpy().flatten()  # Remove batch dimension

    def update(self, states, actions, rewards, next_states, dones):
        # Convert to tensor
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(np.array(dones)).unsqueeze(1)

        # Compute advantage
        with torch.no_grad():
            values = self.critic(states)
            next_values = self.critic(next_states)

        advantages = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO update
        for _ in range(self.epochs):
            action_means = self.actor(states)
            dist = Normal(action_means, torch.full_like(action_means, 0.1))
            new_log_probs = dist.log_prob(actions).sum(dim=1, keepdim=True)
            old_log_probs = new_log_probs.detach()

            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = nn.MSELoss()(self.critic(states), rewards + self.gamma * next_values * (1 - dones))

            # Update actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update critic
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

# Training loop
def train(env, agent, num_episodes):
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        total_reward = 0
        states, actions, rewards, next_states, dones = [], [], [], [], []

        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            state = next_state
            total_reward += reward

        agent.update(states, actions, rewards, next_states, dones)
        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Main execution
if __name__ == "__main__":
    # Load and preprocess your data
    data = load_data("data.csv")
    # scaled_data = scale_data(data)


    # Create the environment
    env = TradingEnv(pd.DataFrame(data))

    # del scale_data

    # Initialize the PPO agent
    state_dim = env.observation_space.shape[0]
    action_dim = 1  # Single continuous action
    agent = PPO(state_dim, action_dim, lr=0.001, gamma=0.99, clip_epsilon=0.2, epochs=5)

    # Train the agent
    train(env, agent, num_episodes=1000)