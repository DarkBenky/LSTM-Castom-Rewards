import tensorflow as tf
from tf_agents.environments import py_environment
from tf_agents.specs import array_spec
from tf_agents.trajectories import time_step as ts
from tf_agents.networks import actor_distribution_network, value_network
from tf_agents.agents.ppo import ppo_agent
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.drivers import dynamic_step_driver
from tf_agents.metrics import tf_metrics
from tf_agents.policies import random_tf_policy
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
import numpy as np
import matplotlib.pyplot as plt
import wandb
from main import load_data

# Initialize WandB
wandb.init(project="PPO_Trading_Model", config={
    "num_iterations": 1000,
    "collect_steps_per_iteration": 100,
    "replay_buffer_capacity": 10000,
    "num_eval_episodes": 10,
    "num_parallel_environments": 1
})
config = wandb.config

# Trading Environment
class TradingEnvironment(py_environment.PyEnvironment):
    def __init__(self, data, window_size=64):
        self._data = data
        self._window_size = window_size
        self._episode_ended = False
        self._current_position = 0
        self._current_step = window_size
        self._initial_balance = 10000
        self._balance = self._initial_balance

        self._action_spec = array_spec.BoundedArraySpec(
            shape=(), dtype=np.int32, minimum=0, maximum=2, name='action')
        self._observation_spec = array_spec.BoundedArraySpec(
            shape=(window_size, data.shape[1]), dtype=np.float32, name='observation')

    def action_spec(self):
        return self._action_spec

    def observation_spec(self):
        return self._observation_spec

    def _reset(self):
        self._current_step = self._window_size
        self._balance = self._initial_balance
        self._current_position = 0
        self._episode_ended = False
        return ts.restart(self._get_observation())

    def _step(self, action):
        if self._episode_ended:
            return self.reset()

        reward = 0
        current_price = self._data[self._current_step][1]

        if action == 1:  # Buy
            if self._current_position == 0:
                self._current_position = 1
                reward -= current_price * 0.001  # Transaction cost
        elif action == 2:  # Sell
            if self._current_position == 1:
                self._current_position = 0
                reward += current_price * 0.999  # Transaction cost

        self._current_step += 1
        new_price = self._data[self._current_step][1]
        
        if self._current_position == 1:
            reward += new_price - current_price

        self._balance += reward

        if self._current_step >= len(self._data) - 1:
            self._episode_ended = True
            return ts.termination(self._get_observation(), reward)
        else:
            return ts.transition(self._get_observation(), reward, discount=0.99)

    def _get_observation(self):
        return self._data[self._current_step - self._window_size:self._current_step]

# Create the environment
data = load_data('data.csv')
train_env = TradingEnvironment(data)
eval_env = TradingEnvironment(data)

# Define the agent
fc_layer_params = (100, 50)
actor_net = actor_distribution_network.ActorDistributionNetwork(
    train_env.observation_spec(),
    train_env.action_spec(),
    fc_layer_params=fc_layer_params)
value_net = value_network.ValueNetwork(
    train_env.observation_spec(),
    fc_layer_params=fc_layer_params)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

train_step_counter = tf.Variable(0)

tf_agent = ppo_agent.PPOAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    optimizer=optimizer,
    actor_net=actor_net,
    value_net=value_net,
    num_epochs=10,
    train_step_counter=train_step_counter)

tf_agent.initialize()

# Replay buffer and dataset
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=tf_agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=config.replay_buffer_capacity)

replay_observer = [replay_buffer.add_batch]

train_metrics = [
    tf_metrics.NumberOfEpisodes(),
    tf_metrics.EnvironmentSteps(),
    tf_metrics.AverageReturnMetric(),
    tf_metrics.AverageEpisodeLengthMetric(),
]

eval_policy = tf_agent.policy
collect_policy = tf_agent.collect_policy

collect_driver = dynamic_step_driver.DynamicStepDriver(
    train_env,
    collect_policy,
    observers=replay_observer + train_metrics,
    num_steps=config.collect_steps_per_iteration)

# Training loop
def train_agent(n_iterations):
    time_step = train_env.reset()
    collect_driver.run(time_step)

    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3, sample_batch_size=64, num_steps=2).prefetch(3)
    iterator = iter(dataset)

    returns = []
    for iteration in range(n_iterations):
        time_step, _ = collect_driver.run(time_step)
        experience, _ = next(iterator)
        train_loss = tf_agent.train(experience=experience)
        
        step = tf_agent.train_step_counter.numpy()

        if iteration % 10 == 0:
            print(f'Iteration: {iteration}, Loss: {train_loss.loss.numpy()}')
            avg_return = compute_avg_return(eval_env, tf_agent.policy, config.num_eval_episodes)
            returns.append(avg_return)
            print(f'Average Return: {avg_return}')
            wandb.log({
                'Iteration': iteration,
                'Loss': train_loss.loss.numpy(),
                'Average Return': avg_return
            })

            plot_returns(returns, iteration)

    return returns

def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for _ in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return
    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]

def plot_returns(returns, iteration):
    plt.figure(figsize=(10, 6))
    plt.plot(range(0, (iteration+1)*10, 10), returns)
    plt.xlabel('Iterations')
    plt.ylabel('Average Return')
    plt.title('Average Return over Iterations')
    plt.savefig(f'returns_plot_{iteration}.png')
    wandb.log({"returns_plot": wandb.Image(f'returns_plot_{iteration}.png')})
    plt.close()

# Run the training
returns = train_agent(config.num_iterations)

# Final evaluation
final_avg_return = compute_avg_return(eval_env, tf_agent.policy, config.num_eval_episodes)
print(f'Final Average Return: {final_avg_return}')
wandb.log({'Final Average Return': final_avg_return})

# Save the trained model
tf_agent.policy.save('ppo_trading_model')

wandb.finish()