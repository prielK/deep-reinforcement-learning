import numpy as np
import random
from collections import deque
import gymnasium as gym
from datetime import datetime
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.summary import create_file_writer

# Hyperparameters and flags for the DQN model
RENDER = False
TEST = False
MAX_STEPS = 10000
EPISODES_FOR_MA = 100
hidden_layers_3 = [32, 32, 32]
hidden_layers_5 = [32, 32, 32, 32, 32]
n_episodes = 1000
batch_size = 256
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 0.99
learning_rate = 0.001
update_freq = 1000
log_dir = f"logs/DQN_3Layers99dec97gam_{datetime.now().strftime('%d%m%Y%H%M%S')}"


class DQNAgent:
    """A Deep Q-Network agent with policy and target networks."""

    def __init__(self, input_size, output_size, hidden_layers):
        """Initializes the DQN agent with policy and target networks."""
        self.policy_net = self.create_model(input_size, output_size, hidden_layers)
        self.target_net = self.create_model(input_size, output_size, hidden_layers)
        self.update_target_net()

    def create_model(self, input_size, output_size, hidden_layers):
        """Creates a Sequential neural network model."""
        model = Sequential()
        model.add(Dense(units=hidden_layers[0], activation="relu", input_shape=(input_size,)))
        for units in hidden_layers[1:]:
            model.add(Dense(units=units, activation="relu"))
        model.add(Dense(units=output_size, activation=None))
        return model

    def update_target_net(self):
        """Updates the target network with weights from the policy network."""
        print("UPDATING TARGET NET")
        self.target_net.set_weights(self.policy_net.get_weights())

    def train(self, states, actions, updated_q_values, n_actions):
        """Trains the policy network using given batch of experiences."""
        masks = tf.one_hot(actions, n_actions)

        with tf.GradientTape() as tape:
            q_values = self.policy_net(states)
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            loss = tf.keras.losses.MSE(updated_q_values, q_action)
        grads = tape.gradient(loss, self.policy_net.trainable_variables)
        return grads, loss


class ReplayBuffer:
    """Buffer to store and sample experiences for the DQN agent."""

    def __init__(self, capacity):
        """Initializes the replay buffer with a specified capacity."""
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Adds an experience to the buffer."""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """Samples a batch of experiences from the buffer."""
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """Returns the current size of the buffer."""
        return len(self.buffer)


def sample_action(state, policy_net, epsilon, n_actions):
    """Samples an action using epsilon-greedy policy."""
    if random.random() > epsilon:
        return np.argmax(policy_net.predict(state, verbose=0))
    else:
        return random.randint(0, n_actions - 1)


def train_agent(
    env,
    agent,
    optimizer,
    experience_replay,
    n_episodes,
    batch_size,
    gamma,
    epsilon_start,
    epsilon_end,
    epsilon_decay,
    update_freq,
    writer,
):
    """Trains the DQN agent in the given environment."""
    global_step = 0
    most_recent_rewards = deque(maxlen=EPISODES_FOR_MA)
    epsilon = epsilon_start

    for episode in range(n_episodes):
        # Initialize variables for each episode
        total_reward = 0
        step = 0
        state = env.reset()[0]
        state = np.expand_dims(state, axis=0)

        while step < MAX_STEPS:
            action = sample_action(state, agent.policy_net, epsilon, env.action_space.n)
            next_state, reward, done, _, _ = env.step(action)
            next_state = np.expand_dims(next_state, axis=0)
            experience_replay.push(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            step += 1
            global_step += 1

            if done:
                break

            if len(experience_replay) > batch_size:
                # Sample a batch and train the agent
                sample_batch = experience_replay.sample(batch_size)
                states, actions, rewards, next_states, dones = zip(*sample_batch)

                states = np.vstack(states)
                actions = np.array(actions)
                rewards = np.array(rewards, dtype=np.float32)
                next_states = np.vstack(next_states)
                dones = np.array(dones, dtype=np.float32)

                future_q_values = agent.target_net.predict(next_states, verbose=0)
                updated_q_values = rewards + gamma * np.max(future_q_values, axis=1) * (1 - dones)

                grads, loss = agent.train(states, actions, updated_q_values, env.action_space.n)
                optimizer.apply_gradients(zip(grads, agent.policy_net.trainable_variables))

                # Log training loss
                with writer.as_default():
                    tf.summary.scalar("Loss", loss, step=global_step)

            # Update the target network
            if global_step % update_freq == 0:
                agent.update_target_net()

        # Decay epsilon
        epsilon = max(epsilon_end, epsilon_decay * epsilon)

        # Logging and print statements for monitoring
        most_recent_rewards.append(total_reward)
        moving_avg_steps = np.mean(most_recent_rewards)
        print(
            f"Episode {episode} - {step} steps - Epsilon {epsilon} - MA of {EPISODES_FOR_MA} episodes: {moving_avg_steps:.2f}"
        )

        # Log total reward and MA after each episode
        with writer.as_default():
            tf.summary.scalar("Total Reward", total_reward, step=episode)
            tf.summary.scalar("MA", moving_avg_steps, step=episode)


def test_agent(env, agent, n_episodes):
    """Tests the trained DQN agent in the given environment."""
    for episode in range(n_episodes):
        print(f"Testing episode {episode}")
        state = env.reset()[0]
        state = np.expand_dims(state, axis=0)
        total_reward = 0

        while True:
            action = sample_action(state, agent.policy_net, 0, env.action_space.n)
            next_state, reward, done, _, _ = env.step(action)
            state = np.expand_dims(next_state, axis=0)
            total_reward += reward

            if done:
                print(f"Test Episode {episode} Total Reward: {total_reward}")
                break


# Environment Setup
if RENDER == True:
    env = gym.make("CartPole-v1", render_mode="human")
else:
    env = gym.make("CartPole-v1")

# Initialize agent, optimizer, replay buffer and tensorboard writer
n_actions = env.action_space.n
n_states = env.observation_space.shape[0]
agent = DQNAgent(n_states, n_actions, hidden_layers_3)
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
experience_replay = ReplayBuffer(100000)
writer = create_file_writer(log_dir)

# Train the agent
train_agent(
    env,
    agent,
    optimizer,
    experience_replay,
    n_episodes,
    batch_size,
    gamma,
    epsilon_start,
    epsilon_end,
    epsilon_decay,
    update_freq,
    writer,
)

# Test the agent if the TEST flag is set
if TEST:
    test_agent(env, agent, 10)

# Close the environment after training/testing
env.close()
