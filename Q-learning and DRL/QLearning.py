import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

# Initialize the FrozenLake environment
env = gym.make("FrozenLake-v1", is_slippery=True)

# Initialize Q-table with zeros
q_table = np.zeros([env.observation_space.n, env.action_space.n])

NUM_EPISODES = 5000
MAX_STEPS = 100  # Maximum steps per episode
# Hyperparameters
alpha = 0.1  # Learning rate
gamma = 0.99  # Discount factor
epsilon = 1.0  # Epsilon-greedy probability
epsilon_end = 0.01  # Minimum epsilon
epsilon_decay = 0.999  # Decay rate as a function of number of episodes
# For stats
rewards_per_episode = []
steps_per_episode = []
q_tables_at_steps = {}  # To store Q-tables at specific steps
steps_per_episode_success = []

# Q-learning algorithm
for episode in range(NUM_EPISODES + 1):
    state = env.reset()[0]
    done = False
    total_reward = 0
    steps = 0

    for _ in range(MAX_STEPS):
        # Epsilon-greedy action selection
        if np.random.uniform(0, 1) < epsilon:
            action = env.action_space.sample()  # Explore action space
        else:
            action = np.argmax(q_table[state])  # Exploit learned values

        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        steps += 1

        # Q-table update
        old_value = q_table[state, action]
        next_max = np.max(q_table[next_state])
        new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
        q_table[state, action] = new_value

        state = next_state

        if done:
            break

    # Decaying epsilon
    epsilon = max(epsilon_end, epsilon_decay * epsilon)

    # Record rewards and steps
    rewards_per_episode.append(total_reward)
    steps_per_episode.append(steps if done else MAX_STEPS)
    if total_reward > 0:
        steps_per_episode_success.append(steps if done else MAX_STEPS)
    # Store Q-tables at specific episodes
    if episode in [500, 2000, NUM_EPISODES]:
        q_tables_at_steps[episode] = q_table.copy()

# Plot of the reward per episode
fig, ax1 = plt.subplots()

color = "tab:blue"
ax1.set_xlabel("Episode")
ax1.set_ylabel("Reward", color=color)
ax1.plot(rewards_per_episode, color=color)
ax1.tick_params(axis="y", labelcolor=color)

# Instantiate a second axes that shares the same x-axis
ax2 = ax1.twinx()

# Plot cumulative reward on the second axis
color = "tab:red"
ax2.set_ylabel("Cumulative Reward", color=color)
cumulative_reward = np.cumsum(rewards_per_episode)
ax2.plot(cumulative_reward, color=color)
ax2.tick_params(axis="y", labelcolor=color)

# Add title
plt.title("Reward per Episode and Cumulative Reward")
plt.show()

# Plot of the average number of steps to the goal over every 100 episodes
average_steps = [
    np.mean(steps_per_episode[i : i + 100]) for i in range(0, len(steps_per_episode), 100)
]
plt.plot(average_steps)
plt.title("Average steps to Goal on all runs (per 100 episodes)")
plt.xlabel("Episode (in hundreds)")
plt.ylabel("Average Steps")
plt.show()

# Plot of the average number of steps to the goal on successful runs over every 100 episodes
average_steps_success = [
    np.mean(steps_per_episode_success[i : i + 100])
    for i in range(0, len(steps_per_episode_success), 100)
]
plt.plot(average_steps_success)
plt.title("Average steps to goal on successful runs(per 100 episodes)")
plt.xlabel("Episode (in hundreds)")
plt.ylabel("Average Steps")
plt.show()

action_labels = ["Left", "Down", "Right", "Up"]  # Action labels for FrozenLake

# Plotting Q-tables at specified episodes
for episode, q_table in q_tables_at_steps.items():
    plt.figure(figsize=(5, 5))
    heatmap = plt.imshow(q_table, cmap="viridis")
    plt.colorbar(heatmap)
    plt.title(f"Q-table after {episode} episodes")
    plt.xlabel("Actions")
    plt.ylabel("States")

    # Set action labels as x-ticks
    plt.xticks(range(len(action_labels)), action_labels, rotation=90)
    plt.yticks(range(len(q_table)), range(len(q_table)))
    plt.show()
