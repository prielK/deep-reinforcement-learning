import gymnasium as gym
import numpy as np
import tensorflow.compat.v1 as tf
import collections
from datetime import datetime

# optimized for Tf2
tf.disable_v2_behavior()
print("tf_ver:{}".format(tf.__version__))

env = gym.make("CartPole-v1")
np.random.seed(1)


class ValueNetwork:
    """Value Network class for reinforcement learning.
    Attributes:
        state: Input state.
        R_t: Total rewards.
        loss: Loss function of the network.
        optimizer: Optimizer used for training.
    """

    def __init__(self, state_size, learning_rate, name="value_network"):
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, state_size], name="state")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            w1 = tf.get_variable("W1", [state_size, 12], initializer=tf2_initializer)
            b1 = tf.get_variable("b1", [12], initializer=tf2_initializer)
            w2 = tf.get_variable("W2", [12, 1], initializer=tf2_initializer)
            b2 = tf.get_variable("b2", [1], initializer=tf2_initializer)

            self.Z1 = tf.add(tf.matmul(self.state, w1), b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, w2), b2)

            self.loss = tf.reduce_mean(tf.square(self.output - self.R_t))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)


class PolicyNetwork:
    """Policy Network class for reinforcement learning.
    Attributes:
        state: Input state.
        action: Action taken.
        R_t: Total rewards.
        loss: Loss function of the network.
        optimizer: Optimizer used for training.
    """

    def __init__(self, state_size, action_size, learning_rate, name="policy_network"):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf2_initializer)
            self.b1 = tf.get_variable("b1", [12], initializer=tf2_initializer)
            self.W2 = tf.get_variable("W2", [12, self.action_size], initializer=tf2_initializer)
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf2_initializer)

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.output, labels=self.action
            )
            self.advantage = tf.placeholder(tf.float32, name="advantage")
            self.loss = tf.reduce_mean(self.neg_log_prob * self.advantage)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(
                self.loss
            )


def run():
    # Define hyperparameters
    state_size = 4
    action_size = env.action_space.n
    max_episodes = 5000
    max_steps = 501
    discount_factor = 0.99
    learning_rate = 0.0004
    render = False

    # Initialize the networks
    tf.reset_default_graph()
    policy = PolicyNetwork(state_size, action_size, learning_rate)
    value_network = ValueNetwork(state_size, learning_rate)

    # Start training the agent with REINFORCE algorithm
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        logdir = f"tensorboard_logs/REINFORCE_{datetime.now().strftime('%d%m%Y%H%M%S')}"
        writer = tf.summary.FileWriter(logdir, sess.graph)

        solved = False
        Transition = collections.namedtuple(
            "Transition", ["state", "action", "reward", "next_state", "done"]
        )
        episode_rewards = np.zeros(max_episodes)
        average_rewards = 0.0

        for episode in range(max_episodes):
            state = env.reset()[0]
            state = state.reshape([1, state_size])
            episode_transitions = []
            policy_loss_accumulated = 0
            value_loss_accumulated = 0

            for step in range(max_steps):
                actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
                action = np.random.choice(
                    np.arange(len(actions_distribution)), p=actions_distribution
                )
                next_state, reward, done, _, _ = env.step(action)
                next_state = next_state.reshape([1, state_size])

                if render:
                    env.render()

                action_one_hot = np.zeros(action_size)
                action_one_hot[action] = 1
                episode_transitions.append(
                    Transition(
                        state=state,
                        action=action_one_hot,
                        reward=reward,
                        next_state=next_state,
                        done=done,
                    )
                )
                episode_rewards[episode] += reward

                if done | (step == max_steps - 1):
                    if episode > 98:
                        # Check if solved
                        average_rewards = np.mean(episode_rewards[(episode - 99) : episode + 1])
                    print(
                        "Episode {} Reward: {} Average over 100 episodes: {}".format(
                            episode, episode_rewards[episode], round(average_rewards, 2)
                        )
                    )
                    if average_rewards > 475:
                        print(" Solved at episode: " + str(episode))
                        solved = True
                    break
                state = next_state

            if solved:
                break

            # Update value function and policy network
            for t, transition in enumerate(episode_transitions):
                total_discounted_return = sum(
                    discount_factor**i * t.reward for i, t in enumerate(episode_transitions[t:])
                )
                value = sess.run(value_network.output, {value_network.state: transition.state})
                advantage = total_discounted_return - value

                # Update the policy network using advantage
                policy_loss, _ = sess.run(
                    [policy.loss, policy.optimizer],
                    {
                        policy.state: transition.state,
                        policy.advantage: advantage,
                        policy.action: transition.action,
                    },
                )
                policy_loss_accumulated += policy_loss

                # Update the value network
                value_loss, _ = sess.run(
                    [value_network.loss, value_network.optimizer],
                    {
                        value_network.state: transition.state,
                        value_network.R_t: total_discounted_return,
                    },
                )
                value_loss_accumulated += value_loss

                # Compute the average loss for the episode
                policy_loss_avg = policy_loss_accumulated / len(episode_transitions)
                value_loss_avg = value_loss_accumulated / len(episode_transitions)

                # Log the average policy loss, value loss, and episode reward
                summary = tf.Summary()
                summary.value.add(tag="Policy Loss", simple_value=policy_loss_avg)
                summary.value.add(tag="Value Loss", simple_value=value_loss_avg)
                summary.value.add(tag="Episode Reward", simple_value=episode_rewards[episode])
                summary.value.add(tag="Moving Average Reward", simple_value=average_rewards)
                writer.add_summary(summary, episode)

        writer.close()


if __name__ == "__main__":
    run()
