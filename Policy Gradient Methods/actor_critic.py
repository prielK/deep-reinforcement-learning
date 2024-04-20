import gymnasium as gym
import numpy as np
import tensorflow.compat.v1 as tf
from datetime import datetime

# Disable eager execution to use tf1.x functions in tf2.x
tf.disable_v2_behavior()
print("tf_ver:{}".format(tf.__version__))

env = gym.make("CartPole-v1")
np.random.seed(1)


# Choosing the action based on the probability distribution
class ActorNetwork:
    def __init__(self, state_size, action_size, learning_rate, name="actor_network"):
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, state_size], name="state")
            self.action = tf.placeholder(tf.int32, [None, action_size], name="action")
            self.delta = tf.placeholder(tf.float32, name="delta")

            # Actor network architecture
            self.W1 = tf.get_variable(
                "W1", [state_size, 32], initializer=tf.keras.initializers.glorot_normal(seed=0)
            )
            self.b1 = tf.get_variable("b1", [32], initializer=tf.constant_initializer(0.0))
            self.layer1 = tf.nn.relu(tf.matmul(self.state, self.W1) + self.b1)
            self.W2 = tf.get_variable(
                "W2", [32, action_size], initializer=tf.keras.initializers.glorot_normal(seed=0)
            )
            self.b2 = tf.get_variable("b2", [action_size], initializer=tf.constant_initializer(0.0))
            self.output = tf.matmul(self.layer1, self.W2) + self.b2

            # Loss and optimizer
            self.actions_distribution = tf.nn.softmax(self.output)
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(
                logits=self.output, labels=self.action
            )

            self.loss = tf.reduce_mean(self.neg_log_prob * self.delta)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)


# Critic network to evaluate the value function
class CriticNetwork:
    def __init__(self, state_size, learning_rate, name="critic_network"):
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, state_size], name="state")
            self.target_value = tf.placeholder(tf.float32, name="target_value")

            # Critic network architecture
            self.W1 = tf.get_variable(
                "W1", [state_size, 128], initializer=tf.keras.initializers.glorot_normal(seed=0)
            )
            self.b1 = tf.get_variable("b1", [128], initializer=tf.constant_initializer(0.0))
            self.layer1 = tf.nn.relu(tf.matmul(self.state, self.W1) + self.b1)
            # Add a 2nd layer
            self.W2 = tf.get_variable(
                "W2", [128, 128], initializer=tf.keras.initializers.glorot_normal(seed=0)
            )
            self.b2 = tf.get_variable("b2", [128], initializer=tf.constant_initializer(0.0))
            self.layer2 = tf.nn.relu(tf.matmul(self.layer1, self.W2) + self.b2)
            # Final layer
            self.W3 = tf.get_variable(
                "W3", [128, 1], initializer=tf.keras.initializers.glorot_normal(seed=0)
            )
            self.b3 = tf.get_variable("b3", [1], initializer=tf.constant_initializer(0.0))
            self.output = tf.matmul(self.layer2, self.W3) + self.b3

            # Loss and optimizer
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_value, self.output))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)


def run():
    state_size = 4
    action_size = env.action_space.n
    max_episodes = 5000
    max_steps = 501
    discount_factor = 0.99
    learning_rate = 0.0004
    render = False

    # Initialize the networks
    tf.reset_default_graph()
    actor_net = ActorNetwork(state_size, action_size, learning_rate)
    critic_net = CriticNetwork(state_size, learning_rate)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        logdir = f"_logs/actor_critic_{datetime.now().strftime('%d%m%Y%H%M%S')}"  #
        writer = tf.summary.FileWriter(logdir, sess.graph)  #

        solved = False
        episode_rewards = np.zeros(max_episodes)
        average_rewards = 0.0

        for episode in range(max_episodes):
            state = env.reset()[0]
            state = state.reshape([1, state_size])
            actor_loss_accumulated = 0  #
            critic_loss_accumulated = 0  #

            for step in range(max_steps):
                actions_distribution = sess.run(
                    actor_net.actions_distribution, {actor_net.state: state}
                )
                actions_distribution = actions_distribution[0]
                actions_distribution = actions_distribution.reshape([1, action_size])

                action = np.random.choice(
                    np.arange(len(actions_distribution[0])), p=actions_distribution[0]
                )
                next_state, reward, done, _, _ = env.step(action)
                next_state = next_state.reshape([1, state_size])
                if render:
                    env.render()

                action_one_hot = np.zeros((1, action_size))
                action_one_hot[0, action] = 1

                # Compute the value of the current state and the value of the next state from the ValueNetwork
                V_s = sess.run(critic_net.output, {critic_net.state: state})
                V_s_next = sess.run(critic_net.output, {critic_net.state: next_state})
                V_s_next = 0 if done else V_s_next

                # Compute TD Target and TD Error (delta) using the CriticNetwork
                td_target = reward + discount_factor * V_s_next
                td_error = td_target - V_s

                # Update the Actor network using the Critic's TD error
                feed_dict_actor = {
                    actor_net.state: state,
                    actor_net.action: action_one_hot,
                    actor_net.delta: td_error,
                }
                actor_loss, _ = sess.run(
                    [actor_net.loss, actor_net.optimizer], feed_dict=feed_dict_actor
                )
                actor_loss_accumulated += actor_loss

                # Update the Critic network
                feed_dict_critic = {critic_net.state: state, critic_net.target_value: td_target}
                critic_loss, _ = sess.run(
                    [critic_net.loss, critic_net.optimizer], feed_dict=feed_dict_critic
                )
                critic_loss_accumulated += critic_loss

                episode_rewards[episode] += reward

                if (done) | (step == max_steps - 1):
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

            # Compute the average loss for the episode
            actor_loss_avg = actor_loss_accumulated / (episode + 1)  #
            critic_loss_avg = critic_loss_accumulated / (episode + 1)  #

            # Log the average policy loss, value loss, and episode reward and MA100
            summary = tf.Summary()  #
            summary.value.add(tag="Actor Loss", simple_value=actor_loss_avg)  #
            summary.value.add(tag="Critic Loss", simple_value=critic_loss_avg)  #
            summary.value.add(tag="Reward", simple_value=episode_rewards[episode])
            summary.value.add(tag="Average Reward", simple_value=average_rewards)
            writer.add_summary(summary, episode)  #
            if solved:
                break
    writer.close()  #


if __name__ == "__main__":
    run()
