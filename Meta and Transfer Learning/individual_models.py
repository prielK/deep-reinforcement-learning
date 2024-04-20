import gymnasium as gym
import numpy as np
import tensorflow.compat.v1 as tf
from datetime import datetime
import time

# Disable eager execution to use tf1.x functions in tf2.x
tf.disable_v2_behavior()
print("tf_ver:{}".format(tf.__version__))
RENDER = False


class ActorNetwork:
    def __init__(
        self, state_size, action_size, learning_rate, continuous=False, name="actor_network"
    ):
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, state_size], name="state")
            init_xavier = tf.keras.initializers.glorot_uniform()
            self.delta = tf.placeholder(tf.float32, name="delta")
            # For MountainCarContinuous-v0
            if continuous:
                self.hidden1 = tf.layers.dense(
                    self.state,
                    256,
                    activation=tf.nn.relu,
                    kernel_initializer=init_xavier,
                    name="layer1",
                )
                self.mu = tf.layers.dense(
                    self.hidden1,
                    action_size,
                    activation=None,
                    kernel_initializer=init_xavier,
                    name="mu",
                )
                self.sigma = tf.layers.dense(
                    self.hidden1,
                    action_size,
                    activation=tf.nn.softplus,
                    kernel_initializer=init_xavier,
                    name="sigma",
                )
                self.sigma = self.sigma + 1e-5
                self.norm_dist = tf.distributions.Normal(self.mu, self.sigma)
                self.action = tf.squeeze(self.norm_dist.sample(1), axis=0)
                self.action = tf.clip_by_value(
                    self.action, -1, 1
                )  # Clip the action to be within valid range for the environment
                self.neg_log_prob = -self.norm_dist.log_prob(self.action) * self.delta
                self.loss = tf.reduce_mean(self.neg_log_prob)
            # For CartPole-v1 and Acrobot-v1
            else:
                self.hidden1 = tf.layers.dense(
                    self.state,
                    32,
                    activation=tf.nn.relu,
                    kernel_initializer=init_xavier,
                    name="layer1",
                )
                self.output = tf.layers.dense(
                    self.hidden1,
                    action_size,
                    activation=tf.nn.relu,
                    kernel_initializer=init_xavier,
                    name="output",
                )
                self.action = tf.placeholder(tf.int32, [None, action_size], name="action")
                self.actions_distribution = tf.nn.softmax(self.output)
                self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=self.output, labels=self.action
                )
                self.loss = tf.reduce_mean(self.neg_log_prob * self.delta)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)


class CriticNetwork:
    def __init__(self, state_size, learning_rate, name="critic_network"):
        with tf.variable_scope(name):
            self.state = tf.placeholder(tf.float32, [None, state_size], name="state")
            self.target_value = tf.placeholder(tf.float32, name="target_value")
            init_xavier = tf.keras.initializers.glorot_uniform()

            self.hidden1 = tf.layers.dense(
                self.state,
                256,
                activation=tf.nn.relu,
                kernel_initializer=init_xavier,
                name="layer1",
            )
            self.hidden2 = tf.layers.dense(
                self.hidden1,
                256,
                activation=tf.nn.relu,
                kernel_initializer=init_xavier,
                name="layer2",
            )
            self.output = tf.layers.dense(
                self.hidden2, 1, activation=None, kernel_initializer=init_xavier, name="output"
            )
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_value, self.output))
            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(self.loss)


# Helper function to pad states
def pad_state(state, target_size):
    padding = np.zeros(target_size - len(state))
    padded_state = np.append(state, padding)

    return padded_state.reshape([1, target_size])


# Run the training on the environment
def run(env_name, max_state_size):
    if RENDER:
        env = gym.make(env_name, render_mode="human")
        env.metadata["render_fps"] = 360
    else:
        env = gym.make(env_name)
    np.random.seed(1)

    continuous = isinstance(env.action_space, gym.spaces.Box)
    action_size = env.action_space.shape[0] if continuous else env.action_space.n

    if continuous:
        learning_rate = 0.0001
        max_steps = 1000
    else:
        learning_rate = 0.0004
        max_steps = 501
    max_episodes = 1000
    discount_factor = 0.99

    tf.reset_default_graph()
    actor_net = ActorNetwork(max_state_size, action_size, learning_rate, continuous)
    critic_net = CriticNetwork(max_state_size, learning_rate)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        start_time = time.time()
        episode_rewards = np.zeros(max_episodes)
        average_rewards = 0.0
        logdir = f"logs/{env_name}_Section1_{datetime.now().strftime('%d%m%Y%H%M%S')}"
        writer = tf.summary.FileWriter(logdir, sess.graph)
        solved = False

        for episode in range(max_episodes):
            state = pad_state(env.reset()[0], max_state_size)
            actor_loss_accumulated = 0  # Initialization here
            critic_loss_accumulated = 0  # Assuming you also accumulate critic loss similarly

            for step in range(max_steps):
                if continuous:
                    # Directly use mu as the deterministic part of the action and add noise for exploration
                    action, _ = sess.run(
                        [actor_net.mu, actor_net.action], feed_dict={actor_net.state: state}
                    )

                else:

                    actions_distribution = sess.run(
                        actor_net.actions_distribution, {actor_net.state: state}
                    )
                    actions_distribution = actions_distribution[0].reshape([1, action_size])
                    action = np.random.choice(
                        np.arange(len(actions_distribution[0])), p=actions_distribution[0]
                    )
                next_state, reward, done, _, _ = env.step(action)
                next_state = pad_state(next_state, max_state_size)

                action_one_hot = np.zeros((1, action_size)) if not continuous else action
                if not continuous:
                    action_one_hot[0, action] = 1

                V_s = sess.run(critic_net.output, {critic_net.state: state})
                V_s_next = sess.run(critic_net.output, {critic_net.state: next_state})
                V_s_next = 0 if done else V_s_next

                td_target = reward + discount_factor * V_s_next
                td_error = td_target - V_s

                feed_dict_actor = {
                    actor_net.state: state,
                    actor_net.action: action_one_hot,
                    actor_net.delta: td_error,
                }
                actor_loss, _ = sess.run(
                    [actor_net.loss, actor_net.optimizer], feed_dict=feed_dict_actor
                )
                actor_loss_accumulated += actor_loss

                feed_dict_critic = {critic_net.state: state, critic_net.target_value: td_target}
                critic_loss, _ = sess.run(
                    [critic_net.loss, critic_net.optimizer], feed_dict=feed_dict_critic
                )
                critic_loss_accumulated += critic_loss

                episode_rewards[episode] += reward
                state = next_state

                # Checking for solved condition
                if (env_name == "MountainCarContinuous-v0" and reward == 100) or (
                    env_name == "Acrobot-v1" and reward >= -100 and done
                ):
                    solved = True
                    print("SOLVED!")
                    break

                if (done) | (step == max_steps - 1):
                    if episode > 98:
                        average_rewards = np.mean(episode_rewards[(episode - 99) : episode + 1])
                    print(
                        "Episode {} Reward: {} Average over 100 episodes: {}".format(
                            episode, episode_rewards[episode], round(average_rewards, 2)
                        )
                    )
                    # Check if solved
                    if env_name == "CartPole-v1" and average_rewards > 475:
                        print(" Solved at episode: " + str(episode))
                        solved = True
                    break

            # Compute the average loss for the episode
            actor_loss_avg = actor_loss_accumulated / (episode + 1)  #
            critic_loss_avg = critic_loss_accumulated / (episode + 1)  #

            # Log the average policy loss, value loss, and episode reward and MA100
            summary = tf.Summary()  #
            summary.value.add(tag="Actor Loss", simple_value=actor_loss_avg)  #
            summary.value.add(tag="Critic Loss", simple_value=critic_loss_avg)  #
            summary.value.add(tag="Reward", simple_value=episode_rewards[episode])
            writer.add_summary(summary, episode)  #
            if solved:
                break

        end_time = time.time()
        running_time = end_time - start_time
        print(
            f"Environment: {env_name}, Time to Train: {running_time:.2f} sec, Episodes to Converge: {episode}"
        )
    writer.close()


if __name__ == "__main__":
    env_cartpole = gym.make("CartPole-v1")
    env_acrobot = gym.make("Acrobot-v1")
    env_mountaincar = gym.make("MountainCarContinuous-v0")
    max_state_size = max(
        env_cartpole.observation_space.shape[0],
        env_acrobot.observation_space.shape[0],
        env_mountaincar.observation_space.shape[0],
    )
    for env_name in ["CartPole-v1", "Acrobot-v1", "MountainCarContinuous-v0"]:
        # for env_name in ["MountainCarContinuous-v0"]:
        print(f"Training on {env_name}")
        run(env_name, max_state_size)
