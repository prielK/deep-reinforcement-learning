import gymnasium as gym
import numpy as np
from datetime import datetime
import time
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Concatenate
from tensorflow.keras.models import load_model
import tensorflow_probability as tfp
from tensorflow.summary import create_file_writer
from tensorflow.keras.optimizers.legacy import Adam

RENDER = False


class ActorNetwork(Model):
    def __init__(
        self, action_size, learning_rate, continuous=True, name="actor_network", hidden_units=256
    ):
        super(ActorNetwork, self).__init__(name=name)
        self.continuous = continuous
        self.learning_rate = learning_rate
        self.optimizer = Adam(learning_rate=learning_rate)
        self.action_size = action_size
        # Glorot Normal Initializer
        init_glorot = tf.keras.initializers.GlorotNormal()
        self.dense_1 = Dense(hidden_units, activation="relu", kernel_initializer=init_glorot)

        if self.continuous == True:
            # Output layers for mu and sigma
            self.mu_output = Dense(action_size, activation=None)  # Linear activation for mu
            self.sigma_output = Dense(
                action_size, activation="softplus"
            )  # Softplus activation for sigma
        else:

            self.output_layer = Dense(action_size, kernel_initializer=init_glorot)

    @tf.function
    def call(self, state):
        x = self.dense_1(state)
        if self.continuous == True:
            mu, sigma = self.mu_output(x), self.sigma_output(x) + 1e-5
            action_dist = tfp.distributions.Normal(mu, sigma)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            return -log_prob, action, None
        else:
            logits = self.output_layer(x)
            action_probs = tf.nn.softmax(logits)
            action = tf.random.categorical(logits, 1)[0, 0]  # Sample based on logits
            neg_log_prob_action = -tf.math.log(action_probs + 1e-10)[0, action]
            return neg_log_prob_action, action, None

    def return_output_from_first_layer(self, state):
        return self.dense_1(state)


class CriticNetwork(Model):
    def __init__(self, learning_rate, name="critic_network", hidden_units=256):
        super(CriticNetwork, self).__init__(name=name)
        self.learning_rate = learning_rate
        self.optimizer = Adam(learning_rate=learning_rate)
        # Glorot Normal Initializer
        init_glorot = tf.keras.initializers.GlorotNormal()
        self.dense_1 = Dense(hidden_units, activation="relu", kernel_initializer=init_glorot)
        self.dense_2 = Dense(hidden_units, activation="relu", kernel_initializer=init_glorot)
        self.output_layer = Dense(1)  # Linear activation

    @tf.function
    def call(self, state):
        x = self.dense_1(state)
        x = self.dense_2(x)
        return self.output_layer(x)

    def return_output_from_first_layer(self, state):
        return self.dense_1(state)

    def return_output_from_second_layer(self, state):
        return self.dense_2(state)


class ModifiedActorNetwork(Model):
    def __init__(
        self,
        source_network_1,
        source_network_2,
        state_size,
        action_size,
        learning_rate,
        continuous=False,
        name="modified_actor_network",
    ):
        super(ModifiedActorNetwork, self).__init__(name=name)
        self.source_network_1 = source_network_1
        self.source_network_2 = source_network_2
        self.state_size = state_size
        self.action_size = action_size
        self.continuous = continuous
        self.learning_rate = learning_rate
        init_glorot = tf.keras.initializers.GlorotNormal()
        self.optimizer = Adam(learning_rate=learning_rate)

        # New layers for the modified network
        self.concat_layer = Concatenate()
        self.dense_mod = Dense(32, activation="relu", kernel_initializer=init_glorot)
        if continuous == True:
            # Output layers for mu and sigma
            self.mu_output = Dense(action_size, activation=None)  # Linear activation for mu
            self.sigma_output = Dense(
                action_size, activation="softplus"
            )  # Softplus activation for sigma
        else:
            self.output_layer = Dense(action_size)

    def call(self, inputs):
        # Extract the first layer output from both source networks
        source_1_output = self.source_network_1.return_output_from_first_layer(inputs)
        source_2_output = self.source_network_2.return_output_from_first_layer(inputs)

        # Concatenate the source outputs with the current inputs
        x = self.dense_mod(inputs)
        concat_output = self.concat_layer([x, source_1_output, source_2_output])

        if self.continuous == True:
            mu, sigma = self.mu_output(concat_output), self.sigma_output(concat_output) + 1e-5
            action_dist = tfp.distributions.Normal(mu, sigma)
            action = action_dist.sample()
            log_prob = action_dist.log_prob(action)
            return -log_prob, action, None
        else:
            logits = self.output_layer(x)
            action_probs = tf.nn.softmax(logits)
            action = tf.random.categorical(logits, 1)[0, 0]  # Sample based on logits
            neg_log_prob_action = -tf.math.log(action_probs + 1e-10)[0, action]
            return neg_log_prob_action, action, None


class ModifiedCriticNetwork(Model):
    def __init__(
        self,
        source_network_1,
        source_network_2,
        state_size,
        learning_rate,
        name="modified_critic_network",
    ):
        super(ModifiedCriticNetwork, self).__init__(name=name)
        self.source_network_1 = source_network_1
        self.source_network_2 = source_network_2
        self.state_size = state_size
        self.learning_rate = learning_rate
        self.optimizer = Adam(learning_rate=learning_rate)
        init_glorot = tf.keras.initializers.GlorotNormal()
        self.source_network_1 = source_network_1
        self.source_network_2 = source_network_2

        self.concat_layer_1 = Concatenate()
        self.concat_layer_2 = Concatenate()
        self.dense_mod_1 = Dense(128, activation="relu", kernel_initializer=init_glorot)
        self.dense_mod_2 = Dense(128, activation="relu", kernel_initializer=init_glorot)
        self.output_layer = Dense(1)

    def call(self, inputs):

        source_1_output_1 = self.source_network_1.return_output_from_first_layer(inputs)
        source_1_output_2 = self.source_network_1.return_output_from_second_layer(inputs)
        source_2_output_1 = self.source_network_2.return_output_from_first_layer(inputs)
        source_2_output_2 = self.source_network_2.return_output_from_second_layer(inputs)
        # Concatenate the source outputs with the current inputs for both layers
        x = self.dense_mod_1(inputs)
        concat_output_1 = self.concat_layer_1([x, source_1_output_1, source_2_output_1])
        x = self.dense_mod_2(concat_output_1)
        concat_output_2 = self.concat_layer_2([x, source_1_output_2, source_2_output_2])
        return self.output_layer(concat_output_2)


@tf.function
def pad_state(state, target_size):
    # Ensure the state tensor is treated as 2D (shape: [1, state_length])
    state = tf.reshape(state, [1, -1])
    # Calculate the padding size
    padding_size = target_size - tf.shape(state)[1]
    # Create a padding tensor of the appropriate size, ensuring it's 2D ([1, padding_size])
    padding = tf.zeros([1, padding_size], dtype=state.dtype)
    # Concatenate the state with the padding tensor along the second axis (columns)
    padded_state = tf.concat([state, padding], axis=1)
    return padded_state


def save_model(model, save_path):
    """
    Save the model to the specified path.

    Args:
    - model: The Keras model instance to be saved.
    - save_path: The path (including filename) where to save the model.
    """
    if not model.built:
        print("Model not built yet. Building model with dummy input.")
        model.build((None, max_state_size))

    model.save(save_path)
    print(f"Model saved to {save_path}")


def load_model_new(load_path):
    """
    Load a model from the specified path.

    Args:
    - load_path: The path (including filename) from where to load the model.

    Returns:
    - The loaded Keras model.
    """
    model = load_model(load_path)
    print(f"Model loaded from {load_path}")
    return model


def initial_setup(env_name, max_state_size, learning_rate):
    if RENDER:
        env = gym.make(env_name, render_mode="human")
        env.metadata["render_fps"] = 1000
    else:
        env = gym.make(env_name)
    continuous = isinstance(env.action_space, gym.spaces.Box)
    action_size = env.action_space.shape[0] if continuous else env.action_space.n
    max_episodes = 1000
    if env_name == "MountainCarContinuous-v0":
        max_steps = 2000
    else:
        max_steps = 500
    discount_factor = 0.99
    learning_rate = learning_rate
    max_state_size = max_state_size
    models_path_prefix = "/"
    # Print summary of setup
    print(f"Environment: {env_name}")
    print(f"Continuous Action Space: {continuous}")
    print(f"Action Size: {action_size}")
    print(f"Max Episodes: {max_episodes}")
    print(f"Max Steps: {max_steps}")
    print(f"Discount Factor: {discount_factor}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Max State Size: {max_state_size}")
    print(f"Models Path Prefix: {models_path_prefix}")

    return (
        env,
        continuous,
        action_size,
        max_episodes,
        max_steps,
        discount_factor,
        learning_rate,
        max_state_size,
        models_path_prefix,
    )


def load_pretrained_source_networks(env_name, learning_rate, models_path_prefix):
    if env_name == "MountainCarContinuous-v0":
        actor_net_source_1 = ActorNetwork(2, learning_rate, False, name="actor_network_CartPole-v1")
        critic_net_source_1 = CriticNetwork(learning_rate, name="critic_network_CartPole-v1")
        # Load pre-trained weights for CartPole-v1
        return_output_from_first_layer_actor = actor_net_source_1.return_output_from_first_layer
        return_output_from_first_layer_critic = critic_net_source_1.return_output_from_first_layer
        return_output_from_second_layer_critic = critic_net_source_1.return_output_from_second_layer
        actor_net_source_1 = load_model_new("actor_weights_CartPole-v1.ckpt")
        critic_net_source_1 = load_model_new("critic_weights_CartPole-v1.ckpt")

        actor_net_source_1.return_output_from_first_layer = return_output_from_first_layer_actor
        critic_net_source_1.return_output_from_first_layer = return_output_from_first_layer_critic
        critic_net_source_1.return_output_from_second_layer = return_output_from_second_layer_critic

        actor_net_source_2 = ActorNetwork(3, learning_rate, False, name="actor_network_Acrobot-v1")
        critic_net_source_2 = CriticNetwork(learning_rate, name="critic_network_Acrobot-v1")
        # Load pre-trained weights for Acrobot-v1
        return_output_from_first_layer_actor = actor_net_source_2.return_output_from_first_layer
        return_output_from_first_layer_critic = critic_net_source_2.return_output_from_first_layer
        return_output_from_second_layer_critic = critic_net_source_2.return_output_from_second_layer

        actor_net_source_2 = load_model_new("actor_weights_Acrobot-v1.ckpt")
        critic_net_source_2 = load_model_new("critic_weights_Acrobot-v1.ckpt")

        actor_net_source_2.return_output_from_first_layer = return_output_from_first_layer_actor
        critic_net_source_2.return_output_from_first_layer = return_output_from_first_layer_critic
        critic_net_source_2.return_output_from_second_layer = return_output_from_second_layer_critic

        actor_net_source_1.trainable = False
        actor_net_source_2.trainable = False
        critic_net_source_1.trainable = False
        critic_net_source_2.trainable = False

    elif env_name == "CartPole-v1":
        actor_net_source_1 = ActorNetwork(3, learning_rate, False, name="actor_network_Acrobot-v1")
        critic_net_source_1 = CriticNetwork(learning_rate, name="critic_network_Acrobot-v1")
        # Load pre-trained weights for CartPole-v1
        return_output_from_first_layer_actor = actor_net_source_1.return_output_from_first_layer
        return_output_from_first_layer_critic = critic_net_source_1.return_output_from_first_layer
        return_output_from_second_layer_critic = critic_net_source_1.return_output_from_second_layer
        actor_net_source_1 = load_model_new("actor_weights_Acrobot-v1.ckpt")
        critic_net_source_1 = load_model_new("actor_weights_Acrobot-v1.ckpt")

        actor_net_source_1.return_output_from_first_layer = return_output_from_first_layer_actor
        critic_net_source_1.return_output_from_first_layer = return_output_from_first_layer_critic
        critic_net_source_1.return_output_from_second_layer = return_output_from_second_layer_critic

        actor_net_source_2 = ActorNetwork(
            1, learning_rate, True, name="actor_network_MountainCarContinuous-v0"
        )
        critic_net_source_2 = CriticNetwork(
            learning_rate, name="critic_network_MountainCarContinuous-v0"
        )
        # Load pre-trained weights for MountainCarContinuous-v0
        return_output_from_first_layer_actor = actor_net_source_2.return_output_from_first_layer
        return_output_from_first_layer_critic = critic_net_source_2.return_output_from_first_layer
        return_output_from_second_layer_critic = critic_net_source_2.return_output_from_second_layer

        actor_net_source_2 = load_model_new("actor_weights_MountainCarContinuous-v0.ckpt")
        critic_net_source_2 = load_model_new("critic_weights_MountainCarContinuous-v0.ckpt")

        actor_net_source_2.return_output_from_first_layer = return_output_from_first_layer_actor
        critic_net_source_2.return_output_from_first_layer = return_output_from_first_layer_critic
        critic_net_source_2.return_output_from_second_layer = return_output_from_second_layer_critic

        actor_net_source_1.trainable = False
        actor_net_source_2.trainable = False
        critic_net_source_1.trainable = False
        critic_net_source_2.trainable = False

    if actor_net_source_1 is None:
        print("Actor Network 1 is None")
    if critic_net_source_1 is None:
        print("Critic Network 1 is None")
    if actor_net_source_2 is None:
        print("Actor Network 2 is None")
    if critic_net_source_2 is None:
        print("Critic Network 2 is None")

    # Print summaries of the source networks
    print("Source Network 1 (Actor Network) Summary:")
    print(actor_net_source_1.summary())
    print("Source Network 1 (Critic Network) Summary:")
    print(critic_net_source_1.summary())
    print("Source Network 2 (Actor Network) Summary:")
    print(actor_net_source_2.summary())
    print("Source Network 2 (Critic Network) Summary:")
    print(critic_net_source_2.summary())

    return actor_net_source_1, critic_net_source_1, actor_net_source_2, critic_net_source_2


def tensor_to_numpy(tensor):
    return tensor.numpy()


def run(env_name, max_state_size, transfer_learning=False):
    # Set up the environment and other parameters
    (
        env,
        continuous,
        action_size,
        max_episodes,
        max_steps,
        discount_factor,
        learning_rate,
        max_state_size,
        models_path_prefix,
    ) = initial_setup(env_name, max_state_size, 0.0005)

    log_dir = f"Assignment 3/logs/new_amazing_{env_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    writer = create_file_writer(log_dir)
    print(f"Transfer Learning ? {transfer_learning}")
    print(f"Environment ? {env_name}")
    if transfer_learning == True:
        (actor_net_source_1, critic_net_source_1, actor_net_source_2, critic_net_source_2) = (
            load_pretrained_source_networks(env_name, learning_rate, models_path_prefix)
        )
        if env_name == "MountainCarContinuous-v0":
            continuous = True
        elif env_name == "CartPole-v1":
            continuous = False
        elif env_name == "Acrobot-v1":
            continuous = False

        actor_net = ModifiedActorNetwork(
            actor_net_source_1,
            actor_net_source_2,
            max_state_size,
            action_size,
            learning_rate,
            continuous,
        )
        critic_net = ModifiedCriticNetwork(
            critic_net_source_1, critic_net_source_2, max_state_size, learning_rate
        )
        # Pass dummy input to build the model
        actor_net.build((None, max_state_size))
        critic_net.build((None, max_state_size))
        print("Modified Actor Network Summary:")
        print(actor_net.summary())
        print("Modified Critic Network Summary:")
        print(critic_net.summary())
    else:
        if (env_name == "CartPole-v1") | (env_name == "Acrobot-v1"):
            actor_net = ActorNetwork(
                action_size, learning_rate, continuous, f"actor_network_{env_name}", 32
            )
            critic_net = CriticNetwork(learning_rate, f"critic_network_{env_name}", 128)
        elif env_name == "MountainCarContinuous-v0":
            actor_net = ActorNetwork(
                action_size, learning_rate, continuous, f"actor_network_{env_name}", 256
            )
            critic_net = CriticNetwork(learning_rate, f"critic_network_{env_name}", 256)

    start_time = time.time()
    episode_rewards = np.zeros(max_episodes)
    average_rewards = 0.0
    solved = False

    for episode in range(max_episodes):
        state = pad_state(env.reset()[0], max_state_size)
        actor_loss_accumulated = 0  # Initialization here
        critic_loss_accumulated = 0  # Assuming you also accumulate critic loss similarly
        for step in range(max_steps):
            with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
                log_prob, action, probabilities = actor_net(state)  # tf.expand_dims(state, axis=0))
                if continuous:
                    action = tf.clip_by_value(action, env.action_space.low, env.action_space.high)
                else:
                    action = tf.squeeze(action)

                # Take a step in the environment
                action_to_env = tensor_to_numpy(action)  # if hasattr(action, 'numpy') else action

                next_state, reward, done, _, _ = env.step(action_to_env)
                next_state = pad_state(
                    next_state, max_state_size
                )  # Pad the state to match the max_state_size
                # Compute critic value for current and next state
                V_s = critic_net(tf.expand_dims(state, axis=0))  # Current state value prediction
                V_s_next = critic_net(
                    tf.expand_dims(next_state, axis=0)
                )  # Next state value prediction
                V_s_next = 0 if done else V_s_next  # Zero out V(s') if done
                # Compute TD target and error
                td_target = reward + discount_factor * V_s_next
                td_error = td_target - V_s

                critic_loss = tf.reduce_mean(tf.square(td_error))
                actor_loss = tf.reduce_mean(log_prob * td_error)

            # Compute and apply gradients for the critic network
            grads_critic = critic_tape.gradient(critic_loss, critic_net.trainable_variables)
            critic_net.optimizer.apply_gradients(zip(grads_critic, critic_net.trainable_variables))
            grads_actor = actor_tape.gradient(actor_loss, actor_net.trainable_variables)
            actor_net.optimizer.apply_gradients(zip(grads_actor, actor_net.trainable_variables))
            actor_loss_accumulated += actor_loss
            critic_loss_accumulated += critic_loss

            episode_rewards[episode] += reward
            state = next_state

            if (done) | (step == max_steps - 1):
                if episode > 100:
                    average_rewards = np.mean(episode_rewards[(episode - 100) : episode + 1])
                print(
                    "Episode {} Reward: {} Average over 100 episodes: {}".format(
                        episode, episode_rewards[episode], round(average_rewards, 2)
                    )
                )
                # Check if solved
                if env_name == "CartPole-v1" and average_rewards > 475:
                    print(" Solved at episode: " + str(episode))
                    solved = True
                elif (
                    env_name == "Acrobot-v1"
                    and episode_rewards[episode] > -100
                    and average_rewards != 0
                ):
                    print(" Solved at episode: " + str(episode))
                    solved = True
                elif env_name == "MountainCarContinuous-v0" and average_rewards > 0:
                    print(" Solved at episode: " + str(episode))
                    solved = True
                break

        print(f"Episode: {episode}, Total Reward: {episode_rewards[episode]}")
        # Compute the average loss for the episode
        actor_loss_avg = actor_loss_accumulated / (episode + 1)  #
        critic_loss_avg = critic_loss_accumulated / (episode + 1)  #

        with writer.as_default():
            tf.summary.scalar("Reward", episode_rewards[episode], step=episode)
            tf.summary.scalar("Average Reward", average_rewards, step=episode)
            tf.summary.scalar("Actor Loss", actor_loss_avg, step=episode)
            tf.summary.scalar("Critic Loss", critic_loss_avg, step=episode)

            writer.flush()

        # if solved save model weights
        if solved | (episode == max_episodes - 1):
            end_time = time.time()
            running_time = end_time - start_time
            print(
                f"Environment: {env_name}, Time to Train: {running_time:.2f} sec, Episodes to Converge: {episode}"
            )
            if transfer_learning == False:
                save_model(actor_net, f"actor_weights_{env_name}.ckpt")
                save_model(critic_net, f"critic_weights_{env_name}.ckpt")
            return solved

    return False


if __name__ == "__main__":
    save_and_reload = True

    global max_state_size

    env_cartpole = gym.make("CartPole-v1")
    env_acrobot = gym.make("Acrobot-v1")
    env_mountaincar = gym.make("MountainCarContinuous-v0")

    max_state_size = max(
        env_cartpole.observation_space.shape[0],
        env_acrobot.observation_space.shape[0],
        env_mountaincar.observation_space.shape[0],
    )

    # Pretraining
    problems = ["CartPole-v1", "Acrobot-v1", "MountainCarContinuous-v0"]
    for problem in problems:
        print(f"Training on {problem}")
        success = run(problem, max_state_size, transfer_learning=False)

    # Task 1
    problem = "CartPole-v1"
    print(f"Training on {problem}")
    success = run(problem, max_state_size, transfer_learning=True)

    # Task 2
    problem = "MountainCarContinuous-v0"
    print(f"Training on {problem}")
    success = run(problem, max_state_size, transfer_learning=True)
