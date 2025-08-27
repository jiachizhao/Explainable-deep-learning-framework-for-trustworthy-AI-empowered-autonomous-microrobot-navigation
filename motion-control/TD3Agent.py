import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import NonNeg


class ReplayBuffer:
    def __init__(self, buffer_size, input_shape, n_actions):
        self.buffer_size = buffer_size
        self.counter     = 0

        self.state_mem      = np.zeros((self.buffer_size, input_shape))
        self.new_state_mem  = np.zeros((self.buffer_size, input_shape))
        self.action_mem     = np.zeros((self.buffer_size, n_actions))
        self.reward_mem     = np.zeros(self.buffer_size)
        self.terminal_mem   = np.zeros(self.buffer_size, dtype=np.bool_)

    def add(self, state, action, reward, next_state, done):
        idx = self.counter % self.buffer_size

        self.state_mem[idx]     = state
        self.action_mem[idx]    = action
        self.reward_mem[idx]    = reward 
        self.new_state_mem[idx] = next_state
        self.terminal_mem[idx]  = done

        self.counter += 1

    def sample(self, batch_size):
        buffer_max_memory = min(self.counter, self.buffer_size)
        batch = np.random.choice(buffer_max_memory, batch_size)

        states      = self.state_mem[batch]
        next_states = self.new_state_mem[batch]
        actions     = self.action_mem[batch]
        rewards     = self.reward_mem[batch]
        dones       = self.terminal_mem[batch]

        return states, actions, rewards, next_states, dones


class ActorNetwork(tf.keras.Model):
    def __init__(self, name):
        super(ActorNetwork, self).__init__()

        self.fcmono1 = tf.keras.layers.Dense(8, activation='tanh', kernel_constraint=NonNeg(), use_bias=False)
        self.fcmono2 = tf.keras.layers.Dense(16, activation='tanh', kernel_constraint=NonNeg(), use_bias=False)
        self.fcmono3 = tf.keras.layers.Dense(32, activation='tanh', kernel_constraint=NonNeg(), use_bias=False)

        self.fc1 = tf.keras.layers.Dense(8, activation='relu')
        self.fc2 = tf.keras.layers.Dense(16, activation='relu')
        self.fc3 = tf.keras.layers.Dense(32, activation='relu')

        self.outputdot = tf.keras.layers.Dot(axes=1)
        self.outputtanh = tf.keras.layers.Activation('tanh')

        self.checkpoint_dir = 'checkpoints'
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)
        self.model_name = name

    def call(self, state):
        xmono = self.fcmono1(state[:, 0:2])
        xmono = self.fcmono2(xmono)
        xmono = self.fcmono3(xmono)

        xshare = self.fc1(state[:, 2:4])
        xshare = self.fc2(xshare)
        xshare = self.fc3(xshare)
        yaw_action = self.outputdot([xmono, xshare])
        yaw_action = self.outputtanh(yaw_action) * 0.25

        return yaw_action


class CriticNetwork(tf.keras.Model):
    def __init__(self, name):
        super(CriticNetwork, self).__init__()
        self.fc1 = tf.keras.layers.Dense(64, activation='leaky_relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='leaky_relu')
        self.q = tf.keras.layers.Dense(1, activation=None)
        self.model_name =  name

        self.checkpoint_dir = 'checkpoints'
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name)

    def call(self, state, action):
        x = tf.concat([state, action], axis=1)
        q1 = self.fc1(x)
        q1 = self.fc2(q1)
        q = self.q(q1)

        return q


class TD3Agent:
    def __init__(self, input_dims, batch_size, n_actions):
        self.gamma  = 0.9
        self.tau    = 0.0001
        self.actionpara = 0.25

        self.memory         = ReplayBuffer(100000, input_dims, n_actions)
        self.batch_size     = batch_size
        self.learn_counter  = 0
        self.time_step      = 0
        self.warmup         = 500
        self.starttrain     = 500
        self.n_actions      = n_actions

        self.actor          = ActorNetwork(name='actor')
        self.critic_1       = CriticNetwork(name='critic_1')
        self.critic_2       = CriticNetwork(name='critic_2')
        self.target_actor   = ActorNetwork(name='target_actor')
        self.target_critic_1 = CriticNetwork(name='target_critic_1')
        self.target_critic_2 = CriticNetwork(name='target_critic_2')

        self.actor.compile(optimizer=Adam(learning_rate=0.001), loss='mean')
        self.critic_1.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        self.critic_2.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        self.target_actor.compile(optimizer=Adam(learning_rate=0.001), loss='mean')
        self.target_critic_1.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        self.target_critic_2.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

        self.noise = 0.2
        self.actor_freq = 2
        self.update(tau = 1)
    
    def choose_action(self, observation):
        if self.time_step < self.warmup:
            mu = 0.25*np.random.uniform(-1, 1, size=(self.n_actions,))
        else:
            state = tf.convert_to_tensor([observation], dtype=tf.float32)
            mu = self.actor(state)[0].numpy()
            mu = mu + self.actionpara*np.random.normal(scale=self.noise, size=(self.n_actions,))
            mu = np.clip(mu, -self.actionpara, self.actionpara)
        self.time_step += 1

        return mu
    
    def predict_action(self, observation):
        state = tf.convert_to_tensor([observation], dtype=tf.float32)
        pre_action = self.actor(state)[0].numpy()

        return pre_action
    
    def remember(self, state, action, reward, new_state, done):
        self.memory.add(state, action, reward, new_state, done)

    def train(self):
        if self.memory.counter < self.starttrain:
            return
        states, actions, rewards, new_states, dones = self.memory.sample(self.batch_size)
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        states_ = tf.convert_to_tensor(new_states, dtype=tf.float32)

        with tf.GradientTape(persistent=True) as grad:
            target_actions = self.target_actor(states_)
            noise = tf.clip_by_value(tf.random.normal(shape=target_actions.shape, stddev=self.noise), -0.2, 0.2)
            target_actions = tf.clip_by_value(target_actions + noise, -self.actionpara, self.actionpara)
        
            q1_ = self.target_critic_1(states_, target_actions)
            q1 = tf.squeeze(self.critic_1(states, actions), 1)
            q1_ = tf.squeeze(q1_, 1)

            q2_ = self.target_critic_2(states_, target_actions)
            q2 = tf.squeeze(self.critic_2(states, actions), 1)
            q2_ = tf.squeeze(q2_, 1)

            critic_value_ = tf.math.minimum(q1_, q2_)
            target = rewards + self.gamma*critic_value_*(1-dones)
            critic_1_loss = tf.keras.losses.MSE(target, q1)
            critic_2_loss = tf.keras.losses.MSE(target, q2)

        critic_1_gradient = grad.gradient(critic_1_loss, self.critic_1.trainable_variables)
        critic_2_gradient = grad.gradient(critic_2_loss, self.critic_2.trainable_variables)

        self.critic_1.optimizer.apply_gradients(zip(critic_1_gradient, self.critic_1.trainable_variables))
        self.critic_2.optimizer.apply_gradients(zip(critic_2_gradient, self.critic_2.trainable_variables))

        self.learn_counter += 1

        if self.learn_counter % self.actor_freq != 0:
            return

        with tf.GradientTape() as grad:
            new_actions = self.actor(states)
            critic_1_value = self.critic_1(states, new_actions)
            actor_loss = -tf.math.reduce_mean(critic_1_value)

        actor_gradient = grad.gradient(actor_loss, self.actor.trainable_variables)
        self.actor.optimizer.apply_gradients(zip(actor_gradient, self.actor.trainable_variables))

        self.update()

    def update(self, tau = None):
        if tau is None:
            tau = self.tau

        network_weights = self.actor.weights
        target_weights  = self.target_actor.weights
        new_weights     = []
        for idx, weight in enumerate(network_weights):
            new_weights.append(weight * tau + target_weights[idx]*(1-tau))
        self.target_actor.set_weights(new_weights)

        network_weights = self.critic_1.weights
        target_weights  = self.target_critic_1.weights
        new_weights     = []
        for idx, weight in enumerate(network_weights):
            new_weights.append(weight * tau + target_weights[idx]*(1-tau))
        self.target_critic_1.set_weights(new_weights)

        network_weights = self.critic_2.weights
        target_weights  = self.target_critic_2.weights
        new_weights     = []
        for idx, weight in enumerate(network_weights):
            new_weights.append(weight * tau + target_weights[idx]*(1-tau))
        self.target_critic_2.set_weights(new_weights)

    def save_models(self, checkpoint_dir):
        print('SAVING')
        self.actor.save_weights(os.path.join(checkpoint_dir, "actor_episode.ckpt"))
        self.critic_1.save_weights(os.path.join(checkpoint_dir, "critic_1_episode.ckpt"))
        self.critic_2.save_weights(os.path.join(checkpoint_dir, "critic_2_episode.ckpt"))
        self.target_actor.save_weights(os.path.join(checkpoint_dir, "target_actor_episode.ckpt"))
        self.target_critic_1.save_weights(os.path.join(checkpoint_dir, "target_critic_1_episode.ckpt"))
        self.target_critic_2.save_weights(os.path.join(checkpoint_dir, "target_critic_2_episode.ckpt"))

    def load_models(self, checkpoint_dir):
        print('LOADING')
        self.actor.load_weights(os.path.join(checkpoint_dir, "actor_episode.ckpt"))
        self.critic_1.load_weights(os.path.join(checkpoint_dir, "critic_1_episode.ckpt"))
        self.critic_2.load_weights(os.path.join(checkpoint_dir, "critic_2_episode.ckpt"))
        self.target_actor.load_weights(os.path.join(checkpoint_dir, "target_actor_episode.ckpt"))
        self.target_critic_1.load_weights(os.path.join(checkpoint_dir, "target_critic_1_episode.ckpt"))
        self.target_critic_2.load_weights(os.path.join(checkpoint_dir, "target_critic_2_episode.ckpt"))