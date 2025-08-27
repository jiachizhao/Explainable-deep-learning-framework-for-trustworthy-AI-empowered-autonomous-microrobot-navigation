import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers
from tensorflow.keras.constraints import NonNeg
import random
from collections import deque


class Network(tf.keras.Model):
    def __init__(self, n_actions):
        super(Network, self).__init__()
        self.n_actions = n_actions

        self.obst1 = layers.Conv1D(filters=8, kernel_size=1, activation='tanh', kernel_constraint=NonNeg(), use_bias=False, name='obst1')
        self.obst2 = layers.Conv1D(filters=8, kernel_size=1, activation='relu', kernel_constraint=NonNeg(), use_bias=False, name='obst2')

        self.target_distance1 = layers.Conv1D(filters=8, kernel_size=1, activation='leaky_relu', name='target_distance1')
        self.target_distance2 = layers.Conv1D(filters=4, kernel_size=1, activation='leaky_relu', name='target_distance2')

        self.others1 = layers.Conv1D(filters=16, kernel_size=1, activation='leaky_relu', kernel_constraint=NonNeg(), name='other1')
        self.others2 = layers.Conv1D(filters=8, kernel_size=1, activation='softplus', kernel_constraint=NonNeg(), name='other2')
        
        self.fc1 = layers.Conv1D(filters=8, kernel_size=1, activation='tanh', kernel_constraint=NonNeg(), use_bias=False, name='fc1')
        self.fc2 = layers.Conv1D(filters=8, kernel_size=1, activation='tanh', kernel_constraint=NonNeg(), use_bias=False, name='fc2')
        self.fc3 = layers.Conv1D(filters=1, kernel_size=1, kernel_constraint=NonNeg(), use_bias=False, name='fc3')

    def call(self, inputs):
        x = inputs

        obst = x[:, :, 0:1]
        obst = tf.nn.tanh(obst)
        obst = self.obst1(obst)
        obst = self.obst2(obst)

        target_distance = x[:, :, 3:4]
        target_distance = self.target_distance1(target_distance)
        target_distance = self.target_distance2(target_distance)
        
        others = x[:, :, 1:3]
        others = tf.concat([others, target_distance], axis=2)
        others = self.others1(others)
        others = self.others2(others)
        
        out = tf.multiply(obst, others)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        
        return tf.squeeze(out, axis=-1)


class DQNAgent_local:
    def __init__(self, state_shape, action_size, memory_capacity=100000, batch_size=32, gamma=0.9, learning_rate=0.0001, epsilon_start=1.0, epsilon_end=0.1, epsilon_decay=0.92, logger=None):
        self.state_shape = state_shape
        self.action_size = action_size
        self.memory = deque(maxlen=memory_capacity)
        self.batch_size = batch_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.learning_rate = learning_rate

        self.model = Network(self.action_size)
        self.target_model = Network(self.action_size)
        self.update_target_model()
        self.model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
        self.target_model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
        self.training_step = 0
        self.logger = logger

    def preprocess_state(self, state):
        state = state / 255.0
        state = np.expand_dims(state, axis=0).astype(np.float32)

        return state

    def choose_action(self, state, avoid_obst, action_global=None):
        avoid_obst = np.ceil(avoid_obst.reshape(self.action_size))
        self.training_step += 1
        if self.training_step % 100 == 0 and self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        if np.random.rand() < self.epsilon:
            random_numbers = [random.randint(1, 100) for _ in range(self.action_size)]
            random_action = random_numbers * avoid_obst
            return  [np.argmax(random_action), True]
        else:
            q_values = self.model.predict(state, verbose=0)
            if action_global is not None:
                q_values = q_values[0] * action_global
            else:
                q_values = q_values[0]
            all_zero = np.any(q_values > 0)

            return [np.argmax(q_values), all_zero]

    def predict_action(self, state, action_global=None):
        q_values = self.model.predict(state, verbose=0)
        if action_global is not None:
            q_values = q_values[0] * action_global
        else:
            q_values = q_values[0]
        all_zero = np.any(q_values > 0)

        return [np.argmax(q_values), all_zero]

    def store_transition(self, state_image, action, reward, next_state_image, done):
        state_image = np.squeeze(state_image)
        next_state_image = np.squeeze(next_state_image)
        self.memory.append((state_image, action, reward, next_state_image, done))

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state_image, action_batch, reward_batch, next_state_image, done_batch = map(np.array, zip(*batch))
        target = self.target_model(state_image).numpy()
        next_q_value = tf.reduce_max(self.target_model(next_state_image), axis=1)
        target_q = reward_batch + (1 - done_batch) * self.gamma * next_q_value
        target[range(self.batch_size), action_batch] = target_q

        with tf.GradientTape() as tape:
            q_pred = self.model(state_image)
            loss = tf.losses.MSE(target, q_pred)
        grads = tape.gradient(loss, self.model.trainable_weights)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
        self.logger.log_scalar('loss', tf.reduce_mean(loss), step=self.training_step)

        tf.keras.backend.clear_session()
        if self.training_step % 200 == 0:
            self.update_target_model()

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())

    def save_model(self, filepath):
        self.model.save_weights(filepath)

    def load_model(self, filepath):
        self.model.load_weights(filepath)
        self.update_target_model()
