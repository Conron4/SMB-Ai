import numpy as np
import cv2
import gym_super_mario_bros
from gym import wrappers
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from collections import deque
import random
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import Nadam

# Define preprocessing functions
def preprocess_frame(frame):
    # Grayscale
    frame = np.mean(frame, axis=2)
    # Resize
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    return frame.astype(np.uint8)

def stack_frames(stacked_frames, state, is_new_episode):
    frame = preprocess_frame(state)
    if is_new_episode:
        stacked_frames = deque([np.zeros((84, 84), dtype=np.uint8) for _ in range(4)], maxlen=4)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
    else:
        stacked_frames.append(frame)
        stacked_state = np.stack(stacked_frames, axis=2)
    return stacked_state, stacked_frames

# Define DQNAgent class
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1e-07  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.990
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(Conv2D(32, (8, 8), strides=(4, 4), activation='relu', input_shape=self.state_size))
        model.add(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Nadam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_agent():
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    state_size = (84, 84, 4)
    action_size = env.action_space.n
    agent = DQNAgent(state_size, action_size)
    batch_size = 32
    episodes = 1000
    for e in range(episodes):
        state = env.reset()
        state, stacked_frames = stack_frames(None, state, True)
        for time in range(500):
            action = agent.act(np.expand_dims(state, axis=0))
            next_state, reward, done, info = env.step(action)
            env.render()
            next_state, stacked_frames = stack_frames(stacked_frames, next_state, False)
            reward = reward if not done else -10
            agent.remember(np.expand_dims(state, axis=0), action, reward, np.expand_dims(next_state, axis=0), done)
            state = next_state
            if done:
                print("episode: {}/{}, score: {}, e: {:.2}"
                      .format(e, episodes, time, agent.epsilon))
                break
            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

if __name__ == "__main__":
    train_agent()

