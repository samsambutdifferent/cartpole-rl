
import gym
import tensorflow as tf
from collections import deque

import os
import datetime
import random
import numpy as np
import math

from tensorflow.keras import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber
from tensorflow.keras.initializers import he_normal
from tensorflow.keras.callbacks import History

dt_now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
performance_dir = f"performance/{dt_now}/"


class DDDQN(tf.keras.Model):
    def __init__(self):
      super(DDDQN, self).__init__()
      self.d1 = tf.keras.layers.Dense(128, activation='relu')
      self.d2 = tf.keras.layers.Dense(128, activation='relu')
      self.v = tf.keras.layers.Dense(1, activation=None)
      self.a = tf.keras.layers.Dense(env.action_space.n, activation=None)

    def call(self, input_data):
      x = self.d1(input_data)
      x = self.d2(x)
      v = self.v(x)
      a = self.a(x)
      Q = v +(a -tf.math.reduce_mean(a, axis=1, keepdims=True))
      return Q

    def advantage(self, state):
      x = self.d1(state)
      x = self.d2(x)
      a = self.a(x)
      return a


class exp_replay():
    def __init__(self, buffer_size= 1000000):
        self.buffer_size = buffer_size
        self.state_mem = np.zeros((self.buffer_size, *(env.observation_space.shape)), dtype=np.float32)
        self.action_mem = np.zeros((self.buffer_size), dtype=np.int32)
        self.reward_mem = np.zeros((self.buffer_size), dtype=np.float32)
        self.next_state_mem = np.zeros((self.buffer_size, *(env.observation_space.shape)), dtype=np.float32)
        self.done_mem = np.zeros((self.buffer_size), dtype=np.bool)
        self.pointer = 0

    def add_exp(self, state, action, reward, next_state, done):
        idx  = self.pointer % self.buffer_size 
        self.state_mem[idx] = state
        self.action_mem[idx] = action
        self.reward_mem[idx] = reward
        self.next_state_mem[idx] = next_state
        self.done_mem[idx] = 1 - int(done)
        self.pointer += 1

    def sample_exp(self, batch_size= 64):
        max_mem = min(self.pointer, self.buffer_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        next_states = self.next_state_mem[batch]
        dones = self.done_mem[batch]
        return states, actions, rewards, next_states, dones


import matplotlib.pyplot as plt
from IPython.display import clear_output

class agent():
      def __init__(self, gamma=0.99, replace=100, lr=0.001):
          self.gamma = gamma
          self.epsilon = 1.0
          self.min_epsilon = 0.01
          self.epsilon_decay = 1e-3
          self.replace = replace
          self.trainstep = 0
          self.memory = exp_replay()
          self.batch_size = 64
          self.q_net = DDDQN()
          self.target_net = DDDQN()
          opt = tf.keras.optimizers.Adam(learning_rate=lr)
          self.q_net.compile(loss='mse', optimizer=opt)
          self.target_net.compile(loss='mse', optimizer=opt)
          self.plot_target = 5

          self.test_state = False

      def act(self, state):
          if not self.test_state and np.random.rand() <= self.epsilon:
              return np.random.choice([i for i in range(env.action_space.n)])

          else:
              actions = self.q_net.advantage(np.array([state]))
              action = np.argmax(actions)
              return action
      
      def update_mem(self, state, action, reward, next_state, done):
          self.memory.add_exp(state, action, reward, next_state, done)

      def update_target(self):
          self.target_net.set_weights(self.q_net.get_weights())     

      def update_epsilon(self):
          self.epsilon = self.epsilon - self.epsilon_decay if self.epsilon > self.min_epsilon else self.min_epsilon
          return self.epsilon

      def train(self):
          if self.memory.pointer < self.batch_size:
             return 
          
          if self.trainstep % self.replace == 0:
             self.update_target()

          states, actions, rewards, next_states, dones = self.memory.sample_exp(self.batch_size)
          target = self.q_net.predict(states, verbose=0)
          next_state_val = self.target_net.predict(next_states, verbose=0)
          max_action = np.argmax(self.q_net.predict(next_states, verbose=0), axis=1)
          batch_index = np.arange(self.batch_size, dtype=np.int32)
          q_target = np.copy(target)  #optional  
          q_target[batch_index, actions] = rewards + self.gamma * next_state_val[batch_index, max_action]*dones
          self.q_net.train_on_batch(states, q_target)
          self.update_epsilon()
          self.trainstep += 1

      def _plot(
          self, 
          episode, 
          rewards, 
          # losses: List[float], 
          # epsilons: List[float],
      ):
          """Plot the training progresses."""
          if not os.path.isdir(performance_dir):
            os.makedirs(performance_dir)


          clear_output(True)
          title = f"episode {episode} score: {np.mean(rewards[-10:])}"
          plt.figure(figsize=(20, 5))
          plt.subplot(131)
          plt.title(title)
          plt.plot(rewards)
          # plt.subplot(132)
          # plt.title('loss')
          # plt.plot(losses)
          # plt.subplot(133)
          # plt.title('epsilons')
          # plt.plot(epsilons)

          plt.show()
          plt.savefig(f"{performance_dir}{title}.png")


if __name__=="__main__":
    env = gym.make("CartPole-v0") # gym.make("ALE/Boxing-v5", difficulty=1)
    all_rewards = []
    agentoo7 = agent()
    steps = 50
    for s in range(steps):
        done = False
        state = env.reset()
        total_reward = 0
        
        while not done:
            # env.render()
            action = agentoo7.act(state)
            next_state, reward, done, _ = env.step(action)
            agentoo7.update_mem(state, action, reward, next_state, done)
            agentoo7.train()
            state = next_state
            total_reward += reward
            
            if done:
                print("total reward after {} episode is {} and epsilon is {}".format(s, total_reward, agentoo7.epsilon))

        all_rewards.append(total_reward)

        if s % agentoo7.plot_target == 0 and s != 0:
            agentoo7._plot(
            s,
            all_rewards
            )

    
    ## VIDEO

    # agent.test_state = True

    # clean_env = env
    # env = gym.wrappers.RecordVideo(env, video_folder="videos")
            
    # state = env.reset()
    # done = False
    # score = 0

    # while not done:
    #     action = agent().act(state)

    #     next_state, reward, done, _ = env.step(action)
    #     state = next_state
    #     score += reward
            
    # print("score: ", score)
    # env.close()        
    # env = clean_env
    # agent.test_state = False