
import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

import matplotlib.pyplot as plt

use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

from multiprocessing_env import SubprocVecEnv

num_envs = 8
env_name = "CartPole-v0"

def make_env():
    def _thunk():
        env = gym.make(env_name)
        return env
    return _thunk

plt.ion()
envs = [make_env() for i in range(num_envs)]
envs = SubprocVecEnv(envs) # 8 env

env = gym.make(env_name) # a single env

class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()
        
        self.critic = nn.Sequential( # network that outputs value
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.actor = nn.Sequential( # network that outputs prob of action
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        )
        
    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value


def test_env(vis=False):
    state = env.reset()
    if vis: env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if vis: env.render()
        total_reward += reward
    return total_reward


def compute_returns(next_value, rewards, masks, gamma=0.99): # calculates Q
    R = next_value
    returns = []
    for step in reversed(range(len(rewards))):
        R = rewards[step] + gamma * R * masks[step]
        returns.insert(0, R)
    return returns

def plot(frame_idx, rewards):
    plt.plot(rewards,'b-')
    plt.title('frame %s. reward: %s' % (frame_idx, rewards[-1]))
    plt.pause(0.0001)


num_inputs  = envs.observation_space.shape[0]
num_outputs = envs.action_space.n

#Hyper params:
hidden_size = 256
lr          = 1e-3
num_steps   = 5

model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
optimizer = optim.Adam(model.parameters())


max_frames   = 20000
frame_idx    = 0
test_rewards = []


state = envs.reset()

while frame_idx < max_frames:

    log_probs = []
    values    = []
    rewards   = []
    masks     = []
    entropy = 0

    # rollout trajectory
    for _ in range(num_steps):
        state = torch.FloatTensor(state).to(device) # get a state from env
        dist, value = model(state) # run the state through the network to get an action distribution and value of state

        action = dist.sample() # pick an action from the action distribution output by the model
        next_state, reward, done, _ = envs.step(action.cpu().numpy()) # take the action, and get a new state and reward

        log_prob = dist.log_prob(action) # log prob of the action
        entropy += dist.entropy().mean() # entropy
        
        log_probs.append(log_prob) # add the log prob to a list
        values.append(value) # add a list of predicted values
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device)) # add to a list of rewards
        masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
        
        state = next_state
        frame_idx += 1
        
        if frame_idx % 100 == 0:
            test_rewards.append(np.mean([test_env() for _ in range(10)]))
            plot(frame_idx, test_rewards)
            
    next_state = torch.FloatTensor(next_state).to(device)
    _, next_value = model(next_state)
    returns = compute_returns(next_value, rewards, masks) # computing returns = getting Q
    
    log_probs = torch.cat(log_probs)
    returns   = torch.cat(returns).detach()
    values    = torch.cat(values)

    advantage = returns - values # advantage function

    actor_loss  = -(log_probs * advantage.detach()).mean() # this is the eqn shown in notes for A2C = -log_prob(action) * advantage function
    critic_loss = advantage.pow(2).mean() # update the critic network (value loss), by using the mean squared loss of the advantage function
                                          # comparing reward gained from one state to the value estimated for that state

    loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy # coefficients are considered as hyperparameters

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

#test_env(True)

# link explaining entropy: https://adventuresinmachinelearning.com/cross-entropy-kl-divergence/?fbclid=IwAR3sFnj5FATQgWTReEJ6sw5yxXh0KuCzRl1QlyR_WZhZBE-8AssQzPJpPhc
# link explaining A2C: https://www.youtube.com/watch?v=O5BlozCJBSE
