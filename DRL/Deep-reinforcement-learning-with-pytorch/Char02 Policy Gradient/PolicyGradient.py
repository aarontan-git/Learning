import argparse
import gym
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.distributions import Categorical

import time

parser = argparse.ArgumentParser(description='PyTorch REINFORCE example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)

# define a policy as a neural network with 4 input (4 states for a cartpole) and 2 output (2 available action)
# the policy maps states to probabilities of actions (softmax)
class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.affine2 = nn.Linear(128, 2)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.affine2(x)
        return F.softmax(action_scores, dim=1)


policy = Policy()
# optimizer updates the weights of the NN
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state): # state = numpy array, state = array([ 0.03184929,  0.00238684, -0.02272496,  0.02353324])
    state = torch.from_numpy(state).float().unsqueeze(0) # converts the numpy array into a tensor
    probs = policy(state) # returns the probability of an action with the given state (eg. output of the NN) = tensor of 1x2
    m = Categorical(probs) # turn the probability of action (categorical = left or right) in to a distribution
    action = m.sample() # sample the action from the distribution computed previously
    policy.saved_log_probs.append(m.log_prob(action)) # log_prob returns the probability of having taken the action
    return action.item()


def finish_episode():
    R = 0
    policy_loss = []
    rewards = []
    for r in policy.rewards[::-1]: # r is the immediate reward based on every action in the episode
        R = r + args.gamma * R
        rewards.insert(0, R) # create a list of rewards
    rewards = torch.tensor(rewards) # turn the list of rewards in to a tensor
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for log_prob, reward in zip(policy.saved_log_probs, rewards):
        policy_loss.append(-log_prob * reward) # probability of the action that was taken and the reward it received
        # loss is negative to do "gradient ascent"
        # "log_prob * reward" is the policy gradient equation in notes
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum() # adds together all the rewards from the episode
    policy_loss.backward() # update the NN based on the rewards (-loss)
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]


def main():
    running_reward = 10 # what is the running reward?
    for i_episode in count(1):
        state = env.reset()
        for t in range(200):  # 10000 steps per episode, when the episode is done, done = True
            action = select_action(state) # select action
            state, reward, done, _ = env.step(action) #obtain new state (numpy array), reward, done (bool)
            if args.render:
                env.render()
                time.sleep(0.05)
            policy.rewards.append(reward)
            # if done:
            #     # env.reset()
            #     break
        env.close()

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
