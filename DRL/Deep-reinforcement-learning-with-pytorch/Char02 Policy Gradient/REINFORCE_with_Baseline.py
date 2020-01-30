import torch.nn as nn
import torch.nn.functional as F
import gym
import torch
from torch.distributions import Categorical
import torch.optim as optim
from copy import deepcopy
import argparse
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from torch.nn.utils import clip_grad_norm_
render = False

#parser = argparse.ArgumentParser(description='PyTorch REINFORCE example with baseline')
# parser.add_argument('--render', action='store_true', default=True,
#                     help='render the environment')
#args = parser.parse_args()

class Policy(nn.Module):
    def __init__(self,n_states, n_hidden, n_output):
        super(Policy, self).__init__()
        self.linear1 = nn.Linear(n_states, n_hidden)
        self.linear2 = nn.Linear(n_hidden, n_output)

        self.reward = []
        self.log_act_probs = []
        self.Gt = []
        self.sigma = []
#这是state_action_func的参数
        # self.Reward = []
        # self.s_value = []

    def forward(self, x):
        x = F.relu(self.linear1(x))
        output = F.softmax(self.linear2(x), dim= 1)
        return output



env = gym.make('CartPole-v0')
n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

# define 2 networks, one for policy and one for value function, V(s)
policy = Policy(n_states, 128, n_actions) # input = states, output = possible actions (move left, move right)
s_value_func = Policy(n_states, 128, 1) # input = each state has 4 inputs (cart and pendulum position/velocity), value of state


alpha_theta = 1e-3 # learning rate
optimizer_theta = optim.Adam(policy.parameters(), lr=alpha_theta)
# alpha_w = 1e-3  #初始化
# optimizer_w = optim.Adam(policy.parameters(), lr=alpha_w)
gamma = 0.99

seed = 1
env.seed(seed)
torch.manual_seed(seed)
live_time = []

def loop_episode():
    state = env.reset()
    if render: env.render()
    policy_loss = []
    s_value = []
    state_sequence = []
    log_act_prob = []
    for t in range(1000):
        state = torch.from_numpy(state).unsqueeze(0).float()  # turn state in to a tensor
        state_sequence.append(deepcopy(state))
        action_probs = policy(state) # get an action distribution
        m = Categorical(action_probs) 
        action = m.sample() # sample from action distribution
        m_log_prob = m.log_prob(action) # log prob of the action
        log_act_prob.append(m_log_prob) # add log prob of the action to a list
        # policy.log_act_probs.append(m_log_prob)
        action = action.item()
        next_state, re, done, _ = env.step(action) # enact the action chosen (Take a step in the environment)
        if render: env.render()
        policy.reward.append(re) # add the reward 
        if done:
            live_time.append(t)
            break
        state = next_state

    R = 0
    Gt = []

    # get Gt value: long term reward (all immediate rewards in an episode)
    for r in policy.reward[::-1]:
        R = r + gamma * R
        Gt.insert(0, R)
        # s_value_func.sigma.insert(0,sigma)
        # policy.Gt.insert(0,R)


    # update step by step
    for i in range(len(Gt)): # iterate over each episode

        # in one episode, G is the total reward obtained from that episode, and V is the state value ( I assume of the first state where that episode began?)
        G = Gt[i]
        V = s_value_func(state_sequence[i])

        # delta is your advantage function here - aaron: not sure if this is considered as an advantage function since it is technically G - V instead of Q - V
        delta = G - V

        # update value network
        # the value network is updated based on the advantage function
        alpha_w = 1e-3  # 初始化
        optimizer_w = optim.Adam(policy.parameters(), lr=alpha_w)
        optimizer_w.zero_grad()
        policy_loss_w =-delta
        policy_loss_w.backward(retain_graph = True)
        clip_grad_norm_(policy_loss_w, 0.1)
        optimizer_w.step()

        # update policy network
        # the policy network is updated based on the log_prob and advantage function
        optimizer_theta.zero_grad()
        policy_loss_theta = - log_act_prob[i] * delta
        policy_loss_theta.backward(retain_graph = True)
        clip_grad_norm_(policy_loss_theta, 0.1)
        optimizer_theta.step()

    del policy.log_act_probs[:]
    del policy.reward[:]


def plot(live_time):
    plt.ion()
    plt.grid()
    plt.plot(live_time, 'g-')
    plt.xlabel('running step')
    plt.ylabel('live time')
    plt.pause(0.000001)



if __name__ == '__main__':

    for i_episode in range(1000):
        loop_episode()
        plot(live_time)
    #policy.plot(live_time)
