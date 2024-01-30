#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
The original code is from the PyTorch DQN tutorial:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
The licence of the original part conforms to that of the site.
"""
import os
import argparse
import json
import gymnasium as gym
import math
import random
from collections import namedtuple, deque

import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class DQN:
    def __init__(self, config, n_observations, n_actions):
        self.config = config
        self.BATCH_SIZE = config['BATCH_SIZE']
        self.GAMMA = config['GAMMA']
        self.EPS_START = config['EPS_START']
        self.EPS_END = config['EPS_END']
        self.EPS_DECAY = config['EPS_DECAY']
        self.TAU = config['TAU']
        self.LR = config['LR']
        self.model_file = config['model_file'] if "model_file" in config else None
        self.n_actions = n_actions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN.DQN_nn(n_observations, n_actions).to(self.device)
        if self.model_file is not None and os.path.isfile(self.model_file):
            self.policy_net.load_state_dict(torch.load(self.model_file))
            self.policy_net.eval()
        self.target_net = DQN.DQN_nn(n_observations, n_actions).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = DQN.ReplayMemory(10000)
        if "steps_done" in config:
            self.steps_done = config['steps_done']
        else:
            self.steps_done = 0


    class ReplayMemory(object):

        def __init__(self, capacity):
            self.Transition = namedtuple('Transition',
                                         ('state', 'action', 'next_state', 'reward'))
            self.memory = deque([], maxlen=capacity)

        def push(self, *args):
            """Save a transition"""
            self.memory.append(self.Transition(*args))

        def sample(self, batch_size):
            return random.sample(self.memory, batch_size)

        def __len__(self):
            return len(self.memory)

    # DQN algorithm

    class DQN_nn(nn.Module):

        def __init__(self, n_observations, n_actions):
            super(DQN.DQN_nn, self).__init__()
            self.layer1 = nn.Linear(n_observations, 128)
            self.layer2 = nn.Linear(128, 128)
            self.layer3 = nn.Linear(128, n_actions)

        # Called with either one element to determine next action, or a batch
        # during optimization. Returns tensor([[left0exp,right0exp]...]).
        def forward(self, x):
            x = F.relu(self.layer1(x))
            x = F.relu(self.layer2(x))
            return self.layer3(x)

    def select_action(self, observation):
        observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                action = self.policy_net(observation).max(1).indices.view(1, 1)
        else:
            action = torch.tensor([[np.random.randint(0, self.n_actions)]], device=self.device, dtype=torch.long)
            pass
        return action

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = self.memory.Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def learn(self, prev_observation, action, observation, reward, terminated, truncated):
        prev_observation = torch.tensor(prev_observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward = torch.tensor([reward], device=self.device)
        done = terminated or truncated

        if terminated:
            observation = None
        else:
            observation = torch.tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)

        # Store the transition in memory
        self.memory.push(prev_observation, action, observation, reward)

        # Perform one step of the optimization (on the policy network)
        self.optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * self.TAU + target_net_state_dict[key] * (1 - self.TAU)
        self.target_net.load_state_dict(target_net_state_dict)

        return done

    def close(self):
        if self.model_file is not None:
            torch.save(self.policy_net.state_dict(), self.model_file)
        print("steps_done:", self.steps_done)

def plot_durations(episode_durations, show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())


def main():
    parser = argparse.ArgumentParser(description='DQN pytorch implementation')
    parser.add_argument('--config', type=str, default='DQN.json', metavar='N',
                        help='Model configuration (default: DQN.json')
    parser.add_argument('--num_episodes', type=int, default=600, metavar='N',
                        help='Number of training episodes (default: 600)')
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    env = gym.make("CartPole-v1")
    # Training
    # Get number of actions from gym action space
    n_actions = env.action_space.n
    # Get the number of state observations
    observation, info = env.reset()
    n_observations = len(observation)

    dqn = DQN(config, n_observations, n_actions)

    episode_durations = []
    # Training loop
    for i_episode in range(args.num_episodes):
        # Initialize the environment and get its observation
        prev_observation, info = env.reset()
        for t in count():
            action = dqn.select_action(prev_observation)
            observation, reward, terminated, truncated, _ = env.step(action.item())
            done = dqn.learn(prev_observation, action, observation, reward, terminated, truncated)
            prev_observation = observation
            if done:
                episode_durations.append(t + 1)
                plot_durations(episode_durations)
                break

    print('Complete')
    dqn.close()
    plot_durations(episode_durations, show_result=True)
    plt.ioff()
    plt.show()


if __name__ == '__main__':
    from itertools import count
    # set up matplotlib
    import matplotlib
    import matplotlib.pyplot as plt
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        from IPython import display
    plt.ion()
    main()
