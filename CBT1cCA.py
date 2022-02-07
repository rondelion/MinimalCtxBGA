#!/usr/bin/env python

import sys
import argparse
from collections import deque
import json
import ast

# from gym import wrappers

import logging
import numpy as np

import brica1.brica_gym

import torch
import torch.utils.data
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

from tensorforce.environments import Environment
from tensorforce.agents import Agent

import gym


class StreamDataSet(torch.utils.data.IterableDataset):
    def __init__(self, stream):
        super(StreamDataSet).__init__()
        self.stream = stream

    def __iter__(self):
        return iter(self.stream)


class Perceptron(torch.nn.Module):
    def __init__(self, in_dim, intra_dim, out_dim, device, epochs):
        super(Perceptron, self).__init__()
        self.seq = torch.nn.Sequential(nn.Linear(in_dim, intra_dim), nn.Sigmoid(),
                                       nn.Linear(intra_dim, out_dim), nn.Sigmoid())
        self.optimizer = None
        self.loss_fn = nn.BCELoss()
        self.device = device
        self.epochs = epochs

    def forward(self, x):
        return self.seq(x)

    def set_optimizer(self, lr):
        self.optimizer = torch.optim.SGD(self.parameters(), lr=lr)

    def learn(self, dataset):
        self.train()
        loss_sum = 0.0
        cnt = 0
        for epoque in range(self.epochs):
            train_loader = torch.utils.data.DataLoader(dataset, batch_size=10)
            for idx, (x, y) in enumerate(train_loader):
                self.optimizer.zero_grad()
                output = self(x)
                loss = self.loss_fn(output, y)
                loss_sum += loss.item()
                loss.backward()
                self.optimizer.step()
                cnt += 1
        return min([loss_sum / cnt, 1.0])    # average loss


class NeoCortex:
    def __init__(self, in_dim, n_action, config):
        self.in_dim = in_dim
        self.action_predictor = NeoCortex.ActionPredictor(in_dim, n_action, config['ActionPredictor'])
        self.moderator = NeoCortex.Moderator(n_action)
        self.selector = NeoCortex.Selector(n_action)
        self.uncertainty = 1.0

    class ActionPredictor:
        def __init__(self, in_dim, out_dim, config):
            super(NeoCortex.ActionPredictor, self).__init__()
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.device = None
            self.batch_size = config['batch_size']
            self.model = Perceptron(in_dim, config['intra_dim'], out_dim, self.device, config['epochs'])
            self.model.set_optimizer(config['lr'])
            self.stream = deque()
            self.cnt = 0
            self.average_loss = 1.0
            self.dump = config['dump']

        def step(self, in_data):
            if in_data.dtype == np.dtype('int16'):
                in_data = np.zeros(self.in_dim, dtype='float64')
            x = in_data.reshape(1, len(in_data))
            data = torch.from_numpy(x.astype(np.float64)).float().to(self.device)
            output = self.model(data)
            return output.to(self.device).detach().numpy().reshape(self.out_dim,)

        def learn(self, in_data, output):
            in_out_torch = (torch.tensor(in_data, dtype=torch.float64).float(),
                            torch.tensor(output, dtype=torch.float64).float())
            in_out = np.append(in_data, output)
            self.stream.append(in_out_torch)
            if self.cnt % self.batch_size == 0 and self.cnt != 0:
                dataset = StreamDataSet(self.stream)
                self.average_loss = self.model.learn(dataset)
                self.stream.clear()
            self.cnt += 1

    class Moderator:
        def __init__(self, dim):
            self.dim = dim
            self.random = np.random.rand(self.dim)
            self.use_prediction = 0

        def step(self, prediction):
            return prediction if self.use_prediction else self.random

        def reset(self, prediction_certainty):
            self.random = np.random.rand(self.dim)
            self.use_prediction = np.random.binomial(1, prediction_certainty)

    class Selector:
        def __init__(self, dim):
            self.selection = np.random.rand(dim)

        def step(self, selector_input, go):
            max_val = np.max(selector_input)
            self.selection = 1 * (np.where(selector_input == max_val, max_val, 0.0) > 0)    # step function
            return self.selection * go

        def get_selection(self):
            return self.selection

    def step(self, go, in_data):
        action_predictor_output = self.action_predictor.step(in_data)
        moderator_output = self.moderator.step(action_predictor_output)
        output = self.selector.step(moderator_output, go)
        return output

    def learn(self, in_data, output):
        self.action_predictor.learn(in_data, output)

    def get_selection(self):
        return self.selector.get_selection()

    def reset(self):
        self.uncertainty = self.action_predictor.average_loss
        self.moderator.reset(1.0 - self.uncertainty)


class BG:
    class BGEnv(Environment):
        def __init__(self, obs_dim, action_dim):
            super(Environment, self).__init__()
            state_dim = obs_dim + action_dim
            self.state_space = dict(type='int', shape=(state_dim,), num_values=2)
            self.action_space = dict(type='int', num_values=2)
            self.state = np.random.random(size=(state_dim,))
            self.reset()
            self.reward = 0.0
            self.done = False
            self.info = {}

        def set_state(self, state):
            self.state = state

        def states(self):
            return self.state_space

        def actions(self):
            return self.action_space

        def reset(self):
            return self.state

        def execute(self, actions):
            if self.done == 1 or self.done:
                terminal = True
            else:
                terminal = False
            return self.state, terminal, self.reward

        def set_reward(self, reward):
            self.reward = reward

        def set_done(self, done):
            self.done = done

    class FLActor:
        def __init__(self, train, config):
            self.state = None
            self.success_count = {}
            self.state_go_count = {}
            self.gone = False
            self.dump = train['dump']
            self.dump_flags = train['dump_flags']
            if config['use_dump']:
                try:
                    with open(config['learning_dump']) as dump_file:
                        self.success_count, self.state_go_count = ast.literal_eval(dump_file.read())
                except:
                    print('learning_dump file error', file=sys.stderr)
                    sys.exit(1)

        def act(self, state):
            st = str(state)
            success_count = self.success_count[st] if st in self.success_count else 0
            state_go_count = self.state_go_count[st] if st in self.state_go_count else 0
            go_probability = (success_count + 1) / (state_go_count + 2)  # average beta distribution
            action = 0 if self.gone else np.random.binomial(1, go_probability)
            if action > 0:
                if not self.gone:
                    self.gone = True
                    self.state = state  # first go state
            if self.dump is not None and "b" in self.dump_flags:
                self.dump.write("state: {0}, go {1}, suc.cnt: {2}, state go cnt: {3}\n"
                                .format(state, action, success_count, state_go_count))
            return action

        def learn(self, reward):
            st = str(self.state)
            self.state = None
            if self.gone:
                if st not in self.state_go_count:
                    self.state_go_count[st] = 1
                else:
                    self.state_go_count[st] += 1
                if reward > 0:
                    if st not in self.success_count:
                        self.success_count[st] = 1
                    else:
                        self.success_count[st] += 1
                self.gone = False

        def reset(self):
            self.gone = False
            self.state = None

    def __init__(self, config, learning_mode, train):
        self.init = True
        self.init_action = config['init_action']
        self.reward_sum = 0.0
        self.learning_mode = learning_mode
        self.rl_go_sum = 0
        self.train = train
        self.prev_action = 0
        if learning_mode == "rl":
            self.env = Environment.create(environment=BG.BGEnv,
                                          max_episode_timesteps=train["episode_count"]*train["max_steps"],
                                          action_dim=config['n_action'], obs_dim=config['in_dim'])
            self.agent = Agent.create(agent=train['rl_agent'], environment=self.env, batch_size=train['rl_batch_size'])
        elif learning_mode == "fl":
            self.fl_actor = BG.FLActor(train, config)
        self.successes = 0
        self.success_rate = config['init_success_rate']
        self.sr_cycle = config['sr_cycle']
        self.sr_counter = 0

    def step(self, in_data, selection, reward, done):
        state = np.append(in_data, selection)
        action = 0
        if self.learning_mode == "rl":
            self.env.set_state(state)
            self.env.set_reward(reward)
            self.env.set_done(done)
            if self.init:
                self.rl_go_sum = 0
                self.agent.timestep_completed[:] = True
                state = self.env.reset()
                action = self.agent.act(states=state)
                self.prev_action = action
                self.init = False
            elif done == 1:
                state, terminal, reward = self.env.execute(actions=self.prev_action)
                self.agent.observe(terminal=terminal, reward=reward)
                self.init = True
            rl_go = self.prev_action
            self.rl_go_sum += rl_go * in_data.max()
            self.dump(self.train['dump'], reward, state, self.prev_action, done, self.train['dump_flags'])
        if self.learning_mode == "fl":
            if self.init:
                if self.init_action:
                    self.prev_action = self.fl_actor.act(state)
                self.init = False
            if done != 1:
                if not self.init_action:
                    action = self.fl_actor.act(state)
                    self.prev_action = action
            else:
                self.fl_actor.learn(reward)
                self.rl_go_sum = 0
                self.init = True
                if self.train['dump'] is not None and "b" in self.train['dump_flags']:
                    self.train['dump'].write("reward: {0}\n".format(reward))
            rl_go = self.prev_action
            self.rl_go_sum += rl_go * in_data.max()
        if self.learning_mode == "rd":
            action = np.random.randint(0,2)
            self.rl_go_sum = action
        if np.max(in_data) == 0:
            return 0
        # success rate
        if done == 1:
            self.successes += reward
            self.sr_counter += 1
            if self.sr_counter % self.sr_cycle == 0 and self.sr_counter != 0:
                self.success_rate = self.successes / self.sr_cycle
                self.successes = 0
        if self.init_action:
            go = 1 if self.rl_go_sum >= 1 else 0
        else:
            go = 1 if action > 0 else 0
        return go

    def reset(self):
        if self.learning_mode == "rl":
            self.env.reset()
        if self.learning_mode == "fl":
            self.fl_actor.reset()

    @staticmethod
    def dump(dump, reward, state, action, done, dump_flags):
        if dump is not None and "b" in dump_flags:
            dump.write("state: {0}, rl_go {1}, reward: {2}, done: {3}\n".format(state, action, reward, done))


class CBT1Component(brica1.Component):

    def __init__(self, learning_mode, train, config):
        super().__init__()
        self.in_dim = config['in_dim']
        self.n_action = config['n_action']    # number of action choices
        self.make_in_port('observation', self.in_dim)
        self.make_in_port('reward', 1)
        self.make_in_port('done', 1)
        self.make_out_port('action', self.n_action)
        self.make_in_port('token_in', 1)
        self.make_out_port('token_out', 1)
        self.make_out_port('done', 1)
        self.token = 0
        self.prev_actions = 0
        self.learning_mode = learning_mode
        self.init = True
        self.neoCortex = NeoCortex(self.in_dim, self.n_action, config['NeoCortex'])
        self.bg = BG(config, learning_mode, train)
        self.go = 0
        self.gone = False
        self.use_success_rate = config['use_success_rate']
        self.dump = train['dump']
        self.dump_learn = config['dump_learn']
        self.learning_dump = config['learning_dump'] if 'learning_dump' in config else None

    def fire(self):
        if self.init:  # TODO initialization error somewhere
            self.results['reward'] = np.array([0.0])
            done = 0
            self.init = False
        else:
            self.results['reward'] = self.inputs['reward']
            done = self.get_in_port('done').buffer[0]
        if self.token + 1 == self.inputs['token_in'][0] or done == 1:
            in_data = self.get_in_port('observation').buffer
            reward = self.get_in_port('reward').buffer[0]
            self.neoCortex.step(self.go, in_data)   # feed the selector
            self.go = self.bg.step(in_data, self.neoCortex.get_selection(), reward, done)
            self.results['action'] = self.neoCortex.step(self.go, in_data)
            self.token = self.inputs['token_in'][0]
        if self.go == 1 and not self.gone:
            self.neoCortex.learn(in_data, self.results['action'])
            self.gone = True
            if self.dump is not None and "b" in self.bg.train['dump_flags']:
                self.dump.write("Gone\n")
        self.results['done'] = np.array([done])
        if done == 1:
            self.results['token_out'] = np.array([0])
            self.results['action'] = np.zeros(self.n_action, dtype=np.int) # np.array([0])
        else:
            self.results['token_out'] = self.inputs['token_in']

    def reset(self):
        self.token = 0
        self.init = True
        self.inputs['token_in'] = np.array([0])
        self.inputs['reward'] = np.array([0.0])
        self.results['token_out'] = np.array([0])
        self.get_in_port('token_in').buffer = self.inputs['token_in']
        self.get_in_port('reward').buffer = self.inputs['reward']
        self.get_out_port('token_out').buffer = self.results['token_out']
        self.neoCortex.reset()
        if self.use_success_rate:
            self.neoCortex.moderator.reset(self.bg.success_rate)
        self.go = 0
        self.gone = False
        if self.learning_mode == "rd":
            self.neoCortex.moderator.use_prediction = 0

    def close(self):
        if self.learning_mode == "fl" and self.dump_learn and self.learning_dump is not None:
            try:
                f = open(self.learning_dump, 'w')
            except OSError as e:
                print(e)
            else:
                f.write("{0},".format(self.bg.fl_actor.success_count))
                f.write("{0}".format(self.bg.fl_actor.state_go_count))
                f.close()

class CognitiveArchitecture(brica1.Module):
    def __init__(self, rl, train, config):
        super(CognitiveArchitecture, self).__init__()
        self.in_dim = config['in_dim']
        self.n_action = config['n_action']    # number of action choices
        self.make_in_port('observation', self.in_dim)
        self.get_in_port('observation').buffer = np.zeros(self.in_dim)
        self.make_in_port('reward', 1)
        self.make_in_port('done', 1)
        self.make_out_port('action', self.n_action)
        self.make_in_port('token_in', 1)
        self.make_out_port('token_out', 1)
        self.cbt1 = CBT1Component(rl, train, config)
        self.add_component('cbt1', self.cbt1)
        self.cbt1.alias_in_port(self, 'observation', 'observation')
        self.cbt1.alias_in_port(self, 'token_in', 'token_in')
        self.cbt1.alias_in_port(self, 'reward', 'reward')
        self.cbt1.alias_in_port(self, 'done', 'done')
        self.cbt1.alias_out_port(self, 'action', 'action')
        self.cbt1.alias_out_port(self, 'token_out', 'token_out')


def main():
    parser = argparse.ArgumentParser(description='BriCA Minimal Cognitive Architecture with Gym')
    parser.add_argument('mode', help='1:random act, 2: reinforcement learning, 3: frequency learning',
                        choices=['1', '2', '3'])
    parser.add_argument('--dump', help='dump file path')
    parser.add_argument('--episode_count', type=int, default=1, metavar='N',
                        help='Number of training episodes (default: 1)')
    parser.add_argument('--max_steps', type=int, default=20, metavar='N',
                        help='Max steps in an episode (default: 20)')
    parser.add_argument('--config', type=str, default='CBT1CA.json', metavar='N',
                        help='Model configuration (default: CBT1CA.json')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dump_flags', type=str, default="",
                        help='m:main, b:bg, o:obs, p:predictor')
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    train = {"episode_count": args.episode_count, "max_steps": args.max_steps, "dump_flags": args.dump_flags}

    if args.dump is not None and args.dump_flags != "":
        try:
            dump = open(args.dump, mode='w')
        except IOError:
            print('Error: No dump path specified', file=sys.stderr)
            sys.exit(1)
    else:
        dump = None
    train["dump"] = dump
    if "e" in args.dump_flags:
        config['env']['dump'] = dump
    else:
        config['env']['dump'] = None
    if "p" in args.dump_flags:
        config['NeoCortex']['ActionPredictor']['dump'] = dump
    else:
        config['NeoCortex']['ActionPredictor']['dump'] = None

    env = gym.make(config['env']['name'], config=config['env'])
    # out_dir = config['gym_monitor_out_dir']
    # env = wrappers.Monitor(env, out_dir, force=True)

    md = args.mode
    model = None
    if md == "1":   # random act
        model = CognitiveArchitecture("rd", train, config)
    elif md == "2":  # act by reinforcement learning
        train['rl_agent'] = config['BG']['rl_agent']
        train['rl_batch_size'] = config['BG']['rl_batch_size']
        model = CognitiveArchitecture("rl", train, config)
    elif md == "3":  # act by custom frequency learning
        model = CognitiveArchitecture("fl", train, config)

    agent = brica1.brica_gym.GymAgent(model, env)
    scheduler = brica1.VirtualTimeSyncScheduler(agent)

    dump_cycle = config["dump_cycle"]
    dump_counter = 0
    reward_sum = 0.0
    reward_go_sum = 0.0
    go_count = 0
    for i in range(train["episode_count"]):
        last_token = 0
        for j in range(train["max_steps"]):
            scheduler.step()
            # time.sleep(config["sleep"])
            current_token = agent.get_out_port('token_out').buffer[0]
            if last_token + 1 == current_token:
                last_token = current_token
                # env.render()
                if "o" in train["dump_flags"]:
                    dump.write(str(agent.get_in_port("observation").buffer.tolist()) + '\n')
            if agent.env.done:
                agent.env.flush = True
                while True:
                    scheduler.step()
                    if agent.get_in_port("done").buffer[0] == 1:
                        scheduler.step()
                        break
                if dump is not None and "m" in args.dump_flags:
                    if model.cbt1.gone:
                        go_count += 1
                        reward_go_sum += agent.get_in_port("reward").buffer[0]
                    reward_sum += agent.get_in_port("reward").buffer[0]
                    if dump_counter % dump_cycle == 0 and dump_counter != 0:
                        if go_count != 0:
                            reward_per_go = reward_go_sum / go_count
                        else:
                            reward_per_go = 0.0
                        average_loss = model.cbt1.neoCortex.action_predictor.average_loss
                        dump.write("{0}: avr. reward: {1:.2f}\treward/go: {2:.2f}\tloss: {3:.2f}\n".
                                   format(dump_counter // dump_cycle,
                                          reward_sum / dump_cycle,
                                          reward_per_go,
                                          average_loss))
                        reward_sum = 0.0
                        reward_go_sum = 0.0
                        go_count = 0
                    dump_counter += 1
                model.cbt1.reset()
                agent.env.reset()
                # agent.env.out_ports['token_out'] = np.array([0])
                agent.env.done = False
                break
    print("Close")
    if dump is not None:
        dump.close()
    env.close()


if __name__ == '__main__':
    main()
