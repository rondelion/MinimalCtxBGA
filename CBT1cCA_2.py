#!/usr/bin/env python

import sys
import argparse
from collections import deque
import json

import logging
import numpy as np

import brica1
import brica1.brica_gym

import torch
import torch.utils.data
import torch.nn as nn

from tensorforce.environments import Environment
from tensorforce.agents import Agent

import gymnasium


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
        self.loss_fn = nn.BCEWithLogitsLoss() # nn.BCELoss()
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
                loss.backward()
                self.optimizer.step()
                if epoque == 0:
                    loss_sum += loss.item()
                    cnt += 1
        return min([loss_sum / cnt, 1.0])  # average loss


class NeoCortex:
    def __init__(self, in_dim, n_action, config):
        self.in_dim = in_dim
        config['ActionPredictor']['device'] = config['device']
        self.action_predictor = NeoCortex.ActionPredictor(in_dim, n_action, config['ActionPredictor'])
        self.moderator = NeoCortex.Moderator(n_action)
        self.selector = NeoCortex.Selector(n_action)
        self.uncertainty = 1.0
        self.blind = config['blind']

    class ActionPredictor:
        def __init__(self, in_dim, out_dim, config):
            super(NeoCortex.ActionPredictor, self).__init__()
            self.in_dim = in_dim
            self.out_dim = out_dim
            self.device = config['device']
            self.batch_size = config['batch_size']
            self.loss_accum_rate = config['loss_accum_rate']
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
            return output.to(self.device).detach().numpy().reshape(self.out_dim, )

        def learn(self, in_data, output):
            in_out_torch = (torch.tensor(in_data, dtype=torch.float64).float(),
                            torch.tensor(output, dtype=torch.float64).float())
            self.stream.append(in_out_torch)
            if self.cnt % self.batch_size == 0 and self.cnt != 0:
                dataset = StreamDataSet(self.stream)
                self.model.learn(dataset)
                self.stream.clear()
            self.cnt += 1

        def accum_loss(self, in_data, output):
            if in_data.dtype == np.dtype('int16'):
                in_data = np.zeros(self.in_dim, dtype='float64')
            x = in_data.reshape(1, len(in_data))
            data = torch.from_numpy(x.astype(np.float64)).float().to(self.device)
            if output.dtype == np.dtype('int16'):
                output = np.zeros(len(output), dtype='float64')
            y = output.reshape(1, len(output))
            out_data = torch.from_numpy(y.astype(np.float64)).float().to(self.device)
            prediction = self.model(data)
            # loss = min(self.model.loss_fn(out_data, prediction).item() / torch.numel(out_data), 1.0)
            loss = np.tanh(self.model.loss_fn(out_data, prediction).item()) #  / torch.numel(out_data))
            self.average_loss = loss * self.loss_accum_rate + self.average_loss * (1.0 - self.loss_accum_rate)

        def reset(self):
            self.stream.clear()
            self.cnt = 0

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
            self.selection = np.zeros(dim, dtype=int)

        def step(self, selector_input):
            max_val = np.max(selector_input)
            self.selection = np.zeros(len(selector_input), dtype=int)
            if max_val > 0.0:
                self.selection[np.argmax(selector_input)] = 1
            else:
                self.selection[np.random.randint(0, dim)] = 1  # one-hot
            return self.selection

        def get_selection(self):
            return self.selection

    def select(self, in_data):
        action_predictor_output = self.action_predictor.step(in_data)
        moderator_output = self.moderator.step(action_predictor_output)
        return self.selector.step(moderator_output)

    def release(self, go):
        return self.selector.get_selection() * go

    def learn(self, in_data, output):
        self.action_predictor.learn(in_data, output)

    def reset(self):
        self.uncertainty = self.action_predictor.average_loss
        # self.action_predictor.reset()
        if self.blind:
            self.moderator.reset(0.0)   # not certain
        else:
            self.moderator.reset(1.0 - self.uncertainty)


class BG:
    class BGEnv(Environment):
        def __init__(self, obs_dim, action_dim, state_type, num_values):
            super(Environment, self).__init__()
            state_dim = obs_dim + action_dim
            if num_values is not None:
                self.state_space = dict(type=state_type, shape=(state_dim,), num_values=num_values)
            else:
                self.state_space = dict(type=state_type, shape=(state_dim,))
            self.action_space = dict(type='int', num_values=2)
            if state_type == "int":
                self.state = np.random.randint(1, 2, state_dim, dtype=int)
            else:
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

    def __init__(self, config, learning_mode, train):
        self.init = True
        self.init_action = config['init_action']
        self.blind = config['blind']
        self.learning_mode = learning_mode
        self.train = train
        self.prev_action = 0
        if learning_mode == "rl":
            state_type = config['BG']['state_type'] if 'state_type' in config['BG'] else 'int'
            num_values = config['BG']['num_values'] if ('num_values' in config['BG'] and state_type == 'int') \
                else (2 if state_type == 'int' else None)
            self.env = Environment.create(environment=BG.BGEnv,
                                          max_episode_timesteps=train["max_steps"],
                                          # train["episode_count"]*train["max_steps"],
                                          action_dim=config['n_action'], obs_dim=config['in_dim'],
                                          state_type=state_type, num_values=num_values)
            if config['BG']['rl_agent'] == "dqn":
                self.agent = Agent.create(agent="dqn", environment=self.env,
                                          batch_size=config['BG']['rl_batch_size'],
                                          memory=train["max_steps"])  # TODO + horizon
            else:
                self.agent = Agent.create(agent=config['BG']['rl_agent'], environment=self.env,
                                          batch_size=config['BG']['rl_batch_size'])
        self.successes = 0
        self.success_rate = config['init_success_rate']
        self.sr_cycle = config['sr_cycle']
        self.sr_counter = 0
        self.one_go_per_episode = config['one_go_per_episode']

    def step(self, in_data, selection, reward, done):
        if self.blind:
            in_data = np.zeros(len(in_data), dtype=float)
        state = np.append(in_data, selection)
        action = 0
        if self.learning_mode == "rl":
            self.env.set_state(state)
            self.env.set_reward(reward)
            self.env.set_done(done)
            if self.init:
                self.agent.timestep_completed[:] = True
                state = self.env.reset()
                action = self.agent.act(states=state)
                self.prev_action = action
                self.init = False
            elif done == 1 or not self.one_go_per_episode:
                state, terminal, reward = self.env.execute(actions=self.prev_action)
                self.agent.observe(terminal=terminal, reward=reward)
                if not self.one_go_per_episode:
                    action = self.agent.act(states=state)
                    self.prev_action = action
                if done == 1:
                    self.init = True
            self.dump(self.train['dump'], reward, state, self.prev_action, done, self.train['dump_flags'])
        elif self.learning_mode == "rd":
            action = np.random.randint(0, 2)
        elif self.learning_mode == "zr":
            action = 0
        if not self.blind and np.max(in_data) == 0:
            return 0
        # success rate
        if done == 1:
            self.successes += reward
            self.sr_counter += 1
            if self.sr_counter % self.sr_cycle == 0 and self.sr_counter != 0:
                self.success_rate = self.successes / self.sr_cycle
                self.successes = 0
        return action

    def reset(self):
        if self.learning_mode == "rl":
            self.env.reset()
        self.init = True
        self.prev_action = 0

    @staticmethod
    def dump(dump, reward, state, action, done, dump_flags):
        if dump is not None and "b" in dump_flags:
            dump.write("state: {0}, rl_go {1}, reward: {2}, done: {3}\n".format(state, action, reward, done))


class CBT1Component(brica1.brica_gym.Component):

    def __init__(self, learning_mode, train, config):
        super().__init__()
        self.in_dim = config['in_dim']
        self.n_action = config['n_action']  # number of action choices
        self.make_in_port('observation', self.in_dim)
        self.make_in_port('reward', 1)
        self.make_in_port('done', 1)
        self.make_out_port('action', self.n_action)
        self.make_in_port('token_in', 1)
        self.make_out_port('token_out', 1)
        self.make_out_port('done', 1)
        self.token = 0
        self.learning_mode = learning_mode
        self.neocortex_learn = config['neocortex_learn']
        self.one_go_per_episode = config['one_go_per_episode']
        self.blind = config['blind']
        config['NeoCortex']['blind'] = config['blind']
        config['NeoCortex']['device'] = config['device']
        self.neoCortex = NeoCortex(self.in_dim, self.n_action, config['NeoCortex'])
        self.bg = BG(config, learning_mode, train)
        self.go = 0
        self.gone = False
        self.use_success_rate = config['use_success_rate']
        self.dump = train['dump']
        self.dump_learn = config['dump_learn']
        self.learning_dump = config['learning_dump'] if 'learning_dump' in config else None
        self.go_cost = config['go_cost']

    def fire(self):
        done = self.get_in_port('done').buffer[0]
        in_data = self.get_in_port('observation').buffer
        reward = self.get_in_port('reward').buffer[0] - self.go * self.go_cost
        selection = self.neoCortex.select(in_data) # feed the selector
        self.go = self.bg.step(in_data, selection, reward, done)
        self.results['action'] = self.neoCortex.release(self.go)
        if not self.one_go_per_episode:
            self.neoCortex.reset()
        if self.go == 1 and not (self.one_go_per_episode and self.gone):
            if self.neocortex_learn:
                action = self.results['action']
                self.neoCortex.action_predictor.accum_loss(in_data, action)
                self.neoCortex.learn(in_data, action)
            self.gone = True
            if self.dump is not None and "b" in self.bg.train['dump_flags']:
                self.dump.write("Gone\n")
        self.results['done'] = np.array([done])

    def reset(self):
        self.token = 0
        self.neoCortex.reset()
        self.bg.reset()
        if self.use_success_rate:
            self.neoCortex.moderator.reset(self.bg.success_rate)
        self.go = 0
        self.gone = False
        if self.learning_mode == "rd":
            self.neoCortex.moderator.use_prediction = 0
        self.results['token_out'] = np.array([0])
        self.results['done'] = np.array([0])
        self.results['action'] = np.array([0])

    def close(self):
        pass

class CognitiveArchitecture(brica1.Module):
    def __init__(self, rl, train, config):
        super(CognitiveArchitecture, self).__init__()
        self.in_dim = config['in_dim']
        self.n_action = config['n_action']  # number of action choices
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
    parser.add_argument('mode', help='0: zero output, 1:random act, 2: reinforcement learning',
                        choices=['0', '1', '2'])
    parser.add_argument('--dump', help='dump file path')
    parser.add_argument('--episode_count', type=int, default=1, metavar='N',
                        help='Number of training episodes (default: 1)')
    parser.add_argument('--max_steps', type=int, default=20, metavar='N',
                        help='Max steps in an episode (default: 20)')
    parser.add_argument('--config', type=str, default='CBT1CA2.json', metavar='N',
                        help='Model configuration (default: CBT1CA2.json')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dump_flags', type=str, default="",
                        help='m:main, b:bg, o:obs, p:predictor')
    parser.add_argument('--use-cuda', default=False,
                        help='uses CUDA for training')
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

    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    config["device"] = device

    env = gymnasium.make(config['env']['name'], config=config['env'])

    md = args.mode
    model = None
    if md == "0":
        model = CognitiveArchitecture("zr", train, config)
    elif md == "1":  # random act
        model = CognitiveArchitecture("rd", train, config)
    elif md == "2":  # act by reinforcement learning
        train['rl_agent'] = config['BG']['rl_agent']
        train['rl_batch_size'] = config['BG']['rl_batch_size']
        model = CognitiveArchitecture("rl", train, config)

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
            current_token = agent.get_out_port('token_out').buffer[0]
            if last_token + 1 == current_token:
                last_token = current_token
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
                    if model.cbt1.one_go_per_episode and model.cbt1.gone:
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
                agent.env.done = False
                break
    print("Close")
    if dump is not None:
        dump.close()
    env.close()


if __name__ == '__main__':
    main()
