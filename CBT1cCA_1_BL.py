#!/usr/bin/env python

import sys
import argparse
from collections import deque
import json
import ast

# from gym import wrappers

import logging
import numpy as np

import brica1
import brica1.brica_gym

import torch
import torch.utils.data
import torch.nn as nn
# from torch.utils.tensorboard import SummaryWriter

from tensorforce.environments import Environment
from tensorforce.agents import Agent

import gym

import brical
import CBT1cCA_1


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
    parser.add_argument('--brical', type=str, default='CBT1CA.brical.json', metavar='N',
                        help='a BriCAL json file')
    args = parser.parse_args()

    with open(args.config) as config_file:
        config = json.load(config_file)

    nb = brical.NetworkBuilder()
    f = open(args.brical)
    nb.load_file(f)
    if not nb.check_consistency():
        sys.stderr.write("ERROR: " + args.brical + " is not consistent!\n")
        exit(-1)

    if not nb.check_grounding():
        sys.stderr.write("ERROR: " + args.brical + " is not grounded!\n")
        exit(-1)
    
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

    md = args.mode
    model = None
    if md == "1":   # random act
        nb.unit_dic['CBT1CA.CBT1Component'].__init__("rd", train, config)
    elif md == "2":  # act by reinforcement learning
        train['rl_agent'] = config['BG']['rl_agent']
        train['rl_batch_size'] = config['BG']['rl_batch_size']
        nb.unit_dic['CBT1CA.CBT1Component'].__init__("rl", train, config)
    elif md == "3":  # act by custom frequency learning
        nb.unit_dic['CBT1CA.CBT1Component'].__init__("fl", train, config)

    nb.make_ports()

    agent_builder = brical.AgentBuilder()
    model = nb.unit_dic['CBT1CA.CognitiveArchitecture']
    agent = agent_builder.create_gym_agent(nb, model, env)

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
                    if nb.unit_dic['CBT1CA.CBT1Component'].gone:
                        go_count += 1
                        reward_go_sum += agent.get_in_port("reward").buffer[0]
                    reward_sum += agent.get_in_port("reward").buffer[0]
                    if dump_counter % dump_cycle == 0 and dump_counter != 0:
                        if go_count != 0:
                            reward_per_go = reward_go_sum / go_count
                        else:
                            reward_per_go = 0.0
                        average_loss = nb.unit_dic['CBT1CA.CBT1Component'].neoCortex.action_predictor.average_loss
                        dump.write("{0}: avr. reward: {1:.2f}\treward/go: {2:.2f}\tloss: {3:.2f}\n".
                                   format(dump_counter // dump_cycle,
                                          reward_sum / dump_cycle,
                                          reward_per_go,
                                          average_loss))
                        reward_sum = 0.0
                        reward_go_sum = 0.0
                        go_count = 0
                    dump_counter += 1
                nb.unit_dic['CBT1CA.CBT1Component'].reset()
                agent.env.reset()
                agent.env.done = False
                break
    print("Close")
    if dump is not None:
        dump.close()
    env.close()


if __name__ == '__main__':
    main()
