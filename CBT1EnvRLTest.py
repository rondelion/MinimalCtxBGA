import sys
import json

from tensorforce.environments import Environment
from tensorforce.agents import Agent

import CBT1Env as cbt1env

# config = {"interaction_period": 1, "delay": 2, "penalty": -0.7}


def train(n, agent, environment):
    for _ in range(n):
        states = environment.reset()
        terminal = False
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            agent.observe(terminal=terminal, reward=reward)


def evaluate(n, agent, environment):
    sum_rewards = 0.0
    for _ in range(n):
        states = environment.reset()
        internals = agent.initial_internals()
        terminal = False
        while not terminal:
            actions, internals = agent.act(states=states, internals=internals, independent=True)
            states, terminal, reward = environment.execute(actions=actions)
            sum_rewards += reward
    return sum_rewards / n


def main():
    with open(sys.argv[1]) as config_file:
        config = json.load(config_file)

    # Create agent and environment
    env = cbt1env.CBT1Env(config)
    environment = Environment.create(environment=env, max_episode_timesteps=10)
    agent = Agent.create(agent=config['algorithm'], environment=environment, batch_size=10)

    for _ in range(100):
        train(100, agent, environment)
        avr_rewards = evaluate(100, agent, environment)
        print('Mean episode reward:', avr_rewards)

    # Close agent and environment
    agent.close()
    environment.close()


if __name__ == '__main__':
    main()
