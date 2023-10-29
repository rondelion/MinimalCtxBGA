import gym
import numpy as np
import sys

class CBT1Env(gym.Env):
    def __init__(self, config):
        self.action_space = gym.spaces.Discrete(4)
        # LOW = [0, 0, 0]
        # HIGH = [1, 1, 1]
        # self.observation_space = gym.spaces.Box(low=0, high=1, shape=(3,))
        self.observation_space = gym.spaces.Box(low=np.array([0, 0, 0], dtype='float32'), high=np.array([1, 1, 1], dtype='float32'))
        self.observation = 0
        self.observations = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 1], [0, 1, 1], [1, 0, 0]])
        self.interaction_period = config["interaction_period"]
        self.delay = config["delay"]
        self.penalty = config["penalty"]
        self.answered = False
        self.done = False
        self.count = 0
        self.action_count = [0, 0]
        self.correct = False

    def reset(self):
        self.answered = False
        self.done = False
        self.count = 0
        self.action_count = [0, 0]
        self.correct = False
        self.observation = np.random.randint(1, 5)
        # print("CBT1Env reset")
        return self.observations[self.observation]

    def step(self, action):
        reward = 0
        self.count += 1
        if self.count <= self.interaction_period + 1:
            if action == 1 or action == 2:
                self.action_count[action-1] += 1
            if (self.observation == 1 or self.observation == 2) and action == 2:
                self.correct = True
            elif (self.observation == 3 or self.observation == 4) and action == 1:
                self.correct = True
        if self.count > 0 and action > 0:
            self.answered = True
        if self.count > self.interaction_period:
            observation = 0
        else:
            observation = self.observation
        if self.count == self.delay + self.interaction_period:
            if self.correct and not (self.action_count[0] > 0 and self.action_count[1] > 0):
                reward = 1
            elif self.action_count[0] > 0 or self.action_count[1] > 0:
                reward = self.penalty
            self.done = True
        # sys.stdout.write("obs: {0}({1}), action: {2}, correct: {3}, done: {4}, count:{5}\n".format(self.observations[self.observation],
        #                                                                                  self.observation, action,
        #                                                                                  self.correct, self.done,
        #                                                                                  self.count))
        return self.observations[observation], reward, self.done, {}

    def render(self):
        pass


def main():
    config = {"interaction_period": 1, "delay": 1}
    env = CBT1Env(config)
    for i in range(20):
        obs = env.reset()
        while True:
            if not np.array_equal(obs, np.array([0, 0, 0])):
                action = np.random.randint(1, 4)
                print(obs, action, end=",")
            else:
                action = 0
            obs, reward, done, info = env.step(action)
            if done:
                print("reward:", reward)
                break


if __name__ == '__main__':
    main()
