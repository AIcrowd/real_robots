import gym
import numpy as np
import time
from real_robots.policy import BasePolicy

class RandomPolicy(BasePolicy):

    def __init__(self, action_space):
        self.action_space = action_space
        self.action = np.zeros(action_space.shape[0])
        self.action += -np.pi*0.5

    def step(self, observation, reward, done):
        self.action += 0.4*np.pi*np.random.randn(self.action_space.shape[0])
        return self.action


env = gym.make("REALRobot-v0")
pi = RandomPolicy(env.action_space)

env.render("human")

observation = env.reset()
reward, done = 0, False
for t in range(40):
    time.sleep(1./1000.)
    a = pi.step(observation, reward, done)
    observation, reward, done, info = env.step(a)
    print(t, reward)
