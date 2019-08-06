# real-robots

Robots that learn to interact with the environment autonomously

## Installation

```bash
pip install -U real_robots
```

## Usage

```python
import gym
import numpy as np
import time
import real_robots


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space
        self.action = np.zeros(action_space.shape[0])
        self.action += -np.pi*0.5

    def act(self):
        self.action += 0.4*np.pi*np.random.randn(self.action_space.shape[0])
        return self.action

env = gym.make("REALRobot-v0")
pi = RandomPolicy(env.action_space)
env.render("human")

observation = env.reset()
for t in range(40):
    time.sleep(1./1000.)
    a = pi.act()
    observation, reward, done, info = env.step(a)
    print(t, reward)
```

-   Free software: MIT license

## Features

-   TODO

## Authors

-   Francesco Mannella
-   Emilio Cartoni
-   Sharada Mohanty
