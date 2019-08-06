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

## Local Evaluation

````python
import gym
import numpy as np
import real_robots

class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space
        self.action = np.zeros(action_space.shape[0])
        self.action += -np.pi*0.5

    def step(self, observation, reward, done):
        self.action += 0.4*np.pi*np.random.randn(self.action_space.shape[0])
        return self.action

result = real_robots.evaluate(
                RandomPolicy,
                intrinsic_timesteps=40,
                extrinsic_timesteps=40,
                extrinsic_trials=5,
                visualize=True,
                goals_dataset_path="./goals.npy.npz",
            )
#  NOTE : You can find a sample goals.npy.npz file at
#
#  https://aicrowd-production.s3.eu-central-1.amazonaws.com/misc/REAL-Robots/goals.npy.npz
print(result)
# {'score_2D': 0.6949320310408206, 'score_2.5D': 0, 'score_3D': 0, 'score_total': 0.23164401034694018}
```

-   Free software: MIT license

## Features

-   TODO

## Authors

-   Francesco Mannella
-   Emilio Cartoni
-   Sharada Mohanty
````
