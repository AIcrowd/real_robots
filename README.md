# real-robots

![https://travis-ci.com/AIcrowd/real_robots.svg?branch=master](https://travis-ci.com/AIcrowd/real_robots.svg?branch=master)

<TABLE " width="100%" BORDER="0">
<TR>
<TD><img src="https://i.imgur.com/ORXaKBB.gif" alt="demo0" width="100%"></TD>
<TD><img src="https://i.imgur.com/w66lz4L.gif" alt="demo1" width="100%"></TD>
<TD><img src="https://i.imgur.com/oYARyZV.gif" alt="demo1" width="100%"></TD>
</TR>
</TABLE>

Robots that learn to interact with the environment autonomously

## Installation

```bash
pip install -U real_robots
```

If everything went well, then you should be able to run :

```
real-robots-demo
```

and it should (eventually) open up a small window with a little robotic arm
doing random stuff.

## Usage

```python
import gym
import numpy as np
import time
import real_robots
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
    action = pi.step(observation, reward, done)
    observation, reward, done, info = env.step(action)
    print(t, reward)
```

## Local Evaluation

```python
import gym
import numpy as np
import real_robots
from real_robots.policy import BasePolicy

class RandomPolicy(BasePolicy):
    def __init__(self, action_space):
        self.action_space = action_space
        self.action = np.zeros(action_space.shape[0])
        self.action += -np.pi*0.5

    def step(self, observation, reward, done):
        self.action += 0.4*np.pi*np.random.randn(self.action_space.shape[0])
        return self.action

result, detailed_scores = real_robots.evaluate(
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
# {'score_total': 0.2327459533906755, 'score_2D': 0.6982378601720265, 'score_2.5D': 0, 'score_3D': 0}
print(detailed_scores)
# {'2D': [0.6646365565451159, 0.6632591441787807, 0.7569003923985664, 0.7167885964780916, 0.6896046112595778]}
```

See also our [FAQ](https://github.com/AIcrowd/real_robots/blob/master/FAQ.md).

-   Free software: MIT license

## Features

The REALRobot environment is a standard gym environment.  
It includes a 7DoF kuka arm with a 2DoF gripper, a table with 3 objects on it and a camera looking at the table from the top. 
For more info on the environment see [environment.md](https://github.com/AIcrowd/real_robots/blob/master/environment.md).

## Authors

-   Francesco Mannella
-   Emilio Cartoni
-   **[Sharada Mohanty](https://twitter.com/MeMohanty)**
