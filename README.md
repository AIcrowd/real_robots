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
        self.action = action_space.sample()

    def step(self, observation, reward, done):
        if np.random.rand() < 0.05:
            self.action = self.action_space.sample()
        return self.action

env = gym.make("REALRobot2020-R2J3-v0")
pi = RandomPolicy(env.action_space)
env.render("human")

observation = env.reset()
reward, done = 0, False
for t in range(40):    
    action = pi.step(observation, reward, done)
    observation, reward, done, info = env.step(action)    
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
        self.action = action_space.sample()

    def step(self, observation, reward, done):
        if np.random.rand() < 0.05:
            self.action = self.action_space.sample()
        return self.action

result, detailed_scores = real_robots.evaluate(
                RandomPolicy,
                environment='R1',
                action_type='macro_action',
                n_objects=1,
                intrinsic_timesteps=1e3,
                extrinsic_timesteps=1e3,
                extrinsic_trials=3,
                visualize=False,
                goals_dataset_path='goals-REAL2020-s2020-50-1.npy.npz'
            )
# NOTE : You can find goals-REAL2020-s2020-50-1.npy.npz file in the REAL2020 Starter Kit repository
# or you can generate one using the real-robots-generate-goals command.
#
print(result)
# {'score_REAL2020': 0.06529471503519801, 'score_total': 0.06529471503519801}
print(detailed_scores)
# {'REAL2020': [0.00024387094790936833, 0.19553060745741896, 0.00010966670026571288]}
```

See also our [FAQ](FAQ.md).

-   Free software: MIT license

## Features

The REALRobot environment is a standard gym environment.  
It includes a 7DoF kuka arm with a 2DoF gripper, a table with 3 objects on it and a camera looking at the table from the top. 
For more info on the environment see [environment.md](environment.md).

## Authors

-   Francesco Mannella
-   Emilio Cartoni
-   **[Sharada Mohanty](https://twitter.com/MeMohanty)**
