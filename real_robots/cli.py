# -*- coding: utf-8 -*-

"""Console script for real_robots."""
import sys
import click
import numpy as np
import gym
from tqdm.auto import trange


class RandomPolicy:
    def __init__(self, action_space):
        self.action_space = action_space
        self.action = np.zeros(action_space.shape[0])
        self.action += -np.pi*0.5

    def step(self, observation, reward, done):
        self.action += 0.4*np.pi*np.random.randn(self.action_space.shape[0])
        return self.action


def run_episode(env, pi, visualize=False):
    steps = 20
    if visualize:
        env.render("human")
        steps = 200
    observation = env.reset()
    reward, done = 0, False
    for t in trange(steps, unit=" steps "):
        action = pi.step(observation, reward, done)
        observation, reward, done, info = env.step(action)


@click.command()
def demo(args=None):
    "Simple demo script to test that everything is installed "
    "and running fine"
    # click.echo("Replace this message by putting your code into "
    #            "real_robots.cli.main")
    # click.echo("See click documentation at http://click.pocoo.org/")
    click.echo(
        """
#####################################################################################################################
#####################################################################################################################
.______       _______     ___       __         .______        ______   .______     ______   .___________.    _______.
|   _  \     |   ____|   /   \     |  |        |   _  \      /  __  \  |   _  \   /  __  \  |           |   /       |
|  |_)  |    |  |__     /  ^  \    |  |        |  |_)  |    |  |  |  | |  |_)  | |  |  |  | `---|  |----`  |   (----`
|      /     |   __|   /  /_\  \   |  |        |      /     |  |  |  | |   _  <  |  |  |  |     |  |        \   \    
|  |\  \----.|  |____ /  _____  \  |  `----.   |  |\  \----.|  `--'  | |  |_)  | |  `--'  |     |  |    .----)   |   
| _| `._____||_______/__/     \__\ |_______|   | _| `._____| \______/  |______/   \______/      |__|    |_______/    
#####################################################################################################################
#####################################################################################################################
        """ # noqa
    )
    click.echo("1) Testing setup without visualisation : ")
    env = gym.make("REALRobot-v0")
    pi = RandomPolicy(env.action_space)
    run_episode(env, pi)
    click.echo("2) Testing setup with visualisation : ")
    env = gym.make("REALRobot-v0")
    run_episode(env, pi, visualize=True)
    click.echo("################ All Good \m/ !! Best of Luck !! ################")  # noqa: E501,W291
    return 0


if __name__ == "__main__":
    sys.exit(demo())  # pragma: no cover
