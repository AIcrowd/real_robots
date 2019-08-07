#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for `real_robots` package."""

from click.testing import CliRunner

import real_robots  # noqa
from real_robots import cli
from real_robots import generate_goals
import gym
import numpy as np
from real_robots.policy import BasePolicy


def test_command_line_interface():
    """Test the CLI."""
    runner = CliRunner()
    result = runner.invoke(cli.demo)
    assert result.exit_code == 0
    help_result = runner.invoke(cli.demo, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output


def test_goal_generation():
    """Test goal generation."""
    runner = CliRunner()
    result = runner.invoke(generate_goals.main, 
                           ['32', '0', '0', '0', '0', '0', '0', '1'])
    assert result.exit_code == 0
    help_result = runner.invoke(generate_goals.main, ['--help'])
    assert help_result.exit_code == 0
    assert '--help  Show this message and exit.' in help_result.output


def test_goals():
    env = gym.make('REALRobot-v0')
    obs = env.reset()

    # Environment starts without goals
    # @TODO Check if we want to provide a goal file and load it immediately
    assert(env.goals is None)
    # goal_idx == -1 also means we are in the intrinsic phase
    assert(env.goal_idx == -1)
    # An all-zeroes matrix is displayed a goal (no goal)
    assert(obs['goal'].min() == 0 and obs['goal'].max() == 0)

    # Setting the goal path should not trigger the extrinsic phase - (Issue 12)
    env.set_goals_dataset_path('goals.npy.npz')
    assert(env.goal_idx == -1)

    # This should trigger the first goal
    env.set_goal()
    obs, _, _, _ = env.step(np.zeros(9))
    assert(not(obs['goal'].min() == 0 and obs['goal'].max() == 0))
    # We check one of the pixels to ensure this is the first goal
    assert(obs['goal'][111, 131, 0] == 118)
    assert(env.goal_idx == 0)

    # This should trigger the first goal
    env.set_goal()
    obs, _, _, _ = env.step(np.zeros(9))
    assert(not(obs['goal'].min() == 0 and obs['goal'].max() == 0))
    # We check one of the pixels to ensure this is the second goal
    assert(obs['goal'][111, 131, 0] == 154)
    assert(env.goal_idx == 1)


def test_local_evaluation():
    class RandomPolicy(BasePolicy):
        def __init__(self, action_space):
            self.action_space = action_space
            self.action = np.zeros(action_space.shape[0])
            self.action += -np.pi*0.5

        def step(self, observation, reward, done):
            self.action += 0.4*np.pi*np.random.randn(
                                self.action_space.shape[0])
            return self.action

    result = real_robots.evaluate(
                    RandomPolicy,
                    intrinsic_timesteps=40,
                    extrinsic_timesteps=40,
                    extrinsic_trials=5,
                    visualize=True,
                    goals_dataset_path="./goals.npy.npz",
                )
    print(result)
