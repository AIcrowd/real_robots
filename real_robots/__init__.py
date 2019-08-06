# -*- coding: utf-8 -*-

"""Top-level package for real-robots."""

__author__ = """S.P. Mohanty"""
__email__ = 'mohanty@aicrowd.com'
__version__ = '0.1.0'

from gym.envs.registration import register

register(id='REALComp-v0',
    entry_point='real_robots.envs:REALRobotEnv',
)

register(id='REALCompSingleObj-v0',
    entry_point='real_robots.envs:REALRobotEnvSingleObj',
)

from real_robots.envs import env as real_robot_env