# -*- coding: utf-8 -*-

"""Top-level package for real-robots."""

__author__ = """S.P. Mohanty"""
__email__ = 'mohanty@aicrowd.com'
__version__ = '0.1.0'

from gym.envs.registration import register

register(id='REALComp-v0',
    entry_point='real_robots.envs:REALCompEnv',
)

register(id='REALCompSingleObj-v0',
    entry_point='real_robots.envs:REALCompEnvSingleObj',
)

from realcomp.envs import realcomp_env