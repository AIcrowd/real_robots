# -*- coding: utf-8 -*-

"""Top-level package for real-robots."""

__author__ = """S.P. Mohanty"""
__email__ = 'mohanty@aicrowd.com'
__version__ = '0.1.1'

import os
from gym.envs.registration import register

register(id='REALRobot-v0',
    entry_point='real_robots.envs:REALRobotEnv',
)

register(id='REALRobotSingleObj-v0',
    entry_point='real_robots.envs:REALRobotEnvSingleObj',
)

from real_robots.envs import env as real_robot_env
from real_robots.evaluate import evaluate


def getPackageDataPath():
    import real_robots
    return os.path.join(
                real_robots.__path__[0],
                "data"
            )

def copy_over_data_into_pybullet(force_copy=False):
    """
    If the package specific data has not already 
    been copied over into pybullet_data, then
    copy them over.
    """
    import pybullet_data

    pybullet_data_path = pybullet_data.getDataPath()
    is_data_absent = "kuka_gripper_description" not in os.listdir(pybullet_data_path)
    if force_copy or is_data_absent:
        import shutil
        source_data_path = os.path.join(
                                getPackageDataPath(),
                                "kuka_gripper_description")
        target_data_path = os.path.join(
                                pybullet_data_path,
                                "kuka_gripper_description")
        print(
            "[REALRobot] Copying over data into pybullet_data_path."
            "This is a one time operation.")
        shutil.copytree(source_data_path, target_data_path)
copy_over_data_into_pybullet()