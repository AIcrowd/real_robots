# -*- coding: utf-8 -*-

"""Top-level package for real-robots."""

__author__ = """S.P. Mohanty"""
__email__ = 'mohanty@aicrowd.com'
__version__ = '0.1.18'

import os
from gym.envs.registration import register

from .evaluate import evaluate  # noqa F401


for n_obj in [1,2,3]:
    for obs, rnd in zip([True, False], ["R1", "R2"]):
        for action_type in ['joints', 'cartesian', 'macro_action']:
            action_str = action_type[0].upper()
            env_id = 'REALRobot2020-{}{}{}-v0'.format(rnd, action_str, n_obj)

            register(id=env_id,
                entry_point='real_robots.envs:REALRobotEnv',
                kwargs={'additional_obs': obs,
                        'objects' : n_obj,
                        'action_type' : action_type
                        },
            )





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
    is_data_absent = \
        "kuka_gripper_description" not in os.listdir(pybullet_data_path)
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
