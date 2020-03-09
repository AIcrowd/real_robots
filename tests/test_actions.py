import real_robots
import numpy as np
import gym
import matplotlib.pyplot as plt
import types
import pybullet
import sys
import os

def enhanceEnvironment(env):

    def renderTarget(self, targetPosition, bullet_client=None):

        if bullet_client is None:
            bullet_client = self._p

        self.targetPosition = targetPosition

        view_matrix = bullet_client.computeViewMatrix(
                cameraEyePosition=self.eyePosition,
                cameraTargetPosition=self.targetPosition,
                cameraUpVector=self.upVector)

        proj_matrix = bullet_client.computeProjectionMatrixFOV(
                fov=self.fov,
                aspect=float(self.render_width)/self.render_height,
                nearVal=0.1, farVal=100.0)

        (_, _, px, _, mask) = bullet_client.getCameraImage(
                width=self.render_width, height=self.render_height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
                )

        rgb_array = np.array(px).reshape(self.render_height,
                                         self.render_width, 4)
        rgb_array = rgb_array[:, :, :3]

        return rgb_array, mask

    no_mask = np.zeros((240,320))

    def get_observation(self, camera_on=False):

        joints = self.robot.calc_state()
        sensors = self.robot.get_touch_sensors()

        if camera_on:
            retina, mask = self.get_retina()
        else:
            retina = self.no_retina
            mask = no_mask

        observation = {
                env.robot.ObsSpaces.JOINT_POSITIONS: joints,
                env.robot.ObsSpaces.TOUCH_SENSORS: sensors,
                env.robot.ObsSpaces.RETINA: retina,
                env.robot.ObsSpaces.GOAL: self.goal.retina}

        observation['mask'] = mask

        observation['object_positions'] = {}
        for obj in env.robot.used_objects[1:]:
            observation['object_positions'][obj] = env.robot.object_bodies[obj].get_pose()

        if getattr(self.goal, "mask", None) is not None:
            observation['goal_mask'] = self.goal.mask
        else:
            observation['goal_mask'] = self.goal.retina
        observation['goal_positions'] = self.goal.final_state

        return observation

    env.get_observation = types.MethodType(get_observation, env)
    env.eyes['eye'].renderTarget = types.MethodType(renderTarget, env.eyes['eye'])


def generateAction(point_1, point_2):
        n_timesteps = 400
        home = np.zeros(9)
        actionsPart1 = np.linspace(home, point_1, n_timesteps / 2)
        actionsPart2 = np.linspace(point_1, point_2, n_timesteps / 2)
        raw_actions = np.vstack([actionsPart1, actionsPart2])
        return raw_actions

def generateXYPush(point_1, point_2):
        n_timesteps = 100
        
        home = np.zeros(9)

        home2 = np.zeros(9)
        home2[5] = np.pi / 2
        home2[6] = np.pi / 2

        point_1_h = goToPosXY(np.hstack([point_1, 0.6]))
        point_1_l = goToPosXY(np.hstack([point_1, 0.46]))
        point_2_h = goToPosXY(np.hstack([point_2, 0.6]))
        point_2_l = goToPosXY(np.hstack([point_2, 0.46]))


        actionsParts = []
        actionsParts += [np.linspace(home, home2, 100)]
        actionsParts += [np.linspace(home2, point_1_h, n_timesteps)]
        actionsParts += [np.linspace(point_1_h, point_1_h, n_timesteps)]
        actionsParts += [np.linspace(point_1_h, point_1_l, n_timesteps)]
        actionsParts += [np.linspace(point_1_l, point_1_l, n_timesteps)]
        actionsParts += [np.linspace(point_1_l, point_2_l, n_timesteps)]
        actionsParts += [np.linspace(point_2_l, point_2_l, n_timesteps)]
        actionsParts += [np.linspace(point_2_l, point_2_h, n_timesteps)]
        actionsParts += [np.linspace(point_2_h, point_2_h, n_timesteps)]
        actionsParts += [np.linspace(point_2_h, home2, n_timesteps)]

        raw_actions = np.vstack(actionsParts)

        return raw_actions

def generateXYGrasp(point_1, point_2):
        n_timesteps = 100
        
        home = np.zeros(9)

        home2 = np.zeros(9)
        home2[5] = np.pi / 2
        home2[6] = np.pi / 2

        point_1_ho = goToPosXYOpen(np.hstack([point_1, 0.6]))
        point_1_hc = goToPosXYClosed(np.hstack([point_1, 0.6]))
        point_1_lo = goToPosXYOpen(np.hstack([point_1, 0.38]))
        point_1_lc = goToPosXYClosed(np.hstack([point_1, 0.42]))
        point_2_ho = goToPosXYOpen(np.hstack([point_2, 0.6]))
        point_2_hc = goToPosXYClosed(np.hstack([point_2, 0.6]))
        point_2_lo = goToPosXYOpen(np.hstack([point_2, 0.38]))
        point_2_lc = goToPosXYClosed(np.hstack([point_2, 0.42]))


        actionsParts = []
        actionsParts += [np.linspace(home, home2, n_timesteps)]
        actionsParts += [np.linspace(home2, point_1_ho, n_timesteps)]
        actionsParts += [np.linspace(point_1_ho, point_1_ho, n_timesteps*2)]
        actionsParts += [np.linspace(point_1_ho, point_1_lo, n_timesteps)]
        actionsParts += [np.linspace(point_1_lo, point_1_lo, n_timesteps)]
        actionsParts += [np.linspace(point_1_lo, point_1_lc, n_timesteps)]
        actionsParts += [np.linspace(point_1_lc, point_1_lc, n_timesteps)]
        actionsParts += [np.linspace(point_1_lc, point_1_hc, n_timesteps)]
        actionsParts += [np.linspace(point_1_hc, point_1_hc, n_timesteps)]
        actionsParts += [np.linspace(point_1_hc, point_2_hc, n_timesteps)]
        actionsParts += [np.linspace(point_2_hc, point_2_hc, n_timesteps*2)]
        actionsParts += [np.linspace(point_2_hc, point_2_lc, n_timesteps)]
        actionsParts += [np.linspace(point_2_lc, point_2_lc, n_timesteps)]
        actionsParts += [np.linspace(point_2_lc, point_2_lo, n_timesteps)]
        actionsParts += [np.linspace(point_2_lo, point_2_ho, n_timesteps)]
        actionsParts += [np.linspace(point_2_ho, home2, n_timesteps)]

        raw_actions = np.vstack(actionsParts)

        return raw_actions

def goToPosXY(coords):
    desiredOrientation = pybullet.getQuaternionFromEuler([0,3.14,-1.57])
    action = pybullet.calculateInverseKinematics(0,7,coords,desiredOrientation,maxNumIterations = 1000, residualThreshold = 0.001)
    return action[:9]


def goToPosXYOpen(coords):
    action = np.array(goToPosXY(coords))
    action[-2:] = np.pi/2
    return action

def goToPosXYClosed(coords):
    action = np.array(goToPosXY(coords))
    action[-2] = np.pi/4
    action[-1] = np.pi/2
    return action

def generateGraspObj(angle, distance, observation):
    cube_position = observation['object_positions']['cube'][:2]

    destination = cube_position.copy()
    destination[0] += np.cos(angle)*distance
    destination[1] += np.sin(angle)*distance

    np.minimum(destination, [-0.1, 0.5], destination)
    np.maximum(destination, [-0.5, -0.5], destination)

    return generateXYGrasp(cube_position, destination)


obj = 1
if obj == 1:
    env = gym.make('REALRobotSingleObj-v0')
else:
    env = gym.make('REALRobot-v0')
enhanceEnvironment(env)
obs = env.reset()

from PIL import Image

for _ in range(10):
    angle = np.random.rand()*np.pi*2
    distance = np.random.rand()*0.14+0.01
    raw_actions = generateGraspObj(angle, distance, obs)
    for i in range(len(raw_actions)):
        obs, _, _, _ = env.step({'joint_command': raw_actions[i], 'render': True}) #render to False shuts off robot camera and makes simulation way faster
        frame = env.render('rgb_array')
        im = Image.fromarray(frame)
        im.save("frames/frame%04d.png" % i)
    input("Press to continue")

    # Go back home
    for i in range(500):
        obs, _, _, _ = env.step({'joint_command': np.zeros(9), 'render': False})


