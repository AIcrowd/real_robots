from pybullet_envs.scene_abstract import SingleRobotEmptyScene
from pybullet_envs.env_bases import MJCFBaseBulletEnv
import numpy as np
import pybullet
import real_robots
from .robot import Kuka
import os
from gym import spaces


def DefaultRewardFunc(observation):
    return 0


class Goal:
    def __init__(self, initial_state=None, final_state=None, retina=None,
                 retina_before=None, challenge=None, mask=None):

        self.initial_state = initial_state
        self.final_state = final_state
        self.retina = retina
        self.retina_before = retina_before
        self.challenge = challenge
        self.mask = mask


class REALRobotEnv(MJCFBaseBulletEnv):
    """ Create a REALCompetion environment inheriting by gym.env

    """

    intrinsic_timesteps = int(1e7)
    extrinsic_timesteps = int(2e3)

    def __init__(self, render=False, objects=3, action_type='joints',
                 additional_obs=True):

        self.robot = Kuka(additional_obs, objects)
        MJCFBaseBulletEnv.__init__(self, self.robot, render)

        self.joints_space = self.robot.action_space

        self.cartesian_space = spaces.Box(
                           low=np.array([-0.25, -0.5, 0.40, -1, -1, -1, -1]),
                           high=np.array([0.25, 0.5, 0.60,  1,  1,  1,  1]),
                           dtype=float)

        self.macro_space = spaces.Box(
                                  low=np.array([[-0.25, -0.5], [-0.25, -0.5]]),
                                  high=np.array([[0.05, 0.5], [0.05, 0.5]]),
                                  dtype=float)

        self.gripper_space = spaces.Box(low=0,
                                        high=np.pi/2, shape=(2,), dtype=float)

        if action_type == 'joints':
            self.action_space = spaces.Dict({
                                "joint_command": self.joints_space,
                                "render": spaces.MultiBinary(1)})
            self.step = self.step_joints

        elif action_type == 'cartesian':
            self.action_space = spaces.Dict({
                                "cartesian_command": self.cartesian_space,
                                "gripper_command": self.gripper_space,
                                "render": spaces.MultiBinary(1)})
            self.step = self.step_cartesian

        elif action_type == 'macro_action':
            self.action_space = spaces.Dict({
                                "macro_action": self.macro_space,
                                "render": spaces.MultiBinary(1)})
            self.step = self.step_macro
            self.requested_action = None
        else:
            raise ValueError("action_type must be one 'joints', 'cartesian' "
                             "or 'macro_action'")

        self._cam_dist = 1.2
        self._cam_yaw = 30
        self._cam_roll = 0
        self._cam_pitch = -30
        self._render_width = 320
        self._render_height = 240
        self._cam_pos = [0, 0, .4]
        self.setCamera()
        self.eyes = {}

        self.reward_func = DefaultRewardFunc

        self.set_eye("eye")

        self.goal = Goal(retina=self.observation_space.spaces[
                                self.robot.ObsSpaces.GOAL].sample()*0)

        # Set default goals dataset path
        #
        # The goals dataset is basically a list of real_robots.envs.env.Goal
        # objects which are stored using :
        #
        # np.savez_compressed(
        #               "path.npy.npz",
        #                list_of_goals)
        #
        self.goals_dataset_path = os.path.join(
                                    real_robots.getPackageDataPath(),
                                    "goals_dataset.npy.npz")
        self.goals = None
        self.goal_idx = -1
        self.no_retina = self.observation_space.spaces[
                         self.robot.ObsSpaces.RETINA].sample()*0

        if additional_obs:
            self.get_observation = self.get_observation_extended
            self.no_mask = self.observation_space.spaces[
                                self.robot.ObsSpaces.MASK].sample()*0

    def setCamera(self):
        ''' Initialize environment camera
        '''
        self.envCamera = EnvCamera(
                distance=self._cam_dist,
                yaw=self._cam_yaw,
                pitch=self._cam_pitch,
                roll=self._cam_roll,
                pos=self._cam_pos,
                width=self._render_width,
                height=self._render_height)

    def set_eye(self, name, eye_pos=[0.01, 0, 1.2], target_pos=[0, 0, 0]):
        ''' Initialize an eye camera
        @name the label of the created eye camera
        '''
        cam = EyeCamera(eye_pos, target_pos)
        self.eyes[name] = cam

    def load_goals(self):
        self.goals = list(np.load(
                self.goals_dataset_path, allow_pickle=True).items())[0][1]

    def set_goals_dataset_path(self, path):
        assert os.path.exists(path), "Non existent path {}".format(path)
        self.goals_dataset_path = path

    def set_goal(self):
        if self.goals is None:
            self.load_goals()

        self.goal_idx += 1
        self.goal = self.goals[self.goal_idx]

        for obj in self.goal.initial_state.keys():
            position = self.goal.initial_state[obj][:3]
            orientation = self.goal.initial_state[obj][3:]
            self.robot.object_bodies[obj].reset_pose(position, orientation)

        return self.get_observation()

    def extrinsicFormula(self, p_goal, p, a_goal, a, w=1):
        pos_dist = np.linalg.norm(p_goal-p)
        pos_const = -np.log(0.25) / 0.05  # Score goes down to 0.25 within 5cm
        pos_value = np.exp(- pos_const * pos_dist)

        orient_dist = min(np.linalg.norm(a_goal-a), np.linalg.norm(a_goal+a))
        orient_const = - np.log(0.25) / 0.30
        # Score goes down to 0.25 within 0.3
        orient_value = np.exp(- orient_const * orient_dist)

        value = w * pos_value + (1-w) * orient_value
        return value

    def evaluateGoal(self):
        initial_state = self.goal.initial_state  # noqa F841
        final_state = self.goal.final_state
        current_state = self.robot.object_bodies
        score = 0
        for obj in final_state.keys():
            if obj not in current_state:
                pass
            p = np.array(current_state[obj].get_position())
            p_goal = np.array(final_state[obj][:3])
            pos_dist = np.linalg.norm(p_goal-p)
            # Score goes down to 0.25 within 10cm
            pos_const = -np.log(0.25) / 0.10
            pos_value = np.exp(- pos_const * pos_dist)
            objScore = pos_value
            # print("Object: {} Score: {:.4f}".format(obj,objScore))
            score += objScore

        # print("Goal score: {:.4f}".format(score))
        return self.goal.challenge, score

    def create_single_player_scene(self, bullet_client):
        return SingleRobotEmptyScene(bullet_client, gravity=9.81,
                                     timestep=0.005, frame_skip=1)

    def reset(self):
        super(REALRobotEnv, self).reset()
        self._p.setGravity(0., 0., -9.81)
        self.camera._p = self._p
        for name in self.eyes.keys():
            self.eyes[name]._p = self._p

        self._p.resetDebugVisualizerCamera(
                self._cam_dist, self._cam_yaw,
                self._cam_pitch, self._cam_pos)

        self.timestep = 0

        return self.get_observation()

    def render(self, mode='human', close=False):
        if mode == "human":
            self.isRender = True
        if mode != "rgb_array":
            return np.array([])

        rgb_array = self.envCamera.render(self._p)
        return rgb_array

    def get_part_pos(self, name):
        # print(self.robot.parts.keys())
        return self.robot.parts[name].get_position()

    def get_obj_pos(self, name):
        return self.robot.object_bodies[name].get_position()

    def get_obj_pose(self, name):
        return self.robot.object_bodies[name].get_pose()

    def get_all_used_objects(self):
        all_positions = {}
        for obj in self.robot.used_objects[1:]:
            all_positions[obj] = self.robot.object_bodies[obj].get_position()
        return all_positions

    def get_contacts(self):
        return self.robot.get_contacts()

    def get_retina(self):
        '''
        :return: the current rgb_array for the eye
        '''
        return self.eyes["eye"].render(
                                       self.robot.object_bodies["table"]
                                       .get_position())

    def control_objects_limits(self):
        '''
        reset positions if an object goes out of the limits
        '''
        for obj in self.robot.used_objects:
            x, y, z = self.robot.object_bodies[obj].get_position()
            if z < self.robot.object_poses['table'][2]:
                self.robot.reset_object(obj)

    def get_observation(self, camera_on=True):

        joints = self.robot.calc_state()
        sensors = self.robot.get_touch_sensors()

        if camera_on:
            retina = self.get_retina()
        else:
            retina = self.no_retina

        observation = {
                Kuka.ObsSpaces.JOINT_POSITIONS: joints,
                Kuka.ObsSpaces.TOUCH_SENSORS: sensors,
                Kuka.ObsSpaces.RETINA: retina,
                Kuka.ObsSpaces.GOAL: self.goal.retina}

        return observation

    def get_observation_extended(self, camera_on=True):

        joints = self.robot.calc_state()
        sensors = self.robot.get_touch_sensors()

        if camera_on:
            retina, mask = self.get_retina()
        else:
            retina = self.no_retina
            mask = self.no_mask

        all_obj_positions = self.get_all_used_objects()

        observation = {
                Kuka.ObsSpaces.JOINT_POSITIONS: joints,
                Kuka.ObsSpaces.TOUCH_SENSORS: sensors,
                Kuka.ObsSpaces.RETINA: retina,
                Kuka.ObsSpaces.MASK: mask,
                Kuka.ObsSpaces.OBJ_POS: all_obj_positions,
                Kuka.ObsSpaces.GOAL: self.goal.retina,
                Kuka.ObsSpaces.GOAL_MASK: self.goal.mask,
                Kuka.ObsSpaces.GOAL_POS: self.goal.final_state
        }

        return observation

    def limitActionByJoint(self, desired_joints):
        current_joints = self.robot.calc_state()
        # The following specifies maximum change requested for each joint
        maxDiff = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3, 0.1, 0.1])
        minDiff = -maxDiff
        diff = np.minimum(maxDiff, desired_joints-current_joints)
        diff = np.maximum(minDiff, diff)
        return current_joints+diff

    def step(self, action):
        return self.step_joint(action)

    def step_joints(self, action):

        joint_action = action['joint_command']
        camera_on = action['render']

        assert(not self.scene.multiplayer)

        joint_action = self.limitActionByJoint(joint_action)

        self.control_objects_limits()
        self.robot.apply_action(joint_action)
        self.scene.global_step()

        observation = self.get_observation(camera_on)
        reward = self.reward_func(observation)

        done = False
        self.timestep += 1
        if self.goal_idx < 0:
            if self.timestep >= self.intrinsic_timesteps:
                done = True
        else:
            if self.timestep >= self.extrinsic_timesteps:
                done = True

        info = {}

        return observation, reward, done, info

    def step_cartesian(self, action):
        coords = action['cartesian_command'][:3]
        desiredOrientation = action['cartesian_command'][3:]
        inv_act = pybullet.calculateInverseKinematics(0, 7, coords,
                                                      desiredOrientation,
                                                      maxNumIterations=1000,
                                                      residualThreshold=0.001)

        joint_action = {"joint_command": inv_act[:9],
                        "render": action['render']}

        return self.step_joints(joint_action)

    def step_macro(self, action):

        macro_action = action['macro_action']

        if macro_action is None:
            joint_action = {"joint_command": np.zeros(9),
                            "render": action['render']}
        else:
            sameAction = np.all(macro_action == self.requested_action)
            if sameAction:
                joints = self.next_step()

            if not sameAction or joints is None:
                self.requested_action = macro_action
                self.generate_plan(macro_action)
                joints = self.next_step()

            joint_action = {"joint_command": joints,
                            "render": action['render']}

        return self.step_joints(joint_action)

    def generate_plan(self, macro_action):

        point_1 = macro_action[0]
        point_2 = macro_action[1]

        home = np.zeros(9)

        home2 = np.zeros(9)
        home2[5] = np.pi / 2
        home2[6] = np.pi / 2

        def goToPosXY(coords):
            desiredOrientation = pybullet.getQuaternionFromEuler([0, 3.14, -1.57])
            action = pybullet.calculateInverseKinematics(0, 7, coords,
                                                         desiredOrientation,
                                                         maxNumIterations=1000,
                                                         residualThreshold=0.001)
            return action[:9]

        def interpolate3D(p1, p2, steps):
            p1 = np.array(p1)
            p2 = np.array(p2)
            dist = np.linalg.norm(p2 - p1)
            pieces = int(dist / 0.05) + 1
            pieces = min(pieces, steps)
            coords = np.linspace(p1, p2, pieces + 1)
            joints = np.zeros((steps, 9))
            chunk = int(steps/pieces)
            for i, coord in enumerate(coords[1:]):
                joints[i*chunk:, :] = goToPosXY(coord)
            return joints

        point_1_h = goToPosXY(np.hstack([point_1, 0.6]))
        point_1_l = goToPosXY(np.hstack([point_1, 0.46]))
        point_2_h = goToPosXY(np.hstack([point_2, 0.6]))
        point_2_l = goToPosXY(np.hstack([point_2, 0.46]))

        actionsParts = []
        actionsParts += [np.tile(home2, (100, 1))]
        actionsParts += [np.tile(point_1_h, (100, 1))]
        actionsParts += [np.tile(point_1_l, (50, 1))]
        actionsParts += [interpolate3D(np.hstack([point_1, 0.46]), np.hstack([point_2, 0.46]), 500)]
        actionsParts += [np.tile(point_2_h, (50, 1))]
        actionsParts += [np.tile(home2, (100, 1))]
        actionsParts += [np.tile(home, (100, 1))]

        raw_actions = np.vstack(actionsParts)

        self.planned_actions = raw_actions
        self.plan_step = -1


    def next_step(self):
        self.plan_step += 1
        if self.plan_step < len(self.planned_actions):
            return self.planned_actions[self.plan_step, :]
        else:
            return None


class EnvCamera:

    def __init__(self, distance, yaw, pitch, roll, pos,
                 fov=80, width=320, height=240):

        self.dist = distance
        self.yaw = yaw
        self.pitch = pitch
        self.roll = roll
        self.pos = pos
        self.fov = fov
        self.render_width = width
        self.render_height = height

    def render(self, bullet_client=None):

        if bullet_client is None:
            bullet_client = self._p

        view_matrix = bullet_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.pos,
                distance=self.dist,
                yaw=self.yaw,
                pitch=self.pitch,
                roll=self.roll,
                upAxisIndex=2)

        proj_matrix = bullet_client.computeProjectionMatrixFOV(
                fov=self.fov,
                aspect=float(self.render_width)/self.render_height,
                nearVal=0.1, farVal=100.0)

        (_, _, px, _, _) = bullet_client.getCameraImage(
                width=self.render_width, height=self.render_height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=pybullet.ER_BULLET_HARDWARE_OPENGL
                )

        rgb_array = np.array(px).reshape(self.render_height,
                                         self.render_width, 4)
        rgb_array = rgb_array[:, :, :3]

        return rgb_array


class EyeCamera:

    def __init__(self, eyePosition, targetPosition,
                 fov=80, width=320, height=240):

        self.eyePosition = eyePosition
        self.targetPosition = targetPosition
        self.upVector = [0, 0, 1]
        self.fov = fov
        self.render_width = width
        self.render_height = height
        self._p = None
        self.pitch_roll = False

    def render(self, *args, **kargs):
        if self.pitch_roll is True:
            return self.renderPitchRoll(*args, **kargs)
        else:
            return self.renderTarget(*args, **kargs)

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
                renderer=pybullet.COV_ENABLE_TINY_RENDERER
                )

        rgb_array = np.array(px).reshape(self.render_height,
                                         self.render_width, 4)
        rgb_array = rgb_array[:, :, :3]

        return rgb_array, mask

    def renderPitchRoll(self, distance, roll, pitch, yaw, bullet_client=None):

        if bullet_client is None:
            bullet_client = self._p

        # self.targetPosition = targetPosition

        view_matrix = bullet_client.computeViewMatrixFromYawPitchRoll(
                cameraTargetPosition=self.pos,
                distance=distance,
                yaw=yaw,
                pitch=pitch,
                roll=roll,
                upAxisIndex=2)

        proj_matrix = bullet_client.computeProjectionMatrixFOV(
                fov=self.fov,
                aspect=float(self.render_width)/self.render_height,
                nearVal=0.1, farVal=100.0)

        (_, _, px, _, _) = bullet_client.getCameraImage(
                width=self.render_width, height=self.render_height,
                viewMatrix=view_matrix,
                projectionMatrix=proj_matrix,
                renderer=pybullet.COV_ENABLE_TINY_RENDERER
                )

        rgb_array = np.array(px).reshape(self.render_height,
                                         self.render_width, 4)
        rgb_array = rgb_array[:, :, :3]

        return rgb_array
