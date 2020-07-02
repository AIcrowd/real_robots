from robot_bases import URDFBasedRobot
import numpy as np
import pybullet_data
import os
import gym
from robot_bases import BodyPart
import pybullet


class Kuka(URDFBasedRobot):

    used_objects = [
            "table",
            "orange",
            "mustard",
            "cube",
            "tomato"]

    object_poses = {
            "table":   [  0.00000,  0.00000,  0.08000, 0.00000, 0.00000, 0.00000],
            "orange":  [  0.20000, -0.15000,  0.45000, 0.00000, 0.00000, 0.00000],
            "mustard": [ -0.10000,  0.30000,  0.45000, 1.57080, 3.14159, 0.00000],
            "cube":    [ -0.10000,  0.00000,  0.45000, 0.00000, 0.00000, 0.00000],
            "tomato":  [ -0.10000, -0.30000,  0.45000, 0.00000, 0.00000, 0.00000]}
    
    num_joints = 9
    num_kuka_joints = 7
    num_gripper_joints = 2
    num_touch_sensors = 4
    eye_width = 320
    eye_height = 240

    class ObsSpaces:
        JOINT_POSITIONS = "joint_positions"
        TOUCH_SENSORS = "touch_sensors"
        RETINA = "retina"
        MASK = "mask"
        OBJ_POS = "object_positions"
        GOAL = "goal"
        GOAL_MASK = "goal_mask"
        GOAL_POS = "goal_positions"

    def __init__(self, additional_obs=False, objects=3):

        self.robot_position = [-0.55, 0, -0.04]
        self.contact_threshold = 0.1

        self.used_objects = ["table", "cube",
                                   "tomato", "mustard"][:objects+1]

        self.action_dim = self.num_joints

        URDFBasedRobot.__init__(self, 'kuka_gripper_description/urdf/'
                                'kuka_gripper.urdf', 'kuka0',
                                action_dim=self.action_dim, obs_dim=1)

        self.min_joints = np.ones(9)*-np.pi*0.944
        self.max_joints = np.ones(9)*np.pi*0.944
        self.min_joints[0] = -np.pi*0.666 # Restricted range (min -0.944)
        self.max_joints[0] = np.pi*0.666 # Restricted range (max 0.944)
        self.min_joints[1:9:2] = -np.pi*0.666
        self.max_joints[1:9:2] = np.pi*0.666
        self.min_joints[6] = -np.pi*0.972
        self.max_joints[6] = np.pi*0.972
        self.min_joints[-2:] = 0
        self.max_joints[-2:] = np.pi/2

        self.action_space = gym.spaces.Box(low=self.min_joints,
                                           high=self.max_joints,
                                           dtype=float)

        if additional_obs:
            obj_obs = {}
            for obj in self.used_objects[1:]:
                high = np.array([np.finfo(np.float32).max,
                                 np.finfo(np.float32).max,
                                 np.finfo(np.float32).max,
                                 1.0,
                                 1.0,
                                 1.0,
                                 1.0])
                obj_obs[obj] = gym.spaces.Box(-high, high, dtype=float)

            self.observation_space = gym.spaces.Dict({
                self.ObsSpaces.JOINT_POSITIONS: gym.spaces.Box(
                    -np.inf, np.inf, [self.num_joints], dtype=float),
                self.ObsSpaces.TOUCH_SENSORS: gym.spaces.Box(
                    0, np.inf, [self.num_touch_sensors], dtype=float),
                self.ObsSpaces.RETINA: gym.spaces.Box(
                    0, 255, [Kuka.eye_height, Kuka.eye_width, 3], dtype=np.uint8),
                self.ObsSpaces.GOAL: gym.spaces.Box(
                    0, 255, [Kuka.eye_height, Kuka.eye_width, 3], dtype=np.uint8),
                self.ObsSpaces.MASK: gym.spaces.Box(
                    0, 255, [Kuka.eye_height, Kuka.eye_width], dtype=np.int32),
                self.ObsSpaces.GOAL_MASK: gym.spaces.Box(
                    0, 255, [Kuka.eye_height, Kuka.eye_width], dtype=np.int32),
                self.ObsSpaces.OBJ_POS: gym.spaces.Dict(obj_obs)
                }
            )
        else:
            self.observation_space = gym.spaces.Dict({
                self.ObsSpaces.JOINT_POSITIONS: gym.spaces.Box(
                    -np.inf, np.inf, [self.num_joints], dtype=float),
                self.ObsSpaces.TOUCH_SENSORS: gym.spaces.Box(
                    0, np.inf, [self.num_touch_sensors], dtype=float),
                self.ObsSpaces.RETINA: gym.spaces.Box(
                    0, 255, [Kuka.eye_height, Kuka.eye_width, 3], dtype=float),
                self.ObsSpaces.GOAL: gym.spaces.Box(
                    0, 255, [Kuka.eye_height, Kuka.eye_width, 3], dtype=float)}
            )

        self.target = "orange"

        self.object_names = dict()
        self.object_bodies = dict()
        self.robot_parts = {}

    def reset(self, bullet_client):
        bullet_client.resetSimulation()
        super(Kuka, self).reset(bullet_client)
        return self.calc_state()

    def reset_object(self, obj_name):
        position = self.object_poses[obj_name][:3]
        eulerOrientation = self.object_poses[obj_name][3:]
        orientation = pybullet.getQuaternionFromEuler(eulerOrientation)
        self.object_bodies[obj_name].reset_pose(position, orientation)

    def get_contacts(self, forces=False):

        contact_dict = {}
        for part_name, part in self.parts.items():
            for contact in part.contact_list():
                if abs(contact[8]) < self.contact_threshold:
                    name = self.object_names[contact[2]]
                    if not forces:
                        if part_name in contact_dict.keys():
                            contact_dict[part_name].append(name)
                        else:
                            contact_dict[part_name] = [name]
                    else:
                        force = contact[9]
                        if part_name in contact_dict.keys():
                            contact_dict[part_name].append([name, force])
                        else:
                            contact_dict[part_name] = [(name, force)]

        return contact_dict

    def get_touch_sensors(self):

        sensors = np.zeros(4)
        contacts = self.get_contacts(forces=True)
        for i, skin in enumerate(["skin_00", "skin_01", "skin_10", "skin_11"]):
            if skin in contacts.keys():
                cnts = contacts[skin]
                if len(cnts) > 0:
                    force = np.max([cnt[1] for cnt in cnts])
                    sensors[i] = force

        return sensors

    def robot_specific_reset(self, bullet_client):

        self.robot_body.reset_position(self.robot_position)

        self.object_bodies["kuka"] = self.robot_body
        self.object_names[0] = "kuka"

        for obj_name in self.used_objects:
            pos = self.object_poses[obj_name]
            obj = get_object(bullet_client,
                             "kuka_gripper_description/urdf/{}.urdf".
                             format(obj_name), *pos)
            self.object_bodies[obj_name] = obj
            self.object_names.update({obj.bodies[0]: obj_name})
            self.reset_object(obj_name)

        for _, joint in self.jdict.items():
            joint.reset_current_position(0, 0)

        for name, part in self.parts.items():
            self.robot_parts.update({part.bodyPartIndex: name})


    def apply_action(self, a):
        assert (np.isfinite(a).all())
        assert(len(a) == self.num_joints)

        a = np.maximum(self.min_joints, (np.minimum(a, self.max_joints)))
        a[-1] = np.maximum(0, np.minimum(2*a[-2], a[-1]))

        for i, j in enumerate(a[:-2]):
            self.jdict["lbr_iiwa_joint_%d" % (i+1)].set_position(j)

        self.jdict["base_to_finger00_joint"].set_position(a[-2])
        self.jdict["base_to_finger10_joint"].set_position(a[-2])
        self.jdict["finger00_to_finger01_joint"].set_position(-a[-1])
        self.jdict["finger10_to_finger11_joint"].set_position(-a[-1])

    def calc_state(self):
        joints = []
        for i in range(self.num_kuka_joints):
            joints.append(self.jdict["lbr_iiwa_joint_%d" %
                          (i+1)].get_position())
        joints.append(self.jdict["base_to_finger00_joint"].get_position())
        joints.append(-self.jdict["finger00_to_finger01_joint"].get_position())

        return joints


def get_object(bullet_client, object_file, x, y, z, roll=0, pitch=0, yaw=0):
    position = [x, y, z]
    orientation = bullet_client.getQuaternionFromEuler([roll, pitch, yaw])
    body = bullet_client.loadURDF(
            fileName=os.path.join(pybullet_data.getDataPath(), object_file),
            basePosition=position,
            baseOrientation=orientation,
            useFixedBase=False,
            flags=bullet_client.URDF_USE_INERTIA_FROM_FILE)
    part_name, _ = bullet_client.getBodyInfo(body)
    part_name = part_name.decode("utf8")
    bodies = [body]
    return BodyPart(bullet_client, part_name, bodies, 0, -1)
