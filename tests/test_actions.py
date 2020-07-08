import real_robots
import numpy as np
import gym
import matplotlib.pyplot as plt
import types
import pybullet
import sys
import os
from mpl_toolkits.mplot3d import Axes3D


def generate_plan(point_1, point_2):

    n_timesteps = 100
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
            print("debug", coord)
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


    xy_parts = []
    xy_parts += [np.linspace(np.array([-0.41, 0.0, 1.14]), np.array([-0.41, 0.0, 1.14]), n_timesteps)]
    xy_parts += [np.linspace(np.hstack([point_1, 0.6]), np.hstack([point_1, 0.6]), n_timesteps)]
    xy_parts += [np.linspace(np.hstack([point_1, 0.46]), np.hstack([point_1, 0.46]), 50)]
    xy_parts += [np.linspace(np.hstack([point_2, 0.46]), np.hstack([point_2, 0.46]), 500)]
    xy_parts += [np.linspace(np.hstack([point_2, 0.6]), np.hstack([point_2, 0.6]), 50)]
    xy_parts += [np.linspace(np.array([-0.41, 0.0, 1.14]), np.array([-0.41, 0.0, 1.14]), n_timesteps)]
    xy_parts += [np.linspace(np.array([-0.55, 0.0, 1.27]), np.array([-0.55, 0.0, 1.27]), n_timesteps)]
    raw_xy = np.vstack(xy_parts)

    checktimes = [199, 249, 749, 849, 999]
#    checkpoints = [point_1_h, point_1_l, point_2_l, point_2_h, home]
    checkpoints = [raw_xy[z] for z in checktimes]


    checklabels = [(100, 'home2'), (200, 'point_1_h'), 
              (250, 'point_1_l'), (750, 'point_2_l'),  
              (800, 'point_2_h'),
              (900, 'home2'),  (1000, 'home')]

    return raw_actions, checktimes, checkpoints, raw_xy, checklabels

n_obj = 3
envString = 'REALRobot2020-R1J{}-v0'.format(n_obj)
env = gym.make(envString)
env.render("human")
obs = env.reset()

def drawPoint():
    x = np.random.rand()*0.8-0.4
    y = np.random.rand()*0.27-0.29
    return (y, x)

render = False

for obj in env.robot.used_objects[1:]:
    env.robot.object_poses[obj][0] += 0.3
    env.robot.object_poses[obj][2] += 0.3
    env.robot.reset_object(obj)

replay = False

x = [-0.5, 0, 0.5]
y = [-0.25, 0.05]
z = [0.46, 0.60]

#Note varie:
#1) lo shelf può andare un paio di centimetri più indietro (ingrandire plane)

perimeter = np.array(np.meshgrid(x, y, z)).T.reshape(-1, 3)

perimeter = np.array(np.meshgrid(y, x)).T.reshape(-1, 2)


allCombinations = []

for p1 in perimeter:
    for p2 in perimeter:
        allCombinations += [(p1,p2)]


while True:
    printThis = False
    if not replay:
        point_1 = drawPoint()
        point_2 = drawPoint()
        point_1 = perimeter[np.random.choice(len(perimeter))]
        point_2 = perimeter[np.random.choice(len(perimeter))]
        if len(allCombinations) > 0:
            point_1, point_2 = allCombinations.pop()
        else:
            break
        render = False
    else:
        render = True

    raw_actions, checktimes, checkpoints, raw_xy, labels = generate_plan(point_1, point_2)
    print("{:.3f} {:.3f} {:.3f} {:.3f}".format(*point_1, *point_2)) 
    record = np.zeros(len(raw_actions))
    record_xy = np.zeros(raw_xy.shape)
    record_xy_diff = np.zeros(len(raw_xy))
    for i in range(len(raw_actions)):
        obs, _, _, _ = env.step({'joint_command': raw_actions[i], 'render': render})
        joints_now = obs['joint_positions']
        pos_now = env.robot.parts['base'].get_position()
        record[i] = np.linalg.norm(joints_now - raw_actions[i])
        record_xy_diff[i] = np.linalg.norm(pos_now - raw_xy[i])
        record_xy[i] = pos_now
        if i in checktimes:
            check = checkpoints[checktimes.index(i)]
            diff = np.linalg.norm(pos_now - np.array(check)) 
            if diff > 0.01:
                print("Failed!", i, diff)
                printThis = True

    if printThis:
        print("Printing failed action")
        plt.plot(record)
        for tl, label in labels:
            plt.annotate(label, # this is the text
                         (tl-1, record[tl-1]), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(0,10), # distance from text to points (x,y)
                         ha='center') # horizontal alignment can be left, right or center
        plt.title("Joints diff plot")
        plt.figure()
        plt.plot(record_xy_diff)
        for tl, label in labels:
            plt.annotate(label, # this is the text
                         (tl-1, record_xy_diff[tl-1]), # this is the point to label
                         textcoords="offset points", # how to position the text
                         xytext=(0,10), # distance from text to points (x,y)
                         ha='center') # horizontal alignment can be left, right or center
        plt.title("Cartesian diff plot")

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(record_xy[:, 0], record_xy[:, 1], record_xy[:, 2])
        for check in checkpoints:
            # draw sphere
            radius = 0.01
            center = check
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x = np.cos(u)*np.sin(v)*radius+center[0]
            y = np.sin(u)*np.sin(v)*radius+center[1]
            z = np.cos(v)*radius+center[2]
            ax.plot_wireframe(x, y, z, color="r")

        X = record_xy[:, 0]
        Y = record_xy[:, 1]
        Z = record_xy[:, 2]
        # Create cubic bounding box to simulate equal aspect ratio
        max_range = np.array([X.max()-X.min(), Y.max()-Y.min(), Z.max()-Z.min()]).max()
        Xb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][0].flatten() + 0.5*(X.max()+X.min())
        Yb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][1].flatten() + 0.5*(Y.max()+Y.min())
        Zb = 0.5*max_range*np.mgrid[-1:2:2,-1:2:2,-1:2:2][2].flatten() + 0.5*(Z.max()+Z.min())
        # Comment or uncomment following both lines to test the fake bounding box:
        for xb, yb, zb in zip(Xb, Yb, Zb):
           ax.plot([xb], [yb], [zb], 'w')
        
        plt.show()
        replay = not replay

    for i in range(100):
        obs, _, _, _ = env.step({'joint_command': np.zeros(9), 'render': render})

print("All perimeter combinations tested.")
