# -*- coding: utf-8 -*-

"""Console script to generate goals for real_robots"""

import click
import numpy as np
from real_robots.envs import Goal
import gym
import matplotlib.pyplot as plt
import math
from sklearn.metrics import pairwise_distances

basePosition = None


def runEnv(env, max_t=1000):
    reward = 0
    done = False
    action = np.zeros(env.action_space.shape[0])
    objects = env.robot.used_objects[1:]

    positions = np.vstack([env.get_obj_pose(obj) for obj in objects])
    still = False
    stable = 0
    for t in range(max_t):
        old_positions = positions
        observation, reward, done, _ = env.step(action)
        positions = np.vstack([env.get_obj_pose(obj) for obj in objects])

        maxPosDiff = 0
        maxOrientDiff = 0
        for i, obj in enumerate(objects):
            posDiff = np.linalg.norm(old_positions[i][:3] - positions[i][:3])
            q1 = old_positions[i][3:]
            q2 = positions[i][3:]
            orientDiff = min(np.linalg.norm(q1 - q2), np.linalg.norm(q1+q2))
            maxPosDiff = max(maxPosDiff, posDiff)
            maxOrientDiff = max(maxOrientDiff, orientDiff)

        if maxPosDiff < 0.0001 and maxOrientDiff < 0.001 and t > 10:
            stable += 1
        else:
            stable = 0

        if stable > 20:
            still = True
            break

    pos_dict = {}
    for obj in objects:
        pos_dict[obj] = env.get_obj_pose(obj)

    print("Exiting environment after {} timesteps..".format(t))
    if not still:
        print("Failed because maxPosDiff:{:.6f},"
              "maxOrientDiff:{:.6f}".format(maxPosDiff, maxOrientDiff))

    return observation['retina'], pos_dict, not still, t


class Position:
    def __init__(self, start_state=None, fixed_state=None, retina=None):
        self.start_state = start_state
        self.fixed_state = fixed_state
        self.retina = retina


def generatePosition(env, obj, fixed=False, tablePlane=None):
    if tablePlane is None:
        min_x = -.2
        max_x = .2
    elif tablePlane:
        min_x = -.2
        max_x = .1  # 0.05 real, .1 prudent
    else:
        min_x = 0  # 0.05 mustard
        max_x = .2

    min_y = -.5
    max_y = .5

    x = np.random.rand()*(max_x-min_x)+min_x
    y = np.random.rand()*(max_y-min_y)+min_y
    z = 0.7

    if fixed:
        orientation = basePosition[obj][3:]
    else:
        orientation = (np.random.rand(3)*math.pi*2).tolist()
        orientation = env._p.getQuaternionFromEuler(orientation)

    pose = [x, y, z] + np.array(orientation).tolist()
    return pose


def generateRealPosition(env, startPositions):
    env.reset()
    runEnv(env)
    # Generate Images
    for obj in startPositions:
        pos = startPositions[obj]
        env.robot.object_bodies[obj].reset_pose(pos[:3], pos[3:])

    actual_image, actual_position, failed, it = runEnv(env)

    return actual_image, actual_position, failed, it


def checkSeparation(state):
    positions = np.vstack([state[obj][:3] for obj in state])
    distances = pairwise_distances(positions)
    clearance = distances[distances > 0].min()
    return clearance


def drawPosition(env, fixedOrientation=False, fixedObjects=[],
                 fixedPositions=None, minSeparation=0, objOnTable=None):

    failed = True
    while failed:
        # skip 1st object, i.e the table
        objects = env.robot.used_objects[1:]

        position = Position()
        startPositions = {}

        for obj in fixedObjects:
            startPositions[obj] = fixedPositions[obj]

        for obj in objects:
            if obj in fixedObjects:
                continue
            while True:
                table = None
                if objOnTable is not None:
                    if obj in objOnTable:
                        table = objOnTable[obj]
                startPose = generatePosition(env, obj,
                                             fixedOrientation,
                                             tablePlane=table)
                startPositions[obj] = startPose
                if len(startPositions) == 1:
                    break
                clearance = checkSeparation(startPositions)
                if clearance >= minSeparation:
                    break
                print("Failed minimum separation ({}), draw again {}.."
                      .format(clearance, obj))

        (a, p, f, it) = generateRealPosition(env, startPositions)
        actual_image = a
        actual_position = p
        failed = f

        if failed:
            print("Failed image generation...")
            continue

        clearance = checkSeparation(actual_position)
        if clearance < minSeparation:
            failed = True
            print("Failed minimum separation ({}) after real generation, "
                  "draw again everything..".format(clearance))
            continue

        if fixedOrientation:
            for obj in objects:
                q1 = startPositions[obj][3:]
                q2 = actual_position[obj][3:]
                orientDiff = min(np.linalg.norm(q1 - q2),
                                 np.linalg.norm(q1+q2))
                # TODO CHECK This - we had to rise it many times
                failed = failed or orientDiff > 0.041
                if failed:
                    print("{} changed orientation by {}"
                          .format(obj, orientDiff))
                    break
                else:
                    print("{} kept orientation.".format(obj))

            if failed:
                print("Failed to keep orientation...")
                continue

        for obj in fixedObjects:
            posDiff = np.linalg.norm(startPositions[obj][:3] -
                                     actual_position[obj][:3])
            q1 = startPositions[obj][3:]
            q2 = actual_position[obj][3:]
            orientDiff = min(np.linalg.norm(q1 - q2), np.linalg.norm(q1+q2))
            failed = failed or posDiff > 0.002 or orientDiff > 0.041
            if failed:
                print("{} changed pos by {} and orientation by {}"
                      .format(obj, posDiff, orientDiff))
                print(startPositions[obj])
                print(actual_position[obj])
                break

        if failed:
            print("Failed to keep objects fixed...")
            continue

    position.start_state = startPositions
    position.fixed_state = actual_position
    position.retina = actual_image

    return position


def checkRepeatability(env, goals):
    for goal in goals:
        _, pos, failed, _ = generateRealPosition(env, goal.initial_state)
        objects = [o for o in goal.initial_state]
        p0 = np.vstack([goal.initial_state[o] for o in objects])
        p1 = np.vstack([pos[o] for o in objects])
        diffPos = np.linalg.norm(p1[:, :3]-p0[:, :3])
        diffOr = np.linalg.norm(p1[:, 3:]-p0[:, 3:])

        print("Replicated diffPos:{} diffOr:{}".format(diffPos, diffOr))
        if failed:
            print("*****************FAILED************!!!!")
            return 1000000
    return diffPos, diffOr


def isOnShelf(obj, state):
    z = state[obj][2]
    if obj == 'cube' and z > 0.55:
        return True
    if obj == 'orange' and z > 0.55:
        return True
    if obj == 'tomato' and z > 0.55:
        return True
    if obj == 'mustard' and z > 0.545:
        return True
    return False


def isOnTable(obj, state):
    z = state[obj][2]
    if obj == 'cube' and z < 0.48:
        return True
    if obj == 'orange' and z > 0.48:
        return True
    if obj == 'tomato' and z < 0.49:
        return True
    if obj == 'mustard' and z < 0.48:  # TODO TO BE CHECKED
        return True
    return False


def generateGoal2D(env, n_objects):

    print("Generating GOAL2D with {} object(s) moving..".format(n_objects))

    objects = env.robot.used_objects[1:]
    objOnTable = {}
    for obj in objects:
        objOnTable[obj] = True

    found = False
    while not(found):
        initial = drawPosition(env, fixedOrientation=True,
                               minSeparation=0.15, objOnTable=objOnTable)
        found = True

        for obj in objects:
            if not(isOnTable(obj, initial.fixed_state)):
                found = False
                print("{} is not on table...".format(obj))

    found = False
    while not(found):
        if n_objects == 1:
            fix_objs = ['tomato', 'mustard']
            final = drawPosition(env, fixedPositions=initial.fixed_state,
                                 fixedObjects=fix_objs, fixedOrientation=True,
                                 minSeparation=0.15, objOnTable=objOnTable)
        elif n_objects == 2:
            fix_objs = ['mustard']
            final = drawPosition(env, fixedPositions=initial.fixed_state,
                                 fixedObjects=fix_objs, fixedOrientation=True,
                                 minSeparation=0.15, objOnTable=objOnTable)
        else:
            final = drawPosition(env, fixedOrientation=True,
                                 minSeparation=0.15, objOnTable=objOnTable)

        found = True

        objects = env.robot.used_objects[1:]
        for obj in objects:
            if not(isOnTable(obj, final.fixed_state)):
                found = False
                print("{} is not on table...".format(obj))

    goal = Goal()
    goal.challenge = '2D'
    goal.subtype = str(n_objects)
    goal.initial_state = initial.fixed_state
    goal.final_state = final.fixed_state
    goal.retina_before = initial.retina
    goal.retina = final.retina

    print("SUCCESSFULL generation of GOAL2D with {} object(s)!!!!"
          .format(n_objects))

    return goal


def generateGoal25D(env, n_objects):
    print("Generating GOAL2.5D with {} object(s) moving..".format(n_objects))

    initial = drawPosition(env, fixedOrientation=True, minSeparation=0.15)

    objects = env.robot.used_objects[1:]
    objOnTable = {}
    for obj in objects:
        objOnTable[obj] = not(isOnTable(obj, initial.fixed_state))

    found = False
    while not(found):
        fix_objs = []
        if n_objects == 1:
            fix_objs = ['tomato', 'mustard']
            final = drawPosition(env, fixedPositions=initial.fixed_state,
                                 fixedObjects=fix_objs, fixedOrientation=True,
                                 minSeparation=0.15, objOnTable=objOnTable)
        elif n_objects == 2:
            fix_objs = ['mustard']
            final = drawPosition(env, fixedPositions=initial.fixed_state,
                                 fixedObjects=fix_objs, fixedOrientation=True,
                                 minSeparation=0.15, objOnTable=objOnTable)
        else:
            final = drawPosition(env, fixedOrientation=True,
                                 minSeparation=0.15, objOnTable=objOnTable)

        found = True
        for obj in objects:
            if obj in fix_objs:
                continue
            wasOnTable = isOnTable(obj, initial.fixed_state)
            isNowOnTable = isOnTable(obj, final.fixed_state)
            if not(wasOnTable ^ isNowOnTable):
                found = False
                print("{} failed to move from shelf to table ({}-{})"
                      .format(obj, wasOnTable, isNowOnTable))

    goal = Goal()
    goal.challenge = '2.5D'
    goal.subtype = str(n_objects)
    goal.initial_state = initial.fixed_state
    goal.final_state = final.fixed_state
    goal.retina_before = initial.retina
    goal.retina = final.retina

    print("SUCCESSFULL generation of GOAL25D with {} object(s)!!!!"
          .format(n_objects))

    return goal


def generateGoal3D(env):

    print("Generating GOAL3D..")

    found = False
    while not(found):
        initial = drawPosition(env, fixedOrientation=False)
        found = True

    found = False
    while not(found):
        final = drawPosition(env, fixedOrientation=False)
        found = True

    goal = Goal()
    goal.challenge = '3D'
    goal.subtype = '3'
    goal.initial_state = initial.fixed_state
    goal.final_state = final.fixed_state
    goal.retina_before = initial.retina
    goal.retina = final.retina

    print("SUCCESSFULL generation of GOAL3D!!!!")

    return goal


@click.command()
@click.option('--seed', type=int,
              help='Generate goals using this SEED for numpy.random')
@click.option('--n1', type=int, default=0,
              help='# of 2D Goals with 1 moving object')
@click.option('--n2', type=int, default=0,
              help='# of 2D Goals with 2 moving objects')
@click.option('--n3', type=int, default=0,
              help='# of 2D Goals with 3 moving objects')
@click.option('--n4', type=int, default=0,
              help='# of 2.5D Goals with 1 moving object')
@click.option('--n5', type=int, default=0,
              help='# of 2.5D Goals with 2 moving objects')
@click.option('--n6', type=int, default=0,
              help='# of 2.5D Goals with 3 moving objects')
@click.option('--n7', type=int, default=0,
              help='# of 3D Goals')
def main(seed=None, n1=0, n2=0, n3=0, n4=0, n5=0, n6=0, n7=0):
    """
        Generates the specified number of goals
        and saves them in a file.\n
        The file is called allgoals{}-{}-{}-{}-{}-{}-{}-{}.npy
        where enclosed brackets are replaced with the
        supplied options (seed, n1...n7) or 0.
    """
    np.random.seed(seed)
    allgoals = []
    env = gym.make('REALRobot-v0')
    env.reset()
    pos = env.robot.object_poses['mustard'][:]
    pos[2] = 0.41
    orient = env._p.getQuaternionFromEuler(pos[3:])
    env.robot.object_bodies['mustard'].reset_pose(pos[:3], orient)
    global basePosition
    _, basePosition, _, _ = runEnv(env)

    # In these for loops, we could add some progress bar...
    for _ in range(n1):
        allgoals += [generateGoal2D(env, 1)]
    for _ in range(n2):
        allgoals += [generateGoal2D(env, 2)]
    for _ in range(n3):
        allgoals += [generateGoal2D(env, 3)]
    for _ in range(n4):
        allgoals += [generateGoal25D(env, 1)]
    for _ in range(n5):
        allgoals += [generateGoal25D(env, 2)]
    for _ in range(n6):
        allgoals += [generateGoal25D(env, 3)]
    for _ in range(n7):
        allgoals += [generateGoal3D(env)]

    np.save('allgoals{}-{}-{}-{}-{}-{}-{}-{}.npy'
            .format(seed, n1, n2, n3, n4, n5, n6, n7), allgoals)

    # checkRepeatability(env, allgoals)

    # WARNING: If true, opens a lots of windows...
    # each window displays one of the generated goals
    if False:
        for goal in allgoals[:]:
            fig, axes = plt.subplots(1, 2)
            axes[0].imshow(goal.retina_before)
            axes[1].imshow(goal.retina)
            fig.suptitle(goal.challenge+str(goal.subtype))
        plt.show()


if __name__ == "__main__":
    main()
