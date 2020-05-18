# -*- coding: utf-8 -*-

"""Console script to generate goals for real_robots"""

import click
import numpy as np
from real_robots.envs import Goal
import gym
import math

basePosition = None
slow = False
render = False


def pairwise_distances(a):
    b = a.reshape(a.shape[0], 1, a.shape[1])
    return np.sqrt(np.einsum('ijk, ijk->ij', a-b, a-b))


def runEnv(env, max_t=1000):
    reward = 0
    done = False
    render = slow
    action = {'joint_command': np.zeros(9), 'render': render}
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
            action['render'] = slow

        if stable > 19:
            action['render'] = True

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

    return observation['retina'], pos_dict, not still, t, observation['mask']


class Position:
    def __init__(self, start_state=None, fixed_state=None, retina=None, mask=None):
        self.start_state = start_state
        self.fixed_state = fixed_state
        self.retina = retina
        self.mask = mask


def generatePosition(env, obj, fixed=False, tablePlane=None):
    if tablePlane is None:
        min_x = -.25
        max_x = .25
    elif tablePlane:
        min_x = -.25
        max_x = .05
    else:
        min_x = .10
        max_x = .25

    min_y = -.45
    max_y = .45

    x = np.random.rand()*(max_x-min_x)+min_x
    y = np.random.rand()*(max_y-min_y)+min_y

    if x <= 0.05:
        z = 0.40
    else:
        z = 0.50

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

    actual_image, actual_position, failed, it, mask = runEnv(env)
    return actual_image, actual_position, failed, it, mask


def checkSeparation(state):
    positions = np.vstack([state[obj][:3] for obj in state])
    if len(positions) > 1:
        distances = pairwise_distances(positions)
        clearance = distances[distances > 0].min()
    else:
        clearance = np.inf
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

        (a, p, f, it, m) = generateRealPosition(env, startPositions)
        actual_image = a
        actual_mask = m
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
    position.mask = actual_mask

    return position


def checkRepeatability(env, goals):
    maxDiffPos = 0
    maxDiffOr = 0
    for goal in goals:
        _, pos, failed, _, _ = generateRealPosition(env, goal.initial_state)
        objects = [o for o in goal.initial_state]
        p0 = np.vstack([goal.initial_state[o] for o in objects])
        p1 = np.vstack([pos[o] for o in objects])
        diffPos = np.linalg.norm(p1[:, :3]-p0[:, :3])
        diffOr = min(np.linalg.norm(p1[:, 3:]-p0[:, 3:]),
                     np.linalg.norm(p1[:, 3:]+p0[:, 3:]))
        maxDiffPos = max(maxDiffPos, diffPos)
        maxDiffOr = max(maxDiffPos, diffOr)
        print("Replicated diffPos:{} diffOr:{}".format(diffPos, diffOr))
        if failed:
            print("*****************FAILED************!!!!")
            return 1000000
    return maxDiffPos, maxDiffOr


def isOnShelf(obj, state):
    z = state[obj][2]
    if obj == 'cube' and z > 0.55 - 0.15:
        return True
    if obj == 'orange' and z > 0.55 - 0.15:
        return True
    if obj == 'tomato' and z > 0.55 - 0.15:
        return True
    if obj == 'mustard' and z > 0.545 - 0.15:
        return True
    return False


def isOnTable(obj, state):
    z = state[obj][2]
    if obj == 'cube' and z < 0.48 - 0.15:
        return True
    if obj == 'orange' and z < 0.48 - 0.15:
        return True
    if obj == 'tomato' and z < 0.49 - 0.15:
        return True
    if obj == 'mustard' and z < 0.48 - 0.15:
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
            fix_objs = [o for o in ['tomato', 'mustard'] if o in objects]
            final = drawPosition(env, fixedPositions=initial.fixed_state,
                                 fixedObjects=fix_objs, fixedOrientation=True,
                                 minSeparation=0.15, objOnTable=objOnTable)
        elif n_objects == 2:
            fix_objs = [o for o in ['mustard'] if o in objects]
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
    goal.mask = final.mask

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
            fix_objs = [o for o in ['tomato', 'mustard'] if o in objects]
            final = drawPosition(env, fixedPositions=initial.fixed_state,
                                 fixedObjects=fix_objs, fixedOrientation=True,
                                 minSeparation=0.15, objOnTable=objOnTable)
        elif n_objects == 2:
            fix_objs = [o for o in ['mustard'] if o in objects]
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
    goal.mask = final.mask

    print("SUCCESSFULL generation of GOAL25D with {} object(s)!!!!"
          .format(n_objects))

    return goal


def generateGoalREAL2020(env):

    print("Generating GOAL..")

    found = False
    while not(found):
        initial = drawPosition(env, fixedOrientation=True)
        found = True

    found = False
    while not(found):
        final = drawPosition(env, fixedOrientation=True)
        found = True

    goal = Goal()
    goal.challenge = 'REAL2020'
    goal.subtype = None
    goal.initial_state = initial.fixed_state
    goal.final_state = final.fixed_state
    goal.retina_before = initial.retina
    goal.retina = final.retina
    goal.mask = final.mask

    print("SUCCESSFULL generation of GOAL!")

    return goal


def visualizeGoalDistribution(all_goals, images=True):
    import matplotlib.pyplot as plt
    challenges = np.unique([goal.challenge for goal in all_goals])  
    fig, axes = plt.subplots(max(2,len(challenges)), 3)
    for c, challenge in enumerate(challenges):
        goals = [goal for goal in all_goals if goal.challenge == challenge]
        if len(goals) > 0:
            if images:
                #Superimposed images view
                tomatos = sum([goal.mask == 2 for goal in goals])
                mustards = sum([goal.mask == 3 for goal in goals])
                cubes = sum([goal.mask == 4 for goal in goals])
                axes[c, 0].imshow(tomatos, cmap='gray')
                axes[c, 1].imshow(mustards, cmap='gray')
                axes[c, 2].imshow(cubes, cmap='gray')
            else:
                #Positions scatter view
                for i, o in enumerate(goals[0].final_state.keys()):
                    positions = np.vstack([goal.final_state[o] for goal in goals])
                    axes[c, i].scatter(positions[:, 0], positions[:, 1])

    plt.show()


@click.command()
@click.option('--seed', type=int,
              help='Generate goals using this SEED for numpy.random')
@click.option('--n_goals', type=int, default=50,
              help='# of goals')
@click.option('--n_obj', type=int, default=3,
              help='# of objects')
def main(seed=None, n_goals=0, n_obj=0):
    """
        Generates the specified number of goals
        and saves them in a file.\n
        The file is called goals-REAL2020-s{}-{}-{}.npy.npz
        where enclosed brackets are replaced with the
        supplied options (seed, n_goals, n_obj) or 0.
    """
    np.random.seed(seed)
    allgoals = []
    env = gym.make('REALRobot2020-R1J{}-v0'.format(n_obj))
    if render:
        env.render('human')
    env.reset()

    global basePosition
    _, basePosition, _, _, _ = runEnv(env)

    # In these for loops, we could add some progress bar...
    for _ in range(n_goals):
        allgoals += [generateGoalREAL2020(env)]

    np.savez_compressed('goals-REAL2020-s{}-{}-{}.npy'
                        .format(seed, n_goals, n_obj), allgoals)

    checkRepeatability(env, allgoals)
    #visualizeGoalDistribution(allgoals)


if __name__ == "__main__":
    main()
