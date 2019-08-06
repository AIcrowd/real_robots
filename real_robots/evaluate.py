# -*- coding: utf-8 -*-

import gym
from real_robots.envs import Goal
import numpy as np


"""Local evaluation helper functions."""

def evaluate(Controller, 
            intrinsic_timesteps=1e7,
            extrinsic_timesteps=2e3,
            extrinsic_trials=350,
            visualize=True,
            goals_dataset_path="./goals.npy.npz"):
    """
    A wrapper function to locally simulate the evaluation process
    as is done for all the submitted controllers.

    Parameters
    ----------
    Controller 
        An example controller which should expose a `step` function, for 
        the evaluator to compute the `action` given observation, reward
        and done info
    
    intrinsic_timesteps : int
        Maximum number of timesteps in the Intrinsic phase
    extrinsic_timesteps: int
        Maximum number of timesteps in the Extrinsic phase
    extrinsic_trials: int
        Total number of trials in the extrinsic phase
    visualize: bool
        Boolean flag which enables or disables the visualizer when 
        running the evaluation
    goals_dataset_path: str
        Path to a goals dataset
    """
    env = gym.make('REALRobot-v0')
    env.set_goals_dataset_path(goals_dataset_path)

    if visualize:
        env.render('human')
    
    controller = Controller(env.action_space)

    env.intrinsic_timesteps = intrinsic_timesteps #default = 1e7
    env.extrinsic_timesteps = extrinsic_timesteps #default = 2e3
    extrinsic_trials = 3

    ##########################################################
    ##########################################################
    # Helper functions
    ##########################################################
    ##########################################################
    scores = {}
    def add_scores(challenge, score):
        if challenge in scores.keys():
            scores[challenge] += [score]
        else:
            scores[challenge] = [score]

    def report_score():
        print("*****************")
        total_score = 0
        challenges = ['2D','2.5D','3D']
        for key in challenges:
            if key in scores.keys():
                results = scores[key]
                formatted_results = ", ".join(["{:.4f}".format(r) for r in results])
                challenge_score = np.mean(results)
            else:
                results = []
                formatted_results = "None"
                challenge_score = 0

            print("Challenge {} - {:.4f}".format(key, challenge_score))
            print("Goals: {}".format(formatted_results))
            total_score += challenge_score
        total_score /= len(challenges)
        print("Overall Score: {:.4f}".format(total_score))  
        print("*****************")
    ##########################################################
    ##########################################################
    
    observation = env.reset() 
    reward = 0 
    done = False

    # intrinsic phase
    while not done:
        # Call your controller to chose action 
        action = controller.step(observation, reward, done)
        # do action
        observation, reward, done, _ = env.step(action)
    
    # extrinsic phase
    print("Starting extrinsic phase")
    totalScore = 0
    for k in range(extrinsic_trials):
        observation = env.reset()
        reward = 0
        done = False

        env.set_goal()
        print("Starting extrinsic trial...")

        while not done:
            action = controller.step(observation, reward, done)
            observation, reward, done, _ = env.step(action)

        add_scores(*env.evaluateGoal())
        print("Current score:")
        report_score()
    
    print("Final score:")
    report_score()
