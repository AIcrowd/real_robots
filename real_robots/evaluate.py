# -*- coding: utf-8 -*-

import gym
from real_robots.envs import Goal
import numpy as np


"""Local evaluation helper functions."""

def evaluate(Controller, 
            extrinsic_trials=350,
            goals_dataset_path="./goals.npy.npz",
            debug=True):
    env = gym.make('REALRobot-v0')
    env.set_goals_dataset_path(goals_dataset_path)
    controller = Controller(env.action_space)

    if debug:
        """
        Debug mode for local testing
        """
        env.intrinsic_timesteps = 100 #default = 1e7
        env.extrinsic_timesteps = 100 #default = 2e3
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
