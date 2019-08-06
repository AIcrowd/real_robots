# -*- coding: utf-8 -*-

import gym
from .envs import Goal
import numpy as np

from tqdm.auto import tqdm

"""Local evaluation helper functions."""


def build_score_object(scores):
    score_string = ""
    total_score = 0
    challenges = ['2D','2.5D','3D']

    score_object = {}
    for key in challenges:
        if key in scores.keys():
            results = scores[key]
            formatted_results = ", ".join(["{:.4f}".format(r) for r in results])
            challenge_score = np.mean(results)
        else:
            results = []
            formatted_results = "None"
            challenge_score = 0
        total_score += challenge_score
        score_object["score_{}".format(key)] = challenge_score
    total_score /= len(challenges)

    score_object["score_total"] = total_score
    return  score_object


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

    ##########################################################
    ##########################################################    
    observation = env.reset()
    reward = 0
    done = False    
    intrinsic_phase_progress_bar = tqdm(
                        total=intrinsic_timesteps,
                        desc="Intrinsic Phase",
                        unit="steps ",
                        leave=False
                        )
    intrinsic_phase_progress_bar.write("######################################################")
    intrinsic_phase_progress_bar.write("# Intrinsic Phase Initiated")
    intrinsic_phase_progress_bar.write("######################################################")

    # intrinsic phase
    steps = 0
    while not done:
        # Call your controller to chose action 
        action = controller.step(observation, reward, done)
        # do action
        observation, reward, done, _ = env.step(action)
        steps += 1
        intrinsic_phase_progress_bar.update(1)
    intrinsic_phase_progress_bar.write("######################################################")
    intrinsic_phase_progress_bar.write("# Intrinsic Phase Complete")
    intrinsic_phase_progress_bar.write("######################################################")
    # extrinsic phase
    # tqdm.write("Starting extrinsic phase")

    extrinsic_phase_progress_bar = tqdm(
                                        total = extrinsic_trials,
                                        desc = "Extrinsic Phase Trials",
                                        unit = "trials ",
                                        leave=True
                                        )
    extrinsic_phase_progress_bar.write("######################################################")
    extrinsic_phase_progress_bar.write("# Extrinsic Phase Initiated")
    extrinsic_phase_progress_bar.write("######################################################")

    totalScore = 0
    for k in range(extrinsic_trials):
        observation = env.reset()
        reward = 0
        done = False
        extrinsic_phase_progress_bar.update(1)
        env.set_goal()

        while not done:
            action = controller.step(observation, reward, done)
            observation, reward, done, _ = env.step(action)

        add_scores(*env.evaluateGoal())
        extrinsic_phase_progress_bar.set_postfix(
                        build_score_object(
                                scores
                            )
                    )

    extrinsic_phase_progress_bar.write("######################################################")
    extrinsic_phase_progress_bar.write("# Extrinsic Phase Complete")
    extrinsic_phase_progress_bar.write("######################################################")
    extrinsic_phase_progress_bar.write(str(build_score_object(scores)))
    return build_score_object(scores)
