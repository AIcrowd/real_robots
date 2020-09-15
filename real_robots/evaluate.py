# -*- coding: utf-8 -*-

import gym
from .envs import Goal  # noqa F401
from .policy import BasePolicy
import numpy as np
import time
from tqdm.auto import tqdm
import aicrowd_api

"""Local evaluation helper functions."""


class EvaluationService:
    """
    A generic class containing all necessary functions for local
    and remote evaluation.

    Parameters
    ----------
    Controller: class
        An example controller which should expose a `step` function and be a
        subclass of BasePolicy
    environment: string
        "R1" or "R2", which represent Round1 and Round2 of the competition
    action_type: string
        "cartesian", "joints" or "macro_action" action type
    n_objects: int
        number of objects on the table: 1, 2 or 3
    intrinsic_timesteps: int
        Number of timesteps in the Intrinsic phase (default 15e6)
    extrinsic_timesteps: int
        Number of timesteps in the Extrinsic phase (default 10e3)
    extrinsic_trials: int
        Total number of trials in the extrinsic phase (default 50)
    visualize: bool
        Boolean flag which enables or disables the visualization GUI
        when running the evaluation
    goals_dataset_path: str
        Path to a goals dataset
    """
    def __init__(self,
                 Controller,
                 environment='R1',
                 action_type='macro_action',
                 n_objects=1,
                 intrinsic_timesteps=15e6,
                 extrinsic_timesteps=10e3,
                 extrinsic_trials=50,
                 visualize=True,
                 goals_dataset_path="./goals.npy.npz"):

        self.ControllerClass = Controller
        self.intrinsic_timesteps = intrinsic_timesteps
        self.extrinsic_timesteps = extrinsic_timesteps
        self.extrinsic_trials = extrinsic_trials
        self.visualize = visualize
        self.goals_dataset_path = goals_dataset_path

        # Start Setup
        self.setup_gym_env(environment, action_type, n_objects)
        self.setup_controller()
        self.setup_evaluation_state()
        self.setup_scores()
        self.setup_aicrowd_helpers()

    def setup_aicrowd_helpers(self):
        self.aicrowd_events = aicrowd_api.events.AIcrowdEvents()

    def setup_evaluation_state(self):
        """
        # Setup Evaluation State
        ##########################################################
        State Transitions:
        overall_state:
            PENDING -> INTRINSIC_PHASE_IN_PROGRESS
            INTRINSIC_PHASE_IN_PROGRESS -> INTRINSIC_PHASE_COMPLETE
            INTRINSIC_PHASE_COMPLETE -> EXTRINSIC_PHASE_IN_PROGRESS
            EXTRINSIC_PHASE_IN_PROGRESS -> EXTRINSIC_PHASE_COMPLETE
            ERROR
            EVALUATION_COMPLETE
        intrinsic_phase_state:
            PENDING
            INTRINSIC_PHASE_IN_PROGRESS
            INTRINSIC_PHASE_COMPLETE
            INTRINSIC_PHASE_SKIPPED
            INTRINSIC_PHASE_ERROR
        extrinsic_phase_state:
            PENDING
            EXTRINSIC_PHASE_IN_PROGRESS
            EXTRINSIC_PHASE_COMPLETE
            EXTRINSIC_PHASE_ERROR
        """
        self.evaluation_state = {  # noqa
            "state": "PENDING",
            "intrinsic_phase_state": "PENDING",
            "extrinsic_phase_state": "PENDING",
            "max_intrinsic_timesteps": self.intrinsic_timesteps,
            "max_extrinsic_timesteps": self.extrinsic_timesteps,
            "current_intrinsic_timestep": 0,
            "max_extrinsic_trials": self.extrinsic_trials,
            "num_extrinsic_trials_complete": 0,
            "progress_in_current_extrinsic_trial": 0,
            "evaluation_score": {
                "score": 0,
                "score_2D": 0,
                "score_2.5D": 0,
                "score_3D": 0,
                "score_total": 0
            },
            "score": {
                "score": 0,
                "score_secondary": 0
            }
        }

    def sync_evaluation_state(self):
        """
        Syncs the evaluation state with the evaluator
        """
        # Determine event type
        event_type = self.aicrowd_events.AICROWD_EVENT_INFO
        if self.evaluation_state["state"] == "ERROR":
            event_type = self.aicrowd_events.AICROWD_EVENT_ERROR
        elif self.evaluation_state["state"] == "EVALUATION_COMPLETE":
            event_type = self.aicrowd_events.AICROWD_EVENT_SUCCESS

        # Register event type
        try:
            self.aicrowd_events.register_event(
                event_type=event_type,
                payload=self.evaluation_state
            )
        except:
            # If evaluation is successful, better to try till overall timeout is reached
            if self.evaluation_state["state"] == "EVALUATION_COMPLETE":
                print("Evaluation succcessful but cant communicate to sourcerer, retrying...")
                time.sleep(10)
                self.sync_evaluation_state()
            pass

    def setup_gym_env(self, environment, action_type, n_objects):

        if environment in ["R1", "R2"]:
            rnd = environment
        else:
            raise Exception("Environment type has to be either R1 or R2")

        if action_type in ['joints', 'cartesian', 'macro_action']:
            act = action_type[0].upper()
        else:
            raise Exception("Action type has to be either 'joints', 'cartesian',"
                            "or 'macro_action'")

        if isinstance(n_objects, int) and 1 <= n_objects <= 3:
            n_obj = n_objects
        else:
            raise Exception("Number of objects has to be 1, 2 or 3.")

        envString = 'REALRobot2020-{}{}{}-v0'.format(rnd, act, n_obj)
        self.env = gym.make(envString)
        self.env.set_goals_dataset_path(self.goals_dataset_path)
        self.env.intrinsic_timesteps = self.intrinsic_timesteps  # default=15e6
        self.env.extrinsic_timesteps = self.extrinsic_timesteps  # default=10e3
        
        if self.visualize:
            self.env.render('human')

    def setup_controller(self):
        if not issubclass(self.ControllerClass, BasePolicy):
            raise Exception(
                    "Supplied Controller is not a Sub-Class of "
                    "real_robots.policy.BasePolicy . Please ensure that "
                    "the supplied controller class is derived from "
                    "real_robots.policy.BasePolicy , as described in the "
                    "example here at: "
                    "https://github.com/AIcrowd/real_robots#usage"
                )

        self.controller = self.ControllerClass(self.env.action_space, self.env.observation_space)

    def setup_scores(self):
        self.scores = {}

    def add_scores(self, challenge, score):
        """
        Simple helper function
        """
        if challenge in self.scores.keys():
            self.scores[challenge] += [score]
        else:
            self.scores[challenge] = [score]

    def run_intrinsic_phase(self):
        try:
            self._run_intrinsic_phase()
        except Exception as e:
            self.evaluation_state["state"] = "ERROR"
            self.evaluation_state["intrinsic_phase_state"] = \
                "INTRINSIC_PHASE_ERROR"
            self.sync_evaluation_state()
            raise e

    def _run_intrinsic_phase(self):
        """
        Runs the intrinsic phase based on the evaluation params
        """
        if not self.intrinsic_timesteps:
            # Set intrinsic_timesteps = 0 if its set as False
            self.intrinsic_timesteps = 0

        if self.intrinsic_timesteps > 0:
            observation = self.env.reset()
            reward = 0
            done = False
            intrinsic_phase_progress_bar = tqdm(
                                total=self.intrinsic_timesteps,
                                desc="Intrinsic Phase",
                                unit="steps ",
                                leave=True
                                )
            intrinsic_phase_progress_bar.write(
                    "######################################################")
            intrinsic_phase_progress_bar.write("# Intrinsic Phase Initiated")
            intrinsic_phase_progress_bar.write(
                    "######################################################")
            self.evaluation_state["intrinsic_phase_state"] = \
                "INTRINSIC_PHASE_IN_PROGRESS"
            self.evaluation_state["state"] = "INTRINSIC_PHASE_IN_PROGRESS"
            self.sync_evaluation_state()

            # intrinsic phase
            steps = 0
            # Notify the controller that the intrinsic phase started
            self.controller.start_intrinsic_phase()


            while not done:
                # Call your controller to chose action
                action = self.controller.step(observation, reward, done)
                # do action
                observation, reward, done, _ = self.env.step(action)
                steps += 1
                intrinsic_phase_progress_bar.update(1)
                self.evaluation_state["current_intrinsic_timestep"] = steps
                self.sync_evaluation_state()

            intrinsic_phase_progress_bar.write(
                "######################################################")
            intrinsic_phase_progress_bar.write("# Intrinsic Phase Complete")
            intrinsic_phase_progress_bar.write(
                "######################################################")
            # Change evaluation state to represent the transition
            self.evaluation_state["intrinsic_phase_state"] = \
                "INTRINSIC_PHASE_COMPLETE"
            self.evaluation_state["state"] = "INTRINSIC_PHASE_COMPLETE"
            self.sync_evaluation_state()
            # Notify the controller that the intrinsic phase ended
            self.controller.end_intrinsic_phase(observation, reward, done)
        else:
            print("[WARNING] Skipping Intrinsic Phase as intrinsic_timesteps = 0 or False")  # noqa
            self.evaluation_state["state"] = "INTRINSIC_PHASE_SKIPPED"
            self.sync_evaluation_state()

        
        

    def run_extrinsic_trial(self, trial_number):
        self.env.reset()
        reward = 0
        done = False
        observation = self.env.set_goal()

        # Notify the controller that an extrinsic trial started
        self.controller.start_extrinsic_trial()

        extrinsic_trial_progress_bar = \
            tqdm(
                total=self.extrinsic_timesteps,
                desc="Extrinsic Trial # {}".format(trial_number),
                unit="steps ",
                leave=False
                )

        steps = 0
        while not done:
            action = self.controller.step(observation, reward, done)
            observation, reward, done, _ = self.env.step(action)
            extrinsic_trial_progress_bar.update(1)
            steps += 1
            progress = float(steps) / self.extrinsic_timesteps
            # Change evaluation state to represent the transition
            self.evaluation_state["progress_in_current_extrinsic_trial"] = \
                progress
            self.sync_evaluation_state()

        # Evaluate Current Goal
        self.add_scores(*self.env.evaluateGoal())

        # Change evaluation state to represent the transition
        self.evaluation_state["num_extrinsic_trials_complete"] = \
            trial_number + 1
        self.sync_evaluation_state()
        # Notify the controller that an extrinsic trial ended
        self.controller.end_extrinsic_trial(observation, reward, done)
        extrinsic_trial_progress_bar.close()
        

    def run_extrinsic_phase(self):
        try:
            self._run_extrinsic_phase()
        except Exception as e:
            self.evaluation_state["state"] = "ERROR"
            self.evaluation_state["extrinsic_phase_state"] = \
                "EXTRINSIC_PHASE_ERROR"
            self.sync_evaluation_state()
            raise e

    def _run_extrinsic_phase(self):
        extrinsic_phase_progress_bar = tqdm(
                                            total=self.extrinsic_trials,
                                            desc="Extrinsic Phase",
                                            unit="trials ",
                                            leave=True
                                            )
        extrinsic_phase_progress_bar.write(
            "######################################################")
        extrinsic_phase_progress_bar.write("# Extrinsic Phase Initiated")
        extrinsic_phase_progress_bar.write(
            "######################################################")
        # Change evaluation state to represent the transition
        self.evaluation_state["extrinsic_phase_state"] = \
            "EXTRINSIC_PHASE_IN_PROGRESS"
        self.evaluation_state["state"] = "EXTRINSIC_PHASE_IN_PROGRESS"
        self.sync_evaluation_state()
        # Notify the controller that the extrinsic phase started
        self.controller.start_extrinsic_phase()
        
        
        
        for trial in range(self.extrinsic_trials):
            
            self.run_extrinsic_trial(trial)

            extrinsic_phase_progress_bar.update(1)
            extrinsic_phase_progress_bar.set_postfix(
                            self.build_score_object()
                        )

        extrinsic_phase_progress_bar.write(
            "######################################################")
        extrinsic_phase_progress_bar.write("# Extrinsic Phase Complete")
        extrinsic_phase_progress_bar.write(
            "######################################################")
        extrinsic_phase_progress_bar.write(
                                        str(self.build_score_object()))
        # Change evaluation state to represent the transition
        self.evaluation_state["extrinsic_phase_state"] = \
            "EXTRINSIC_PHASE_COMPLETE"
        self.evaluation_state["state"] = "EXTRINSIC_PHASE_COMPLETE"
        self.evaluation_state["score"] = {
            "score": self.evaluation_state["evaluation_score"]["score_total"],
            "score_secondary" : self.evaluation_state["evaluation_score"]["score_2D"]  # noqa
        }
        self.evaluation_state["meta"] = self.evaluation_state["evaluation_score"]  # noqa
        self.evaluation_state["state"] = "EVALUATION_COMPLETE"
        self.sync_evaluation_state()
        # Notify the controller that the extrinsic phase ended
        self.controller.end_extrinsic_phase()

        
        return self.build_score_object()

    def build_score_object(self):
        total_score = 0
        total_results = []
        challenges = ['2D', '2.5D', '3D']

        score_object = {}
        for key in challenges:
            if key in self.scores.keys():
                results = self.scores[key]
                challenge_score = np.mean(results)
            else:
                results = []
                challenge_score = 0
            total_results += results
            score_object["score_{}".format(key)] = challenge_score
        total_score = np.mean(total_results)

        score_object["score_total"] = total_score
        # Mark Changes in evaluation_state
        self.evaluation_state["evaluation_score"] = score_object
        self.sync_evaluation_state()
        return score_object


def evaluate(Controller,
             environment='R1',
             action_type='macro_action',
             n_objects=1,
             intrinsic_timesteps=15e6,
             extrinsic_timesteps=10e3,
             extrinsic_trials=50,
             visualize=True,
             goals_dataset_path="./goals.npy.npz"):

    evaluation_service = EvaluationService(
        Controller,
        environment,
        action_type,
        n_objects,
        intrinsic_timesteps,
        extrinsic_timesteps,
        extrinsic_trials,
        visualize,
        goals_dataset_path
        )
    
    evaluation_service.run_intrinsic_phase()
    evaluation_service.run_extrinsic_phase()
    
    return evaluation_service.build_score_object(), evaluation_service.scores
