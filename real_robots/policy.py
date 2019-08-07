import numpy as np


class Policy:

    def step(self, observation, reward, done):
        """
        The step function will receive the observation, reward and done signals
        from the environment and computes the next action to take.

        Parameters
        ----------
        observation : dict
            The dictionary will contain the following entries:
                - "joint_positions"
                    Values of the joints position of the arm,
                    including the gripper.
                - "touch_sensors"
                    Values recorded by the touch sensors.
                - "retina"
                    Image of the environment from the camera
                    above the table.
                - "goal"
                    Image of the goal, showing how the objects
                    should be arranged in the environment.
        reward: float
            This will be always zero.
        done: bool
            This will be True when:
                - intrisic phase ends
                - an extrinsic trial ends
            otherwise it will always be false.
        """
        return np.zeros(9)

    def setIntrisicPhase(self):
        """
        The evaluator will call this function to signal the start of the
        Intrinsic Phase.
        Next observation will be from the intrinsic phase.
        """
        pass

    def setExtrinsicPhase(self):
        """
        The evaluator will call this function to signal the start of the
        Extrinsic Phase.
        Next observation will be from the extrinsic phase.
        """
        pass

    def startExtrinsicTrial(self):
        """
        The evaluator will call this function to signal the start of each
        extrinsic trial.
        Next observation will have a (new) goal.
        """
        pass

    def endExtrinsicTrial(self):
        """
        The evaluator will call this function to signal the end of each
        extrinsic trial.
        """
        pass
