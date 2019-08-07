import numpy as np


class BasePolicy:

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
        raise NotImplementedError(
            "the step function has not been implemented. "
            "Please derive your custom controller from this class "
            "and implement atleast the step function. And optionally "
            "the other available interfaces."
        )

    def start_intrinsic_phase(self):
        """
        The evaluator will call this function to signal the start of the
        Intrinsic Phase.
        Next observation will be from the intrinsic phase.
        """
        pass

    def end_intrinsic_phase(self):
        """
        The evaluator will call this function to signal the end of the
        Intrinsic Phase.
        """
        pass


    def start_extrinsic_phase(self):
        """
        The evaluator will call this function to signal the start of the
        Extrinsic Phase.
        Next observation will be from the extrinsic phase.
        """
        pass

    def end_extrinsic_phase(self):
        """
        The evaluator will call this function to signal the end of the
        Extrinsic Phase.
        """
        pass

    def start_extrinsic_trial(self):
        """
        The evaluator will call this function to signal the start of each
        extrinsic trial.
        Next observation will have a (new) goal.
        """
        pass

    def end_extrinsic_trial(self):
        """
        The evaluator will call this function to signal the end of each
        extrinsic trial.
        """
        pass
