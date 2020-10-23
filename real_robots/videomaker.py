from .envs import EnvCamera
import numpy as np
import time
import cv2
from PIL import ImageFont, ImageDraw, Image

VIDEO_WIDTH = int(1920 / 1)
VIDEO_HEIGHT = int(1080 / 1)

class VideoMaker:
    """
    A class to create videos of the intrinsic and extrinsic phase.

    Parameters
    ----------
    intrinsic_timesteps: int, bool
        Maximum number of timesteps in the Intrinsic phase.
        If set to False, then


    """
    def __init__(self):
#    def __init__(self, distance, yaw, pitch, roll, pos, fov=80, width=320, height=240)
        self.camera = EnvCamera(1.0, 90, -45, 0, [-0.3, 0, .4], fov=90, width=VIDEO_WIDTH, height=VIDEO_HEIGHT)
        self.seed = np.random.randint(100000)
        self.current = None
        self.font = ImageFont.truetype('/home/summer/anaconda3/envs/rlkit/lib/python3.6/site-packages/pygame/examples/data/sans.ttf', 36)

    def start_intrinsic(self):
        time_string = time.strftime("%Y,%m,%d,%H,%M").split(',')
        filename = "Simulation-{}-y{}-m{}-d{}-h{}-m{}-intrinsic.avi".format(self.seed, *time_string)
        self.video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 10, (VIDEO_WIDTH, VIDEO_HEIGHT), isColor=True)

    def update_intrinsic(self, steps):
        if steps < 10000:
            if steps % 50 == 0:
                camera = Image.fromarray(self.camera.render())
                self.video.write(cv2.cvtColor(np.array(camera.getdata()).reshape((VIDEO_HEIGHT, VIDEO_WIDTH, 3)).astype(np.uint8), cv2.COLOR_RGB2BGR))

    def end_intrinsic(self):
        cv2.destroyAllWindows()
        self.video.release()

    def getGoal(self, observation):
        goal = observation['goal']
        goal = Image.fromarray(goal)
        g_width = 640
        g_height = 480
        goal = goal.resize((g_width, g_height))
        d = ImageDraw.Draw(goal)
        w, h = d.textsize("GOAL", font = self.font)
        d.text((int((g_width-w)/2), int((g_height*0.75)-h/2)), "GOAL", fill=(0, 0, 0), font = self.font)
        self.goal = goal

    def end_trial(self):
        cv2.destroyAllWindows()
        self.video.release()

    def start_trial(self, observation, trial_number):
        self.trial_number = trial_number
        time_string = time.strftime("%Y,%m,%d,%H,%M").split(',')
        filename = "Simulation-{}-y{}-m{}-d{}-h{}-m{}-trial-{}.avi".format(self.seed, *time_string, self.trial_number)
        self.video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 30, (VIDEO_WIDTH, VIDEO_HEIGHT), isColor=True)
        self.getGoal(observation)

    def extrinsic_trial(self, observation, action, steps, score_object):
        if self.trial_number not in [20,21]:
            return

        if steps % 15 == 0:
            camera = Image.fromarray(self.camera.render())
            camera.paste(self.goal, (VIDEO_WIDTH-640, 0))
            self.video.write(cv2.cvtColor(np.array(camera.getdata()).reshape((VIDEO_HEIGHT, VIDEO_WIDTH, 3)).astype(np.uint8), cv2.COLOR_RGB2BGR))
