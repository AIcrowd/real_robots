from .envs import EnvCamera
import numpy as np
import time
import cv2
from PIL import ImageFont, ImageDraw, Image
from interval import interval

VIDEO_WIDTH = int(320)
VIDEO_HEIGHT = int(240)

class VideoMaker:
    """
    A class to create videos of the intrinsic and extrinsic phase.

    Parameters
    ----------
    intrinsic_timesteps: int, bool
        Maximum number of timesteps in the Intrinsic phase.
        If set to False, then


    """
    def __init__(self, env, intrinsic=None, extrinsic=None, debug=False):
        self.env = env
        self.camera = EnvCamera(1.0, 90, -45, 0, [-0.3, 0, .4], fov=90, width=VIDEO_WIDTH, height=VIDEO_HEIGHT)
        self.seed = np.random.randint(100000)
        self.current = None
        self.font = ImageFont.load_default()
        self.video_fps = 25
        self.speed_up = 1 #some speed up values will be rounded, see frame_freq
        self.frame_freq = int((200.0 / self.video_fps) * self.speed_up)
        self.debug = debug

        print(intrinsic, extrinsic, debug) # DEBUG

        if intrinsic:
            if type(intrinsic) == interval:
                self.intrinsic_frames = intrinsic
            elif type(intrinsic) == bool:
                self.intrinsic_frames = self.get_intrinsic_frames()
            else:
                raise Exception("VideoMaker intrinsic param has to be either" +
                                "None/False, an interval or True")

        if extrinsic:
            if type(extrinsic) == interval:
                self.extrinsic_trials = extrinsic
            elif type(extrinsic) == bool:
                self.extrinsic_trials = self.get_extrinsic_trials()
            else:
                raise Exception("VideoMaker extrinsic param has to be either" +
                                "None/False, an interval or True")

    def get_intrinsic_frames(self):
        int_steps = self.env.intrinsic_timesteps
        one_min_frames = 60 * self.video_fps * self.frame_freq
        return interval([0, one_min_frames], 
                        [int_steps / 2, int_steps + one_min_frames],
                        [int_steps - one_min_frames, int_steps])

    def get_extrinsic_trials(self):
        ext_trials = self.env.extrinsic_trials
        if ext_trials > 0:
            random_trials = np.random.randint(0, ext_trials, 5)
            return interval(*random_trials)
        else:
            return interval()

    def start_intrinsic(self):
        time_string = time.strftime("%Y,%m,%d,%H,%M").split(',')
        filename = "Simulation-{}-y{}-m{}-d{}-h{}-m{}-intrinsic.avi".format(self.seed, *time_string)
        self.video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), self.video_fps, (VIDEO_WIDTH, VIDEO_HEIGHT), isColor=True)

    def update_intrinsic(self, steps):
        if steps in self.intrinsic_frames:
            if steps % self.frame_freq == 0:
                camera = Image.fromarray(self.camera.render(self.env._p))
                self.video.write(cv2.cvtColor(np.array(camera.getdata()).reshape((VIDEO_HEIGHT, VIDEO_WIDTH, 3)).astype(np.uint8), cv2.COLOR_RGB2BGR))

    def end_intrinsic(self):
        cv2.destroyAllWindows()
        self.video.release()

    def getGoal(self, observation):
        goal = observation['goal']
        goal = Image.fromarray(goal)
        g_width = int(VIDEO_WIDTH / 3)
        g_height = int(VIDEO_HEIGHT / 3)
        goal = goal.resize((g_width, g_height))
        d = ImageDraw.Draw(goal)
        w, h = d.textsize("GOAL", font = self.font)
        d.text((int((g_width-w)/2), int((g_height*0.75)-h/2)), "GOAL", fill=(0, 0, 0), font = self.font)
        self.goal = goal

    def end_trial(self):
        if self.trial_number in self.extrinsic_trials:
            cv2.destroyAllWindows()
            self.video.release()

    def start_trial(self, observation, trial_number):
        self.trial_number = trial_number
        if self.trial_number in self.extrinsic_trials:
            time_string = time.strftime("%Y,%m,%d,%H,%M").split(',')
            filename = "Simulation-{}-y{}-m{}-d{}-h{}-m{}-trial-{}.avi".format(self.seed, *time_string, self.trial_number)
            self.video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), self.video_fps, (VIDEO_WIDTH, VIDEO_HEIGHT), isColor=True)
            self.getGoal(observation)

    def extrinsic_trial(self, observation, action, steps, score_object):
        if self.trial_number in self.extrinsic_trials:
            if steps % self.frame_freq == 0:
                camera = Image.fromarray(self.camera.render(self.env._p))
                camera.paste(self.goal, (VIDEO_WIDTH-int(VIDEO_WIDTH / 3), 0))
                self.video.write(cv2.cvtColor(np.array(camera.getdata()).reshape((VIDEO_HEIGHT, VIDEO_WIDTH, 3)).astype(np.uint8), cv2.COLOR_RGB2BGR))
