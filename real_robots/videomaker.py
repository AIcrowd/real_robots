from .envs import EnvCamera
import numpy as np
import time
import os
import cv2
from PIL import Image, ImageDraw, ImageFilter

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
        self.camera = EnvCamera(1.2,30,-30,0,[0, 0, .4],width=960, height=720) 
        self.seed = np.random.randint(100000)
        self.current = None


    def getGoal(self, observation):
        goal = observation['goal']
        goal = Image.fromarray(goal)        
        goal = goal.resize((96,72))
        d = ImageDraw.Draw(goal)
        d.text((int(96*0.4),int(72*0.8)), "GOAL", fill=(0,0,0))
        self.goal = goal

    def updateCurrentTrialStatus(self, observation):
        retina = observation['retina']
        self.current = Image.fromarray(retina)
        d = ImageDraw.Draw(self.current)

        n_obj = len(observation['object_positions'].keys())
        if n_obj == 1:
            d.text((int(320*0.35),int(240*0.75)), "CURRENT DISTANCE:", fill=(0,0,0)) 
        else:
            d.text((int(320*0.35),int(240*0.75)), "CURRENT DISTANCES:", fill=(0,0,0)) 
        string = ""
        intial_dist = {}
        for key in observation['object_positions'].keys():
            intial_dist[key] = np.linalg.norm(observation['object_positions'][key][:3]-observation['goal_positions'][key][:3]) * 100
            string = string + str(key).upper() + ": " + str(intial_dist[key])[:4] + " cm; " 
        n_obj = len(observation['object_positions'].keys()) 
        d.text((int(320*(0.37 - 0.14 * (n_obj-1))),int(240*0.8)), string, fill=(0,0,0)) 

        if n_obj == 1:
            d.text((int(320*0.35),int(240*0.85)), "INITIAL DISTANCE:", fill=(0,0,0)) 
        else:
            d.text((int(320*0.35),int(240*0.85)), "INITIAL DISTANCES:", fill=(0,0,0)) 
        string = ""
        for key in observation['object_positions'].keys():
            string = string + str(key).upper() + ": " + str(intial_dist[key])[:4] + " cm; " 
        d.text((int(320*(0.37 - 0.14 * (n_obj-1))),int(240*0.9)), string, fill=(0,0,0)) 

        self.current.paste(self.goal,(224,0))         


    def end_trial(self):
        cv2.destroyAllWindows()
        self.video.release()  

    def start_trial(self, observation, trial_number):
        self.trial_number = trial_number
        time_string = time.strftime("%Y,%m,%d,%H,%M").split(',')
        filename = "Simulation-{}-y{}-m{}-d{}-h{}-m{}-trial-{}.avi".format(self.seed, *time_string, self.trial_number)
        self.video = cv2.VideoWriter(filename, cv2.VideoWriter_fourcc(*'XVID'), 10, (960,720),isColor=True)
        self.getGoal(observation)
        self.updateCurrentTrialStatus(observation)
        
    def extrinsic_trial(self, observation, action, steps, score_object):
        if action['render']:
            self.updateCurrentTrialStatus(observation)

        if steps % 50 == 0:
            camera = Image.fromarray(self.camera.render())
            camera.paste(self.current,(640,0))
            
            d = ImageDraw.Draw(camera)
            d.text((int(960*0.75),int(720*0.65)), "Action: \n" + str(action['macro_action']), fill=(0,0,0)) 
            d.text((int(960*0.75),int(720*0.75)), "Trial: " + str(self.trial_number) + " Step: " + str(steps), fill=(0,0,0)) 
            if self.trial_number:
                d.text((int(960*0.7),int(720*0.8)), "Total score: " + str(score_object["score_total"])[:5], fill=(0,0,0)) 
                d.text((int(960*0.7),int(720*0.85)), "Score 2D: " + str(score_object['score_2D'])[:5] + " Score 2.5D: " + str(score_object['score_2.5D'])[:5] + " Score 3D: " + str(score_object['score_3D'])[:5], fill=(0,0,0))

            self.video.write(cv2.cvtColor(np.array(camera.getdata()).reshape((720,960,3)).astype(np.uint8),cv2.COLOR_RGB2BGR))

