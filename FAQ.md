# FAQ

* Is this competition really so cool?  
Yes.

* Can I generate different goals for local testing?  
Yes, see [here](https://github.com/AIcrowd/real_robots/wiki/Script-for-generating-new-goal-sets).  
You will be able to generate any amount of goals of each category (2D, 2.5D, 3D) and also to specify how many objects have to be moved from the initial state to reach the goal in the 2D and 2.5D categories.

* FileNotFoundError - goals_dataset.npy.npz  
The environment assumes there is a `goals_dataset.npy.npz` file in the current directory, unless you use the `set_goals_dataset_path` function to specificy otherwise.
You can either generate a goal dataset using the [command `real-robots-generate-goals`](https://github.com/AIcrowd/real_robots/wiki/Script-for-generating-new-goal-sets) or you can download one at the following link: https://aicrowd-production.s3.eu-central-1.amazonaws.com/misc/REAL-Robots/goals.npy.npz 

## Known issues

* __render('human') and visualization=True render shadows and background both in the GUI and in the robot camera__  
If you use `env.render('human')` or put `visualization=True` in the `evaluate` function it will open a GUI showing the environment. The GUI renders the environment using shadows and also shows a bluish background.
As a side effect, using the GUI will also make the environment render the shadows and the background on the robot camera.  
___This means that robots trained on images with the GUI on might have a different performance than when the GUI is off, since images will be different.___  
It is possible to disable shadows from the GUI using the shortcut `s` or also by setting shadows off using `pybullet.configureDebugVisualizer(pybullet.COV_ENABLE_SHADOWS,0)`, however the background will still be bluish on the robot camera (without the GUI it will be white).

