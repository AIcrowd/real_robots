# REALRobot environment

The REALRobot environment is a standard gym environment.
It includes a 7DoF kuka arm with a 2Dof gripper, a table with 3 objects on it and a camera looking at the table from the top. 
The gripper has four touch sensors on the inner part of its links.

#### Action
The ```action```attribute  of ```env.step``` must be a  vector of 9 joint positions in radiants.
The first 7 joints have a range between -Pi/2 and +Pi/2.
The two gripper joints have a range between 0 and +Pi/2. They are also coupled so that the second joint will be at most twice the angle of the first one.


<TABLE " width="100%" BORDER="0">
<TR>
<TD>
       
| index |  joint name               |
| ----- | ------------------------- |
|  0    |  lbr_iiwa_joint_1         |
|  1    |  lbr_iiwa_joint_2         |
|  2    |  lbr_iiwa_joint_3         |
|  3    |  lbr_iiwa_joint_4         |
|  4    |  lbr_iiwa_joint_5         |
|  5    |  lbr_iiwa_joint_6         |
|  6    |  lbr_iiwa_joint_7         |
|  7    |  base_to_finger0_joint    |
|  8    |  finger0_to_finger1_joint |

</TD>
<TD><img src="https://raw.githubusercontent.com/GOAL-Robots/REALCompetitionStartingKit/1e66f1986bd8049c0fee4bd470599b8c22f8dd15/docs/figs/kuka_full_joints.png" alt="kuka_full_joints" width="80%"></TD>
<TD><img src="https://raw.githubusercontent.com/GOAL-Robots/REALCompetitionStartingKit/1e66f1986bd8049c0fee4bd470599b8c22f8dd15/docs/figs/kuka_gripper_joints.png" alt="kuka_gripper_joints" width="80%"></TD>
</TR>
</TABLE>

#### Observation
The ```observation``` object returned by```env.step``` is a dictionary:
* observation["joint_positions"] is a vector containing the current angles of the 9 joints
* observation["touch_sensors"] is a vector containing the current touch intensity at the four touch sensors (see figure below)
* observation["retina"] is a 240x320x3 array with the current top camera image
* observation["goal"] is a 240x320x3 array with the target top camera image (all zeros except for the extrinsic phase, see below)

<TABLE " width="100%" BORDER="0">
<TR>
</TD>
<TD><img src="https://raw.githubusercontent.com/GOAL-Robots/REALCompetitionStartingKit/1e66f1986bd8049c0fee4bd470599b8c22f8dd15/docs/figs/kuka_gripper_sensors.png" alt="kuka_sensors" width="40%"></TD>
</TR>
</TABLE>
For each sensor, intensity is defined as the maximum force that was exerted on it at the current timestep.

#### Reward

The ```reward```  value returned by```env.step``` is always put to 0.

#### Done

The ```done```  value returned by```env.step``` is  set to ```True``` only when the intrinsic phase or an extrinsic trial is concluded (see below). 

### Intrinsic and extrinsic phases

The environment is set to run as an "intrinsic phase" for a certain number of timesteps (`env.intrinsic_timesteps` , default 15M).  
During the intrinsic phase, no goal is observed.  
After `env.intrinsic_timesteps` have passed the intrinsic phase ends (`done` is set to `True`).  

When using `real_robots.evaluate`, after the intrinsic phase ends, a number of extrinsic trials will be run.  
Each extrinsic trial lasts `env.extrinsic_timesteps` (default: 10000).  
During each extrinsic trial, a different goal is set and it will be displayed in the observation.  
Each goal consists in moving the objects from a certain starting position to another position on the table.  
The goal observation shows how the objects should appear when reaching the final position.  
At the end of each extrinsic trial, `real_robots.evaluate` calls `env.evaluateGoal` to score that goal achievement.

Goals are loaded from an external goal dataset file (which can be chosen using `env.set_goals_dataset_path`).  
A new goal dataset can be generated using the `real-robots-generate-goals` utility.
