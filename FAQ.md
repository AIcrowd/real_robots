# FAQ

* Is this competition really so cool?  
Yes.

* Can I generate different goals for local testing?  
Yes, see [here](../../wiki/Script-for-generating-new-goal-sets).  
You will be able to generate any amount of goals and also to specify how many objects are present in the environment.

* FileNotFoundError - goals_dataset.npy.npz  
The environment assumes there is a `goals_dataset.npy.npz` file in the current directory, unless you use the `set_goals_dataset_path` function to specificy otherwise.
You can either generate a goal dataset using the [command `real-robots-generate-goals`](../../wiki/Script-for-generating-new-goal-sets) or you can find one in the REAL2020 Starter Kit.
