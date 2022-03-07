I have created master cdv files syncing data of all the three sensors in different manner to fetch them needed.
Due to size constraint, I am not able to upload those.
These will automatically be created when the code file: synchronous_data.py runs

There are total four code files that needs to be run synchronously:
1. pr2_utils.py
2. synchronous_data.py
3. motion_model.py
4. Particle_filter.py

-First one is already given
-Second one needs to be run in order to generate the cdv files that are used in the codes later
-third file predicts the motion model without noise and plots the trajectory.
-fourth file is the mail file that contains all the transformation functions, map functions, plotting functions etc. and the mail loop of particle filter functions. It also plots the LiDAR scan visualization with or without noise.

