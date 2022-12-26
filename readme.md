# Particle Filter SLAM

<p align="justify">
Autonomous vehicles are equipped with sensors that are used to collect information about the motion and surroundings of the vehicle. With the help of the information collected, our goal is to build a virtual map of the surroundings while simultaneously locating the vehicle in that map. A popular technique which is used for this task is called SLAM: Simultaneous Localization and Mapping. This project focuses on one of the approaches to solve the SLAM problem for a moving car and the advantages as well as disadvantages of the given implementation.
</p>

## Project Report
[Orish Jindal, 'Implementation of Particle Filter SLAM for creating a map of the environment from the sensor data collected from a moving car.
', ECE 276A, Course Project, UCSD](https://github.com/ojindal/Particle_filter_SLAM/blob/main/Report-Particlefilter.pdf)


## Dead reckoning path.
<p align="center">
  <img width="500" alt="image" src="https://user-images.githubusercontent.com/89351094/209570477-4fef8cdb-fed7-4f32-8d22-aede8d034250.jpg">
 </p>

 
 ## Results of particle flter SLAM.
<p align="center">
  <img src = "https://user-images.githubusercontent.com/89351094/209570616-e5df8473-d3dc-44d0-bbad-baffb51bf294.png"/>
</p>


## I have created master cdv files syncing data of all the three sensors in different manner to fetch them needed.
These will automatically be created when the code file: synchronous_data.py runs

There are total four code files that needs to be run synchronously:
1. pr2_utils.py
2. synchronous_data.py
3. motion_model.py
4. Particle_filter.py

- First includes some utility functions required further.
- Second one needs to be run in order to generate the cdv files that are used in the codes later
- third file predicts the motion model without noise and plots the trajectory.
- fourth file is the mail file that contains all the transformation functions, map functions, plotting functions etc. and the mail loop of particle filter functions. It also plots the LiDAR scan visualization with or without noise.
