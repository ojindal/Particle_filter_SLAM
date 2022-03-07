import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import time
from tqdm import tqdm
from pr2_utils import *
from synchronous_data import *



# Reading master file with synced data having angles already summed
fol1 = '/Users/orish/wi22/ece276/ECE276A_PR2/code/data/sensor_data'
file1 = fol1 + '/cummulative_angles.csv'
a, b = read_data_from_csv(file1)

fog_cumulative = np.zeros((96444, 3))
fog_cumulative = b[:,:3]
diff_enc_l = b[:,3]
diff_enc_r = b[:,4]
rays_deg = b[:,5:]

resol = 4096
l_dia = 0.623479
r_dia = 0.622806

velocity = np.array((l_dia*diff_enc_l*2*np.pi + r_dia*diff_enc_r*2*np.pi)/(2*resol*np.array(a))) #

# A function returning the list of (x,y) tuples for the position of vehilcle at each time
def cordinates():
    
    x = np.array([0,0])
    t = 0
    for t in range(len(a)-1):
        yaw = fog_cumulative[t][2]
        xn = np.array([a[t]*velocity[t]*np.cos(yaw),a[t]*velocity[t]*np.sin(yaw)])
        x = np.vstack((x,x[t-1]+xn))
    return x

trajectory = cordinates()

# Plotting the trajectory
def plotpath(trajectory):

    x = [step[0] for step in trajectory[1:]]
    y = [step[1] for step in trajectory[1:]]

    plt.plot(x,y)
    plt.gca().set_aspect("equal")
    plt.show(block = True)
        
plotpath(trajectory)

