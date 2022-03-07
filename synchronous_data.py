# Imporing the files
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import time
from tqdm import tqdm
from pr2_utils import *

# A function to sync the Encoder data with FOG data
# Return a csv file
def new_csv():
    
    fol1 = '/Users/orish/wi22/ece276/ECE276A_PR2/code/data/sensor_data'
    
    file1 = fol1 + '/fog.csv'
    fog_t, fog_d = read_data_from_csv(file1)
    
    file2 = fol1 + '/encoder.csv'
    encod_t, encod_d = read_data_from_csv(file2)
    
    x = np.zeros(6)
    increment = 0
    for t in range(len(encod_t)-1):
        temp = 0
        temp_t = 0
        counter = 0
        for i in range(25):
            temp+=fog_d[increment+t+i]
            counter = i
            temp_t = fog_t[increment+t+i]
            if fog_t[increment+t+i] >= encod_t[t]:
                break
        increment += counter
        tim = (temp_t + encod_t[t])/2
        new = np.hstack((tim,np.hstack((temp,encod_d[t+1]-encod_d[t]))))
        x = np.vstack((x,new))
    
    pd.DataFrame(x).to_csv(fol1 +'/intermediate_sync.csv',header = False, index = False)

file = new_csv()

# A function to sync the already synchronus Encoder and FOG data with the Lidar data
# Return a csv file
def new_csv_full():
    
    fol1 = '/Users/orish/wi22/ece276/ECE276A_PR2/code/data/sensor_data'
    
    file1 = fol1 + '/intermediate_sync.csv'
    a, b = read_data_from_csv(file1)

    file2 = fol1 + '/lidar.csv'
    lidar_t, lidar_d = read_data_from_csv(file2)

    increment1 = 0
    increment2 = 0

    x = np.zeros((200000,(len(b[1,:])+len(lidar_d[1,:])+1)))
    l = 0
    for t in range(min([len(a),len(lidar_t)])-1):
        if a[t]<lidar_t[t]:
            temp = 0
            temp_t = 0
            counter1 = 0
            for i in range(25):
                temp+=b[increment1+t+i]
                counter1 = i
                temp_t = a[increment1+t+i]
                if a[increment1+t+i] >= lidar_t[t]:
                    break
            increment1 += counter1
            tim = (temp_t + lidar_t[t])/2
            x[t][0] = tim
            x[t][1:] = np.hstack((temp, lidar_d[t]))
            l+=1

        else:
            temp = 0
            temp_t = 0
            counter2 = 0
            for i in range(25):
                temp = lidar_d[increment2+t+i]
                counter = i
                temp_t = lidar_t[increment2+t+i]
                if lidar_t[increment2+t+i] >= a[t]:
                    break
            increment2 += counter2
            tim = (temp_t + a[t])/2
            x[t][0] = tim
            x[t][1:] = np.hstack((b[t],temp))
            l+=1
    ans = x[:l][:]
    pd.DataFrame(ans).to_csv(fol1 +'/full_sync.csv',header = False, index = False)
    
ans = new_csv_full()

# A function to return csv file of cummulative sum of angles at each time and time difference, made from mater csv - full_sync
def cumangles():
    fol1 = '/Users/orish/wi22/ece276/ECE276A_PR2/code/data/sensor_data'
    file1 = fol1 + '/full_sync.csv'
    a, b = read_data_from_csv(file1)
    
    for i in range(1,len(a)):
        b[i][0:3] = b[i][0:3]+b[i-1][0:3]
        a[i-1] = a[i]-a[i-1]
    
    ans = np.hstack((np.array(a).reshape(len(a),1),np.array(b)))
        
    pd.DataFrame(ans).to_csv(fol1 +'/cummulative_angles.csv',header = False, index = False)
ans = cumangles()

# A function to return csv file of change in angles (without cumsum) at each time and time difference, made from mater csv - full_sync
def tau():
    fol1 = '/Users/orish/wi22/ece276/ECE276A_PR2/code/data/sensor_data'
    file1 = fol1 + '/full_sync.csv'
    a, b = read_data_from_csv(file1)
    
    for i in range(1,len(a)):
        a[i-1] = a[i]-a[i-1]
    
    ans = np.hstack((np.array(a).reshape(len(a),1),np.array(b)))
        
    pd.DataFrame(ans).to_csv(fol1 +'/tau_diffYaw.csv',header = False, index = False)
ans = tau()