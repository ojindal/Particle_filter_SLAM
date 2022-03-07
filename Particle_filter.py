
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt; plt.ion()
from mpl_toolkits.mplot3d import Axes3D
import time
from tqdm import tqdm
from pr2_utils import *
from motion_model import *
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

# %%
'''
FOG parameters
'''
FOV = 190
start_angle = -5. #degree
end_angle = 185. #degree
resol = 0.666
rrange = 80.
n = int((FOV//resol)+1) # number of rays

vRl = np.array([[0.00130201, 0.796097, 0.605167],
                [0.999999, -0.000419027, -0.00160026],
                [-0.00102038, 0.605169, -0.796097]])
vTl = np.array([0.8349,-0.0126869, 1.76416 ])

'''
functions
'''

# Rotation matrix about the z axis given yaw angle
def rotation_z(yaw):
    ans = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                    [np.sin(yaw), np.cos(yaw), 0],
                    [0, 0, 1]])
    return ans

# Transformation operation using pose
def transform(R,T,x):
    '''
    t and x are row vectors
    outpot is also a row vector
    '''
    pose = np.identity(4)
    pose[:3,:3] = R
    pose[:3,3] = T
    x = np.array(x).T
    n = len(x[0,:])
    if len(x[:,0]) == 3:
        x = np.vstack((x,np.ones(n))) #1,4
    ans = np.matmul(pose,x).T
    return ans #1,4

# Filtering the rays of lidar with our constraints
def angles_array(t):
    '''
    To calculate valid rays and corresponding angles at time t
    '''
    angles = np.linspace(-5, 185, 286) / 180 * np.pi
    rays = rays_deg[t,:]
    indValid = np.logical_and((rays < 75), (rays > 2))
    rays = rays[indValid]
    angles = angles[indValid]
            
    return angles, rays #(n,)

# Ray ends from polar to cartesian coordinates w.r.t. lidar frame at time t.   
def cartesian(t):
    '''
    To calculate cartesian coordinates of end points of valid rays at time t
    '''
    # t - time
    angle,r = angles_array(t) #(n,)
    x = r*np.cos(angle)
    y = r*np.sin(angle)
    z = np.zeros(len(r)) # to make zero z vector
    ans = np.stack((x,y,z), axis=1) #n,3
    return ans

def ray_start(wXv,yaw):
    '''
    coordinates, overall yaw of vehicle wrt world
    '''
    t_len,gg = angles_array(t)
    t_ln = len(t_len)
    Xl = np.tile([0,0,0],(t_ln,1)) 
    vXl = transform(vRl,vTl,Xl)
    wRv = rotation_z(yaw)
    wXl = transform(wRv, wXv, vXl)
    
    return wXl #4, = 1,4 np array

def ray_end(t, wXv,yaw):
    '''
    time for indValid, coordinates, overall yaw of vehicle wrt world
    '''
    Xl = cartesian(t) #n,3 - in in lidar frame
    vXl = transform(vRl,vTl,Xl) # lidar to vehicle
    wRv = rotation_z(yaw)
    wXl = transform(wRv,wXv,vXl) # in world frame
    
    return wXl #n,4

# Initiallizing the MAP
def define_map():
    MAP = {}
    MAP['res'] = 1  # meters
    MAP['xmin'] = -100  # meters
    MAP['ymin'] = -1200
    MAP['xmax'] = 1300
    MAP['ymax'] = 200
    MAP['sizex'] = int(np.ceil((MAP['xmax'] - MAP['xmin']) / MAP['res'] + 1))  # cells along x
    MAP['sizey'] = int(np.ceil((MAP['ymax'] - MAP['ymin']) / MAP['res'] + 1))  # cells along y
    MAP['map'] = np.zeros((MAP['sizex'], MAP['sizey']), dtype=np.float32)  # DATA TYPE: char or int8

    return MAP

# Updating the MAP at a given position and yaw angle in the world frame
def map_updation(t,wXv,yaw):
    '''
    time for indValid, coordinates, overall yaw of vehicle wrt world, MAP
    '''
    xy_start = ray_start(wXv,yaw)
    x_start = xy_start[:,0]
    y_start = xy_start[:,1]

    sx = np.ceil((x_start - MAP['xmin']) / MAP['res']).astype(np.int16)-1
    sy = np.ceil((y_start - MAP['ymin']) / MAP['res']).astype(np.int16)-1

    xy_end = ray_end(t,wXv,yaw)
    x_end = xy_end[:,0]
    y_end = xy_end[:,1]
    ex = np.ceil((x_end - MAP['xmin']) / MAP['res']).astype(np.int16)-1
    ey = np.ceil((y_end - MAP['ymin']) / MAP['res']).astype(np.int16)-1

    for n in range(len(x_end)):
        bresenham_points = bresenham2D(sx[n], sy[n], ex[n], ey[n]) #n can be removed from sx by changing lidar function
        bresenham_points_x = bresenham_points[0, :].astype(np.int16)
        bresenham_points_y = bresenham_points[1, :].astype(np.int16)

        indGood = np.logical_and(np.logical_and(np.logical_and((bresenham_points_x > 1), (bresenham_points_y > 1)), (bresenham_points_x < MAP['sizex'])), (bresenham_points_y < MAP['sizey']))

        # decrease log-odds if cell observed free
        MAP['map'][bresenham_points_x[indGood], bresenham_points_y[indGood]] -= np.log(4)

        # increase log-odds if cell observed occupied, also *2 for giving priority to obstacles
        if ((ex[n] > 1) and (ex[n] < MAP['sizex']) and (ey[n] > 1) and (ey[n] < MAP['sizey'])):
            MAP['map'][ex[n], ey[n]] += 2*np.log(4)

    # clip range to prevent over-confidence
    MAP['map'] = np.clip(MAP['map'], -10*np.log(4), 10*np.log(4))

    return MAP

# For plotting and saving
def plot_map(MAP):

    plt.imshow(MAP['map'], cmap='gray')
    
    plt.title("Particle Filter")
    plt.xlabel("y grid-cell coordinates")
    plt.ylabel("x grid-cell coordinates")
    plt.savefig("full.jpg", dpi=1200)
    plt.show()

# For plotting motion model   
def plot_superimp(x):
    MAP = define_map()
    x = np.array(x)
    sx = np.ceil((x[:,0] - MAP['xmin']) / MAP['res']).astype(np.int16)-1
    sy = np.ceil((x[:,1] - MAP['xmin']) / MAP['res']).astype(np.int16)-1
    return sx,sy


# Plotting the Dead Reckoning Map (obtained from lidar scan)
x = cordinates()
height = len(x[:,0])
all_wXv = np.zeros((height,3))
all_wXv[:,:2] = x

all_yaw = fog_cumulative[:,2]
MAP = define_map()
#sx,sy = plot_superimp(x)

for t in tqdm(range(0,len(a))):
    wXv = all_wXv[t]
    yaw = all_yaw[t]

    map_updation(t,wXv,yaw)

plot_map(MAP)



# Reading master file with synced data having angles without sum, representing change in angles at each time stamp

fol_d = '/Users/orish/wi22/ece276/ECE276A_PR2/code/data/sensor_data'
file_d = fol_d + '//tau_diffYaw.csv'
times, others = read_data_from_csv(file_d)
diff_yaw = others[:,2]

# Predict function - to add noise
def predict(u, t,numParticles): #dont need v and yaw, already inside func as a function of t
    '''
    predicts the next with added noise state at a certain time
    '''
    tau = a[t]
    
    for i in range (numParticles):
        
        noisy_yaw = diff_yaw[t]+np.random.normal(0, np.abs(diff_yaw[t]/2))
        noise_v = velocity[t] + np.random.normal(0, np.abs(velocity[t]/2))
        u[2,i] = u[2,i]+noisy_yaw # use full_sync csv here..as we cant use cumulative theta
        
        differential=np.array([noise_v*np.cos(u[2,i]), noise_v*np.sin(u[2,i])])
        differential=np.hstack((differential, np.zeros(1)))
        u[:,i]=u[:,i]+tau*differential*0.5
    return u
    

# Assign weights to particles according to the co-relation index
def weight_approximate(MAP, state, weights, t, particle_count): 

    x_im = np.arange(MAP['xmin'], MAP['xmax'] + MAP['res'], MAP['res'])  
    y_im = np.arange(MAP['ymin'], MAP['ymax'] + MAP['res'], MAP['res']) 

    # 9x9 grid around particle
    x_range = np.arange(-4 * MAP['res'], 5 * MAP['res'], MAP['res']) 
    y_range = np.arange(-4 * MAP['res'], 5 * MAP['res'], MAP['res']) 

    correlation = np.zeros(particle_count)

    for i in range(particle_count):
        # wXv - (3,)
        wXv = [state[0,i],state[1,i],0]
        yaw = state[2,i]
        
        xy_end = ray_end(t,wXv,yaw)
        x_end = xy_end[:,0].reshape(len(xy_end[:,0]),1)
        y_end = xy_end[:,1].reshape(len(xy_end[:,0]),1)
        
        vp = np.stack((x_end, y_end))

        # co-relation
        c = mapCorrelation(MAP['map'], x_im, y_im,vp,x_range, y_range)
        

        # highest correlation
        correlation[i] = np.max(c)
        
    # update particle weights to estimate best particle
    d = np.max(correlation)
    beta = np.exp(correlation - d)
    p_h = beta / beta.sum()
    weights *= p_h / np.sum(weights * p_h)

    return weights

# Resampling step
def resampling(state, weights, particle_count):

    state_new = np.zeros((3, particle_count))
    weights_new = np.tile(1 / particle_count, particle_count).reshape(1, particle_count) #1,n
    j = 0
    weights = weights.reshape(1,particle_count)
    counter_w = weights[0,0]

    for i in range(particle_count):
        z = np.random.uniform(0, 1/particle_count)
        beta = z + i/particle_count
        while beta > counter_w:
            j += 1
            counter_w += weights[0,j]
        state_new[:, i] = state[:, j]

    return state_new, weights_new[0] # to avoid becoming list of lists

# Main function to plot the particle filter slam
numParticles = 10
weights=np.ones(numParticles)/numParticles #1,4 - particle weight
u=np.zeros([3,numParticles])
MAP = define_map()

# Main loop for each timestamp
for t in tqdm(range(0,len(times))):
    '''
    co-relation
    '''  
    u = predict(u, t,numParticles)
    '''
    co-relation
    '''
    n_weights = weight_approximate(MAP, u, weights ,t, numParticles)
    best_particle_ind = np.argmax(n_weights)
    best_state = u[:, best_particle_ind]

    '''
    map updation
    '''
    wXv = [best_state[0],best_state[1],0]
    yaw = best_state[2]

    MAP = map_updation(t,wXv,yaw)
    '''
    resample
    '''
    u,weights = resampling(u, n_weights, numParticles)
        
plot_map(MAP)

