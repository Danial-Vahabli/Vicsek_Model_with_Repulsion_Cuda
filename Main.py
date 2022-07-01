# -*- coding: utf-8 -*-
"""
Created on Sat Jun 12 18:56:47 2021

@author: Danial Vahabli
"""

from numba import cuda
import numpy as np
import math
import matplotlib.pyplot as plt

import time
import os

import cv2
from matplotlib.colors import Normalize

import Vicsek_Model_Cuda_lib






flottyped = "float32"
    
factor = 1/5 #Scales Alignment Velocity and Repulsion Coefficent
    
N = int(3600) #number of particles
NO = int(1600) #number of obstacles
TPB = 4 #Threads per Blocks 

TOTN = N + NO
L = 60 #Length
H = 60 #Height

V_al =0.1*factor #alignment velocity
R_al = 2 #alignment radius

RR = 0.25 #Repulsion Radius
PRR = RR #Particle Repulsion Radius
ORR = RR #Obstacle Repulsion Radius

C_R = 0.1*factor/3 #Repulsion Coefficent
C_core = 1 #hard-core repulsion coefficent


A = 0.4342 
r_cut_off = -math.log(1/10)*A #Maximum radius of soft-repulsion.
                            #A is tuned such that this is 1
                            #as stated in the text, we drop the repulsion velocity if   
                            #its value is less than 1/10 of C_rep*exp((r_i+r_k)/A)



NOISE = 1 #Noise Boolean
if NOISE:
    noiseR = 0.01 #noise interval [-noiseR/2,noiseR/2]
    noisetext = str(noiseR)
else:
    noisetext = '0'
    
    
    
t0 = 0 #initial time
tf = 2000#final time


PLOT =1 #Plot Boolean


tf_print = 4/5*tf  #starts plotting at t = tf_print
ts = 1 #time step
plot_step =10 #plots every plot_step step

###plot avergaging
t_av_plot = 100 #uses the last t_av_plot frames to assign a color to agent
rotate_data_plot = np.zeros((N,t_av_plot)) 

#Plotting Parameters
PS = 8000000/L/H*PRR*PRR #particle size, you may need to change the constant base on your 
                        #screen to plot the particles and agents in their actual size

OS = 8000000/L/H*ORR*ORR #obstacle size

ArrowSize = 4.5*factor # Velocity size

ArrowWidth = 0.0024 #velocity width
fgs = 35 #figure size
LabelSize = 30 #Label size

major_ticks = np.arange(0, L, 5) #ticks for grid
minor_ticks = np.arange(0, L, 1)


if PLOT:
    save = 1# if 0 dont save and generate video if 1 saves and generates
    EXPORT = 1#exports video
else:
    save = 0
    EXPORT = 0
    
#path to save the video and frames
path = 'video/Particle = {0} Obstacle = {1} R_al = {2} RR = {3} C_R = {4:.3f} V_al = {5:.3f}  noise = {6}'.format(N,NO,R_al,RR,C_R,V_al,noisetext)

    
img_array = []




if save:
    Vicsek_Model_Cuda_lib.createFolder('./{0}'.format(path))


start_time = time.time()

##Generate random points
x_pos = np.random.uniform(low=0.0, high=L, size=N).astype(np.float32)# initial X position | low included high excluded
y_pos = np.random.uniform(low=0.0, high=H, size=N).astype(np.float32) # initial y position
x_pos_b = np.random.uniform(low=0.0, high=L, size=NO) # initial X position | low included high excluded
y_pos_b = np.random.uniform(low=0.0, high=H, size=NO)


#Position matrix
pos = np.zeros((N+NO, 18), dtype=flottyped) 
pos[:,0]=  np.concatenate((x_pos,x_pos_b)) 
pos[:,1] =  np.concatenate((y_pos,y_pos_b)) 


#Angle
angle = np.random.uniform(low=0, high=np.pi*2, size=N) # initial angles

#initial velocities
x_v = V_al*np.cos(angle)
y_v = V_al*np.sin(angle)

#velocity matrix
vel = np.zeros((N, 2), dtype=flottyped)
vel[:,0]= x_v
vel[:,1] = y_v



#Copy to CUDA
pos_device = cuda.to_device(pos) #TOTN*18
vel_device = cuda.to_device(vel) #N*2
anglelist_device = cuda.to_device(angle)  #N*1


#distance matrix
dist_device = cuda.device_array((N,N+NO), dtype= "float32"  , strides=None, order='C', stream=0)
#Index matrix for PBC
ind_device = cuda.device_array((N,N+NO), dtype= "float32"  , strides=None, order='C', stream=0)

#new velocity, empty
newvel_device = cuda.device_array((N,2), dtype= "float32"  , strides=None, order='C', stream=0)

#Rotation
rotate_device = cuda.device_array((N,1), dtype= "float32"  , strides=None, order='C', stream=0)


#Rotation Storage



T = []
for t in range (int((tf-t0)/ts)):

    T.append(t*ts) # creating a list of time steps
    
    Vicsek_Model_Cuda_lib.pbs[math.ceil(TOTN/TPB ),  (TPB ,16)](pos_device,H,L,N) #Generating copies of position for PBC

    Vicsek_Model_Cuda_lib.get_distance[(math.ceil(N/TPB ), math.ceil(TOTN/TPB )), (TPB , TPB )](pos_device, dist_device,ind_device,TOTN) #Calculatin the pair-wise distance

    
    if NOISE: #Noise
    
        noise = np.random.randint(low= int(-noiseR/2*(10**8)), high=int(noiseR/2*(10**8)  + 1), size=N,dtype=int)/(10**8) #Generating noise as integers   in [a,b+1) interval to ensure that the noise is generated on [a,b] 
                                                                                                                                # https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html
    else:
        noise = np.zeros((N), dtype=flottyped  )
    
    noise_device = cuda.to_device(noise) #Copying the noise array to CUDA memory
    

    Vicsek_Model_Cuda_lib.get_v[math.ceil(N/TPB ),  TPB ](pos_device,dist_device,noise_device,vel_device,ind_device,newvel_device,C_core,C_R,N,NO,V_al,R_al,A,PRR,ORR,r_cut_off) #calculate velocities

    Vicsek_Model_Cuda_lib.get_angle[math.ceil(N/TPB ), TPB ](newvel_device, anglelist_device,N)


    if t !=0:
        Vicsek_Model_Cuda_lib.get_rotate[math.ceil(N/TPB ),  TPB ](anglelist_device,angleold_device,rotate_device,N)

    
        rotate = rotate_device.copy_to_host() #copy to host
        

        
        
    anglelist = anglelist_device.copy_to_host() #get new angles #copy to host
    
    #swap angles
    angleold = anglelist
    angleold_device = cuda.to_device(angleold)
    
    posnew = pos_device.copy_to_host() #copy to host
    
    velnew = newvel_device.copy_to_host() #copy to host
    
    if t > 0:
        rotate_data_plot[:,np.mod(t,t_av_plot)] = rotate[:,0] #store the "last t_av_plot" frames 'rotation for colorcode calculations
    if PLOT:
       if (np.mod(t*ts+1,plot_step) == 0 and (t> tf_print )):
           Vicsek_Model_Cuda_lib.Plot(posnew,velnew,save,rotate_data_plot,anglelist,path,N,fgs,PS,OS,ArrowSize,ArrowWidth,LabelSize,L,H,major_ticks,minor_ticks,save,t,EXPORT,img_array)
    
    Vicsek_Model_Cuda_lib.next_pos[math.ceil(N/TPB ),  TPB ](pos_device,newvel_device,N,L,H,ts) #Next Position
    
    
    velold = velnew
    vel_device = cuda.to_device(velold)




if EXPORT:
    height, width, layers = img_array[0].shape
    size = (width,height)
    out = cv2.VideoWriter('{0}/video.avi'.format(path),cv2.VideoWriter_fourcc(*'DIVX'), 7, size)
     
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()
    
    
