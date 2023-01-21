# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 13:18:27 2022

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

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)


 
@cuda.jit
def get_distance(pos, dist,ind,TOTN):
    x, y = cuda.grid(2)

    if x<TOTN and y<TOTN:
        temp = math.sqrt((pos[x,0]-pos[y,0])**2 + (pos[x,1]-pos[y,1])**2)
        
        tempL = math.sqrt((pos[x,0]-pos[y,2])**2 + (pos[x,1]-pos[y,3])**2)
        tempU = math.sqrt((pos[x,0]-pos[y,4])**2 + (pos[x,1]-pos[y,5])**2)
        tempR = math.sqrt((pos[x,0]-pos[y,6])**2 + (pos[x,1]-pos[y,7])**2)
        tempD = math.sqrt((pos[x,0]-pos[y,8])**2 + (pos[x,1]-pos[y,9])**2)
        
        tempLU = math.sqrt((pos[x,0]-pos[y,10])**2 + (pos[x,1]-pos[y,11])**2)
        tempRU = math.sqrt((pos[x,0]-pos[y,12])**2 + (pos[x,1]-pos[y,13])**2)
        tempRD = math.sqrt((pos[x,0]-pos[y,14])**2 + (pos[x,1]-pos[y,15])**2)
        tempLD = math.sqrt((pos[x,0]-pos[y,16])**2 + (pos[x,1]-pos[y,17])**2)
        disttemp = min(temp,tempL,tempU,tempR,tempD,tempLU,tempRU,tempRD,tempLD)
    
    
        if temp==disttemp:
            ind[x,y] = 0
        if tempL==disttemp:
            ind[x,y] = 1
        if tempU==disttemp:
           ind[x,y] = 2
        if tempR==disttemp:
            ind[x,y] = 3
        if tempD==disttemp:
            ind[x,y] = 4
            
        if tempLU==disttemp:
            ind[x,y] = 5
        if tempRU==disttemp:
            ind[x,y] = 6
        if tempRD==disttemp:
            ind[x,y] = 7
        if tempLD==disttemp:
            ind[x,y] = 8
            
        dist[x,y]  = disttemp
     
               

@cuda.jit
def pbs(pos,H,L,N):
    x, y = cuda.grid(2)
        #org,L,U,R,D,LU,RU,RD,LD
    if x<N:
                
        if y %2== 0:  #x
            if (y==4 or y==10 or y==12):
                pos[x,y+2] = pos[x,0]+L
            if(y==0 or y==8 or y==14):
                pos[x,y+2] = pos[x,0]-L
            if (y==2 or y==6 ):
                pos[x,y+2] = pos[x,0]
        else: #y
            if (y==3 or y==9 or y==11):
                pos[x,y+2] = pos[x,1]+H
            if (y==7 or y==15 or y==13):
                pos[x,y+2] = pos[x,1]-H
            if(y==1 or y==5 ):
                pos[x,y+2] = pos[x,1]
            
            
        
  


@cuda.jit
def get_angle(v, anglelist,N):
    k, y = cuda.grid(2)
    if k<N:
        ratio = v[k,1]/v[k,0]
        ang = math.atan(ratio)
    
        if v[k,0] <=0:
            ang += np.pi
        if ang < 0:
            ang += 2*np.pi
        anglelist[k] = ang
        
@cuda.jit
def get_rotate(anglenew, angleold,rotate,N):
    i, y,u = cuda.grid(3)
    if i<N:
    
        if (math.pi*3/2 <anglenew[i] and angleold[i] <math.pi/2):
            
            rotate[i] = -(2*math.pi - abs((anglenew[i]-angleold[i])))
        elif (math.pi*3/2 <angleold[i] and anglenew[i] <math.pi/2):
            rotate[i] = 2*math.pi - abs((anglenew[i]-angleold[i]))
        else:
            rotate[i] = anglenew[i]-angleold[i]
            


    
@cuda.jit
def next_pos(pos, vel,N,L,H,ts):
    i, y,k = cuda.grid(3)
    if i <N:

        pos[i,0] = (pos[i,0] + vel[i,0]*ts) % L
        pos[i,1] = (pos[i,1] + vel[i,1]*ts) % H
        

     

@cuda.jit
def get_v(pos,dist,noise, vel,ind,velocity,C_core,C_R,N,NO,V_al,R_al,A,PRR,ORR,r_cut_off):
    i, y,k = cuda.grid(3)
    if i <N:

        tempx = 0.0 #neigh v
        tempy = 0.0
        
        tempxr = 0.0 #rep v
        tempyr = 0.0
        
        for j in range(N+NO):
            if j < N:
    
                if dist[i,j] < R_al: #neigh V
                    tempx += vel[j,0]
                    tempy += vel[j,1]
                    
    

            if j<N:
                min_distance = PRR+PRR #particle-particle
            else:
                min_distance = PRR+ORR #particle_obstacle
                
            if dist[i,j] <= (min_distance):#overlapp
                if i!=j: 
                    tempxr = tempxr - (C_R*(math.exp(((min_distance-dist[i,j])/A))+C_core*(min_distance-dist[i,j])) *(pos[j,int(ind[i,j]*2)]-pos[i,0])/dist[i,j])
                    tempyr = tempyr - (C_R*(math.exp(((min_distance-dist[i,j])/A))+C_core*(min_distance-dist[i,j]))*(pos[j,int(ind[i,j]*2+1)]-pos[i,1])/dist[i,j])
        
                    
            else:
               if i!=j:
                   if dist[i,j] <r_cut_off:
                       tempxr = tempxr - (C_R*math.exp((min_distance-dist[i,j])/A)*(pos[j,int(ind[i,j]*2)]-pos[i,0])/dist[i,j])
                       tempyr = tempyr - (C_R*math.exp((min_distance-dist[i,j])/A)*(pos[j,int(ind[i,j]*2+1)]-pos[i,1])/dist[i,j])
    

        if R_al == 0:
            ratio = vel[i,1]/vel[i,0]
            ang = math.atan(ratio)
            
            if vel[i,0] <=0:
                ang += np.pi
            if ang < 0:
                ang += 2*np.pi
        
            ang = ang + noise[i]
            velocity[i,0] =   V_al*math.cos(ang) + tempxr 
            
            velocity[i,1] =  V_al*math.sin(ang) + tempyr 
        
        else:
            ratio = tempy/tempx
            ang = math.atan(ratio)
            
            if tempx <=0:
                ang += np.pi
            if ang < 0:
                ang += 2*np.pi
        
            ang = ang + noise[i]
            velocity[i,0] =   V_al*math.cos(ang) + tempxr 
            
            velocity[i,1] =  V_al*math.sin(ang) + tempyr
            
def Plot(pos,vel,SAVE,rotate_data_plot,anglelist,path,N,fgs,PS,OS,ArrowSize,ArrowWidth,LabelSize,L,H,major_ticks,minor_ticks,save,t,EXPORT,img_array):
    

              
            fig, ax = plt.subplots(1,1,figsize = (fgs,fgs))


            sizevel = np.sqrt(vel[:,0]*vel[:,0] + vel[:,1]*vel[:,1]) #magnitute of velocity vectors
           
            
           #colormap
            colortemp = np.arange(N*4,dtype='float16')
            colortemp = colortemp.reshape((N, 4))
            colorlist = np.ones_like(colortemp)
            
               
            norm = Normalize()
            norm.autoscale(sizevel)

            rotate_av_ps = np.sum(rotate_data_plot, axis = 1)
            for c in range (0,N):
                if abs(rotate_av_ps[c])<2: #limit for green
                    
                    colorlist[c][2] = 0
                    colorlist[c][0] = 0
                    colorlist[c][1] = 1


                else:
                        
                    if rotate_av_ps[c]>0: #red
                    
                        colorlist[c][2] = 0
                        tempcol =    anglelist[c]/(2*np.pi)*0.7 #Scales the colors
                        colorlist[c][1] = tempcol
                        
                    elif rotate_av_ps[c]==0: #previous color
                        tempcol =    anglelist[c]/(2*np.pi)*0.7
                        colorlist[c][1] = tempcol
                        
                            
                    else: #blue
                        colorlist[c][0] = 0
                        tempcol =    anglelist[c]/(2*np.pi)*0.7
                        colorlist[c][1] = tempcol
                        
                        
            ax.scatter(pos[0:N,0],pos[0:N,1], c='slategray' ,label='0',alpha =0.8,s = PS) #Particles
            ax.scatter(pos[N:,0],pos[N:,1], c='k' ,label='-1',alpha = 1,s = OS) #Obstacles
            
            ax.quiver(pos[0:N,0],pos[0:N,1], vel[:,0],vel[:,1],color=colorlist,scale =ArrowSize,width = ArrowWidth) #Arrows


            ax.set_ylabel('Y position',fontsize = LabelSize)
            ax.set_xlabel('X position',fontsize = LabelSize)
            ax.tick_params(axis='both',labelsize = LabelSize)
            ax.set(xlim=(0, L), ylim=(0, H))
            ax.set_aspect('equal')
            ax.set_xticks(major_ticks)
            ax.set_xticks(minor_ticks, minor=True)
            ax.set_yticks(major_ticks)
            ax.set_yticks(minor_ticks, minor=True)
            ax.grid(which='minor', alpha=0.2)
            ax.grid(which='major', alpha=0.5)


            if save:
                plt.savefig("{1}/{0}" .format(t+1,path))
                if EXPORT:
                    img = cv2.imread("{1}/{0}.png" .format(t+1,path))
                    height, width, layers = img.shape
                    img_array.append(img)
                
            plt.close()


