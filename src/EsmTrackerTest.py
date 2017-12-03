#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:46:27 2017

@author: matthewxfz
"""
#%%
import numpy as np
import cv2
import glob
from matplotlib import pyplot as plt
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import pinv, inv
from numpy.linalg import LinAlgError
from scipy.linalg import expm
from numpy.linalg import norm

import sys
sys.path.append("/Users/matthewxfz/cs512gits/project/src")
import util

import time

#initialize param
path_image_1 = "/Users/matthewxfz/Workspaces/gits/course/reference/ESMkitMac_0_4_1/seq/im004.pgm"
path_image_2 = "/Users/matthewxfz/Workspaces/gits/course/reference/ESMkitMac_0_4_1/seq/im005.pgm"

maxIte = 100
minLamP = 1e-8

maxIte = 1
minLamP = 1e-8
def normlize(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = img.astype(np.float32)
    #mean, stddev = cv2.meanStdDev(img) 
    #img = img - mean;
    #img = (img/stddev)
    #img = cv2.pyrDown(img)
    img = cv2.blur(img, (3,3)) 
    return img

Pose = np.eye(2,3,dtype = np.float)#2,3
img1_o= cv2.imread('/Users/matthewxfz/Workspaces/gits/course/reference/ESMkitMac_0_4_1/seq/im002.pgm')
img2_o= cv2.imread('/Users/matthewxfz/Workspaces/gits/course/reference/ESMkitMac_0_4_1/seq/im003.pgm')

templThumb = normlize(img1_o)
compThumb = normlize(img2_o)

plt.imshow(compThumb, cmap='gray'),plt.show()
plt.imshow(templThumb, cmap='gray'),plt.show()

# Select ROI
#rang = cv2.selectROI(templThumb)
rang = np.array([200,200,100,100])
r = rang
# Crop image
temple = templThumb[int(rang[0]):int(rang[0]+rang[2]),int(rang[1]):int(rang[1]+rang[3])]
xd = int(rang[0]);
yd = int(rang[1]);
maxIte1=1
scoreA = 1000
#r = rang
#vertexO = np.array([[int(r[0]),int(r[1]),1],[int(r[0]+r[2]),int(r[1]+r[3]),1]])
#cv2.rectangle(warpImage,(vertexO[0,0],vertexO[0,1]),(vertexO[1,0],vertexO[1,1]),(255,0,0),4)
##%%
Pose = np.eye(3,3,dtype = np.float)#2,3
Pose[0,2] = xd;
Pose[1,2] = yd;
WPose = Pose
#

temGx = cv2.Sobel(temple,cv2.CV_64F,1,0,ksize=5)
temGy = cv2.Sobel(temple,cv2.CV_64F,0,1,ksize=5)

x = temple.shape[0]
y = temple.shape[1]

img = util.warp(templThumb.astype(np.float32), 
             WPose.astype(np.float32),
             x,
             y)
plt.imshow(img.astype(np.uint8), cmap='gray'),plt.show()  
plt.imshow(temple, cmap='gray'),plt.show()

print(Pose)

    
#%%
ndxs = np.empty((10000,1),dtype = np.float32)
for k in range(0,100):
    Pose = WPose;
    print k
    for i in range(0,100):
        SD, diff, warpImage = util.updateHx3(compThumb.astype(np.float32),
                                    Pose.astype(np.float32),
                                    temple.astype(np.float32),
                                    temGx.astype(np.float32),
                                    temGy.astype(np.float32),
                                    xd,
                                    yd)
        delta = -2*np.linalg.inv(SD.transpose().dot(SD)).dot(SD.transpose().dot(diff))
        update_auxA = np.empty((3,3), dtype = np.float32)
        update_auxA[0,0] = delta[6,0];
        update_auxA[0,1] = delta[0,0];
        update_auxA[0,2] = delta[1,0];
        update_auxA[1,0] = delta[2,0];
        update_auxA[1,1] = -delta[6,0] - delta[7,0];
        update_auxA[1,2] = delta[3,0];
        
        update_auxA[2,0] = delta[4,0];
        update_auxA[2,1] = delta[5,0];
        update_auxA[2,2] = delta[7,0];
        
        Pose = np.dot(Pose, expm(update_auxA))
#        print Pose
        #print("ndx: %f, ssd: %f, my difference" %(norm(delta), sum(diff)), norm(warpImage - temple))
        
        ndxs[100*k+i,0] = norm(delta)
        
        if(norm(delta) < 1e-10):
            break;
        else:
            oimg = util.draw_rectangle(Pose.astype(np.float32),
                                           compThumb.astype(np.float32), 
                                           x, y, 255, 0, 0, 5).astype(np.uint8)
            plt.imshow(util.draw_rectangle(WPose.astype(np.float32),
                                           oimg.astype(np.float32), 
                                           x, y, 0, 0, 255, 5).astype(np.uint8), 
                       cmap='gray'),plt.show() 
    
img = util.warp(compThumb.astype(np.float32), 
         Pose.astype(np.float32),
         x,
         y)
plt.imshow(img.astype(np.uint8), cmap='gray'),plt.show()  
plt.imshow(temple, cmap='gray'),plt.show()
#%%
from matplotlib import pyplot as plt
ndxs
t = np.arange(100, 0, -1)
for k in range(0,100):
    plt.plot(t,ndxs[k*100:(k+1)*100])
    
plt.show()