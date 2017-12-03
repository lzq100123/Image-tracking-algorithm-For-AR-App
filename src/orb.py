#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 10:54:14 2017

@author: matthewxfz
"""
#%%
import numpy as np
import cv2
from matplotlib import pyplot as plt

def greyI(image):
    outImage = np.ones((image.shape[0],image.shape[1]),np.uint8)
    for i in range(0,image.shape[0],1):
        for j in range(0,image.shape[1],1):
            outImage[i][j] = (image[i][j][0] * 49 + image[i][j][1]*31 + image[i][j][0]*2)/100
            
    return outImage

#%%
winName =  'matching'

img1_o= cv2.imread('/Users/matthewxfz/Workspaces/gits/course/cs512-f17-fangzhou-xiong/project/data/zeal10.jpg')
img2_o= cv2.imread('/Users/matthewxfz/Workspaces/gits/course/cs512-f17-fangzhou-xiong/project/data/calib03.jpg')
cliba3_o= cv2.imread('/Users/matthewxfz/Workspaces/gits/course/cs512-f17-fangzhou-xiong/project/data/calib03.jpg')

img1 = cv2.cvtColor(img1_o, cv2.COLOR_BGR2GRAY)
img2 = cv2.cvtColor(img2_o, cv2.COLOR_BGR2GRAY)
cali3 = cv2.cvtColor(cliba3_o, cv2.COLOR_RGB2GRAY)
# Initiate STAR detector
orb = cv2.ORB_create(500)

#%%
# find the keypoints with ORB
kp1, des1  = orb.detectAndCompute(img1,None)
kp2, des2  = orb.detectAndCompute(img2,None)

# compute the descriptors with ORB
#kp1, des1 = orb.compute(img1, kp)
#cv2.drawKeypoints(img1,kp,,color=(255,0,0), flags=0)

bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
# Match descriptors.
matches = bf.match(des1,des2)

# Sort them in the order of their distance.
matches = sorted(matches, key = lambda x:x.distance)

# Draw first 10 matches.
img3 = cv2.drawMatches(img1,kp1,img2,kp2,matches[:50] ,None, flags=2)


plt.imshow(img3, cmap='gray'),plt.show()

#%% project with homography
ms = matches[:53]
#%%
pts_dst = np.array([p.pt for p in kp1], dtype = np.uint16)
pts_src = np.array([p.pt for p in kp2], dtype = np.uint16)

query = [m.queryIdx for m in ms];
train = [m.trainIdx for m in ms];
#%%
pts_dst = pts_dst[query,:]
pts_src = pts_src[train,:]

#%%
#h, status = cv2.findHomography(pts_src, pts_dst)
#im_out = cv2.warpPerspective(img2, h, (img1.shape[1],img1.shape[0]))
#plt.imshow(im_out),plt.show()

#%%
tmp = pts_src
pts_src = pts_dst
pts_dst = tmp

h, status = cv2.findHomography(pts_src, pts_dst)

#im_out = cv2.warpPerspective(img2, h, (img1.shape[1],img1.shape[0]))
im_temp = cv2.warpPerspective(img1, h, (img2.shape[1],img2.shape[0]))

im_dst = img2.astype(np.uint16)
# Black out polygonal area in destination image.
cv2.fillConvexPoly(im_dst, pts_dst.astype(np.uint16), 0, 16);

# Add warped source image to destination image.
im_dst = (im_dst + im_dst);

#%%
plt.imshow(im_temp, cmap='gray'),plt.show()
#plt.imshow(im_dst),plt.show()

#%%
T = img1;
kmax = 100;
I  = cali3
H = h;



