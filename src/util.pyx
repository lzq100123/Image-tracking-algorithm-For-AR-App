    #!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 20 14:18:53 2017

@author: matthewxfz
"""
from __future__ import division
import numpy as np
import cv2
cimport numpy as np
DTYPE = np.int
FLOAT = np.float32

ctypedef np.int_t DTYPE_t
ctypedef np.float DTYPE_f

cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function

def get_s(np.ndarray image):
    assert image.dtype == DTYPE 
    cdef np.ndarray h = np.zeros([6, 6], dtype=FLOAT)
    cdef np.ndarray pi = np.zeros([1, 6], dtype=FLOAT)
     
    cdef int vmax = image.shape[0]
    cdef int wmax = image.shape[1]
    cdef int x, y, s, t, v, w
    
    for x in range(vmax):
        for y in range(wmax):
            if(image[x,y] > 0):
                 h += get_P_cor(x,y)
    return h;
                
def get_P_cor(np.int x, np.int y):
    cdef np.ndarray pi = np.zeros([1, 6], dtype=FLOAT)
    pi[0,0] = x*x
    pi[0,1] = x*y
    pi[0,2] = y*y
    pi[0,3] = x
    pi[0,4] = y
    pi[0,5] = 1
    return pi.transpose().dot(pi)

def get_s_la(np.ndarray image, np.ndarray gradient):
    assert image.dtype == DTYPE 
    cdef np.ndarray h = np.zeros([6, 6], dtype=FLOAT)
    cdef np.ndarray pi = np.zeros([1, 6], dtype=FLOAT)
     
    cdef int vmax = image.shape[0]
    cdef int wmax = image.shape[1]
    cdef int x, y, s, t, v, w
    cdef float g = 0.0
    
    for x in range(vmax):
        for y in range(wmax):
            if(image[x,y] > 0):
                 h += get_P_cor(x,y)
                 g += gradient[x,y]
    h = h /g 
    return h;

#warp compImage into the pose, 
#@ x is the length of template in row
#@ y is the length of template in columne

def warp(np.ndarray[float, ndim=2] compImage, 
         np.ndarray[float, ndim=2] Pose,
         int x, 
         int y):
    
    cdef np.ndarray[float, ndim=2] img = np.empty((x,y), dtype = FLOAT)
    cdef int u,v,i,j;
    cdef float d;
    for u in range(0, x):#row
            for v in range(0, y):#column
                d = Pose[2,0]*u+Pose[2,1]*v+Pose[2,2];
                i = int((Pose[0,0]*u+Pose[0,1]*v+Pose[0,2])/d);
                j = int((Pose[1,0]*u+Pose[1,1]*v+Pose[1,2])/d);
                
                if(i>=compImage.shape[0]):
                    i = compImage.shape[0]-1
                elif(i < 0):
                    i = 0
                    
                if(j>=compImage.shape[1]):
                    j = compImage.shape[0]-1
                elif(j < 0):
                    j = 0
                     
                img[u,v] = compImage[i,j]
                
    return img;

def transform(np.ndarray[float, ndim=2] pose, 
              np.ndarray[float, ndim=3] p):
    cdef np.ndarray[float, ndim=3] out = np.empty((p.shape[0],p.shape[1],p.shape[2]), dtype = FLOAT);
    cdef int i
    for i,row in enumerate(p):
        d = pose[2,0]*row[0,0]+pose[2,1]*row[0,1]+pose[2,2];
        out[i,0,0] = float((pose[0,0]*row[0,0]+pose[0,1]*row[0,1]+pose[0,2])/d)
        out[i,0,1] = float((pose[1,0]*row[0,0]+pose[1,1]*row[0,1]+pose[1,2])/d)
        
    return out;

def transform2(np.ndarray[float, ndim=2] pose, 
              np.ndarray[float, ndim=3] p):
    cdef np.ndarray[float, ndim=3] out = np.empty((p.shape[0],p.shape[1],p.shape[2]), dtype = FLOAT);
    cdef int i
    for i,row in enumerate(p):
        d = pose[2,0]*row[0,0]+pose[2,1]*row[0,1]+pose[2,2];
        out[i,0,0] = float((pose[0,0]*row[0,0]+pose[0,1]*row[0,1]+pose[0,2])/d)
        out[i,0,1] = float((pose[1,0]*row[0,0]+pose[1,1]*row[0,1]+pose[1,2])/d)
        
    return out;

def swapxy(np.ndarray[float, ndim=3] p):
    cdef float temp;
    cdef int i;
    cdef np.ndarray[float, ndim=2] row;
    for i,row in enumerate(p):
        temp = row[0,0]
        row[0,0]  = row[0,1]
        row[0,1] = temp
    
    return p;

def swapM(np.ndarray[float, ndim=2] pose):
    cdef float temp;
    cdef int i,j;
    for i in range(0,3):
        temp = pose[0,i]
        pose[0,i]  = pose[1,i]
        pose[1,i] = temp
        
    return pose;

def shiftPt(np.ndarray[float, ndim=3] p, 
            int xd, 
            int yd, ):
    cdef np.ndarray[float, ndim=3] out = np.empty((p.shape[0],p.shape[1],p.shape[2]), dtype = FLOAT);
    cdef int i
    for i,row in enumerate(p):
        out[i,0,0] = row[0,0] - xd
        out[i,0,1] = row[0,1] - yd
        
    return out;
        
def draw_rectangle(np.ndarray[float, ndim=2]  Pose, np.ndarray[float, ndim=2] img, int x, int y, int r, int g, int b, int thickness):
    #(1,1), (1, y-2), (x-2,y-2), (x-2, 1)
    
    cdef int x1 = (int)((Pose[0,0]*1+Pose[0,1]*1+Pose[0,2])/(Pose[2,0]*1+Pose[2,1]*1+Pose[2,2]))
    cdef int y1 = (int)((Pose[1,0]*1+Pose[1,1]*1+Pose[1,2])/(Pose[2,0]*1+Pose[2,1]*1+Pose[2,2]))
    
    cdef int x2 = (int)((Pose[0,0]*1+Pose[0,1]*(y-2)+Pose[0,2])/(Pose[2,0]*1+Pose[2,1]*(y-2)+Pose[2,2]))
    cdef int y2 = (int)((Pose[1,0]*1+Pose[1,1]*(y-2)+Pose[1,2])/(Pose[2,0]*1+Pose[2,1]*(y-2)+Pose[2,2]))

    cdef int x3 = (int)((Pose[0,0]*(x-2)+Pose[0,1]*(y-2)+Pose[0,2])/(Pose[2,0]*(x-2)+Pose[2,1]*(y-2)+Pose[2,2]))
    cdef int y3 = (int)((Pose[1,0]*(x-2)+Pose[1,1]*(y-2)+Pose[1,2])/(Pose[2,0]*(x-2)+Pose[2,1]*(y-2)+Pose[2,2]))
    
    cdef int x4 = (int)((Pose[0,0]*(x-2)+Pose[0,1]*1+Pose[0,2])/(Pose[2,0]*(x-2)+Pose[2,1]*1+Pose[2,2]))
    cdef int y4 = (int)((Pose[1,0]*(x-2)+Pose[1,1]*1+Pose[1,2])/(Pose[2,0]*(x-2)+Pose[2,1]*1+Pose[2,2]))
    
    
    cv2.line(img, (y1,x1),(y2,x2), (r,g,b), thickness)
    cv2.line(img, (y2,x2),(y3,x3), (r,g,b), thickness)
    cv2.line(img, (y3,x3),(y4,x4), (r,g,b), thickness)
    cv2.line(img, (y4,x4),(y1,x1), (r,g,b), thickness)
    
    return img

def draw_rectangle_ONCOLOR(np.ndarray[float, ndim=2]  Pose, np.ndarray[float, ndim=3] img, int x, int y, int r, int g, int b, int thickness):
    #(1,1), (1, y-2), (x-2,y-2), (x-2, 1)
    
    cdef int x1 = (int)((Pose[0,0]*1+Pose[0,1]*1+Pose[0,2])/(Pose[2,0]*1+Pose[2,1]*1+Pose[2,2]))
    cdef int y1 = (int)((Pose[1,0]*1+Pose[1,1]*1+Pose[1,2])/(Pose[2,0]*1+Pose[2,1]*1+Pose[2,2]))
    
    cdef int x2 = (int)((Pose[0,0]*1+Pose[0,1]*(y-2)+Pose[0,2])/(Pose[2,0]*1+Pose[2,1]*(y-2)+Pose[2,2]))
    cdef int y2 = (int)((Pose[1,0]*1+Pose[1,1]*(y-2)+Pose[1,2])/(Pose[2,0]*1+Pose[2,1]*(y-2)+Pose[2,2]))

    cdef int x3 = (int)((Pose[0,0]*(x-2)+Pose[0,1]*(y-2)+Pose[0,2])/(Pose[2,0]*(x-2)+Pose[2,1]*(y-2)+Pose[2,2]))
    cdef int y3 = (int)((Pose[1,0]*(x-2)+Pose[1,1]*(y-2)+Pose[1,2])/(Pose[2,0]*(x-2)+Pose[2,1]*(y-2)+Pose[2,2]))
    
    cdef int x4 = (int)((Pose[0,0]*(x-2)+Pose[0,1]*1+Pose[0,2])/(Pose[2,0]*(x-2)+Pose[2,1]*1+Pose[2,2]))
    cdef int y4 = (int)((Pose[1,0]*(x-2)+Pose[1,1]*1+Pose[1,2])/(Pose[2,0]*(x-2)+Pose[2,1]*1+Pose[2,2]))
    
    cv2.line(img, (y1,x1),(y2,x2), (r,g,b), thickness)
    cv2.line(img, (y2,x2),(y3,x3), (r,g,b), thickness)
    cv2.line(img, (y3,x3),(y4,x4), (r,g,b), thickness)
    cv2.line(img, (y4,x4),(y1,x1), (r,g,b), thickness)
    
    return img

def draw_rectangle_ONCOLOR2(np.ndarray[float, ndim=2]  Pose, np.ndarray[float, ndim=3] img, int x, int y, int r, int g, int b, int thickness):
    #(1,1), (1, y-2), (x-2,y-2), (x-2, 1)
    
    cdef int x1 = (int)((Pose[0,0]*1+Pose[0,1]*1+Pose[0,2])/(Pose[2,0]*1+Pose[2,1]*1+Pose[2,2]))
    cdef int y1 = (int)((Pose[1,0]*1+Pose[1,1]*1+Pose[1,2])/(Pose[2,0]*1+Pose[2,1]*1+Pose[2,2]))
    
    cdef int x2 = (int)((Pose[0,0]*1+Pose[0,1]*(y-2)+Pose[0,2])/(Pose[2,0]*1+Pose[2,1]*(y-2)+Pose[2,2]))
    cdef int y2 = (int)((Pose[1,0]*1+Pose[1,1]*(y-2)+Pose[1,2])/(Pose[2,0]*1+Pose[2,1]*(y-2)+Pose[2,2]))

    cdef int x3 = (int)((Pose[0,0]*(x-2)+Pose[0,1]*(y-2)+Pose[0,2])/(Pose[2,0]*(x-2)+Pose[2,1]*(y-2)+Pose[2,2]))
    cdef int y3 = (int)((Pose[1,0]*(x-2)+Pose[1,1]*(y-2)+Pose[1,2])/(Pose[2,0]*(x-2)+Pose[2,1]*(y-2)+Pose[2,2]))
    
    cdef int x4 = (int)((Pose[0,0]*(x-2)+Pose[0,1]*1+Pose[0,2])/(Pose[2,0]*(x-2)+Pose[2,1]*1+Pose[2,2]))
    cdef int y4 = (int)((Pose[1,0]*(x-2)+Pose[1,1]*1+Pose[1,2])/(Pose[2,0]*(x-2)+Pose[2,1]*1+Pose[2,2]))
    
    cv2.line(img, (x1,y1),(x2,y2), (r,g,b), thickness)
    cv2.line(img, (x2,y2),(x3,y3), (r,g,b), thickness)
    cv2.line(img, (x3,y3),(x4,y4), (r,g,b), thickness)
    cv2.line(img, (x4,y4),(x1,y1), (r,g,b), thickness)
    
    return img

def draw_cube_ONCOLOR(np.ndarray[float, ndim=2]  Pose, np.ndarray[float, ndim=3] img, int x, int y, int r, int g, int b, int thickness):
    #(1,1), (1, y-2), (x-2,y-2), (x-2, 1)
    cdef int h = 0;
    if(x > y):
        h = y-2;
    else:
        h = x-2;
    
    cdef int x1 = (int)((Pose[0,0]*1+Pose[0,1]*1+Pose[0,2])/(Pose[2,0]*1+Pose[2,1]*1+Pose[2,2]))
    cdef int y1 = (int)((Pose[1,0]*1+Pose[1,1]*1+Pose[1,2])/(Pose[2,0]*1+Pose[2,1]*1+Pose[2,2]))
    
    cdef int x2 = (int)((Pose[0,0]*1+Pose[0,1]*(y-2)+Pose[0,2])/(Pose[2,0]*1+Pose[2,1]*(y-2)+Pose[2,2]))
    cdef int y2 = (int)((Pose[1,0]*1+Pose[1,1]*(y-2)+Pose[1,2])/(Pose[2,0]*1+Pose[2,1]*(y-2)+Pose[2,2]))

    cdef int x3 = (int)((Pose[0,0]*(x-2)+Pose[0,1]*(y-2)+Pose[0,2])/(Pose[2,0]*(x-2)+Pose[2,1]*(y-2)+Pose[2,2]))
    cdef int y3 = (int)((Pose[1,0]*(x-2)+Pose[1,1]*(y-2)+Pose[1,2])/(Pose[2,0]*(x-2)+Pose[2,1]*(y-2)+Pose[2,2]))
    
    
    cv2.line(img, (y1,x1),(y2,x2), (r,g,b), thickness)
    cv2.line(img, (y2,x2),(y3,x3), (r,g,b), thickness)
#    cv2.line(img, (y3,x3),(y4,x4), (r,g,b), thickness)
#    cv2.line(img, (y4,x4),(y1,x1), (r,g,b), thickness)
    
#    cv2.line(img, (y5,x5),(y6,x6), (r,g,b), thickness)
#    cv2.line(img, (y6,x6),(y7,x7), (r,g,b), thickness)
#    cv2.line(img, (y7,x7),(y8,x8), (r,g,b), thickness)
#    cv2.line(img, (y8,x8),(y5,x5), (r,g,b), thickness)
#    
#    cv2.line(img, (y1,x1),(y5,x5), (r,g,b), thickness)
#    cv2.line(img, (y2,x2),(y6,x6), (r,g,b), thickness)
#    cv2.line(img, (y3,x3),(y7,x7), (r,g,b), thickness)
#    cv2.line(img, (y4,x4),(y8,x8), (r,g,b), thickness)
    
    return img
    

def updateHx(np.ndarray[float, ndim=2] compImage, 
             np.ndarray[float, ndim=2] Pose,
             np.ndarray[float, ndim=2] template, 
             np.ndarray[float, ndim=2] temGx, 
             np.ndarray[float, ndim=2] temGy):
    cdef int x, y
    x = template.shape[0]
    y = template.shape[1]
    
    #warp image
    cdef np.ndarray[float, ndim=2] warpImage =  warp(compImage, Pose, x, y);
    
    #gradient of the compared image
    cdef np.ndarray[float, ndim=2] comGx = (cv2.Sobel(warpImage,cv2.CV_64F,1,0,ksize=5).astype(FLOAT))
    cdef np.ndarray[float, ndim=2] comGy = (cv2.Sobel(warpImage,cv2.CV_64F,0,1,ksize=5).astype(FLOAT))
    #init
    
    cdef np.ndarray[float, ndim=2] J = np.empty((x*y,8),dtype=FLOAT);
    cdef np.ndarray[float, ndim=2] detaS = np.empty((x*y,1),dtype=FLOAT);#∆S
    
    cdef np.ndarray[float, ndim=2] dx = np.empty((8,1),dtype=FLOAT);
    cdef np.ndarray[float, ndim=2] Adx  = np.empty((3,3),dtype=FLOAT);
    
    
    cdef int i, j, k, d;
    cdef float Ix, Iy, size, diff, temp, u, v

    #find J and ∆s
    for i in range(0, x):#row  in x direction
       for j in range(0, y):#column in y direction
            k = i*y+j
            u = float(i - x/2);
            v = float(j - y/2);
            
            Iy = comGx[i,j]+temGx[i,j]
            Ix = comGy[i,j]+temGy[i,j]
            
            J[k,0] = v*Ix
            J[k,1] = Ix
            J[k,2] = u*Iy
            J[k,3] = Iy
            J[k,4] = -u*(u)*Ix-u*(v)*Iy
            J[k,5] = -(u)*v*Ix-v*(v)*Iy
            J[k,6] = u*Ix-v*Iy
            J[k,7] = (u)*Ix-(v+v)*Iy
            
            detaS[k,0] = warpImage[i,j]- template[i,j]

    dx = -2*np.linalg.inv(J.transpose().dot(J)).dot(J.transpose().dot(detaS))

    #A(∆x)
    Adx[0,0] = dx[6,0]
    Adx[0,1] = dx[0,0]
    Adx[0,2] = dx[1,0]
    
    Adx[1,0] = dx[2,0]
    Adx[1,1] =  -dx[6,0]-dx[7,0]
    Adx[1,2] = dx[3,0]
    
    Adx[2,0] = dx[4,0]
    Adx[2,1] = dx[5,0]
    Adx[2,2] = dx[7,0]
    
    return(Adx, np.linalg.norm(dx), warpImage)
    
def updateHx3(np.ndarray[float, ndim=2] compImage, 
             np.ndarray[float, ndim=2] Pose,
             np.ndarray[float, ndim=2] template, 
             np.ndarray[float, ndim=2] temGx, 
             np.ndarray[float, ndim=2] temGy, int xd, int yd):
    cdef int x, y
    x = template.shape[0]
    y = template.shape[1]
    
    #warp image
    cdef np.ndarray[float, ndim=2] warpImage =  warp(compImage, Pose, x, y);
    
    #gradient of the compared image
    cdef np.ndarray[float, ndim=2] comGx = (cv2.Sobel(warpImage,cv2.CV_64F,1,0,ksize=5).astype(FLOAT))
    cdef np.ndarray[float, ndim=2] comGy = (cv2.Sobel(warpImage,cv2.CV_64F,0,1,ksize=5).astype(FLOAT))
    #init
    
    cdef np.ndarray[float, ndim=2] dx;
    
    cdef np.ndarray[float, ndim=2] J = np.empty(((x)*(y),8),dtype=FLOAT);
    cdef np.ndarray[float, ndim=2] detaS = np.empty(((x)*(y),1),dtype=FLOAT);#∆S
    
    cdef np.ndarray[float, ndim=2] Adx  = np.empty((3,3),dtype=FLOAT);
    
    
    cdef int i, j, k, d;
    cdef float Ix, Iy, size, diff, temp, u, v

    #find J and ∆s
    for i in range(0, x):#row  in x direction
       for j in range(0, y):#column in y direction
            k = i*y+j
            u = float(i - x/2);
            v = float(j - y/2);
            
            Iy = comGx[i,j]+temGx[i,j]
            Ix = comGy[i,j]+temGy[i,j]
            
            J[k,0] = v*Ix
            J[k,1] = Ix
            J[k,2] = u*Iy
            J[k,3] = Iy
            J[k,4] = -u*(u)*Ix-u*(v)*Iy
            J[k,5] = -(u)*v*Ix-v*(v)*Iy
            J[k,6] = u*Ix-v*Iy
            J[k,7] = (u)*Ix-(v+v)*Iy
            
            detaS[k,0] = warpImage[i,j]- template[i,j]
            
    return(J, detaS, warpImage)