#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 15:46:27 2017

@author: matthewxfz
"""
#%%
import numpy as np
import cv2
from matplotlib import pyplot as plt
from scipy.linalg import expm
import sys
sys.path.append("/Users/matthewxfz/cs512gits/project/src")
import util

import time

#%%
maxIte = 100
minNorm = 3e-3
orb_p_num = 500
orb_m_num = 50
SID = 0;

winName = 'esm'
winTemplateName = 'template'
winWaprName = 'warped Image'

def help():
    print("This program is demo for OF, ESM, ORB and hybrid function\n"+
          "Usage:\n"+
          "i - reaload\n"+
          "w - save to file out<x>.jpg"+"\n"+
          "o - crop the image as template and show ORB\n"+
          "e - crop the image as template and do esm tracking\n"+
          "f - crop the image as template and do OF tracking \n"+
          "k - crop the image as template and do hybird algorithm tracking\n"+
          "c - crop the image only\n"+
          "w - test warp image\n"+
          "r - test draw rectangle \n"+
          "h - help"+"\n");
          
def normlize(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #img = img.astype(np.float32)
    #mean, stddev = cv2.meanStdDev(img) 
    #img = img - mean;
    #img = (img/stddev)
    #img = cv2.blur(img, (3,3)) 
    return img

def esm(img, template, temGx, temGy, Pose):
    Rotation_B = np.eye(3,3,dtype = np.float);
    Rotation_B[2,2] = 0;
    global maxIte
    global minNorm
    ite = 0
    for i in range(0,maxIte):
        adx, diff, warpImage = util.updateHx(img.astype(np.float32),
                                            Pose.astype(np.float32),
                                            template.astype(np.float32),
                                            temGx.astype(np.float32),
                                            temGy.astype(np.float32))
    
        Pose += Rotation_B;
        Pose = np.dot(Pose, expm(adx))
        Pose -= Rotation_B;
        if(diff < minNorm):
            break;
        else:
            ite+=1;
            
    return(Pose, ite, warpImage)
        
def OFA(old_frame, frame_gray, p0):
    params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_frame, frame_gray, p0, None, **params)
    M, h = cv2.findHomography(p0, p1, cv2.RANSAC,5.0)
    return M

def detectOBR(frame,template):
    orb = cv2.ORB_create()
    kp2, des2 = orb.detectAndCompute(template,None)#keypoint in scene
    kp1, des1 = orb.detectAndCompute(frame,None)#keypoint in scene
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1,des2)# find matched keypoint index
    matches = sorted(matches, key = lambda x:x.distance)
    matches = matches[:20]
    p0 = np.array([np.array([np.array([kp1[i.queryIdx].pt[0],kp1[i.queryIdx].pt[1]],dtype = np.float32)],dtype = np.float32) for i in matches],dtype = np.float32)
    return p0

def getOBR(frame):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(frame,None)#keypoint in scene
    return kp1;

def crop():
    global template
    global Pose
    global WPose
    global temGx
    global temGy
    global x
    global y
    global img_p
    global img
    global croped
    global H1
    
    showCrosshair = False
    fromCenter = False
    r= cv2.selectROI("Image", img_p, fromCenter, showCrosshair)
    
    #update template inofrmation
    template = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    xd = int(r[1]);#x coordinate of the start of template
    yd = int(r[0]);#y coordinate of the start of template
    Pose = np.eye(3,3,dtype = np.float)#2,3
    Pose[0,2] = xd;
    Pose[1,2] = yd;
    WPose = Pose
    H1 = Pose
    
    #udpate gradient of template
    temGx = cv2.Sobel(template,cv2.CV_64F,1,0,ksize=5)  #derivative of template in x and y direction
    temGy = cv2.Sobel(template,cv2.CV_64F,0,1,ksize=5)
    #size of template
    x = template.shape[0] # length of template in x and y direction
    y = template.shape[1]
    #check the template and warp image
    warp = util.warp(img.astype(np.float32), 
             WPose.astype(np.float32),
             x,
             y)
    cv2.imshow(winTemplateName,template)
    cv2.imshow(winWaprName, warp.astype(np.uint8))
    
    croped = True;
    print("update Pose")
    print(Pose)

#%% Main
cap = cv2.VideoCapture(0)

#Esm init
temGx = np.empty((600,600), dtype = np.float)
temGy = np.empty((600,600), dtype = np.float)
template = np.empty((600,600), dtype = np.float)
Pose = np.eye(3,3,dtype = np.float)#2,3
WPose = Pose
H1 = Pose;

#control init
croped = False;
flag = 'i';
save = 0;

#OF init
pre_img = np.zeros([1,1],dtype = float)   #previous image
pre_img_p = None #previous image with color
P1 = None; #old p
P2 = None; # current p
Pt = None
params = dict( winSize  = (15,15),
              maxLevel = 2,
              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
t = 0
mask = None;
fast = cv2.FastFeatureDetector_create()


while cap.isOpened():
    retval,image = cap.read();
    if not retval:
        break
    else:
        img_o = image;
        img_p = cv2.pyrDown(img_o)
#        img_p = img_o
        img = normlize(img_p);
        if(len(pre_img) == 1):
            pre_img = img
            pre_img_p = img_p
            continue;

    key = cv2.waitKey(5)
    if key == 27:
        break
    elif key ==  ord('s'):
        save = 1;
    elif key != -1:
        flag = key
        
    if key == ord('h'):
        help();
        
    if key == ord('c'):
        crop();
        croped = False;
        
    if key == ord('w'):
        warp = util.warp(img.astype(np.float32), 
             WPose.astype(np.float32),
             x,
             y)
        cv2.imshow(winWaprName, warp.astype(np.uint8))
    
    if flag == ord('r'):
       if(croped):  
           img = util.draw_rectangle(Pose.astype(np.float32),
                                       img.astype(np.float32), 
                                        x,y, 255, 0, 0, 2).astype(np.uint8)
       else:
            pass;
            
    if flag == ord('f'):
        mask;
        if(croped):
            t += 1
            P2, st, err = cv2.calcOpticalFlowPyrLK(pre_img, img, P1 , None, **params)
            good_new = P2[st==1]
            good_old = P1[st==1]
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
                mask = cv2.line(mask, (a,b),(c,d), (0, 255, 0), 2)
                img = cv2.circle(img,(a,b),5,(0,0,255),-1)
                img = cv2.circle(img, (c,d), 5, (255,0,0),-1)
            if t == 2:
                mask = np.zeros_like(img)
                t = 0
            img_p = cv2.add(img,mask)
#            M, h = cv2.findHomography(p0, p1, cv2.RANSAC,5.0)
            P1 = P2;
        else:
            crop();
            Pt = detectOBR(img, template)
            P1 = Pt
            P2 = None;
            mask = np.zeros_like(img)
            
    if flag == ord('k'):
        mask;
        H1 = np.eye(3,3,dtype = np.float)#2,3
        if(croped):
            start_time = time.time()
            
            P2, st, err = cv2.calcOpticalFlowPyrLK(pre_img, img, P1 , None, **params)
            deltaM, h = cv2.findHomography(util.swapxy(P1), util.swapxy(P2), cv2.RANSAC,5.0)
            H2 = deltaM.dot(Pose)
            if(deltaM is None):
                H2  = Pose
                print "bingo"
            else:
                H2 = deltaM.dot(Pose)
            
            H2hat, ite, warpImage = esm(img, template, temGx, temGy, H2)
            
            P2hat = util.transform(WPose.astype(np.float32), util.swapxy(Pt.astype(np.float32)))
            P1 = util.swapxy(P2hat);
            Pose = H2hat
            
            cv2.imshow(winWaprName, warpImage.astype(np.uint8))
            
            img_p = util.draw_rectangle(WPose.astype(np.float32),
                                          img.astype(np.float32), 
                                          x, y, 255, 0, 0, 2).astype(np.uint8)
            good_new = Pt[st==1]
            good_old = Pt[st==1]
            for i,(new,old) in enumerate(zip(good_new,good_old)):
                a,b = new.ravel()
                c,d = old.ravel()
#                mask = cv2.line(mask, (a,b),(c,d), (0, 255, 0), 2)
                img_p = cv2.circle(img_p,(a,b),5,(0,0,255),-1)
                img_p = cv2.circle(img_p, (c,d), 5, (255,0,0),-1)
#            img_p = cv2.add(img_p,mask)
#            print("----Optimal takes %s  seconds with %s iteration----" % (time.time() - start_time,ite))
#            print time.time() - start_time;
        else:
            crop();
            Pt = detectOBR(img, template)
#            Pt = util.transform(WPose.astype(np.float32), Pt.astype(np.float32))
            invPose = WPose
            invPose[0,2] = -invPose[0,2]
            invPose[1,2] = -invPose[0,2]
            b = invPose[0,]
            invPose[0,] = -invPose[1,]
            invPose[1,] = b
            Pt = util.transform(invPose.astype(np.float32), Pt.astype(np.float32))
            P1 = Pt
            P2 = None;
            mask = np.zeros_like(img)
            
            
    if flag == ord('i'): 
        croped = False;
        flag = -1;
        
    if key == ord('o'): #test with orb
        # find the keypoints with ORB
        if(croped):
            orb = cv2.ORB_create(orb_p_num)
            kp1, des1  = orb.detectAndCompute(template,None)
            kp2, des2  = orb.detectAndCompute(img,None)
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(des1,des2)
            # Sort them in the order of their distance.
            matches = sorted(matches, key = lambda x:x.distance)
            # Draw first matches.
            img_c = cv2.drawMatches(template,kp1,img,kp2,matches[:20] ,None, flags=2)
            
            plt.imshow(img_c),plt.show()
        else:
            crop();
            
    if flag == ord('e'): #test with esm
        if(croped):
            
            ite = 0;
            start_time = time.time()
            Pose, ite, warpImage = esm(img, template, temGx, temGy, Pose)
#            print("----Esm takes %s  seconds with %s iteration----" % (time.time() - start_time,ite))
            print time.time() - start_time;
            cv2.imshow(winWaprName, warpImage.astype(np.uint8))
            
            img_p = util.draw_rectangle_ONCOLOR(Pose.astype(np.float32),
                                          img_p.astype(np.float32), 
                                          x, y, 0, 255, 0, 2).astype(np.uint8)  
        else:
            crop();
        
    if(save == 1):
        cv2.imwrite("out"+str(SID)+".jpg",img_p)
        SID+=1;
        save = 0;
    cv2.imshow(winName,img_p)
    
    pre_img = img
    pre_img_p = img_p

cap.release()
cv2.destroyAllWindows()