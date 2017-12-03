#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 11:31:37 2017

@author: matthewxfz
"""
#%%
from __future__ import print_function

import numpy as np
import cv2
import video
from common import draw_str
from video import presets

from scipy.linalg import expm
import sys
sys.path.append("/Users/matthewxfz/cs512gits/project/src")
import util

import time

#%%

lk_params = dict( winSize  = (19, 19),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

feature_params = dict( maxCorners = 1000,
                       qualityLevel = 0.01,
                       minDistance = 8,
                       blockSize = 19 )

def checkedTrace(img0, img1, p0, back_threshold = 1.0):
    p1, _st, _err = cv2.calcOpticalFlowPyrLK(img0, img1, p0, None, **lk_params)
    p0r, _st, _err = cv2.calcOpticalFlowPyrLK(img1, img0, p1, None, **lk_params)
    d = abs(p0-p0r).reshape(-1, 2).max(-1)
    status = d < back_threshold
    return p1, status

def crop(img):
    global template
    global Pose
    global WPose
    global temGx
    global temGy
    global x
    global y
    global croped
    
    showCrosshair = False
    fromCenter = False
    r= cv2.selectROI("Image", img, fromCenter, showCrosshair)
    
    #update template inofrmation
    template = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
    xd = int(r[1]);#x coordinate of the start of template
    yd = int(r[0]);#y coordinate of the start of template
    Pose = np.eye(3,3,dtype = np.float)#2,3
    Pose[0,2] = xd;
    Pose[1,2] = yd;
    WPose = Pose
    
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
            
    return(Pose, ite, warpImage, diff)

green = (0, 255, 0)
red = (0, 0, 255)

maxIte = 200
minNorm = 2e-3

winName = 'esm'
winTemplateName = 'template'
winWaprName = 'warped Image'

#Esm init
temGx = np.empty((600,600), dtype = np.float)
temGy = np.empty((600,600), dtype = np.float)
template = np.empty((600,600), dtype = np.float)
Pose = np.eye(3,3,dtype = np.float)#2,3
WPose = Pose
x = 0
y = 0

#control init
croped = False;

class App:
    def __init__(self, video_src):
        self.cam = self.cam = video.create_capture(video_src, presets['book'])
        self.p0 = None
        self.use_ransac = True

    def run(self):
        global x
        global y
        global Pose
        global WPose
        
        while True:
            _ret, frame = self.cam.read()
            frame = cv2.pyrDown(frame)
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            vis = frame.copy()
            if self.p0 is not None:
                start_time = time.time()
                p2, trace_status = checkedTrace(self.gray1, frame_gray, self.p1)

                self.p1 = p2[trace_status].copy()
                self.p0 = self.p0[trace_status].copy()
                self.gray1 = frame_gray

                if len(self.p0) < 4:
                    self.p0 = None
                    print("continue")
                    continue
                
                H, status = cv2.findHomography(self.p0, self.p1, 0, 10.0)
                
                h, w = frame.shape[:2]
                overlay = cv2.warpPerspective(self.frame0, H, (w, h))
                vis = cv2.addWeighted(vis, 0.5, overlay, 0.5, 0.0)
                
                for (x0, y0), (x1, y1), good in zip(self.p0[:,0], self.p1[:,0], status[:,0]):
                    if good:
                        cv2.line(vis, (x0, y0), (x1, y1), (0, 128, 0))
                    cv2.circle(vis, (x1, y1), 2, (red, green)[good], -1)
                draw_str(vis, (20, 20), 'track count: %d' % len(self.p1))
                
#                Pose = H.dot(Pose)
                Pose = H.dot(Pose)
                Pose, ite, warpImage,diff = esm(frame_gray, template, temGx, temGy, Pose)
                
                warpImage = util.warp(frame_gray.astype(np.float32), 
                     Pose.astype(np.float32),
                     template.shape[0],
                     template.shape[1])
                
                if(diff < minNorm or  True ):
                    self.frame0 = frame.copy()
                    self.p0 = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
                    if self.p0 is not None:
                        self.p1 = self.p0
#                        self.p1 = util.transform(Pose, self.p0)
                        self.gray0 = frame_gray
                        self.gray1 = frame_gray
                    
                cv2.imshow(winWaprName, warpImage.astype(np.uint8))
            
                vis = util.draw_rectangle_ONCOLOR(Pose.astype(np.float32),
                                          vis.astype(np.float32), 
                                          template.shape[0], template.shape[1],  0, 255, 0, 2).astype(np.uint8)
                print("----Esm takes %s  seconds with %s iteration----" % (time.time() - start_time,ite))
                
            else:
                p = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
                if p is not None:
                    for x, y in p[:,0]:
                        cv2.circle(vis, (x, y), 2, green, -1)
                    draw_str(vis, (20, 20), 'feature count: %d' % len(p))

            cv2.imshow('lk_homography', vis)

            ch = cv2.waitKey(1)
            if ch == 27:
                break
            if ch == ord(' '):
                if(croped):
                    self.frame0 = frame.copy()
                    self.p0 = cv2.goodFeaturesToTrack(frame_gray, **feature_params)
                    if self.p0 is not None:
                        self.p1 = self.p0
                        self.gray0 = frame_gray
                        self.gray1 = frame_gray
                else:
                    crop(frame_gray);
            
def main():
    import sys
    try:
        video_src = sys.argv[1]
    except:
        video_src = 0

    print(__doc__)
    App(video_src).run()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()