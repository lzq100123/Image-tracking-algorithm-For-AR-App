# Image-tracking-algorithm-For-AR-App

## Author:
### [Fanghzou Xiong](https://github.com/matthewxfz),  [Zhiquan Li](http://github.com/lzq100123)  

## Status: In process  


## Note
##### This project implements Efficient Second Order Minimization(ESM) Algorithm with an efficient way in paper[1] , and a Hibrid Algorithm using ESM and OF proposing in paper[2], with python and cython. The OF algorithm is directly using function from OpenCV3.  

## ESM
##### To run th#e ESM program, simply clone the program, run ./src/EsmTracker.py, using the command below to play with it:
```bash
Usage:  
i - reaload  
w - save to file out<x>.jpg  
o - crop the image as template and show ORB  
e - crop the image as template and do esm tracking  
f - crop the image as template and do OF tracking   
k - crop the image as template and do hybird algorithm tracking  
c - crop the image only  
w - test warp image  
r - test draw rectangle   
h - help  
```
  
##### * You basicly need to crop every time for tracking template before tracking.
##### * The window are packed together needs to be rearranged.

## Hibrid Algorithm
Run ./src/Hibridy.py
```bash
Usage:
space - crop image, hit it again to start tracking.
```

## Todolist:
1, ESM implemented √  
2, Hibrid implemented √  
3, Refine Algorithm  


## Dependencies: 
##### OpenCV3 3.1.0, python 2.7.0.

## Demo
<a href="https://youtu.be/k-OKT9mJxOA
" target="_blank"><img src="https://youtu.be/k-OKT9mJxOA/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="240" height="180" border="10" /></a>


## Refference:
##### [1] S. Benhimane I.N.R.I.A. France, E. Malis I.N.R.I.A. France, Real-time image-based tracking of planes using efficient second-order minimization.

##### [2] Ievgen M. Gorovyi, Dmytro S. Sharapov. Advanced image tracking approach for augmented reality applications

