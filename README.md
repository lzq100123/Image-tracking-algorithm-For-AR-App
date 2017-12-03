# Image-tracking-algorithm-For-AR-App

## Author:
### [Fanghzou Xiong](https://github.com/matthewxfz),  [Zhiquan Li](http://github.com/lzq100123)  

## Status: In process  


## Note
##### This project implements Efficient Second Order Minimization(ESM) Algorithm with an efficient way in paper[1] , and a Hibrid Algorithm using ESM and OF proposing in paper[2], with python and cython. The OF algorithm is directly using function from OpenCV3. We have a report describing implmentaiton details in ./doc

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
3, Testing and Refine in process


## Dependencies: 
##### * OpenCV3 3.1.0  
##### * python 2.7.0.

## Demo
<a href="https://youtu.be/k-OKT9mJxOA" target="_blank"><img src="https://00e9e64bac987e1aa0971205ff348b7a7f98dbc0eb4a57f984-apidata.googleusercontent.com/download/storage/v1/b/cs522image/o/youtube.png?qk=AD5uMEvDcQLScvXYvHR7ui8mjoiROVIQ9nph4qbrNQLPHlcXMPrrY1KVQsI6pwBchBiw83O8VhRHdnxbkzosVgOugEScYyKxRXgoPe9p4e5k3vWPoU-0PhaPbdkT9IKh_mSgH3xizjZ3TWbFJmv8_azvjrcq6MLNwlUb7HfFa2azmm4vBxD7jSRRWRCsVfBX04zMBsHHzf7x83glyxt-v7AN3nJ4W5J_BjubRr4z40xGLXeyMeQ-dqQ-X3pbLSPh43dLzdj0lPKjGYhMgLBwEERQSNJjmJhulrmhIPdB-fNL3QM48J0mlGjKm0zvSyR7wt16KBi_pcXpxbWLXRMj1bKT39oBL2xdNjm_u5rNVkp-9yJfc7ZjuL1CT6qhhNIiYw7Fx_1ZOnKsizN3mRRk9QrR506fVVFY16hV9aI8xgEOzhhGxW6miLkSSSUwXedMtnJkI7Cx-t7S8ijrIiT0QEQ3yPLNJqIgkrsE5qt9GEH1EVFmbuu6TYWbWnXUGLMBZkiw_PLzEdLryiKOg0I_3n903-5xtJiS8YpD_h2zAJWezw_N8ZkY_qRG2kFLnsHoq8QwR6mCSBJaAEBX_mrt_Q7P5shSbwiQJNQ8n4VzfYeiB6YA35ysgTmA46E4N44VW_9IkL8TeQ0oivnv1_0jN3l3sOvr6zMx15alJIzQSWJ1WxZYd2D3TxZONbIrTzg8L3gLt_nKkuHx0bIXgNYUAZ8CNOVnAg9rn-x159fexhpkfyhQMBDyrac" 
alt="ESM and Hibrid algorithm demo" width="355" height="186" border="10" /></a>


## Refference:
##### [1] S. Benhimane I.N.R.I.A. France, E. Malis I.N.R.I.A. France, Real-time image-based tracking of planes using efficient second-order minimization.

##### [2] Ievgen M. Gorovyi, Dmytro S. Sharapov. Advanced image tracking approach for augmented reality applications

