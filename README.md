# joint-cascade-face-detection-and-alignment

This project is a C++ reimplementation of joint cascade face detection and alignment in the ECCV 2014 

This project start with the code from https://github.com/yulequan/face-alignment-in-3000fps

You should read yulequan's codes first, then compare them with mine.

There are some differences between paper's and mine:

1. Without SIFT+SVM
2. Without multi-scale pixel difference feature
3. Without non-maxmium suppression

I do not make sure the implementation is right, but in my experiment, i get the right face with right keypoints at 50FPS on i7-3770K (optimization version).

The version here is only for training. You shuold implement a version for testing by yourself.

Sorry for my codestyle : (
