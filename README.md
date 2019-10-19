# Real Time Object Detection

To apply real-time object detection, we are using deep learning and OpenCV to work with video streams and video files. This will be accomplished using the highly efficient VideoStream class.</br>
Thus, we will get detections in actual video streams and measure the FPS processing rate.

## Setup
1. Install OpenCV 3.3.0 or higher version on your system. 
```
For Ubuntu, you can refer https://www.learnopencv.com/install-opencv3-on-ubuntu/
For Mac, you can refer https://www.learnopencv.com/install-opencv3-on-macos/
```
2. Create a Python 3 virtual environment for OpenCV called cv:</br>
```mkvirtualenv cv -p python3```

3. pip install OpenCV into your new environment:</br>
```pip install opencv-contrib-python```

## Demo
To run, type this in your command prompt-
python real_time.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel</br>

I’m using a Macbook Pro. A framerate of 6 FPS is pretty good using a CPU on a laptop.

