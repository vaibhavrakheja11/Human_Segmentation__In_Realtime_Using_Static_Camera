# Background Substraction in YOLIO
# cvlib
A high level easy-to-use open source Computer Vision library for Python.

# Installation

# Installing dependencies (installed already with MaskRCNN)

* OpenCV
* TensorFlow

If you don't have them already installed, you can install through pip

# Installing cvlib
`pip install cvlib`

# Real time object detection
`YOLOv3` is actually a heavy model to run on CPU. If you are working with real time webcam / video feed and doesn't have GPU, try using `tiny yolo` which is a smaller version of the original YOLO model. It's significantly fast but less accurate.
`bbox, label, conf = cv.detect_common_objects(img, confidence=0.25, model='yolov3-tiny')`

# Utils
# Video to frames
`get_frames( )` method can be helpful when you want to grab all the frames from a video. Just pass the path to the video, it will return all the frames in a list. Each frame in the list is a numpy array.
`
import cvlib as cv
frames = cv.get_frames('~/Downloads/demo.mp4')
`
Optionally you can pass in a directory path to save all the frames to disk.
`
frames = cv.get_frames('~/Downloads/demo.mp4', '~/Downloads/demo_frames/')
`

# Creating gif
`animate( )` method lets you create gif from a list of images. Just pass a list of images or path to a directory containing images and output gif name as arguments to the method, it will create a gif out of the images and save it to disk for you.

`
cv.animate(frames, '~/Documents/frames.gif')
`

# Background Subtraction
`mask.py` and `maskMOG.py` provides two methods of background subtraction. `mask.py` simply uses frame differencing, while `maskMOG.py` uses MOG. Feel free to modify these two methods in `object_detect.py`.

from maskMOG import draw_mask
``
or
``
from mask import draw_mask
``
To run the mask generation, use following command
``
python yolo_bg.py --image_path
``
Before running mask generation, make sure set the background image in `yolo_bg.py`

``
image_bg=cv2.imread('imgpath')
``


