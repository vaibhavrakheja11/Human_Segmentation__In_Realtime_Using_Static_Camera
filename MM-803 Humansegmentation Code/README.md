### This is a combined README for 3 different set approaches used to achieve or experiment segementation in realtime.
### The segeregated readme files can be found in individual directories. 





# MASKRCNN

FOLDER NAME: MaskRcnn

## Installation

- Create a new env for MaskRCNN.
	`conda create -n MaskRCNN python=3.6 pip`


- Install the requirements.txt inside the zip folder pip install -r requirements.txt
	`actvitate MaskRCNN`
	`pip install -r requirements.txt`

-Download the coco preTrained Weights	
	Go here "https://github.com/matterport/Mask_RCNN/releases"
	download the "mask_rcnn_coco.h5" file
	place the file in the Mask_RCNN directory
	
	
- Open jupyter notebook
	`jupyter notebook`
	
- Run ipynb file ( MaskRcnn-Image.ipynb, HumanSegementationVideo.ipynb, RealTimeMaskRcnn.ipynb ) for Image, Video and Real time segmentation).





# You Only Look At Coefficients (YOLACT)
A simple, fully convolutional model for real-time instance segmentation.

FOLDER NAME: YOLACT




## Installation
 - Set up a Python3 environment.
 - Install [Pytorch](http://pytorch.org/) 1.0.1 (or higher) and TorchVision.
 - Install some other packages:
   Shell
   ### Cython needs to be installed before pycocotools
   pip install cython
   pip install opencv-python pillow pycocotools matplotlib 
   
 - Clone this repository and enter it:
   Shell
   git clone https://github.com/dbolya/yolact.git
   cd yolact
   
 - If you'd like to train YOLACT, download the COCO dataset and the 2014/2017 annotations. Note that this script will take a while and dump 21gb of files into ./data/coco.
   Shell
   sh data/scripts/COCO.sh
   
 - If you'd like to evaluate YOLACT on test-dev, download test-dev with this script.
   Shell
   sh data/scripts/COCO_test.sh
   



## Training
By default, we train on COCO. Make sure to download the entire dataset using the commands above.
 - To train, grab an imagenet-pretrained model and put it in ./weights.
   - For Resnet101, download resnet101_reducedfc.pth from [here](https://drive.google.com/file/d/1tvqFPd4bJtakOlmn-uIA492g2qurRChj/view?usp=sharing).
   - For Resnet50, download resnet50-19c8e357.pth from [here](https://drive.google.com/file/d/1Jy3yCdbatgXa5YYIdTCRrSV0S9V5g1rn/view?usp=sharing).
   - For Darknet53, download darknet53.pth from [here](https://drive.google.com/file/d/17Y431j4sagFpSReuPNoFcj9h7azDTZFf/view?usp=sharing).
 - Run one of the training commands below.
   - Note that you can press ctrl+c while training and it will save an *_interrupt.pth file at the current iteration.
   - All weights are saved in the ./weights directory by default with the file name <config>_<epoch>_<iter>.pth.
Shell
-Trains using the base config with a batch size of 8 (the default).
python train.py --config=yolact_base_config

### Trains yolact_base_config with a batch_size of 5. For the 550px models, 1 batch takes up around 1.5 gigs of VRAM, so specify accordingly.
python train.py --config=yolact_base_config --batch_size=5

### Resume training yolact_base with a specific weight file and start from the iteration specified in the weight file's name.
python train.py --config=yolact_base_config --resume=weights/yolact_base_10_32100.pth --start_iter=-1

### Use the help option to see a description of all available command line arguments
python train.py --help




## Citation
If you use YOLACT or this code base in your work, please cite

@inproceedings{bolya-iccv2019,
  author    = {Daniel Bolya and Chong Zhou and Fanyi Xiao and Yong Jae Lee},
  title     = {YOLACT: {Real-time} Instance Segmentation},
  booktitle = {ICCV},
  year      = {2019},
}






FOLDER NAME: YOLACT




#Installation
# cvlib
A high level easy-to-use open source Computer Vision library for Python.

# Installation

# Installing dependencies (installed already with MaskRCNN)

* OpenCV
* TensorFlow

If you don't have them already installed, you can install through pip

# Installing cvlib
pip install cvlib

# Real time object detection
YOLOv3 is actually a heavy model to run on CPU. If you are working with real time webcam / video feed and doesn't have GPU, try using tiny yolo which is a smaller version of the original YOLO model. It's significantly fast but less accurate.
bbox, label, conf = cv.detect_common_objects(img, confidence=0.25, model='yolov3-tiny')

# Utils
# Video to frames
get_frames( ) method can be helpful when you want to grab all the frames from a video. Just pass the path to the video, it will return all the frames in a list. Each frame in the list is a numpy array.

import cvlib as cv
frames = cv.get_frames('~/Downloads/demo.mp4')

Optionally you can pass in a directory path to save all the frames to disk.

frames = cv.get_frames('~/Downloads/demo.mp4', '~/Downloads/demo_frames/')


# Creating gif
animate( ) method lets you create gif from a list of images. Just pass a list of images or path to a directory containing images and output gif name as arguments to the method, it will create a gif out of the images and save it to disk for you.


cv.animate(frames, '~/Documents/frames.gif')


# Background Subtraction
mask.py and maskMOG.py provides two methods of background subtraction. mask.py simply uses frame differencing, while maskMOG.py uses MOG. Feel free to modify these two methods in object_detect.py.

from maskMOG import draw_mask

or

from mask import draw_mask

To run the mask generation, use following command

python yolo_bg.py --image_path

Before running mask generation, make sure set the background image in yolo_bg.py


image_bg=cv2.imread('imgpath')




