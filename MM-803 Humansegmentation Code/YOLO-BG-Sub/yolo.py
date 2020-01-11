

import cvlib as cv
import cv2
from cvlib.object_detection import draw_bbox
import sys

img=cv2.imread('in000216.jpg')
bbox,label, conf = cv.detect_common_objects(img)
output_image = draw_bbox(img, bbox, label, conf)
cv2.imshow('',output_image)
cv2.waitKey()
cv2.imwrite('output.jpg',output_image)