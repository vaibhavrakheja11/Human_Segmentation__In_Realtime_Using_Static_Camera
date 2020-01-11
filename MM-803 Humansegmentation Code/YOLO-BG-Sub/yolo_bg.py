

# object detection example
# usage: python object_detection.py <input_image>

# import necessary packages
import cvlib as cv
from cvlib.object_detection import draw_bbox
import sys
import cv2
import numpy as np
import time
from maskMOG import draw_mask

# read input image
image = cv2.imread(sys.argv[1])
image_bg=cv2.imread('testImageBGSub.jpg')

# apply object detection
a=time.time()
bbox, label, conf = cv.detect_common_objects(image)

#print(bbox, label, conf)
#print(len(label))
ls=[]
for i in range(len(label)):
    #print(label[i])
    if label[i]!='person':
        ls.append(i)
#print(ls)

for index in sorted (ls,reverse=True):
    #print(label[index])
    del label[index]
    del conf[index]
    del bbox[index]


# draw bounding box over detected objects

mask_out = draw_mask(image, image_bg, bbox, label)


out = draw_bbox(image, bbox, label, conf)

#print("length mask out",len(mask_out))
b = time.time()
c = b - a
print('Time per frame', c)

# display output
# press any key to close window
cv2.imshow("object_detection", out)

#for i in range(len(mask_out)):

    #cv2.imshow("",mask_out[i])
    #cv2.waitKey()
    #print(mask_out[i].shape)
cv2.imshow("",mask_out)
cv2.waitKey()
# save output
#cv2.imwrite("object_detection.jpg", out)

# release resources
cv2.destroyAllWindows()

