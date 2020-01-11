import cvlib as cv
from cvlib.object_detection import draw_bbox
import cv2
import sys
import glob


#size=(720,576)
#codec = cv2.VideoWriter_fourcc(*'XVID')
#test_output = cv2.VideoWriter('ped_test.avi', codec, 30, size)
file="C:/Users/jessi/Documents/MM/MM803/project/object-detection-opencv-master/803data/pedestrians/input/in000568.jpg"
#for file in glob.glob("C:/Users/jessi/Documents/MM/MM803/project/object-detection-opencv-master/803data/pedestrians/input/*.jpg"):
frame = cv2.imread(file)
    #cv2.imshow('ini',frame)

bbox, label, conf = cv.detect_common_objects(frame, confidence=0.9, model='yolov3-tiny')

print(bbox, label, conf)

    # draw bounding box over detected objects
    #mask_out = np.array(draw_mask(frame, bbox, label))
    #b = time.time()
    #c = b - a
print('label'+str(label))
print('bbox'+str(bbox))
print('conf'+str(conf))

out = draw_bbox(frame, bbox, label, conf, write_conf=True)

    # display output
cv2.imshow('',out)
cv2.waitKey()



# release resources
#test_output.release()
#cv2.imwrite('detectionresult.jpg',out)
cv2.destroyAllWindows()




