import cv2
import sys
import numpy as np



def draw_mask(img, image_bg, bbox, labels):

    global mask

    global masked_img

    alpha = 0.95

    mask=[]


    #print('bboxt: ' + str(type(bbox)) + '\n')
    bbox=np.array(bbox)
    #img = np.array(img)
    print('bbox: ' + str(bbox) + '\n')
    #print('labels: '+str(labels)+'\n')
    #print('img: '+str(img.shape)+'\n')
    masked_bg = np.full((img.shape[0], img.shape[1]), 0, dtype="uint8")

    for i,l in enumerate(labels):

        #print('bbox: '+str(bbox[i])+'\n')

        crop_obj = img[bbox[i][1]:bbox[i][3], bbox[i][0]:bbox[i][2]]
        crop_obj = np.array(crop_obj,dtype=np.uint8)
        #cv2.imshow('', crop_obj)
        #cv2.waitKey()

        #crop_obj = cv2.normalize(crop_obj.astype('uint8'), None, 0.0, 1.0, cv2.NORM_MINMAX)

        crop_bg = image_bg[bbox[i][1]:bbox[i][3], bbox[i][0]:bbox[i][2]]

        #cv2.imshow('', crop_bg)
        #cv2.waitKey()

        #crop_bg = cv2.normalize(crop_bg.astype('uint8'), None, 0.0, 1.0, cv2.NORM_MINMAX)

        crop_bg = np.array (crop_bg,dtype=np.uint8)
        #crop_bg = cv2.cvtColor((crop_bg).astype("uint8"), cv2.COLOR_BGR2GRAY)

        #cv2.imshow('crop_bg', crop_bg)

        #cv2.waitKey()

        backgroundSubtractor = cv2.bgsegm.createBackgroundSubtractorMOG()
        backgroundSubtractor.apply(crop_bg, learningRate=0.5)

        fgmask = backgroundSubtractor.apply(crop_obj, learningRate=0)
        masked_bg[bbox[i][1]:bbox[i][3], bbox[i][0]:bbox[i][2]] = fgmask
        masked_img=cv2.cvtColor(masked_bg,cv2.COLOR_GRAY2RGB)

        #cv2.imshow("", masked_img)
        #cv2.waitKey()

    return masked_img