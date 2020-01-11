import cv2

import numpy as np


def draw_mask(img, image_bg, bbox, labels):

    global mask

    global masked_img

    alpha = 0.95

    mask=[]

    masked_img_final=[]

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

        #cv2.imshow('',crop_obj)
        #cv2.waitKey()

        crop_obj = cv2.normalize(crop_obj.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

        crop_bg = image_bg[bbox[i][1]:bbox[i][3], bbox[i][0]:bbox[i][2]]

        #cv2.imshow('', crop_bg)
        #cv2.waitKey()

        crop_bg = cv2.normalize(crop_bg.astype('float'), None, 0.0, 1.0, cv2.NORM_MINMAX)

        #cv2.imshow('', crop_bg)
        #cv2.waitKey()

        try:
            crop_bg_w = alpha * crop_bg + (1 - alpha) * crop_obj
        except:
            print('NoneType!')
            print('bg:' + str(crop_bg_w) + '\nobj:' + str(crop_obj))
            exit()

        mask=cv2.cvtColor((abs(crop_obj-crop_bg)*255).astype(np.uint8),cv2.COLOR_BGR2GRAY)

        #cv2.imshow('', mask)
        #cv2.waitKey()
        #Otsu Thresholding
        #ret, th = cv2.threshold(mask, 0, 255, cv2.THRESH_OTSU)
        #mask = np.invert(th)

        # Otsu's thresholding after Gaussian filtering
        blur = cv2.GaussianBlur(mask, (5, 5), 0)
        #mask = cv2.medianBlur(mask, 5)
        ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask=th3
        #mask = np.invert(th3)
        #print(mask)
        #cv2.imshow('', mask)
        #cv2.waitKey()
        #masked_img = np.full((img.shape[0], img.shape[1]),0,dtype="uint8")
        #masked_img = np.zeros((img.shape[0], img.shape[1]), dtype="uint8")

        #print('mask shape: '+str(mask.shape)+', img: '+str(masked_img.shape)+'\n')

        #print('mask_sum: ' + str(np.sum(mask))+'\n')

        #for r,row in enumerate(range(bbox[i][1], min([bbox[i][3],bbox.shape[0]]))):
            #for c,col in enumarate(range(bbox[i][0], min([bbox[i][2],bbox.shape[1]]))):
                #masked_img[row, col] = mask[r, c]
                #masked_img[row, col] = 255
        #print("masked_bg.shape",masked_bg.shape)
        #print("mask",mask.shape)
        masked_bg[bbox[i][1]:bbox[i][3],bbox[i][0]:bbox[i][2]] = mask
        masked_img=cv2.cvtColor(masked_bg,cv2.COLOR_GRAY2RGB)
        #print('masked_img_sum: ' + str(np.sum(masked_img)) + '\n')
        #cv2.imshow('', masked_img)
        #cv2.waitKey()
        #masked_img_final=np.hstack([masked_img_final,masked_img])

        #print('masked_img',masked_img)
        #masked_img_final.append(masked_img)
        #print('masked_img_final',masked_img_final)
    #masked_img_final=sum(masked_img_final)
    return masked_img