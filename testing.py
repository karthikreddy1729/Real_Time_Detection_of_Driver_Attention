# -*- coding: utf-8 -*-
"""
Created on Sat Jul 25 13:33:36 2020

@author: karth
"""


import cv2
import os
import numpy as np
from pygame import mixer
import time
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image



INPUT_PATH_TEST = "./dataset/test/"
MODEL_PATH = "models/model3.h5"    # Full path of model


WIDTH, HEIGHT = 256, 256        # Size images to train
CLASS_COUNTING = True           # Test class per class and show details each 
BATCH_SIZE = 32                 # How many images at the same time, change depending on your GPU
CLASSES = ['00None', '01Infarct']   # Classes to detect. they most be in same position with output vector


heart_model = load_model(MODEL_PATH)


mixer.init()
sound = mixer.Sound('alarm.wav')


thicc=2
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
c=0

while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # x = frame.resize((256,256))
    x = cv2.resize(frame, (WIDTH, HEIGHT))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = heart_model.predict(x)
    result = array[0]
    answer = np.argmax(result)
    # cv2.putText(frame,'HEART_ATTACK',(300,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
    if(CLASSES[answer]=='01Infarct'):
        c+=1
    else:
        c=0
    cv2.putText(frame,'count : '+str(c),(300,height-20), font, 1,(0,0,255),1,cv2.LINE_AA)

    if(c>10):
        cv2.putText(frame,'HEART_ATTACK',(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
        cv2.imwrite(os.path.join(path,'image.jpg'),frame)
        try:
            sound.play()
            
        except:  # isplaying = False
            pass
        if(thicc<16):
            thicc= thicc+2
        else:
            thicc=thicc-2
            if(thicc<2):
                thicc=2
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
