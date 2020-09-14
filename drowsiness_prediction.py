# -*- coding: utf-8 -*-
"""
Created on Sat May  2 05:44:41 2020

@author: karth
"""

import cv2
import os
import numpy as np
from pygame import mixer
import time
from keras.models import model_from_json
import imutils
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from PIL import Image




mixer.init()
sound = mixer.Sound('alarm.wav')

face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')
detection_model_path = 'haar cascade files\haarcascade_frontalface_default.xml'
emotion_model_path = 'models/model2.hdf5'
INPUT_PATH_TEST = "./dataset/test/"
heart_model_path = "models/model3.h5"
eye_model_path =  'models/model1.h5'


face_detection = cv2.CascadeClassifier(detection_model_path)
EMOTIONS = ["angry" ,"disgust","scared", "happy", "sad", "surprised","neutral"]
WIDTH, HEIGHT = 256, 256        # Size images to train
CLASS_COUNTING = True           # Test class per class and show details each 
BATCH_SIZE = 32                 # How many images at the same time, change depending on your GPU
CLASSES = ['Infrant', 'None']   # Classes to detect. they most be in same position with output vector

# def loadModel(model_path, weight_path):
#     json_file = open(model_path, 'r')
#     loaded_model_json = json_file.read()
#     json_file.close()
#     model = model_from_json(loaded_model_json)
#     # load weights into new model
#     model.load_weights(weight_path)
#     # evaluate loaded model on test data
#     model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#     return model

# model = loadModel('models/model1.json', "models/model1.h5")

emotion_model = load_model(emotion_model_path, compile=False)
model = load_model(eye_model_path)
heart_model = load_model(heart_model_path)

# cv2.namedWindow('your_face')
path = os.getcwd()
cap = cv2.VideoCapture(0)
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
count=0
score=0
fscore=0
escore=0
thicc=2
rpred=[99]
lpred=[99]
c=0


while(True):
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    height,width = frame.shape[:2] 

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    faces = face_detection.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=5,minSize=(30,30),flags=cv2.CASCADE_SCALE_IMAGE)
    
    canvas = np.zeros((250, 300, 3), dtype="uint8")
    frameClone = frame.copy()
    
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)
    
    frame_c = frame.copy()
    x = cv2.resize(frame_c, (WIDTH, HEIGHT))
    x = img_to_array(x)
    x = np.expand_dims(x, axis=0)
    array = heart_model.predict(x)
    result = array[0]
    answer = np.argmax(result)

    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        count=count+1
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict_classes(r_eye)
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        count=count+1
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        break
    
    if(CLASSES[answer]=='Infrant'):
        c+=1
    else:
        c=0
        
    cv2.putText(frame,'body_score:'+str(c),(10,75), font, 1,(236,216,31),1,cv2.LINE_AA)
    
    if len(faces) > 0:
        faces = sorted(faces, reverse=True,
        key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
        (fX, fY, fW, fH) = faces

        roi = gray[fY:fY + fH, fX:fX + fW]
        roi = cv2.resize(roi, (64, 64))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)
        
        
        preds = emotion_model.predict(roi)[0]
        emotion_probability = np.max(preds)
        label = EMOTIONS[preds.argmax()]
        cv2.putText(frame,label,(500,40), font, 1,(0,0,255),1,cv2.LINE_AA)
        if label=="scared" or label=="disgust":
            escore+=2
        else:
            if escore>0:
                escore-=1
            else:
                escore=0
    cv2.putText(frame,'Emotion_Score:'+str(escore),(10,100), font, 1,(236,216,31),1,cv2.LINE_AA)
    
    if len(faces)==0:
        fscore+=1
        cv2.putText(frame,"Cannot find face",(400,20), font, 1,(0,0,255),1,cv2.LINE_AA)
    elif(rpred[0]==0 and lpred[0]==0):
        fscore-=1
        score=score+1
        cv2.putText(frame,"Closed",(500,20), font, 1,(0,0,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        fscore-=1
        score=score-1
        cv2.putText(frame,"Open",(500,20), font, 1,(0,0,255),1,cv2.LINE_AA)
     
    if score<0:
        score=0
    if fscore<0:
        fscore=0
    cv2.putText(frame,'Eye_Score:'+str(score),(10,25), font, 1,(236,216,31),1,cv2.LINE_AA)
    cv2.putText(frame,'Face_Score:'+str(fscore),(10,50), font, 1,(236,216,31),1,cv2.LINE_AA)
    
    if (score>15 or fscore>20 or escore>3 or (c>15 and (len(faces)==0 or (rpred==0 and lpred==0)))):
        #person is feeling sleepy so we beep the alarm
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
        cv2.rectangle(frame,(0,0),(width,height),(21,240,80),thicc) 
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
