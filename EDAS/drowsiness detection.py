import cv2
import os
from keras.models import load_model
import numpy as np
from pygame import mixer
import time
import dlib
from imutils import face_utils

mixer.init()
sound = mixer.Sound('alarm.wav')

#for Eye Blink
face = cv2.CascadeClassifier('haar cascade files\haarcascade_frontalface_alt.xml')
leye = cv2.CascadeClassifier('haar cascade files\haarcascade_lefteye_2splits.xml')
reye = cv2.CascadeClassifier('haar cascade files\haarcascade_righteye_2splits.xml')

#for Yawn
detector = cv2.CascadeClassifier("haar cascade files\haarcascade_frontalface_alt.xml")
predictor = dlib.shape_predictor('models/shape_predictor_68_face_landmarks.dat')


lbl=['Close','Open']
model = load_model('models/cnncat2.h5')
#path = os.getcwd()
cap = cv2.VideoCapture("2.mp4")

codec = cv2.VideoWriter_fourcc(*'XVID')
video_fps =int(cap.get(cv2.CAP_PROP_FPS))
video_width,video_height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter('results.avi', codec, video_fps, (video_width, video_height))


font = cv2.FONT_HERSHEY_COMPLEX_SMALL
score=0
thicc=2
rpred=[99]
lpred=[99]
YAWN_THRESH = 30

while(True):
    ret, frame = cap.read()
    height,width = frame.shape[:2] 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face.detectMultiScale(gray,minNeighbors=5,scaleFactor=1.1,minSize=(25,25))
    left_eye = leye.detectMultiScale(gray)
    right_eye =  reye.detectMultiScale(gray)
    cv2.rectangle(frame, (0,height-50) , (200,height) , (0,0,0) , thickness=cv2.FILLED )

    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y) , (x+w,y+h) , (100,100,100) , 1 )

    for (x,y,w,h) in right_eye:
        r_eye=frame[y:y+h,x:x+w]
        r_eye = cv2.cvtColor(r_eye,cv2.COLOR_BGR2GRAY)
        r_eye = cv2.resize(r_eye,(24,24))
        r_eye= r_eye/255
        r_eye=  r_eye.reshape(24,24,-1)
        r_eye = np.expand_dims(r_eye,axis=0)
        rpred = model.predict_classes(r_eye)
        if(rpred[0]==1):
            lbl='Open' 
        if(rpred[0]==0):
            lbl='Closed'
        break

    for (x,y,w,h) in left_eye:
        l_eye=frame[y:y+h,x:x+w]
        l_eye = cv2.cvtColor(l_eye,cv2.COLOR_BGR2GRAY)  
        l_eye = cv2.resize(l_eye,(24,24))
        l_eye= l_eye/255
        l_eye=l_eye.reshape(24,24,-1)
        l_eye = np.expand_dims(l_eye,axis=0)
        lpred = model.predict_classes(l_eye)
        if(lpred[0]==1):
            lbl='Open'   
        if(lpred[0]==0):
            lbl='Closed'
        break

    if(rpred[0]==0 and lpred[0]==0):
        score=score+1
        cv2.putText(frame,"Closed",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    # if(rpred[0]==1 or lpred[0]==1):
    else:
        score=score-1
        cv2.putText(frame,"Open",(10,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    
        
    if(score<0):
        score=0   
    cv2.putText(frame,'Score:'+str(score),(100,height-20), font, 1,(255,255,255),1,cv2.LINE_AA)
    if(score>8):
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
        cv2.putText(frame, "Drowsiness Alert", (50, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.rectangle(frame,(0,0),(width,height),(0,0,255),thicc)


    rects = detector.detectMultiScale(gray, scaleFactor=1.1,minNeighbors=5, minSize=(30, 30),flags=cv2.CASCADE_SCALE_IMAGE)
    for (x, y, w, h) in rects:

        rect = dlib.rectangle(int(x), int(y), int(x + w), int(y + h))
        shape1 = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape1)
        top_lip = shape[50:53]
        top_lip = np.concatenate((top_lip, shape[61:64]))
        low_lip = shape[56:59]
        low_lip = np.concatenate((low_lip, shape[65:68]))
        top_mean = np.mean(top_lip, axis=0)
        low_mean = np.mean(low_lip, axis=0)
        distance = abs(top_mean[1] - low_mean[1])
        score_lip = distance + 10
        lip = shape[49:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        if (score_lip > YAWN_THRESH):
            cv2.putText(frame, "Yawn Alert", (50, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if (thicc < 16):
                thicc = thicc + 2
            else:
                thicc = thicc - 2
                if (thicc < 2):
                    thicc = 2
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 255), thicc)

    cv2.imshow('frame',frame)
    out.write(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
out.release()
cv2.destroyAllWindows()