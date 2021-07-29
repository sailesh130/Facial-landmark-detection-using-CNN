#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 10 17:13:29 2020

@author: sailesh
"""

import cv2
from detector import Detector

cap = cv2.VideoCapture(0)
dl = Detector(model='./models/model1.pkl',is_pickle= True)

while(True):
    
    try:
        # Capture frame-by-frame
        ret, frame = cap.read()
    
        frame = dl.detect_landmark(frame)
        # Display the resulting frame
        cv2.imshow('frame',frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
