import numpy as np
import pickle
from keras.models import load_model
import cv2

class Detector():
    
    def __init__(self,model,is_pickle= True, dimensions= (96, 96)):
        
        self.face_cascade= cv2.CascadeClassifier('./models/haarcascade_frontalface_default.xml')
        self.dimensions = dimensions
        
        if is_pickle:
            self.model = pickle.load(open(model, 'rb'))
            
        else:
            self.model = load_model(model)
 
    def plot_landmark(self, gray,lands):
        
        for n,land in enumerate(lands):
            
            if n < 17 :
                color = (0, 255, 0)
                
            elif n >= 17 and n < 27 :
                color= (255, 0, 0)
                
            elif n >= 27 and n < 36 :
               color= (0, 0, 255)
                
            elif n >= 36 and n < 48 :
                color= (255, 255, 0)
            
            elif n >= 48 and n < 68 :
                color= (255, 0, 255)
                
            else:
                color = (255,255,255)
                  
            cv2.circle(gray, (land[0],land[1]), 2, (color), -1)
    
        mask = np.zeros(gray.shape)    
        
        points = np.array(lands, np.int32)
        convexhull = cv2.convexHull(points)
        cv2.fillConvexPoly(mask, convexhull, 255)
    
        gray = cv2.addWeighted(gray, 0.8, mask.astype('uint8'),0.3, 0)
        
        return gray     
    
    def detect_landmark(self, frame,filter_code=0):
        
        try:   
            faces = self.face_cascade.detectMultiScale(frame, 1.25, 6) 
            
            x,y,w,h = faces[0]
            y += 55
            h -= 35
            x += 10
            w -= 18
    
            gray = frame[y:y+h, x:x+w]
            imgd = cv2.resize(gray,self.dimensions)
            imgd = cv2.medianBlur(imgd ,5)
            imgd = cv2.GaussianBlur(imgd,(5,5),0)
            imgd = cv2.cvtColor(imgd, cv2.COLOR_BGR2GRAY)
            
            imgd = imgd.reshape(-1,self.dimensions[0],self.dimensions[1],1)
            
            lands = self.model.predict(imgd)
    
            lands = ((lands.reshape(-1, 2) * 48) + 48).astype('int16')       
            lands = (lands * [(w/96),(h/96)]).astype('int16')
            
            if filter_code == 0:           
                gray = self.plot_landmark(gray.copy(),lands)      
                frame[y:y+h, x:x+w,] = gray
            
                return frame
            
            else:
                return lands + [x,y]
    
        except:
            return frame
    
    

