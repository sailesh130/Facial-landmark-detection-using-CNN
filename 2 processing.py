#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 19:10:30 2020
@author: sailesh
"""

# working with mainly reading, convert to grayscale, resizing images
import cv2
# dealing with arrays                  
import numpy as np  
import pandas as pd
# dealing with directories         
import os  
from tqdm import tqdm

class CreateData:
    '''
    This class take .csv landmark files from landmark directory,
    extract the names from csv to read images,
    crop region of interest and reshape it into given dimensions,
    and rescale landmark points,
    save image data and landmark.
    '''
    
    def __init__(self, image_dir,landmark_dir, dimensions):
        
        self.image_dir = image_dir
        self.landmark_dir = landmark_dir
        self.dimensions= dimensions
        self.train =[]
        self.land =[]
        
    def traindata(self):
        
        landmark = pd.read_csv(self.landmark_dir)
        shape = landmark.shape
        
        for f in tqdm(range(shape[0])):
            
            try:           
                data = landmark.iloc[f,:].values
                image_name = data[0]
            
                path = os.path.join(self.image_dir, image_name)
                
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                
                box = data[137:]    # 4
                data = data[1:137] #136
                img = img[box[1]:box[3],box[0]:box[2]]
                h,w = img.shape
                
                img = cv2.resize(img, self.dimensions)
                
                data[::2] = (data[::2] - box[0]) / (w/self.dimensions[0])
                data[1::2] = (data[1::2] - box[1]) / (h/self.dimensions[1])
                
                data = np.where( data > -2 , data,-2)
                data = np.where( data < 98 , data,98)

                self.train.append(img)
                self.land.append(data)
            
            except:
                pass
    
        self.train = np.array(self.train,dtype='int8')
        self.land = np.array(self.land,dtype='int8')
        
        self.save_data()
        
    def save_data(self):
                
        train_name = '/home/rjn/Music/train_name.npy'
        land_name = '/home/rjn/Music/land_name.npy'
        np.save(train_name, self.train)
        np.save(land_name, self.land)
        
    def get_train(self):
        
        return self.train
    
    def get_land(self):
        
        return self.land 

if __name__ == '__main__':
    
    data = CreateData(image_dir='/home/rjnp2/Music/img_align_celeba', 
                      landmark_dir = '/home/rjnp2/Music/combine.csv', 
                      dimensions= (96,96))
    
    data.traindata()
