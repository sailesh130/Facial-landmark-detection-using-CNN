# !/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 24 10:26:19 2020
@author: rjnp2
"""

import numpy as np
import pandas as pd

def bbox_csv(txt_dir , save_dir):
    
    """

    Parameters
    ----------
    txt_dir : string 
        location of txt file of boundary box of face of person.
    save_dir : string
        location of csv file to save of boundary box.

    Returns
    -------
    data : Dataframe
        csv file.

    """   
    
    data = np.loadtxt(txt_dir, dtype='str')
    col = ['image_id', 'x', 'y', 'width','height']
    
    data = pd.DataFrame(data, columns=col)
    data.iloc[:, 1:] = (data.iloc[:, 1:].values).astype('int16')
    data.to_csv(save_dir, index=False)
    return data
    
    
def landmark_csv(txt_dir , save_dir):
    
    """

    Parameters
    ----------
    txt_dir : string 
        location of txt file of landmark of face.
    save_dir : string
        location of csv file to save of landmark.

    Returns
    -------
    data : Dataframe
        csv file.

    """ 
    
    data = np.loadtxt(txt_dir, dtype='str')
    
    col = ['image_id',]
    for i in range(68):
        
        col.append('x_'+ str(i))
        col.append('y_'+ str(i))
    
    data = pd.DataFrame(data, columns=col)
    data.iloc[:, 1:] = (data.iloc[:, 1:].values).astype('int16')
    data.to_csv(save_dir, index=False)
    return data
    

def combine_csv(csv1 , csv2,save_dir):
    
    """

    Parameters
    ----------
    csv1 : DataFrame 
        csv file of vvox of face.
        
    csv2 : DataFrame 
        csv file of landmark of face.
    save_dir : string
        location of csv file to save of landmark.

    Returns
    -------
    data : Dataframe
        csv file.

    """ 
    
    data = pd.concat((csv2,csv1.iloc[:,1:]), axis=1)
    data.iloc[:, 1:] = (data.iloc[:, 1:].values).astype('int16')
    data.to_csv(save_dir, index=False)
    
if __name__ == '__main__':
    
    box = bbox_csv('/home/rjnp2/Music/bbox_align.txt',
                    save_dir='/home/rjnp2/Music/bbox.csv')
    landmark = landmark_csv('/home/rjnp2/Music/landmark_align.txt', 
                            '/home/rjnp2/Music/landmark.csv')
    
    combine_csv(landmark, box, '/home/rjnp2/Music/combine.csv')
