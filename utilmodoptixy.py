import numpy as np
import pandas as pd
import cv2
from random import shuffle

df = pd.read_csv('training.csv')
c1 = 0
d1 = 0
for index, row in df.iterrows():
    path = './images/' + row['image_name']
    img = cv2.imread(path)
    img = cv2.resize(img, (224,224))
    img = cv2.flip(img, -1)
    edges = cv2.Canny(img,100,255)
    #print(edges[:,:,np.newaxis].shape)
    img = np.concatenate((img,edges[:,:,np.newaxis]), axis = 2)
    #print(img.shape)
    if c1 == 0:
        data_img = img[np.newaxis,:]
        c1 = 1
    else:
        img = img[np.newaxis,:]
        data_img = np.concatenate((data_img,img), axis = 0)
    labelx = np.subtract(1, np.divide([row['x1'], row['x2']],640))
    labely = np.subtract(1, np.divide([row['y1'], row['y2']],480))
    label = np.concatenate((labelx,labely), axis = None)
    label = np.multiply(label,1)
    #print(label)
    if d1 == 0:
        data_lab = label[np.newaxis,:]
        d1 = 1
    else:
        label = label[np.newaxis,:]
        data_lab = np.concatenate((data_lab,label), axis = 0)
    print(index)
    if (index+1)%1000 == 0:
        """ if index  == 12814:
            index = idxx + 1 """
        ax = 'data_imgtrain'+str(index+1+72000)+'.npy'
        al = 'data_lab'+str(index+1+72000)+'.npy'
        np.save(ax,data_img)
        np.save(al,data_lab)
        data_img = None
        data_lab = None
        c1 = 0
        d1 = 0
        idxx = index + 1
