# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 21:37:40 2021

@author: balag
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imshow

################################# Do this at the end #######################################
#x_train = pd.read_csv("X_train_sat4.csv", chunksize=100000, header= None, iterator = True)
#x_train_new = pd.concat(x_train, ignore_index=True)
############################################################################################
chunk_size=50000
batch_no=1
for chunk in pd.read_csv('X_train_sat4.csv',chunksize=chunk_size, header = None):
    chunk.to_csv('chunk_x_train_'+str(batch_no)+'.csv',index=False)
    batch_no+=1
    
df_x_train = pd.read_csv("chunk_x_train_1.csv")
df_x_train.head()
reshaped_x_train = df_x_train.values.reshape(-1, 28,28,4).astype(float)
reshaped_x_train_new = reshaped_x_train/255
x_train_grey = []
######################### Display images and convert them to gray scale  ################################
for i in range(0, len(reshaped_x_train_new)):
    #imshow(np.squeeze(reshaped_x_train_new[i,:,:,0:3]).astype(float))
    grey = np.dot(reshaped_x_train_new[i,:,:,0:3],[0.2989, 0.5870, 0.1140])
    x_train_grey.append(grey)
    #imshow(np.squeeze(grey))
    #plt.show()
########################### Reshaping the grey scale images ###################################
x_train_grey_np = np.asarray(x_train_grey)
reshaped_x_train_grey = x_train_grey_np.reshape(-1,28,28,1)



############### Loading Y train ################
batch_no=1
for chunk in pd.read_csv('y_train_sat4.csv',chunksize=chunk_size, header = None):
    chunk.to_csv('chunk_y_train_'+str(batch_no)+'.csv',index=False)
    batch_no+=1
df_y_train = pd.read_csv("chunk_y_train_1.csv")
df_y_train.head()

print(df_y_train.shape)
#    break
