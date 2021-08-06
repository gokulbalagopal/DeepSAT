# -*- coding: utf-8 -*-
"""
Created on Mon Jun 14 21:37:40 2021

@author: balag
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.io import imshow
import scipy

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
reshaped_x_train_new = reshaped_x_train/255.0
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
df_y_train['Labels'] = "NA"
for ix in range(0,len(df_y_train)):
    if(df_y_train.iloc[ix,0] == 1):
        df_y_train.iloc[ix,4] = "Barren Land"
    elif(df_y_train.iloc[ix,1] == 1):
        df_y_train.iloc[ix,4] = "Trees"
    elif(df_y_train.iloc[ix,2] == 1):
        df_y_train.iloc[ix,4] = "Grassland"
    else:
        df_y_train.iloc[ix,4] = "None"

        #    break
svd_feat = []
df_y_train = df_y_train['Labels']
for i in range(0,len(reshaped_x_train_grey)):
    temp = np.asmatrix(reshaped_x_train_grey[i,:,:,:])
    svd_feat.append(scipy.linalg.svdvals(temp))

svd_arr = np.array(svd_feat)
svd_arr.shape

plt.plot(range(0,28),svd_feat[1],"-ok")
plt.xlabel('Number ')
plt.ylabel('Features')
plt.title('Number vs features')

r_mean = []
g_mean = []
b_mean = []

for i in range(0,len(reshaped_x_train_new)):
    r_mean.append(np.mean(reshaped_x_train_new[i,:,:,0]))
    g_mean.append(np.mean(reshaped_x_train_new[i,:,:,1]))
    b_mean.append(np.mean(reshaped_x_train_new[i,:,:,2]))
    
df_means = pd.DataFrame(list(zip(r_mean,g_mean,b_mean)),columns =['R Mean', 'G Mean','B Mean'])
svd_feat_np  = np.asarray(svd_feat)
df_svd_feat = pd.DataFrame(svd_feat_np)

df_X_train = pd.concat([df_svd_feat, df_means], axis=1)


######################### Logistic Regression Classifier ######################
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(df_X_train, df_y_train)
print('Accuracy of Logistic regression classifier on training set: {:.2f}'.format(logreg.score(df_X_train, df_y_train)))
#print('Accuracy of Logistic regression classifier on test set: {:.2f}'
#     .format(logreg.score(X_test, y_test)))



######################### Decision Tree Classifier ######################
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier().fit(df_X_train, df_y_train)
print('Accuracy of Decision Tree classifier on training set: {:.2f}'
     .format(clf.score(df_X_train, df_y_train)))
#print('Accuracy of Decision Tree classifier on test set: {:.2f}'
#    .format(clf.score(X_test, y_test)))


########################## K nearest Neighbours #########################

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(df_X_train, df_y_train)
print('Accuracy of K-NN classifier on training set: {:.2f}'
     .format(knn.score(df_X_train, df_y_train)))
#print('Accuracy of K-NN classifier on test set: {:.2f}'
#     .format(knn.score(X_test, y_test)))



###################### Linear Discriminant Analysis ######################

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis()
lda.fit(df_X_train, df_y_train)
print('Accuracy of LDA classifier on training set: {:.2f}'
     .format(lda.score(df_X_train, df_y_train)))
#print('Accuracy of LDA classifier on test set: {:.2f}'
#     .format(lda.score(X_test, y_test)))


#################### Guassian Naive Bayes Classifier ###########################
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(df_X_train, df_y_train)
print('Accuracy of GNB classifier on training set: {:.2f}'
     .format(gnb.score(df_X_train, df_y_train)))
#print('Accuracy of GNB classifier on test set: {:.2f}'
#     .format(gnb.score(X_test, y_test)))

######################## SVC ################################################
from sklearn.svm import SVC
svm = SVC()
svm.fit(df_X_train, df_y_train)
print('Accuracy of SVM classifier on training set: {:.2f}'
     .format(svm.score(df_X_train, df_y_train)))
#print('Accuracy of SVM classifier on test set: {:.2f}'
#     .format(svm.score(X_test, y_test)))

######################### Random Forest Classifier #########################################

from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators = 50)
rfc.fit(df_X_train, df_y_train)
print('Accuracy of RFC classifier on training set: {:.2f}'
     .format(rfc.score(df_X_train, df_y_train)))
#y_pred = rfc.predict(X_test)

######################## Confusion Matrix ########################################
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
pred = clf.predict(X_test)
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))