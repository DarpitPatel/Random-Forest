'''
__author__ = "Darpit Patel"
__version__ = "1.0.1"
__maintainer__ = "Darpit Patel"
__GitHub__ = https://github.com/DarpitPatel/
'''
import numpy as np
import pandas as pd 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle
from sklearn import tree

#read csv using pandas
data=pd.read_csv('input.csv')
#Select all row and second and third column as input of model
df_x=data.iloc[:,[1,2]]
df_x=df_x.astype('int')
#Select all row and first column as output of model
df_y=data.iloc[:,0]
df_y=df_y.astype('int')
x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.7, random_state=4)
            
clf = RandomForestClassifier()
trained_model = clf.fit(x_train, y_train)
                
predictions = trained_model.predict(x_test) 
#print accuracy of model               
print("Train Accuracy : ", accuracy_score(y_train, trained_model.predict(x_train)))
print("Test Accuracy  : ", accuracy_score(y_test, predictions))

#Save the model for persistence
filename="pickel_model"
with open(filename, 'wb') as fid:
    pickle.dump(clf, fid)

#Make point prediction for a single input
print("Point Prediction : ", trained_model.predict([[78,2]]) )


