# -*- coding: utf-8 -*-
"""
@author: arjun
"""
# *********************************************************************************************************************
# Finding the association between people leaving a bank using neural networks
# ********************************************************************************************************************
# Data Preprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('Churn_Modelling.csv')

X = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

# Categorical Features
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
lblEncoder_country = LabelEncoder()
X[:,1] = lblEncoder_country.fit_transform(X[:,1])
lblEncoder_gender = LabelEncoder()
X[:,2] = lblEncoder_gender.fit_transform(X[:,2])

ohe_country = OneHotEncoder(categorical_features=[1])
X = ohe_country.fit_transform(X).toarray()

# Removing one variable to avoid dummy variable trap
X = X[:,1:]

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X = sc_X.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=0)


# Importing Keras library and packages
import keras

# Importing the 2 modules, sequential module to initialize the neural network and the dense module that is used to build the layers of our neural network
from keras.models import Sequential
from keras.layers import Dense

# Initializing the ANN
neuralClassifier = Sequential()

# Building the hidden layers
neuralClassifier.add(Dense(6, kernel_initializer='uniform', activation='relu', input_dim = 11))
neuralClassifier.add(Dense(6, kernel_initializer='uniform', activation='relu'))

# Building the output layer
neuralClassifier.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))

# Compiling the ANN, applying stochastic gradient descent on ANN
neuralClassifier.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

# Fitting the ANN to the training set
neuralClassifier.fit(X_train,y_train,batch_size=10, epochs=100)

# Predicting result
# This gives the probability
y_pred_probability = neuralClassifier.predict(X_test)
# Converting the probabilities to 1/0, by taking a threshold of 0.5
y_pred = (y_pred_probability > 0.5)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)