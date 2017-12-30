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