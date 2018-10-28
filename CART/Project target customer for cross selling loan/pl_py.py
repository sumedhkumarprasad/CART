# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 08:14:33 2018

@author: sumedh
"""

import os
import pandas as pd

# Set working directory
os.chdir("D:/Great Lakes PGPDSE/Great Lakes/10 Supervised Learning - Classification/Supervised Learning Classification/Mini_Project")
#Loading Dataset
pl = pd.read_csv("PL_XSELL.csv")

pl.isnull().sum() # There is No Missing value in the dataset
pl.drop(pl.columns[[0,2,9,10,12,13,14,15,16,17,18,19,21,22,23,24,25,28,29,30,31,32,33,34,35,36,37,39]],axis=1,inplace=True)

import numpy as np
import matplotlib.pyplot as plt

X= pl[["GENDER","BALANCE","OCCUPATION","AGE_BKT","SCR","HOLDING_PERIOD","LEN_OF_RLTN_IN_MNTH","FLG_HAS_CC","AMT_L_DR",
       "FLG_HAS_ANY_CHGS","FLG_HAS_OLD_LOAN"]]
y=pl["TARGET"]
#Categorical Variable to Numerical Variables
X_train = pd.get_dummies(X)
X_train.columns



# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)
#Decision Tree
#Loading the library
from sklearn.tree import DecisionTreeClassifier

#Setting the parameter
clf = DecisionTreeClassifier(criterion = "gini" , 
                             min_samples_split = 100,
                             min_samples_leaf = 10,
                             max_depth = 50)

#Calling the fit function to built the tree
clf.fit(X_train,y_train)

import pydot
from sklearn.tree import tree
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO 
dot_data = StringIO()
feature_list = list(X_train.columns.values)
export_graphviz(clf, 
                out_file = dot_data, 
                feature_names = feature_list)
graph=pydot.graph_from_dot_data(dot_data.getvalue())
graph[0].write_pdf("D:/Great Lakes PGPDSE/Great Lakes/10 Supervised Learning - Classification/Supervised Learning Classification/Mini_Project.pl_py.pdf")
# Nodes 
Nodes = pd.DataFrame(clf.tree_.__getstate__()["nodes"])
Nodes

feature_importance = pd.DataFrame([X_train.columns,
                               clf.tree_.compute_feature_importances()])
feature_importance.T

## Let us see how good is the model
pred_y_train = clf.predict(X_train )
pred_y_train