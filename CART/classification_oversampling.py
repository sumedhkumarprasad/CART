# -*- coding: utf-8 -*-
"""
Created on Thu Jun 14 22:22:38 2018

@author: sumedh
"""

import os
import pandas as pd

# Set working directory


#Load the Dataset

CTDF_dev = pd.read_csv("DEV_SAMPLE.csv")

CTDF_holdout = pd.read_csv("HOLDOUT_SAMPLE.csv")

print( len(CTDF_dev),  len(CTDF_holdout))

CTDF_dev.head()

import numpy as np

import matplotlib.pyplot as plt

from collections import Counter

from sklearn.datasets import make_classification

from imblearn.over_sampling import RandomOverSampler

X =  CTDF_dev[['Age', 'Gender', 'Balance', 'Occupation','No_OF_CR_TXNS', 'AGE_BKT', 'SCR', 'Holding_Period']]

#Categorical Variable to Numerical Variables

X_train = pd.get_dummies(X)

X_train.columns

y=CTDF_dev["Target"]

X,y = make_classification(n_classes=2, class_sep=2,
                           weights=[0.9, 0.1], n_informative=3, n_redundant=1, flip_y=0,
                            n_features=20, n_clusters_per_class=1, n_samples=14000, random_state=20)

print('Original dataset shape {}'.format(Counter(y)))

y_train = y

CTDF_dev["Target"].value_counts()

print (type(X_train) , type(y_train))

 

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
graph[0].write_pdf("D:/Great Lakes PGPDSE/Great Lakes/10 Supervised Learning - Classification/Supervised Learning Classification/CART/claasification_oversampling_0.9.pdf")

Nodes = pd.DataFrame(clf.tree_.__getstate__()["nodes"])

Nodes

feature_importance = pd.DataFrame([X_train.columns,

                               clf.tree_.compute_feature_importances()])

feature_importance.T

## Let us see how good is the model

pred_y_train = clf.predict(X_train )

pred_y_train

## Let us see the classification accuracy of our model

from sklearn.metrics import accuracy_score

from sklearn.metrics import roc_curve

from sklearn.metrics import auc

score = accuracy_score(y_train, pred_y_train)

score

y_train_prob = clf.predict_proba(X_train)

## AUC

fpr, tpr, thresholds = roc_curve(y_train, y_train_prob[:,1])

auc(fpr, tpr)

## Let us see how good is the model

X_holdout =  CTDF_holdout[['Age', 'Gender', 'Balance', 'Occupation',

               'No_OF_CR_TXNS', 'AGE_BKT', 'SCR', 'Holding_Period']]

X_test = pd.get_dummies(X_holdout)

y_test = CTDF_holdout["Target"]

pred_y_test = clf.predict(X_test)

score_h = accuracy_score(y_test, pred_y_test)

score_h

 

y_test_prob = clf.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_test_prob[:,1])

auc(fpr, tpr)


#Cross validation function

from sklearn.cross_validation import cross_val_score

scores = cross_val_score(clf, X_train , y_train, cv = 10, scoring='roc_auc')

scores.mean()

scores.std()

y_train_prob = clf.predict_proba(X_train)

fpr, tpr, thresholds = roc_curve(y_train, y_train_prob[:,1])

auc(fpr, tpr)

y_test_prob = clf.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_test_prob[:,1])

auc(fpr, tpr)

## Tuning the Classifier using GridSearchCV

from sklearn.grid_search import GridSearchCV

help(GridSearchCV)

 

param_dist = {"criterion": ["gini","entropy"],

              "max_depth": np.arange(3,10),

              }

tree = DecisionTreeClassifier(min_samples_split = 100,min_samples_leaf = 10)


tree_cv  = GridSearchCV(tree, param_dist, cv = 10,

                        scoring = 'roc_auc', verbose = 100)

 

tree_cv.fit(X_train,y_train)

## Building the model using best combination of parameters

print("Tuned Decision Tree parameter : {}".format(tree_cv.best_params_))

classifier = tree_cv.best_estimator_

classifier.fit(X_train,y_train)

#predicting probabilities

y_train_prob = classifier.predict_proba(X_train)

fpr, tpr, thresholds = roc_curve(y_train, y_train_prob[:,1])

auc_d = auc(fpr, tpr)

auc_d

y_test_prob = classifier.predict_proba(X_test)

fpr, tpr, thresholds = roc_curve(y_test, y_test_prob[:,1])

auc_h = auc(fpr, tpr)

auc_h

## Rank Ordering

Prediction = classifier.predict_proba(X_train)

CTDF_dev["prob_score"] = Prediction[:,1]

 

#scoring step

#decile code

def deciles(x):

    decile = pd.Series(index=[0,1,2,3,4,5,6,7,8,9])

    for i in np.arange(0.1,1.1,0.1):

        decile[int(i*10)]=x.quantile(i)

    def z(x):

        if x<decile[1]: return(1)

        elif x<decile[2]: return(2)

        elif x<decile[3]: return(3)

        elif x<decile[4]: return(4)

        elif x<decile[5]: return(5)

        elif x<decile[6]: return(6)

        elif x<decile[7]: return(7)

        elif x<decile[8]: return(8)

        elif x<decile[9]: return(9)

        elif x<=decile[10]: return(10)

        else:return(np.NaN)

    s=x.map(z)

    return(s)
    
def Rank_Ordering(X,y,Target):
    X['decile']=deciles(X[y])
    Rank=X.groupby('decile').apply(lambda x: pd.Series([
        np.min(x[y]),
        np.max(x[y]),
        np.mean(x[y]),
        np.size(x[y]),
        np.sum(x[Target]),
        np.size(x[Target][x[Target]==0])]))
    index=([["min_resp","max_resp","avg_resp","cnt","cnt_resp","cnt_non_resp"]]).reset_index()

    Rank = Rank.sort_values(by='decile',ascending=False)

    Rank["rrate"] = Rank["cnt_resp"]*100/Rank["cnt"]

    Rank["cum_resp"] = np.cumsum(Rank["cnt_resp"])

    Rank["cum_non_resp"] = np.cumsum(Rank["cnt_non_resp"])

    Rank["cum_resp_pct"] = Rank["cum_resp"]/np.sum(Rank["cnt_resp"])

    Rank["cum_non_resp_pct"]=Rank["cum_non_resp"]/np.sum(Rank["cnt_non_resp"])

    Rank["KS"] = Rank["cum_resp_pct"] - Rank["cum_non_resp_pct"]
    Rank
    return(Rank)
    
Rank=Rank_Ordering(CTDF_dev,"prob_score","Target")

Rank

## Let us see the Rank Ordering on Hold-Out Dataset

Prediction_h = classifier.predict_proba(X_test)

CTDF_holdout["prob_score"] = Prediction_h[:,1]

Rank_h = Rank_Ordering(CTDF_holdout,"prob_score","Target")

Rank_h

y_freq = np.bincount(y_test)

y_val = np.nonzero(y_freq)[0]

np.vstack((y_val,y_freq[y_val])).T

