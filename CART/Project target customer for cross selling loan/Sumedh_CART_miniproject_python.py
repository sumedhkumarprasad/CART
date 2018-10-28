# -*- coding: utf-8 -*-
"""
Created on Sun Jun 17 08:14:33 2018

@author: sumedh
"""
import os
import pandas as pd
import numpy as np
## 	Split data in Training (Development Sample) 70% and Hold-out Sample 30%
# Set working directory
os.chdir("D:/Great Lakes PGPDSE/Great Lakes/10 Supervised Learning - Classification/Supervised Learning Classification/Mini_Project")
#Loading Dataset
pl = pd.read_csv("PL_XSELL.csv")

pl['split'] = np.random.randn(pl.shape[0], 1)
br = np.random.rand(len(pl)) <= 0.7
train = pl[br]
test = pl[~br]

# Writing Development_Sample and Hold_Out_Sample into two different csv files

train.to_csv("Development_Sample.csv", encoding='utf-8', index=False)
test.to_csv("Hold_out_Sample .csv", encoding='utf-8', index=False)

## Classification Tree on Unbalanced Dataset
import os
import pandas as pd

#Set the working directory
os.chdir("D:/Great Lakes PGPDSE/Great Lakes/10 Supervised Learning - Classification/Supervised Learning Classification/Mini_Project")

#Load the Dataset
CTDF_dev = pd.read_csv("Development_Sample.csv")
CTDF_holdout = pd.read_csv("Hold_out_Sample .csv")

del CTDF_dev['split']
del CTDF_holdout['split']

print( len(CTDF_dev),  len(CTDF_holdout))         ###13993 6007
CTDF_dev.head()

import numpy as np
import matplotlib.pyplot as plt

#Data Preprocessing
#Splitting into features and response variables

CTDF_dev.columns
X =  CTDF_dev[['GENDER', 'BALANCE', 'OCCUPATION',
       'AGE_BKT', 'SCR', 'HOLDING_PERIOD', 'ACC_TYPE',
       'LEN_OF_RLTN_IN_MNTH', 'NO_OF_L_CR_TXNS', 'NO_OF_L_DR_TXNS',
       'TOT_NO_OF_L_TXNS','FLG_HAS_CC','AMT_L_DR', 'FLG_HAS_ANY_CHGS',
       'AMT_MIN_BAL_NMC_CHGS','FLG_HAS_OLD_LOAN']]

#Categorical Variable to Numerical Variables
X_train = pd.get_dummies(X)
X_train.columns

y_train = CTDF_dev["TARGET"]

print (type(X_train) , type(y_train))

#Decision Tree
#Loading the library
from sklearn.tree import DecisionTreeClassifier

#Setting the parameter
clf = DecisionTreeClassifier(criterion = "gini" , 
                             min_samples_split = 100,
                             min_samples_leaf = 10,
                             max_depth = 100)

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
graph[0].write_pdf("D:/Great Lakes PGPDSE/Great Lakes/10 Supervised Learning - Classification/Supervised Learning Classification/Mini_Project.classification_tree_output.pdf")

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
score         ##0.8891588651468592

y_train_prob = clf.predict_proba(X_train)
## AUC for training     
fpr, tpr, thresholds = roc_curve(y_train, y_train_prob[:,1])
auc(fpr, tpr)           ## 0.8909147836850357

## Let us see how good is the model
X_holdout =  CTDF_holdout[['GENDER', 'BALANCE', 'OCCUPATION',
       'AGE_BKT', 'SCR', 'HOLDING_PERIOD', 'ACC_TYPE',
       'LEN_OF_RLTN_IN_MNTH', 'NO_OF_L_CR_TXNS', 'NO_OF_L_DR_TXNS',
       'TOT_NO_OF_L_TXNS','FLG_HAS_CC','AMT_L_DR', 'FLG_HAS_ANY_CHGS',
       'AMT_MIN_BAL_NMC_CHGS','FLG_HAS_OLD_LOAN']]
X_test = pd.get_dummies(X_holdout)
y_test = CTDF_holdout["TARGET"]

pred_y_test = clf.predict(X_test)
score_h = accuracy_score(y_test, pred_y_test)
score_h               ##  0.874812718495089

y_test_prob = clf.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob[:,1])
auc(fpr, tpr)        ## 0.7990326076444083

# AUC_diff = (0.8909147836850357-0.7990326076444083)/0.8909147836850357
# AUC_diff = 0.1031323957388841

y_freq = np.bincount(y_train)
y_val = np.nonzero(y_freq)[0]
np.vstack((y_val,y_freq[y_val])).T

#array([[    0, 12251],
#       [    1,  1742]], dtype=int64) 1742/(12251+1742)=12.44

#Cross validation function
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(clf, X_train , y_train, cv = 10, scoring='roc_auc')
scores.mean()       # 0.7879884317540934
scores.std()        # 0.0147261362940826

y_train_prob = clf.predict_proba(X_train)
fpr, tpr, thresholds = roc_curve(y_train, y_train_prob[:,1])
auc(fpr, tpr)       # 0.8909147836850357


y_test_prob = clf.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob[:,1])
auc(fpr, tpr)       #  0.7990326076444083

## Tuning the Classifier using GridSearchCV
from sklearn.grid_search import GridSearchCV
#help(GridSearchCV)

param_dist = {"criterion": ["gini","entropy"],
              "max_depth": np.arange(3,10),
              }

tree = DecisionTreeClassifier(min_samples_split = 100,
                             min_samples_leaf = 10)

tree_cv  = GridSearchCV(tree, param_dist, cv = 10, 
                        scoring = 'roc_auc', verbose = 100)

tree_cv.fit(X_train,y_train)

## Building the model using best combination of parameters
print("Tuned Decision Tree parameter : {}".format(tree_cv.best_params_))
# Tuned Decision Tree parameter : {'criterion': 'entropy', 'max_depth': 9}
classifier = tree_cv.best_estimator_

classifier.fit(X_train,y_train)

#predicting probabilities
y_train_prob = classifier.predict_proba(X_train)
fpr, tpr, thresholds = roc_curve(y_train, y_train_prob[:,1])
auc_prob_d = auc(fpr, tpr)
auc_prob_d     # 0.8208470013132318

y_test_prob = classifier.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob[:,1])
auc_prob_h = auc(fpr, tpr)
auc_prob_h   # 0.7508230646573208

#(auc_prob_d-auc_prob_h)/auc_prob_d # 0.0853069287502826

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
        np.size(x[Target][x[Target]==0]),
        ],
        index=(["min_resp","max_resp","avg_resp",
                "cnt","cnt_resp","cnt_non_resp"])
        )).reset_index()
    Rank = Rank.sort_values(by='decile',ascending=False)
    Rank["rrate"] = Rank["cnt_resp"]*100/Rank["cnt"]
    Rank["cum_resp"] = np.cumsum(Rank["cnt_resp"])
    Rank["cum_non_resp"] = np.cumsum(Rank["cnt_non_resp"])
    Rank["cum_resp_pct"] = Rank["cum_resp"]/np.sum(Rank["cnt_resp"])
    Rank["cum_non_resp_pct"]=Rank["cum_non_resp"]/np.sum(Rank["cnt_non_resp"])
    Rank["KS"] = Rank["cum_resp_pct"] - Rank["cum_non_resp_pct"]
    Rank
    return(Rank)

Rank = Rank_Ordering(CTDF_dev,"prob_score","TARGET")
Rank           # Considere the highest KS value among all available values 0.461771 

## Let us see the Rank Ordering on Hold-Out Dataset
Prediction_h = classifier.predict_proba(X_test)
CTDF_holdout["prob_score"] = Prediction_h[:,1]

Rank_h = Rank_Ordering(CTDF_holdout,"prob_score","TARGET")
Rank_h        # On Holding dataset highest KS value to be considered i.e.0.362128

#Rank = (Rank - Rank_h) / Rank
# (0.461771-0.362128)/0.461771 = 0.2157844472693174591
y_freq = np.bincount(y_test)
y_val = np.nonzero(y_freq)[0]
np.vstack((y_val,y_freq[y_val])).T
# array([[   0, 5237],
#      [   1,  770]], dtype=int64)  770/(770+5237) = 12.8183