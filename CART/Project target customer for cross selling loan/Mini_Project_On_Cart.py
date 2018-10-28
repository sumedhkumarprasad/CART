# -*- coding: utf-8 -*-
"""
Created on Sat Jun 16 11:21:35 2018

@author: PRIYANKA
"""
### •	Split data in Training (Development Sample) 70% and Hold-out Sample 30%
import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/prianka bhoyar/Desktop/Data Science/Clustering/Mini Project/PL_XSELL.csv')
df['split'] = np.random.randn(df.shape[0], 1)
msk = np.random.rand(len(df)) <= 0.7
train = df[msk]
test = df[~msk]

##from numpy.random import RandomState
##rng = RandomState()
##train = df.sample(frac=0.7), random_state=rng)
##test = df.loc[~data.index.isin(train.index)]

train.to_csv("Development_Sample.csv", encoding='utf-8', index=False)
test.to_csv("Hold_out_Sample .csv", encoding='utf-8', index=False)

## Classification Tree on Unbalanced Dataset
import os
import pandas as pd

#Set the working directory
os.chdir("C:/Users/prianka bhoyar/Desktop/Data Science/Clustering/Mini Project")

#Load the Dataset
CTDF_dev = pd.read_csv("Development_Sample.csv")
CTDF_holdout = pd.read_csv("Hold_out_Sample .csv")

del CTDF_dev['split']
del CTDF_holdout['split']

print( len(CTDF_dev),  len(CTDF_holdout))         ###14013 5987
CTDF_dev.head()

import numpy as np
import matplotlib.pyplot as plt


#Data Preprocessing
#Splitting into features and response variables

CTDF_dev.columns
X =  CTDF_dev[['AGE', 'GENDER', 'BALANCE', 'OCCUPATION',
       'AGE_BKT', 'SCR', 'HOLDING_PERIOD', 'ACC_TYPE',
       'LEN_OF_RLTN_IN_MNTH', 'NO_OF_L_CR_TXNS', 'NO_OF_L_DR_TXNS',
       'TOT_NO_OF_L_TXNS', 'NO_OF_BR_CSH_WDL_DR_TXNS', 'NO_OF_ATM_DR_TXNS',
       'NO_OF_NET_DR_TXNS', 'NO_OF_MOB_DR_TXNS', 'NO_OF_CHQ_DR_TXNS',
       'FLG_HAS_CC', 'AMT_ATM_DR', 'AMT_BR_CSH_WDL_DR', 'AMT_CHQ_DR',
       'AMT_NET_DR', 'AMT_MOB_DR', 'AMT_L_DR', 'FLG_HAS_ANY_CHGS',
       'AMT_OTH_BK_ATM_USG_CHGS', 'AMT_MIN_BAL_NMC_CHGS',
       'NO_OF_IW_CHQ_BNC_TXNS', 'NO_OF_OW_CHQ_BNC_TXNS', 'AVG_AMT_PER_ATM_TXN',
       'AVG_AMT_PER_CSH_WDL_TXN', 'AVG_AMT_PER_CHQ_TXN', 'AVG_AMT_PER_NET_TXN',
       'AVG_AMT_PER_MOB_TXN', 'FLG_HAS_NOMINEE', 'FLG_HAS_OLD_LOAN', 'random']]

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
graph[0].write_pdf(".\classification_tree_output.pdf")



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
score         ##0.8893170627274674

y_train_prob = clf.predict_proba(X_train)
## AUC        
fpr, tpr, thresholds = roc_curve(y_train, y_train_prob[:,1])
auc(fpr, tpr)           ##0.9013858824364737

## Let us see how good is the model
X_holdout =  CTDF_holdout[['AGE', 'GENDER', 'BALANCE', 'OCCUPATION',
       'AGE_BKT', 'SCR', 'HOLDING_PERIOD', 'ACC_TYPE',
       'LEN_OF_RLTN_IN_MNTH', 'NO_OF_L_CR_TXNS', 'NO_OF_L_DR_TXNS',
       'TOT_NO_OF_L_TXNS', 'NO_OF_BR_CSH_WDL_DR_TXNS', 'NO_OF_ATM_DR_TXNS',
       'NO_OF_NET_DR_TXNS', 'NO_OF_MOB_DR_TXNS', 'NO_OF_CHQ_DR_TXNS',
       'FLG_HAS_CC', 'AMT_ATM_DR', 'AMT_BR_CSH_WDL_DR', 'AMT_CHQ_DR',
       'AMT_NET_DR', 'AMT_MOB_DR', 'AMT_L_DR', 'FLG_HAS_ANY_CHGS',
       'AMT_OTH_BK_ATM_USG_CHGS', 'AMT_MIN_BAL_NMC_CHGS',
       'NO_OF_IW_CHQ_BNC_TXNS', 'NO_OF_OW_CHQ_BNC_TXNS', 'AVG_AMT_PER_ATM_TXN',
       'AVG_AMT_PER_CSH_WDL_TXN', 'AVG_AMT_PER_CHQ_TXN', 'AVG_AMT_PER_NET_TXN',
       'AVG_AMT_PER_MOB_TXN', 'FLG_HAS_NOMINEE', 'FLG_HAS_OLD_LOAN', 'random']]
X_test = pd.get_dummies(X_holdout)
y_test = CTDF_holdout["TARGET"]


pred_y_test = clf.predict(X_test)
score_h = accuracy_score(y_test, pred_y_test)
score_h               ##0.8730582929680976

y_test_prob = clf.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob[:,1])
auc(fpr, tpr)        ##0.7987504183588345

##AUC  = (90 - 80)/90 = 0.111

y_freq = np.bincount(y_train)
y_val = np.nonzero(y_freq)[0]
np.vstack((y_val,y_freq[y_val])).T

###array([[    0, 12268],
  ##     [    1,  1745]], dtype=int64)
  
##1745/14013 = 12.45

#Cross validation function
from sklearn.cross_validation import cross_val_score
scores = cross_val_score(clf, X_train , y_train, cv = 10, scoring='roc_auc')
scores.mean()       ### 0.7794564016934735
scores.std()        ### 0.044578044904966294

y_train_prob = clf.predict_proba(X_train)
fpr, tpr, thresholds = roc_curve(y_train, y_train_prob[:,1])
auc(fpr, tpr)       ###0.9013763998493995


y_test_prob = clf.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob[:,1])
auc(fpr, tpr)       ###0.7987504183588345


## Tuning the Classifier using GridSearchCV
from sklearn.grid_search import GridSearchCV
help(GridSearchCV)

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

classifier = tree_cv.best_estimator_

classifier.fit(X_train,y_train)



#predicting probabilities
y_train_prob = classifier.predict_proba(X_train)
fpr, tpr, thresholds = roc_curve(y_train, y_train_prob[:,1])
auc_d = auc(fpr, tpr)
auc_d     ###0.8562159759637438
y_test_prob = classifier.predict_proba(X_test)
fpr, tpr, thresholds = roc_curve(y_test, y_test_prob[:,1])
auc_h = auc(fpr, tpr)
auc_h   ###0.7635691878093983

###(auc_d-auc_h)/auc_d = 0.10820492814335032

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
Rank           ###0.546566

## Let us see the Rank Ordering on Hold-Out Dataset
Prediction_h = classifier.predict_proba(X_test)
CTDF_holdout["prob_score"] = Prediction_h[:,1]

Rank_h = Rank_Ordering(CTDF_holdout,"prob_score","TARGET")
Rank_h        ###0.393683

##Rank = (Rank - Rank_h) / Rank
## (55-40)/55  = 0.2727272727272727

y_freq = np.bincount(y_test)
y_val = np.nonzero(y_freq)[0]
np.vstack((y_val,y_freq[y_val])).T

##array([[   0, 5220],
      ## [   1,  767]], dtype=int64)
      
## 767/5987= 0.12811090696509103
     
###Res Rate (Dev) = 18.658280922431867
###Res Rate (Test)=10.328997704667177

###18.66-10.33

RR(Test)= 38.752/13 = 2.981
RR(Dev)=47.41/12.45 = 3.808

(3.808 - 2.981) / 3.808 = 0.217  =22%