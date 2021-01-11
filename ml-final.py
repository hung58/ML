# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
pd.options.mode.chained_assignment = None
import sys
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import datatable as dt

import time

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import xgboost as xgb

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

import janestreet
env = janestreet.make_env() 
iter_test = env.iter_test()
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

%%time
#insert data
train_dt = dt.fread('../input/jane-street-market-prediction/train.csv')
train = train_dt.to_pandas()
train = train.astype({c: np.float32 for c in train.select_dtypes(include='float64').columns})

train=train[train['weight']!=0]

%%time
f1 = train['feature_1'].mean()
train['feature_1'] = train['feature_1'].fillna(f1)
index = train[ 'feature_1'] >8
train['feature_1' ][index]=8
index = train['feature_1'] <0
train['feature_1' ] [index]=0

f3 = train[ 'feature_3'].mean()
train['feature_3'] = train['feature_3' ].fillna(f3)
index = train['feature_3'] <-2
train['feature_3' ][ index ]=-2

f13 = train[ 'feature_13' ].mean()
train['feature_13'] = train['feature_13' ].fillna( f13)
index = train['feature_13'] < 0
train[ 'feature_13' ][ index]=0
index = train['feature_13'] > 0
train[ 'feature_13' ] [index]=1

f14 = train[ 'feature_14' ].mean()
train['feature_14'] = train['feature_14' ].fillna(f14)
index = train['feature_14'] < 0.5
train['feature_14' ][index]=-1
index = train['feature_14' ] > 0.5
train[ 'feature_14' ][index]=0.5

f15 = train[ 'feature_15' ].mean()
train['feature_15'] = train['feature_15' ].fillna(f15)
index = train['feature_15'] < -0.5
train['feature_15' ] [ index]=-0.5
index = train['feature_15'] > -0.5
train[ 'feature_15' ] [index]=0.5

f51 = train[ 'feature_51' ].mean()
train['feature_51'] = train['feature_51' ].fillna(f51)
index = train['feature_51'] < -1
train['feature_51' ][ index]=-1
index = train['feature_51'] > -0.5
train['feature_51' ][index]=-0.5

f88 = train['feature_88' ].mean()
train['feature_88'] = train['feature_88'].fillna(f88)
index = train['feature_88'] < 0
train['feature_88' ] [ index]=-2
index = train['feature_88' ] > 0
train['feature_88' ][index]-0

f91 = train['feature_91'].mean()
train['feature_91'] = train['feature_91'].fillna(f91)
index = train['feature_91'] < 0
train['feature_91' ][ index]=-2

f111 = train['feature_111' ].mean()
train['feature_111'] = train['feature_111'].fillna(f111)
index=train['feature_111']<-1
train['feature_111'][index]=-1

f5 = np.product(train['feature_5'].mode())
train['feature_5'] = train['feature_5'].fillna(f5)

f7 = np.product(train['feature_7'].mode())
train['feature_7'] = train['feature_7'].fillna(f7)

f17 = np.product(train['feature_17'].mode())
train['feature_17'] = train['feature_17'].fillna(f17)


train['action']=(train['resp']>0).astype('int')


X_train = train.loc[:, train.columns.str.contains('feature')]
y_train = train.loc[:, 'action']

clf = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=11,
    min_child_weight=9.15,
    gamma=0.59,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.7,
    alpha=10.4,
    nthread=5,
    missing=-999,
    random_state=2020,
    tree_method='gpu_hist'  # THE MAGICAL PARAMETER
)
%time clf.fit(X_train, y_train)

%%time
for (test_df, sample_prediction_df) in iter_test:
    X_test = test_df.loc[:, test_df.columns.str.contains('feature')]
    y_preds = clf.predict(X_test)
    sample_prediction_df.action = y_preds
    env.predict(sample_prediction_df)

X_train, X_test, y_train, y_test = train_test_split(X_train,y_train, test_size=0.2, stratify=y_train,random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,random_state=42, stratify=y_train)

Y_prediction_LogReg = clf.predict(X_test)
Y_prediction_LogReg[Y_prediction_LogReg > 0.5] = 1
Y_prediction_LogReg[Y_prediction_LogReg <= 0.5] = 0
print(classification_report(y_test, Y_prediction_LogReg))
print("Accuracy: ", accuracy_score (y_test, Y_prediction_LogReg))