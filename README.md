# glass.csv

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

df = pd.read_csv('glass.csv')
df.head()
df.shape

# Load the data from “glass.csv” and make a bar plot of different types of glasses.
y = df.Type
N = len(y)
x = np.arange(N)
plt.bar(x,y)
plt.title('Types of glass')
plt.ylabel('No of glasses')
plt.legend(labels=['Glass'])

# Make a train_test split and fit a single decision tree classifier
X = df.iloc[:,:-1].values
y = df['Type']

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=0)

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn import metrics

clf = DecisionTreeClassifier()
clf.fit(X_train,y_train)
y_pred=clf.predict(X_test,check_input=True)

#Accuracy
print(clf.score(X_test,y_test, sample_weight=None))
print('Accuracy is : ',metrics.accuracy_score(y_pred,y_test))

f1 = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(f1)

cross_val_score(clf, X_train, y_train, cv=10)
clf.get_params(deep=True)
ppred = clf.predict(X_test)

# Make a k-fold split with 3 splits and measure the accuracy score with each split
from sklearn.model_selection import KFold
X = df.iloc[:,:-1].values
y = df['Type']
kf= KFold(n_splits=3)
kf.get_n_splits(X)

for train_index,test_index in kf.split(X):
    X_train,X_test = X[train_index],X[test_index]
    y_train,y_test = y[train_index],y[test_index]
    clf.fit(X_train,y_train)
    y_pred=clf.predict(X_test,check_input=True)
    print(clf.score(X_test,y_test, sample_weight=None))
    print('Accuracy is : ',metrics.accuracy_score(y_pred,y_test))

# Use gridSearchCV from sklearn for finding out a suitable number of estimators for a RandomForestClassifer alongwith a 10-fold cross validation.

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param = {'n_estimators':[50,100,150,200]}
# create and fit a ridge regression model, testing each alpha
model = RandomForestClassifier()
grid = GridSearchCV(estimator=model, param_grid=param,cv=10)
grid.fit(X, y)
print(grid)
# summarize the results of the grid search
print(grid.best_score_)
print(grid.best_params_)

from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.model_selection import cross_val_score
model=RandomForestClassifier(n_estimators=100)# a simple random forest model
model.fit(X_train,y_train)
prediction=model.predict(X_test)
print(metrics.accuracy_score(prediction,y_test))
print(cross_val_score(model,X,y,cv=10,scoring='accuracy'))
