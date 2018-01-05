# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
#Import the data
df = pd.read_csv('train.csv')
df.head()

df.corr()
#Check total number of rows in train dataset
df.index

#Check for number of missing values in data
df.isnull().sum()

#Check for the best value to replace missing data in Age

df['Age'].mean()
df['Age'].median()
new_age = df['Age'].mode()

# See age distribution
df.hist(column='Age', figsize = (9,6), bins = 20)

#Fill NaN values with mode value
type(new_age)
new_age = float(new_age)
df['Age'] = df.Age.fillna(new_age)
new_age


#Get training data
df['Embarked'] =pd.factorize(df.Embarked)[0]
df['Sex'] =pd.factorize(df.Sex)[0]



#Selection of columns
x = df.drop(['PassengerId','Survived','Name','Ticket','Cabin','Fare'], axis =1)
x.head()
y = df['Survived']

#Data analysis
from matplotlib import cm
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)
cmap = cm.get_cmap('gnuplot')
scatter = pd.plotting.scatter_matrix(x_train, c= y_train, marker = 'o', s=40, hist_kwds={'bins':15}, figsize=(9,9), cmap=cmap)
plt.show()

#Comparison models
cols1 = ['Pclass','Sex','Age','SibSp','Parch','Fare','Embarked' ]
X = df[cols1]
cols2 = ['Pclass','Sex','Age','SibSp','Parch','Embarked']
X2 = df[cols2]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.25, random_state = 0)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2,y, test_size = 0.25, random_state = 0)

test1 = []
train1 = []
test2 = []
train2 = []
for n in range(1,10):
    knn = KNeighborsClassifier(n_neighbors = n)
    knn.fit(X_train, y_train)
    train1.append(knn.score(X_train, y_train))
    test1.append(knn.score(X_test, y_test))
    knn.fit(X2_train, y2_train)
    train2.append(knn.score(X2_train, y2_train))
    test2.append(knn.score(X2_test, y2_test))
 
plt.scatter(range(1,10), test1, c='r')
plt.scatter(range(1,10), test2, c='b')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.show()

x = df.drop(['PassengerId','Survived','Name','Ticket','Cabin','Fare'], axis =1)
x.head()
y = df['Survived']

randomstates = [0,1,5]
kvalues = [3,4,5]

minmax_scaler = preprocessing.MinMaxScaler()
for rs in randomstates:
    x_train, x_test, y_train, y_test = train_test_split(x, y,train_size = 0.75,test_size=0.25, random_state=rs)
    x_train_minmax = minmax_scaler.fit_transform(x_train)
    x_train_minmax
    x_test = minmax_scaler.fit_transform(x_test)
    for k in kvalues:
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(x_train_minmax, y_train)
        score = knn.score(x_test, y_test)
        print('Data split of 75/25 and random state=' + str(rs) + ' k= '+ str(k) +' kscore =' + str(score))

for rs in randomstates:
    x_train, x_test, y_train, y_test = train_test_split(x, y,train_size = 0.80,test_size=0.20, random_state=rs)
    x_train_minmax = minmax_scaler.fit_transform(x_train)
    x_train_minmax
    x_test = minmax_scaler.fit_transform(x_test)
    for k in kvalues:
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(x_train_minmax, y_train)
        score = knn.score(x_test, y_test)
        print('Data split of 80/20 and random state=' + str(rs) + ' k= '+ str(k) +' kscore =' + str(score))
   
for rs in randomstates:
    x_train, x_test, y_train, y_test = train_test_split(x, y,train_size = 0.70,test_size=0.30, random_state=rs)
    x_train_minmax = minmax_scaler.fit_transform(x_train)
    x_train_minmax
    x_test = minmax_scaler.fit_transform(x_test)
    for k in kvalues:
        from sklearn.neighbors import KNeighborsClassifier
        knn = KNeighborsClassifier(n_neighbors = k)
        knn.fit(x_train_minmax, y_train)
        score = knn.score(x_test, y_test)
        print('Data split of 70/30 and random state=' + str(rs) + ' k= '+ str(k) +' kscore =' + str(score))

#Selected knn is 75/25 with k = 4
x_train, x_test, y_train, y_test = train_test_split(x, y,train_size = 0.90,test_size=0.10, random_state=5)
x_train_minmax = minmax_scaler.fit_transform(x_train)
x_train_minmax
x_test = minmax_scaler.fit_transform(x_test)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(x_train_minmax, y_train)
score = knn.score(x_test, y_test)

      
#predict for new dataset
dftest = pd.read_csv('test.csv')
dftest.head()

#Pre-processing of test data

dfx = dftest.drop(['PassengerId','Name','Ticket','Cabin','Fare'], axis =1)
dfx.head()
dfx['Embarked'] =pd.factorize(dfx.Embarked)[0]
dfx['Sex'] =pd.factorize(dfx.Sex)[0]
dfx['Age'] = dfx.Age.fillna(new_age)

dfx.head()
#scaling of test data
from sklearn import preprocessing
minmax_scaler = preprocessing.MinMaxScaler()
dfx = minmax_scaler.fit_transform(dfx)

#predict the values for the final data
predvar = knn.predict(dfx)
final = zip(dftest['PassengerId'],predvar)
final =  list(final)
final.insert(0, ['PassengerId','Survived'])

#Write final output file

import csv
# W for mac and wb for windows
with open('Titanic_prediction_.csv','w') as resultFile:
    wr = csv.writer(resultFile, dialect='excel')
    wr.writerows(final)
