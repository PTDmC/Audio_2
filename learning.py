
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import csv
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('C:/Users/siura/source/repos/Project_2022_2/Project_2022_2/learning_data/data.csv', header=None)
dataset = df.rename(columns={0: 'Judge'})

dataX = dataset.drop(['Judge'], axis=1)
dataY = dataset['Judge']

X_train, X_test, y_train, y_test = train_test_split(dataX, dataY, test_size=0.2, stratify=dataY)

clf = LogisticRegression()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

filename = 'model_sample.pickle'
pickle.dump(clf, open(filename, 'wb'))