'''
1 import all the necessary modules and lib (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, pandas, linear_model - logistic regression.)
2 import the dataset
3 import sklearn.preprocessing for standardscaler which is used to avoid iteration errors
4 create class_bool(if two classes)
5 import split dataset function
6 split the data into train val and test
7 pick feature columns
8 separate x_train and y_train from train data
9 separate x_val and y_val from val data
10 define logistic regression model
11 train the model using X_train and y_train
12 predict on basis of X_val
13 store the prediction in "predictions"
14 compute accuracy precision f1 recall and roc_auc on the basis of y_val and predictions
'''

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, f1_score, roc_auc_score, recall_score)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("../../Data/student_life.csv")
train_data, val_data, test_data = split_dataset(data)

features = ["Age","Study_Hours","Attendance","Practice_Tests","Sleep_Hours"]

X_train = train_data[features]
y_train = train_data["Passed"]
X_val = val_data[features]
y_val = val_data["Passed"]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.fit_transform(X_val)

model = LogisticRegression()
model.fit(X_train_scaled,y_train)

predictions = model.predict(X_val_scaled)

accuracy = accuracy_score(y_val, predictions)
precision = precision_score(y_val, predictions)
recall = recall_score(y_val, predictions)
f1 = f1_score(y_val, predictions)
roc_auc = roc_auc_score(y_val, predictions)

print(f"Accuracy Score is {accuracy}.\nPrecision Score is {precision}.\nRecall Score = {recall}.\nF1 Score is {f1}.\nROC_AUC Score is {roc_auc_score}.")

