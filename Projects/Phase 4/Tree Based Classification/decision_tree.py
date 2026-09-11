from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
import pandas as pd

data = pd.read_csv("../../Data/student_pass_fail.csv")

train_data, val_data, test_data = split_dataset(data)
features = ["hours_studied","attendance_pct","prev_score","sleep_hours"]
target = ["passed"]

X_train = train_data[features]
y_train = train_data["passed"]
X_val = val_data[features]
y_val = val_data["passed"]

model = DecisionTreeClassifier(max_depth = 6)
model.fit(X_train,y_train)
predictions = model.predict(X_val)

accuracy = accuracy_score(y_val, predictions)
precision = precision_score(y_val, predictions)
recall = recall_score(y_val, predictions)
f1 = f1_score(y_val, predictions)
print(f"Accuracy score is : {accuracy:.2%}")
print(f"Precision score is : {precision:.2%}")
print(f"Recall score is : {recall:.2%}")
print(f"F1 score is : {f1:.2%}")

# train_predictions = model.predict(X_train)
# accuracy_train = accuracy_score(y_train, train_predictions)
# print(f"Train accuracy is : {accuracy_train:.2%}")
# print(f"Validation accuracy is : {accuracy:.2%}")