from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
from sklearn.inspection import permutation_importance
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

data = pd.read_csv("../../Data/student_pass_fail.csv")

train_data, val_data, test_data = split_dataset(data)

features = ["hours_studied","attendance_pct","prev_score","sleep_hours"]

X_train = train_data[features]
y_train = train_data["passed"]
X_val = val_data[features]
y_val = val_data["passed"]
X_test = test_data[features]
y_test = test_data["passed"]

model = GradientBoostingClassifier(learning_rate = 0.1, n_estimators = 50, random_state = 1)
model.fit(X_train, y_train)
predictions = model.predict(X_val)

accuracy = accuracy_score(y_val, predictions)
precision = precision_score(y_val, predictions)
recall = recall_score(y_val, predictions)
f1 = f1_score(y_val, predictions)
print(f"Accuracy score is : {accuracy:.2%}")
print(f"Precision score is : {precision:.2%}")
print(f"Recall score is : {recall:.2%}")
print(f"F1 score is : {f1:.2%}")

train_pred = model.predict(X_train)
train_accuracy = accuracy_score(y_train, train_pred)
val_accuracy = accuracy_score(y_val, predictions)

print(f"Train accuracy is : {train_accuracy:.2%}")
print(f"Validation accuracy is : {val_accuracy:.2%}")

y_test_pred = model.predict(X_test)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"Test Accuracy score is : {test_accuracy:.2%}")
print(f"Test Precision score is : {test_precision:.2%}")
print(f"Test Recall score is : {test_recall:.2%}")
print(f"Test F1 score is : {test_f1:.2%}")
