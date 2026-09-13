import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (accuracy_score, recall_score, f1_score, precision_score)
import joblib
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import confusion_matrix


data = pd.read_csv("../../Data/jee.csv")
def create_feature(X):
    X = X.copy()
    X["Study_hours_per_age"] = X["Study_Hours"] / X["Age"]
    return X

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy = "most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown = "ignore"))
])

preprocessor = ColumnTransformer(
    transformers = [
        ("num", num_pipeline, ["Age","Study_Hours","Study_hours_per_age"]),
        ("cat", cat_pipeline, ["City","Internet_Access"])
    ]
)

feature_engineering = FunctionTransformer(create_feature)

model_pipeline = Pipeline([
    ("feature_engineering", feature_engineering),
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(random_state = 1, max_iter=1000))
])

train_data, val_data, test_data = split_dataset(data)

features = ["Age","City","Study_Hours","Internet_Access"]
X_train = train_data[features]
y_train = train_data["Passed"]
X_val = val_data[features]
y_val = val_data["Passed"]
X_test = test_data[features]
y_test = test_data["Passed"]

model_pipeline.fit(X_train, y_train)
predictions = model_pipeline.predict(X_val)

accuracy = accuracy_score(y_val, predictions)
precision = precision_score(y_val, predictions)
recall = recall_score(y_val, predictions)
f1 = f1_score(y_val, predictions)
print(f"Accuracy score is : {accuracy:.2%}")
print(f"Precision score is : {precision:.2%}")
print(f"Recall score is : {recall:.2%}")
print(f"F1 score is : {f1:.2%}")

joblib.dump(model_pipeline, "model_pipeline.joblib")

loaded_pipeline = joblib.load("model_pipeline.joblib")
loaded_predictions = loaded_pipeline.predict(X_val)
print((loaded_predictions == predictions).all())

y_proba = model_pipeline.predict_proba(X_val)[:, 1]
threshold_result = []
for threshold in np.arange(0.50, 1.0, 0.05):
    y_pred = (y_proba >= threshold).astype(int)
    
    pred_recall = recall_score(y_val, y_pred)
    pred_precision = precision_score(y_val, y_pred)
    pred_f1 = f1_score(y_val, y_pred)

    threshold_result.append({
        "Threshold" : threshold,
        "Precision" : pred_precision,
        "Recall" : pred_recall,
        "F1" : pred_f1
    })
threshold_df = pd.DataFrame(threshold_result)

final_threshold = 0.5
test_proba = model_pipeline.predict_proba(X_test)[:,1]
y_test_pred = (test_proba >= final_threshold).astype(int)

test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"Test Accuracy score is : {test_accuracy:.2%}")
print(f"Test Precision score is : {test_precision:.2%}")
print(f"Test Recall score is : {test_recall:.2%}")
print(f"Test F1 score is : {test_f1:.2%}")

print("Positive min:", y_proba[y_val == 1].min())
print("Positive max:", y_proba[y_val == 1].max())
print("Negative min:", y_proba[y_val == 0].min())
print("Negative max:", y_proba[y_val == 0].max())

# errors = X_val[y_val != y_pred].copy()
# print("Number of Errors :",len(errors))
# errors["Actual"] = y_val[y_val != y_pred]
# errors["Predicted"] = y_pred[y_val != y_pred]
# print(errors)

y_val_pred = model_pipeline.predict(X_val)
errors = X_val[y_val != y_val_pred].copy()
print("Number of Errors :",len(errors))
errors["Actual"] = y_val[y_val != y_val_pred]
errors["Predicted"] = y_val_pred[y_val != y_val_pred]
print(errors)

fp_mask =  (y_val == 0) & (y_val_pred == 1)
fp_proba = y_proba[fp_mask]
print(fp_proba)

print(data[data["Passed"] == 0].describe())
print(data[data["Passed"] == 0]["Study_Hours"].value_counts().sort_index())
print(data[data["Passed"] == 1]["Study_Hours"].value_counts().sort_index())

cm = confusion_matrix(y_val, predictions)
cm

print(threshold_df)
print(threshold_df.to_string(index = False))
