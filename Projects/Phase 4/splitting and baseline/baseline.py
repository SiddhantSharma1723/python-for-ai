def majority_baseline(train_data, target_column, dataset_name):
    majority_class_func = (train_data[target_column].value_counts(normalize=True).max())

    print(f"Baseline accuracy: {majority_class_func:.2%}")


    with open("baseline_report_classification.md","w") as f:
        f.write("Baseline Report:\n\n")
        f.write(f"Dataset name: {dataset_name}\n")
        f.write("Method: Always predict major class\n")
        f.write(f"Baseline Accuracy: {majority_class_func:.2%}")
    print("REPORT SAVED")


import numpy as np
from sklearn.metrics import (mean_absolute_error,root_mean_squared_error)

def mean_baseline(dataset_name,train_data, val_data, target_column):
    mean_func = (train_data[target_column].mean())

    l = len(val_data)
    pred = np.full(l,mean_func)

    mae = mean_absolute_error(val_data[target_column],pred)
    rmse = root_mean_squared_error(val_data[target_column],pred)
    print(f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}")
    with open("baseline_report_regression.md","w") as f:
        f.write("This is the baseline report.\n")
        f.write(f"MAE = {mae:.2f}\nRMSE = {rmse:.2f}")
    print("REPORT SAVED")

    return mae,rmse