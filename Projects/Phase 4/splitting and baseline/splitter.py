import pandas as pd

data = pd.read_csv("../../Data/rice_practice_dataset_1.csv")
def split_dataset(data, train_frac = 0.8, val_frac = 0.1, seed = 1):
    shuffled = data.sample(frac=1, random_state = seed).reset_index(drop=True)
    n = len(shuffled)
    train_end = int(n * train_frac)
    val_end = int(n * (train_frac + val_frac))
    train = shuffled.iloc[:train_end]
    val = shuffled.iloc[train_end:val_end]
    test = shuffled.iloc[val_end:]

    return train, val, test
train_data, val_data, test_data = split_dataset(data)
print(f"Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
