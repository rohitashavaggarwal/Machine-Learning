import numpy as np
import os
import pandas as pd


housing_path = "/Users/rohitashavaggarwal/PycharmProjects/training/training"
def load_housing_data(housing_path):
    csv_path = os.path.join (housing_path, "housing.csv")
    dataframe = pd.read_csv(csv_path)
    # print(dataframe.head())
    return dataframe

df =  load_housing_data(housing_path)

def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    print("*******")
    print(shuffled_indices)
    print("*******")
    print(len(shuffled_indices))
    print("*******")
    test_set_size = int(len(data) * test_ratio)
    print(test_set_size)
    test_indices = shuffled_indices[:test_set_size]
    print("*******")
    print(test_indices)
    print("*******")
    train_indices = shuffled_indices[test_set_size:]
    print(train_indices)
    return data.iloc[train_indices], data.iloc[test_indices]

train,test =split_train_test(df, 0.2)

