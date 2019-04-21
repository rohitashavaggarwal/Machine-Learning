import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

housing_path = "/Users/rohitashavaggarwal/PycharmProjects/training/training"
def load_housing_data(housing_path):
    csv_path = os.path.join (housing_path, "housing.csv")
    dataframe = pd.read_csv(csv_path)
    # print(dataframe.head())
    return dataframe

df =  load_housing_data(housing_path)

def split_train_test(data, train_ratio):
    shuffled_indices = np.random.permutation(len(data))
    train_set_size = int(len(data) * train_ratio)
    train_indices = shuffled_indices[:train_set_size]
    print(train_indices)
    return train_indices

split_train_test(df, 0.8)
print(len(split_train_test(df, 0.8)))
print(df.iloc[split_train_test(df, 0.8)])

def train_set(df):
    return df.iloc[split_train_test(df, 0.8)]


df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

df.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
s=df["population"]/100, label="population",
c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)

plt.legend()
plt.show()