import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelBinarizer


housing_path = "/Users/rohitashavaggarwal/PycharmProjects/training/training"
def load_housing_data(housing_path):
    csv_path = os.path.join (housing_path, "housing.csv")
    dataframe = pd.read_csv(csv_path)
    # print(dataframe.head())
    return dataframe

df =  load_housing_data(housing_path)
print("*******")
print(df.iloc[:5])
housing_cat= df["ocean_proximity"]

encoder = LabelBinarizer(sparse_output=True)
housing_cat_1hot = encoder.fit_transform(housing_cat)
print(housing_cat_1hot)


