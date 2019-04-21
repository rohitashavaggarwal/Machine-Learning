import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt


housing_path = "/Users/rohitashavaggarwal/PycharmProjects/training/training"
def load_housing_data(housing_path):
    csv_path = os.path.join (housing_path, "housing.csv")
    dataframe = pd.read_csv(csv_path)
    # print(dataframe.head())
    return dataframe

housing =  load_housing_data(housing_path)

# print(housing["median_income"].value_counts())

housing["median_income"].hist()
plt.show()
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)

# print(housing["income_cat"].value_counts())
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

print(housing["income_cat"].value_counts())
# housing["income_cat"].where(housing["income_cat"] < 5, 5.0)
# n = np.array([1.2, 4.6, 5.3])
# print(np.ceil(n))

print(housing["median_income"].value_counts(sort=True))