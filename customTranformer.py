import numpy as np
import os
import pandas as pd


from sklearn.base import BaseEstimator, TransformerMixin
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        # print("room_ix")
        # print(rooms_per_household)
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
         bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
         return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        print("********")
        print("room_ix")
        print("********")
        print(rooms_per_household)
        print("********")
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
         bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
         return np.c_[X, rooms_per_household, population_per_household,bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

# creating column header list from dataframe
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values


housing_path = "/Users/rohitashavaggarwal/PycharmProjects/training/training"
def load_housing_data(housing_path):
    csv_path = os.path.join (housing_path, "housing.csv")
    dataframe = pd.read_csv(csv_path)
    # print(dataframe.head())
    return dataframe

df =  load_housing_data(housing_path)
# print(df.info())
attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
dataframe_selector = DataFrameSelector()
dfSelector = dataframe_selector.transform(df)
print("*********")
print(dfSelector)
housing_extra_attribs = attr_adder.transform(df.values)
print(housing_extra_attribs[:5])