# import necessary libraries to execute the Model
import numpy as np
import os
import pandas as pd

# housing.csv file location
housing_path = "/Users/rohitashavaggarwal/PycharmProjects/training/training"

# read the file data and return dataFrame
def load_housing_data(housing_path):
    csv_path = os.path.join (housing_path, "housing.csv")
    dataframe = pd.read_csv(csv_path)
    # print(dataframe.head())
    return dataframe
housing =  load_housing_data(housing_path)

# split the data into test (20%) and train (80%) sets and shuffle them randomly and get same results every time.
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# Run the following commands to make sure that your code runs so far.
# print("test set:")
# print(test_set.info())
# print("train set:")
# print(train_set.info())

# separating the numerical and categorical attributes from data file
train_num = train_set.drop("median_house_value", axis=1)
housing_labels = train_set["median_house_value"].copy()

housing_num = train_num.drop("ocean_proximity", axis=1)
housing_cat = housing["ocean_proximity"]

from sklearn.preprocessing import Imputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder

# from sklearn.preprocessing import LabelBinarizer
# encoder = LabelBinarizer()
# housing_cat_1hot = encoder.fit_transform(housing_cat)

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names, factorize=False):
        self.attribute_names = attribute_names
        self.factorize = factorize

    def fit(self, X, y=None):
        return self

    def transform(self, X, y= None):
        selection = X[self.attribute_names]
        if self.factorize:
            selection = selection.apply(lambda p: pd.factorize(p)[0] + 1)
        return selection.values

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                        bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


from sklearn.pipeline import FeatureUnion

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attribs, False)),
        ('imputer', Imputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
])
cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attribs, True)),
        # ('label_binarizer', CustomBinarizer()),
        ('one_hot_encoder', OneHotEncoder())
])

full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
])
housing_prepared = full_pipeline.fit_transform(train_set)
print(housing_prepared.shape)

# Finally, apply the grid search model to predict the House median value
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
scoring='neg_mean_squared_error')
grid_search.fit(housing_prepared, housing_labels)

cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):

    print(np.sqrt(-mean_score), params)