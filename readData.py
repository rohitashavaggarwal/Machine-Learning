import pandas as pd
import numpy
import os
import matplotlib.pyplot as plt

housing_path = "/Users/rohitashavaggarwal/PycharmProjects/training/training"
def load_housing_data(housing_path):
    csv_path = os.path.join (housing_path, "housing.csv")
    dataframe = pd.read_csv(csv_path)
    # print(dataframe.head())
    return dataframe

df =  load_housing_data(housing_path)
print(df.info())
print("***************")
df_html = df.describe()
df_html.to_csv("output.csv")
print(df.describe(include=[numpy.object, numpy.number]))


print("*********")
print(df["population"].max())
print("*********")


print("*********")
print(df["housing_median_age"].max())
print("*********")




# arranged= numpy.arange(0, df["population"].max(), 200 )
# # change the scale of axes (x and y)
# df.hist(bins = 100, column="population", figsize=[9,9])
# ax = plt.gca()
# ax.set_xlim([0, 20000])
df["housing_median_age"].hist(bins=10, figsize=(5,5))
ax = plt.gca()
ax.set_xlim([0, 60])
plt.show()




