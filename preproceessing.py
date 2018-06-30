import sys

import scipy
import numpy as np
import matplotlib
import pandas as pd
import sklearn

from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split
from sklearn import preprocessing

import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

f_data = open("/home/anirudh/Downloads/Geographical_Origin_of_Music/default_features_1059_tracks.txt", "r")


data_array = np.ones(shape=(1, 70), dtype=float)

for line in f_data.readlines():
    data_in_line = line.split(',')
    f_array = np.array([data_in_line])
    data_array = np.row_stack((data_array, f_array))

data_array = data_array[1:, :]
X_original_data = data_array[:, :68]
Y_original_data = data_array[:, 68:]


entire_dataframe = pd.DataFrame(data_array, columns=column_name)


# Drop columns with more than 50%(424) examples not available
entire_dataframe = entire_dataframe.dropna(axis=1, thresh=424)

# Drop rows with more than 50%(34) features not available
entire_dataframe = entire_dataframe.dropna(axis=0, thresh=34)

# Seperate dataframe into 30% as test cases
train_dataframe, test_dataframe = train_test_split(entire_dataframe, test_size=0.3)

# Fit the classifier on the examples with no missing instance
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()

data_without_na = train_dataframe.dropna()

train_data_x = data_without_na.iloc[:, :68]
train_data_y = data_without_na.iloc[:, 68:]

knn.fit(train_data_x, train_data_y)

filling_train = train_dataframe.iloc[:, :68]
filling_test = test_dataframe.iloc[:, :68]

# Use the classifier to predict the missing values
predicted_train = pd.DataFrame(knn.predict(filling_train))
predicted_test = pd.DataFrame(knn.predict(filling_test))

# Replace the missing values with the values predicted by the classifier
train_dataframe.fillna(predicted_train, inplace=True)
test_dataframe.fillna(predicted_test, inplace=True)

# Completely filled data
train_data_x = train_dataframe.iloc[:, :68]
train_data_y = train_dataframe.iloc[:, 68:]

test_data_x = test_dataframe.iloc[:, :68]
test_data_y = test_dataframe.iloc[:, 68:]

# Use feature scaling and mean normalization so that all data is between [-1,1] with its mean as 0
ss = preprocessing.StandardScaler()
train_data_x = ss.fit_transform(train_data_x)
test_data_x = ss.fit_transform(test_data_x)


# Visualize the correlation between each pair of features
normalized_data = np.row_stack((train_data_x, test_data_x))
correlation_dataframe = pd.DataFrame(normalized_data, columns=column_name[:68])

correlation = correlation_dataframe.corr()
plt.figure(figsize=(68, 68))
sns.heatmap(correlation, vmax=1, square=True, annot=True, cmap='cubehelix')
plt.title('Correlation between different fearures')


# Use Principal Component Analysis to reduce number of features and retain 95% of the total variance
pca = PCA(.95)
pca.fit(train_data_x)

train_data_x = pca.transform(train_data_x)
test_data_x = pca.transform(test_data_x)


# using list comprehensions to label the rows and columns in the final dataset
column_name = ["x{}".format(i + 1) for i in range(40)]
column_name.append("y1")
column_name.append("y2")

row_name_train = [i + 1 for i in range(741)]
row_name_test = [i + 1 for i in range(318)]


train_normal_dataframe = pd.DataFrame(np.column_stack((train_data_x, train_data_y)), index=row_name_train, columns=column_name)

test_normal_dataframe = pd.DataFrame(np.column_stack((test_data_x, test_data_y)), index=row_name_test, columns=column_name)
