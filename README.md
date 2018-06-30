# music-preprocessing

I have preprocessed the data from the dataset in the following steps:

1.Removing the columns and rows in which more than 50% of the values were missing-
The dropna() method is used to do this by setting the respective threshold values for columns and rows

2.Splitting the dataset into test and train datasets-
The train_test_split method splits the data into train and test cases in which I have used 30% of data as test cases

3.Imputing missing values-
K Nearest Neighbors classifier is used to train on the examples which had all their instances present and then used to predict and fill in the misssing values in both datasets.

4.Mean normalization-
The Standard Scaler method transform the data to a mean of 0 and standard deviation of 1

5.Correlation Visualisation-
The heatmap method is used to visualize the correlation between different features; and thus decide if using PCA would be useful 

6.Applying Principal Component Analysis-
PCA is applied on the training data to retain 95% of the original variance. The result is then used to transform both the training and test sets.
