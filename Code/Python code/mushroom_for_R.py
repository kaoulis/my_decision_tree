import csv
import time
import numpy as np
import pandas as pd
import pydotplus
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from IPython.display import Image
from sklearn.utils import resample

from MyDecisionTree import MyDecisionTree
import sys

sys.setrecursionlimit(15000)

data_array = np.genfromtxt("agaricus-lepiota.data", dtype='str', delimiter=",")
dfull = pd.DataFrame(data=data_array)

# The data to analyse in R. Same code with mushroom.py with the difference that gets metrics for 20 * 10 subsets of Mushroum data set
csv_dict = {'Time': [], 'Data_proportion': [], 'Algorithm': []}
for times in range(20):
    for i in range(10):
        # resample a subset of dataframe
        df = resample(dfull, replace=True, n_samples=int(len(dfull)-0.1*i*len(dfull)))

        # Handle missing values
        df = df.replace('?', np.nan)
        df[11].fillna(df[11].mode()[0], inplace=True)

        # Encoding data preparation
        df[0] = LabelEncoder().fit_transform(df[0])
        df = pd.get_dummies(df)
        df.drop(['4_f', '8_b', '10_e', '16_p'], axis=1, inplace=True)
        labels = df[0].values
        df.drop(0, axis=1, inplace=True)
        selected_features = df[['5_n', '5_f', '8_n']]
        features = selected_features.to_numpy()


        skdc_scores = {'Time': []}
        mydc_scores = {'Time': []}
        skdc = DecisionTreeClassifier()
        mydc = MyDecisionTree(impurity_threshold=0.001)
        cv = KFold(n_splits=10)
        for train_index, test_index in cv.split(features):
            x_train, x_test, y_train, y_test = features[train_index], features[test_index], labels[train_index], labels[
                test_index]
            # Sklearn
            start = time.time()
            fit = skdc.fit(x_train, y_train)
            skdc_scores['Time'].append(time.time() - start)
            y_pred_dc = skdc.predict(x_test)
            # My alggorithm
            start = time.time()
            tree = mydc.create_tree(x_train.tolist(), y_train.tolist())
            mydc_scores['Time'].append(time.time() - start)
            y_pred_mydc = mydc.predict(x_test.tolist(), tree)

        csv_dict['Time'].append(np.mean(skdc_scores['Time']))
        csv_dict['Data_proportion'].append(100-10*i)
        csv_dict['Algorithm'].append('Sklearn')

        csv_dict['Time'].append(np.mean(mydc_scores['Time']))
        csv_dict['Data_proportion'].append(100-10*i)
        csv_dict['Algorithm'].append('Mine')

# Save the data for R
with open('timecsv.csv', mode='w', newline='') as f:
    w = csv.writer(f)
    w.writerow(csv_dict.keys())
    for i in list(map(list, zip(*list(csv_dict.values())))):
        w.writerow(i)

