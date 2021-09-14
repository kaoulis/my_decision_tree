import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
import csv

from MyDecisionTree import MyDecisionTree

data_array = np.genfromtxt("iris.data", dtype='str', delimiter=",")
df = pd.DataFrame(data=data_array)
df[[0, 1, 2, 3]] = df[[0, 1, 2, 3]].apply(pd.to_numeric)

labels = df[4].values
df.drop(4, axis=1, inplace=True)
features = df.values

# Lets use the splitting randomness of skearn decision tree in order to collect some metrics and compare later in R.
csv_dict = {'Accuracy': [], 'Precision': [], 'Recall': [], 'Time': [], 'Depth': [], 'Algorithm': []}
for d in range(6):
    for i in range(20):
        skdc_scores = {'Accuracy': [], 'Precision': [], 'Recall': [], 'Time': []}
        mydc_scores = {'Accuracy': [], 'Precision': [], 'Recall': [], 'Time': []}

        max_depth = d+1
        skdc = DecisionTreeClassifier(max_depth=max_depth) #, splitter='random' https://stackoverflow.com/questions/39158003/confused-about-random-state-in-decision-tree-of-scikit-learn
        mydc = MyDecisionTree(max_depth=max_depth)

        cv = KFold(n_splits=10, shuffle=True)
        for train_index, test_index in cv.split(features):
            x_train, x_test, y_train, y_test = features[train_index], features[test_index], labels[train_index], labels[
                test_index]

            start = time.time()
            skdc.fit(x_train, y_train)
            y_pred_skdc = skdc.predict(x_test)

            skdc_scores['Time'].append(time.time() - start)
            skdc_scores['Accuracy'].append(metrics.accuracy_score(y_test, y_pred_skdc))
            skdc_scores['Precision'].append(metrics.precision_score(y_test, y_pred_skdc, average='macro'))
            skdc_scores['Recall'].append(metrics.recall_score(y_test, y_pred_skdc, average='macro'))

            start = time.time()
            tree = mydc.create_tree(x_train.tolist(), y_train.tolist())

            y_pred_mydc = mydc.predict(x_test.tolist(), tree)
            mydc_scores['Time'].append(time.time() - start)
            mydc_scores['Accuracy'].append(metrics.accuracy_score(y_test, y_pred_mydc))
            mydc_scores['Precision'].append(metrics.precision_score(y_test, y_pred_mydc, average='macro'))
            mydc_scores['Recall'].append(metrics.recall_score(y_test, y_pred_mydc, average='macro'))

        csv_dict['Accuracy'].append(np.mean(skdc_scores['Accuracy']))
        csv_dict['Precision'].append(np.mean(skdc_scores['Precision']))
        csv_dict['Recall'].append(np.mean(skdc_scores['Recall']))
        csv_dict['Time'].append(np.mean(skdc_scores['Time']))
        csv_dict['Depth'].append(max_depth)
        csv_dict['Algorithm'].append('Sklearn')

        csv_dict['Accuracy'].append(np.mean(mydc_scores['Accuracy']))
        csv_dict['Precision'].append(np.mean(mydc_scores['Precision']))
        csv_dict['Recall'].append(np.mean(mydc_scores['Recall']))
        csv_dict['Time'].append(np.mean(mydc_scores['Time']))
        csv_dict['Depth'].append(max_depth)
        csv_dict['Algorithm'].append('Mine')

print(list(map(list, zip(*list(csv_dict.values())))))
with open('mycsvfile.csv', mode='w', newline='') as f:
    w = csv.writer(f)
    w.writerow(csv_dict.keys())
    for i in list(map(list, zip(*list(csv_dict.values())))):
        w.writerow(i)
