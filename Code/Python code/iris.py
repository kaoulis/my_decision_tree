import time
import matplotlib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sklearn
from sklearn import metrics
from sklearn.model_selection import train_test_split, KFold
from sklearn.tree import DecisionTreeClassifier
import pydotplus
from IPython.display import Image
import csv
from MyDecisionTree import MyDecisionTree
from Node import unique_counts

# Load the data
data_array = np.genfromtxt("iris.data", dtype='str', delimiter=",")
df = pd.DataFrame(data=data_array)
df[[0, 1, 2, 3]] = df[[0, 1, 2, 3]].apply(pd.to_numeric)

# Seperate class labels and features
labels = df[4].values
df.drop(4, axis=1, inplace=True)
features = df.values
# features = StandardScaler().fit_transform(features)

skdc_scores = {'Accuracy': [], 'Precision': [], 'Recall': [], 'Time': []}
mydc_scores = {'Accuracy': [], 'Precision': [], 'Recall': [], 'Time': []}

# Set the threshold
max_depth = 1000
impurity_threshold = 0
skdc = DecisionTreeClassifier(max_depth=max_depth, random_state=7) #, splitter='random' https://stackoverflow.com/questions/39158003/confused-about-random-state-in-decision-tree-of-scikit-learn
mydc = MyDecisionTree(max_depth=max_depth)

# Start cross validation to collect metrics
cv = KFold(n_splits=10, shuffle=True, random_state=5)
for train_index, test_index in cv.split(features):
    x_train, x_test, y_train, y_test = features[train_index], features[test_index], labels[train_index], labels[
        test_index]
    
    # SKLEARN ALGORITHM
    start = time.time()
    skdc.fit(x_train, y_train)
    skdc_scores['Time'].append(time.time() - start)
    y_pred_skdc = skdc.predict(x_test)

    # # Visualize sklearn tree in order to test case and compare with my tree
    # dot_data = sklearn.tree.export_graphviz(skdc, out_file=None,
    #                                         feature_names=['Feature 0', ' Feature 1', ' Feature 2', ' Feature 3'],
    #                                         class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
    #                                         filled=True)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # Image(graph.create_png())
    # graph.write_png("iris.png")

    skdc_scores['Accuracy'].append(metrics.accuracy_score(y_test, y_pred_skdc))
    skdc_scores['Precision'].append(metrics.precision_score(y_test, y_pred_skdc, average='macro'))
    skdc_scores['Recall'].append(metrics.recall_score(y_test, y_pred_skdc, average='macro'))

    # IMPLEMENTED ALGORITHM
    start = time.time()
    tree = mydc.create_tree(x_train.tolist(), y_train.tolist())

    # # Visualize my tree in order to test case and compare with sklearn tree
    # mydc.print_tree(tree)
    mydc_scores['Time'].append(time.time() - start)
    y_pred_mydc = mydc.predict(x_test.tolist(), tree)
    mydc_scores['Accuracy'].append(metrics.accuracy_score(y_test, y_pred_mydc))
    mydc_scores['Precision'].append(metrics.precision_score(y_test, y_pred_mydc, average='macro'))
    mydc_scores['Recall'].append(metrics.recall_score(y_test, y_pred_mydc, average='macro'))

    # print('~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!~!')

print('Sklearn Decision Tree Metrics')
print('Time: ', np.mean(skdc_scores['Time']))
print('Accuracy: ', np.mean(skdc_scores['Accuracy']))
print('Precision: ', np.mean(skdc_scores['Precision']))
print('Recall: ', np.mean(skdc_scores['Recall']))
print('F1-Score: ', 2 * np.mean(skdc_scores['Precision']) * np.mean(skdc_scores['Recall']) /
      (np.mean(skdc_scores['Precision']) + np.mean(skdc_scores['Recall'])))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

print('My Decision Tree Metrics')
print('Time: ', np.mean(mydc_scores['Time']))
print('Accuracy: ', np.mean(mydc_scores['Accuracy']))
print('Precision: ', np.mean(mydc_scores['Precision']))
print('Recall: ', np.mean(mydc_scores['Recall']))
print('F1-Score: ', 2 * np.mean(mydc_scores['Precision']) * np.mean(mydc_scores['Recall']) /
      (np.mean(mydc_scores['Precision']) + np.mean(mydc_scores['Recall'])))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

