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


from MyDecisionTree import MyDecisionTree
import sys

sys.setrecursionlimit(15000)
print(sys.getrecursionlimit())

# Load the data
data_array = np.genfromtxt("agaricus-lepiota.data", dtype='str', delimiter=",")
df = pd.DataFrame(data=data_array)

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

# Cross validation and collection of metrics.
skdc_scores = {'Accuracy': [], 'Precision': [], 'Recall': [], 'AUC': [], 'Time': []}
mydc_scores = {'Accuracy': [], 'Precision': [], 'Recall': [], 'AUC': [], 'Time': []}
skdc = DecisionTreeClassifier(random_state=7)
mydc = MyDecisionTree(impurity_threshold=0.001)
cv = KFold(n_splits=10)
for train_index, test_index in cv.split(features):
    x_train, x_test, y_train, y_test = features[train_index], features[test_index], labels[train_index], labels[
        test_index]
    
    # SKLEARN ALGORITHM
    start = time.time()
    skdc.fit(x_train, y_train)
    print(skdc.fit(x_train, y_train).get_depth())
    # # Visualize sklearn tree in order to test case and compare with my tree
    # dot_data = sklearn.tree.export_graphviz(skdc, out_file=None,
    #                                         feature_names=['5_n', '5_f', '8_n'],
    #                                         class_names=['p', 'e'],
    #                                         filled=True)
    # graph = pydotplus.graph_from_dot_data(dot_data)
    # Image(graph.create_png())
    # graph.write_png("mushroom.png")
    skdc_scores['Time'].append(time.time() - start)
    y_pred_dc = skdc.predict(x_test)

    skdc_scores['Accuracy'].append(metrics.accuracy_score(y_test, y_pred_dc))
    skdc_scores['Precision'].append(metrics.precision_score(y_test, y_pred_dc))
    skdc_scores['Recall'].append(metrics.recall_score(y_test, y_pred_dc))

    # IMPLEMENTED ALGORITHM
    start = time.time()
    tree = mydc.create_tree(x_train.tolist(), y_train.tolist())

    # # Visualize my tree in order to test case and compare with sklearn tree
    # mydc.print_tree(tree)

    mydc_scores['Time'].append(time.time() - start)
    y_pred_mydc = mydc.predict(x_test.tolist(), tree)
    mydc_scores['Accuracy'].append(metrics.accuracy_score(y_test, y_pred_mydc))
    mydc_scores['Precision'].append(metrics.precision_score(y_test, y_pred_mydc))
    mydc_scores['Recall'].append(metrics.recall_score(y_test, y_pred_mydc))


print('Time: ', np.mean(skdc_scores['Time']))
print('Accuracy: ', np.mean(skdc_scores['Accuracy']))
print('Precision: ', np.mean(skdc_scores['Precision']))
print('Recall: ', np.mean(skdc_scores['Recall']))
print('F1-Score: ', 2 * np.mean(skdc_scores['Precision']) * np.mean(skdc_scores['Recall']) /
      (np.mean(skdc_scores['Precision']) + np.mean(skdc_scores['Recall'])))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')

print('Time: ', np.mean(mydc_scores['Time']))
print('Accuracy: ', np.mean(mydc_scores['Accuracy']))
print('Precision: ', np.mean(mydc_scores['Precision']))
print('Recall: ', np.mean(mydc_scores['Recall']))
print('F1-Score: ', 2 * np.mean(mydc_scores['Precision']) * np.mean(mydc_scores['Recall']) /
      (np.mean(mydc_scores['Precision']) + np.mean(mydc_scores['Recall'])))
print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
