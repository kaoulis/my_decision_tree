# Splitting the records based on a feature value and a feature index
# FOR EXAMPLE : is feature 2 greater than 3.75? -> true_records, false_records
def split_records(split_point, records, feature_index):
    true_records = [[], []]
    false_records = [[], []]
    # Handle numerical features
    if isinstance(split_point, int) or isinstance(split_point, float):
        index = 0
        for rec in records[0]:
            if rec[feature_index] >= split_point:
                true_records[0].append(rec)
                true_records[1].append(records[1][index])
            else:
                false_records[0].append(rec)
                false_records[1].append(records[1][index])
            index += 1
    # Handle categorical features
    else:
        index = 0
        for rec in records[0]:
            if rec[feature_index] == split_point:
                true_records[0].append(rec)
                # del true_records[0][-1][feature_index]
                true_records[1].append(records[1][index])
            else:
                false_records[0].append(rec)
                # del false_records[0][-1][feature_index]
                false_records[1].append(records[1][index])
            index += 1
    return true_records, false_records

# Number of rows.
def n_rows(records):
    return len(records[1])

# Creates dictionary with keys: classes and values: the count of them 
def unique_counts(lst):
    dict = {}
    for i in range(len(lst)):
        if lst[i] not in dict:
            dict[lst[i]] = 0
        dict[lst[i]] += 1
    return dict

# Calculates the gini impurity of the node
def gini(records):
    returned_gini = 1
    for val in unique_counts(records[1]).values():
        returned_gini -= (val / (n_rows(records))) ** 2
    return returned_gini

# Return the most frequent class label. So it predicts
def get_prediction(leaf):
    return max(unique_counts(leaf[1]), key=unique_counts(leaf[1]).get)

# This class demonstrates the node of a tree and contains all the relative information.
class Node:

    def __init__(self, split_point, records, feature_index, threshold):
        self.split_point = split_point  # numeric or str value to split
        self.records = records  # list[features[[],[],[]], labels[0,1,1]]
        self.feature_index = feature_index
        self.true_records, self.false_records = split_records(split_point, records, feature_index)  # sublist of records
        self.impurity = gini(records)
        self.left_node = None
        self.right_node = None
        self.threshold = threshold
    # The impurity decrease. The threshold is the minimum impurity decrease
    def info_gain(self):
        info = self.impurity - \
               (gini(self.true_records) * n_rows(self.true_records) / (n_rows(self.records))) - \
               (gini(self.false_records) * n_rows(self.false_records) / (n_rows(self.records)))
        if info <= self.threshold:
            return 0
        return info
