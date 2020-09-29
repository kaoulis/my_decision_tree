from Node import Node, unique_counts, get_prediction

# iterates all the features and rows to find the best split. Keeps track the node with higher info_gain.
def best_split(x_train, y_train, threshold):
    try:
        node = None
        best_gain = 0
        if not x_train:
            return node
        for feature_index in range(len(x_train[0])):
            unique_points = []
            for row in x_train:
                if row[feature_index] not in unique_points:
                    unique_points.append(row[feature_index])
                    temp_node = Node(row[feature_index], [x_train, y_train], feature_index, threshold)
                    # This if else statement keeps the latest best split node.
                    # For testing it was changed to keep the first best split.
                    if temp_node.info_gain() >= best_gain and temp_node.info_gain() != 0:
                        best_gain = temp_node.info_gain()
                        node = Node(row[feature_index], [x_train, y_train], feature_index, threshold)

        return node
    except Exception as error:
        print('ERROR! Diagnostics:')
        print(x_train)
        print(y_train)
        print(error)
        new = ''
        while new == '':
            new = input('Press a button! \n')


# This class contains the decision tree algorithm.
class MyDecisionTree:

    # the parameters as maximum depth threshold and minimum impurity decrease threshold
    def __init__(self, max_depth=100000, impurity_threshold=0):
        self.max_depth = max_depth
        self.impurity_threshold = impurity_threshold
        
    #Creates a tree by fitting the data. Recursive constauction of the tree
    def create_tree(self, x_train, y_train, depth=0):
        tree = best_split(x_train, y_train, self.impurity_threshold)
        if tree is None or depth >= self.max_depth:
            return

        tree.left_node = self.create_tree(tree.true_records[0], tree.true_records[1], depth + 1)
        tree.right_node = self.create_tree(tree.false_records[0], tree.false_records[1], depth + 1)
        return tree
        
    # Iterates the tree by choosing the matched branches and predicting the test set.
    def predict(self, x_test, tree):
        y_pred = []
        for row in x_test:
            # Based on feature_index and splitting point of tree node choose branch
            # Iterate until the chosen branch is None and return the prediction
            tree_climber = tree
            while True:
                index = tree_climber.feature_index
                value = tree_climber.split_point
                # Numerical test case
                if isinstance(value, int) or isinstance(value, float):
                    if row[index] >= value:
                        if tree_climber.left_node is not None:
                            tree_climber = tree_climber.left_node
                            continue
                        else:
                            y_pred.append(get_prediction(tree_climber.true_records))
                            break
                    else:
                        if tree_climber.right_node is not None:
                            tree_climber = tree_climber.right_node
                            continue
                        else:
                            y_pred.append(get_prediction(tree_climber.false_records))
                            break
                # Categorical test case
                else:
                    if value == row[index]:
                        if tree_climber.left_node is not None:
                            tree_climber = tree_climber.left_node
                            continue
                        else:
                            y_pred.append(get_prediction(tree_climber.true_records))
                            break
                    else:
                        if tree_climber.right_node is not None:
                            tree_climber = tree_climber.right_node
                            continue
                        else:
                            y_pred.append(get_prediction(tree_climber.false_records))
                            break

        return y_pred

    #returns the tree depth(debugginf reasons)
    def get_depth(self, node):
        if node is None:
            return 0
        else:
            left_depth = self.get_depth(node.left_node)
            right_depth = self.get_depth(node.right_node)

            if left_depth > right_depth:
                return left_depth + 1
            else:
                return right_depth + 1
    
    # Prints the tree(debugging reasons)
    def print_tree(self, tree, space='', orientation='Root--> ', leaf=None):
        if tree is None:
            print(space, orientation, 'Leaf')
            print((len(space) + len(orientation)) * ' ', '-Samples:', len(leaf[1]))
            print((len(space) + len(orientation)) * ' ', '-value:', unique_counts(leaf[1]))
            print((len(space) + len(orientation)) * ' ', '-class:', max(unique_counts(leaf[1]), key=unique_counts(leaf[1]).get))
            return
        print(space, orientation, 'Is feature ', tree.feature_index, ' <= ', tree.split_point)
        print((len(space) + len(orientation)) * ' ', '-Impurity:', round(tree.impurity, 3))
        print((len(space) + len(orientation)) * ' ', '-Samples:', len(tree.records[1]))
        print((len(space) + len(orientation)) * ' ', '-value:', unique_counts(tree.records[1]))
        print((len(space) + len(orientation)) * ' ', '-class:', get_prediction(tree.records))

        self.print_tree(tree.right_node, space + '    ', 'Left--> ', tree.false_records)
        self.print_tree(tree.left_node, space + '    ', 'Right--> ', tree.true_records)
    

