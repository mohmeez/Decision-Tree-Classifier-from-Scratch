# Project 2 for programming for AI: Decision tree classifier for lenses dataset by Mohsin Mohammed
from collections import Counter
import matplotlib.pyplot as plt
import pandas as pd
import operator
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix


class Leaf:
    def __init__(self, labels, value):
        self.labels = Counter(labels)
        self.value = value


class Internal_Node:
    def __init__(self, feature, branches, value):
        self.feature = feature
        self.branches = branches
        self.value = value


def print_tree(node, spacing=""):
    """function that prints our tree."""
    question_dict = {0: "age", 1: "spectacle prescription", 2: "astigmatic", 3: "tear production rate"}
    # Base case: we've reached a leaf
    if isinstance(node, Leaf):
        print(spacing + str(node.labels))
        return

    # Print the question at this node
    print(spacing + "Splitting on " + question_dict[node.feature])

    for i in range(len(node.branches)):
        print(spacing + '--> Branch ' + str(node.branches[i].value) + ':')
        print_tree(node.branches[i], spacing + "  ")


# reading the csv file in
lenses = pd.read_csv('meFile.csv')
print(lenses.head())
# print(lenses)

# renaming the column headings
lenses.columns = ["age", "spectacle prescription", "astigmatic", "tear production rate", "contact lenses"]

print(lenses.head())

# Creaing a new dataframe for the class we are trying to predict
target_labels = lenses[['contact lenses']]

# converting the target column of the dataframe into a list so we can work with the list
myList = target_labels.values.tolist()
final_labels_list = [val for item in myList for val in item]
print("target label list:\n", final_labels_list)

# creating a data frame with all the features that we will use for predicting our target feature
data = lenses[["age", "spectacle prescription", "astigmatic", "tear production rate"]]
print(data.head())

# creating a list containing all the the rows from the data frame: data which contains our features to predict
# the target variable
data_rows = (data.iloc[:]).values.tolist()
print("row values \n", data_rows)
print()


# this is our split functions tha splits our dataset based on the index or column
def split(dataset, labels, column):
    """
    :param dataset: dataset that contains our rows from the data frame
    :param labels: the target column that determines contact lenses
    :param column: is the index or the column feature which will be used to determine the split
    :return: two lists containing the data subset and label subset
    """
    data_subsets = []
    label_subsets = []
    counts = list(set([data[column] for data in dataset]))
    counts.sort()
    for k in counts:
        new_data_subset = []
        new_label_subset = []
        for i in range(len(dataset)):
            if dataset[i][column] == k:
                new_data_subset.append(dataset[i])
                new_label_subset.append(labels[i])
        data_subsets.append(new_data_subset)
        label_subsets.append(new_label_subset)
    return data_subsets, label_subsets


# function to calculate the gini impurity
def gini(dataset):
    impurity = 1
    label_counts = Counter(dataset)
    for label in label_counts:
        prob_of_label = label_counts[label] / len(dataset)
        impurity -= prob_of_label ** 2
    return impurity


# function that calculates the information by calling the gini function
def information_gain(starting_labels, split_labels):
    info_gain = gini(starting_labels)
    for subset in split_labels:
        # Multiply gini(subset) by the correct percentage below
        info_gain -= gini(subset) * len(subset) / len(starting_labels)
    return info_gain


# splitting the data based on the 3rd index or column feature as it gives us the best info gain.
test_split_data, split_labels = split(data_rows, final_labels_list, 3)
print("test split data: \n", test_split_data)
# print(len(split_data))
# print(split_data[0])

print()
# printing the final labels list after the split
print("Split Labels: \n", split_labels)

# We need to loop through all the features to find the best split so we loop through the indices 0 through 3
print()
print("Printing the information gain for all four features:")
for i in range(0, 4):
    split_data, split_labels = split(data_rows, final_labels_list, i)
    print(information_gain(final_labels_list, split_labels))


# function that finds the best feature to split on and provides a split
def find_best_split(dataset, labels):
    """
    :param dataset: dataset that contains our rows from the data frame
    :param labels: the target column that determines contact lenses
    :return: the index of the feature that causes best split and best gain
    """
    best_gain = 0
    best_feature = 0
    for feature in range(len(dataset[0])):
        data_subsets, label_subsets = split(dataset, labels, feature)
        gain = information_gain(labels, label_subsets)
        if gain > best_gain:
            best_gain, best_feature = gain, feature
    return best_feature, best_gain


# testing the code for best feature to split on and printing the best gain
best_feature, best_gain = find_best_split(data_rows, final_labels_list)
print()
print('Best feature is at index:', best_feature, '& ' "Best information gain is:", best_gain)
print()


# function that builds our tree
def build_tree(data, labels, value=""):
    best_feature, best_gain = find_best_split(data, labels)
    if best_gain == 0:
        return Leaf(Counter(labels), value)
    data_subsets, label_subsets = split(data, labels, best_feature)
    branches = []
    for i in range(len(data_subsets)):
        branch = build_tree(data_subsets[i], label_subsets[i], data_subsets[i][0][best_feature])
        branches.append(branch)
    return Internal_Node(best_feature, branches, value)


# printing the Decision Tree:

tree = build_tree(data_rows, final_labels_list)
print("Printing Decision Tree: ")
print_tree(tree)


# This is our tree classifier
def classify(datapoint, tree):
    if isinstance(tree, Leaf):
        return max(tree.labels.items(), key=operator.itemgetter(1))[0]
    value = datapoint[tree.feature]

    for branch in tree.branches:
        if branch.value == value:
            return classify(datapoint, branch)


print()

model_predictions = []
print("Printing our data rows followed by our model's predictions:")
for i in range(len(data_rows)):
    print("feature row: ", data_rows[i])
    predictions = classify(data_rows[i], tree)
    print("Model's prediction: ", predictions)
    model_predictions.append(predictions)
print()
print("model predictions:", model_predictions)
print("Actual values--->:", final_labels_list)
print()
print('*' * 100)

print("This is the confusion matrix using sklearn on the model's predictions and actual output:")

model_confusion_matrix = confusion_matrix(final_labels_list, model_predictions)
print(model_confusion_matrix)

print()

print("Classification report:")
report = classification_report(final_labels_list, model_predictions)
print(report)

