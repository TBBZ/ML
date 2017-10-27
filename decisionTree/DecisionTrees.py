'''
Created on 2015.10

@author: joeyqzhou
'''

import numpy as np
import copy
from matplotlib.mlab import entropy


# return the majority of the label
def majority(data, attributes, target):
    target_index = attributes.index(target)

    valFreq = {}
    for i in range(data.shape[0]):

        if data[i, target_index] in valFreq:
            valFreq[data[i, target_index]] += 1
        else:
            valFreq[data[i, target_index]] = 1

    maxLabel = 0
    major = ""
    for label in valFreq.keys():
        if valFreq[label] > maxLabel:
            major = label
            max = valFreq[label]

    return major


def get_entropy_data(data, attributes, target, rows):
    data_len = data.shape[0]
    target_index = attributes.index(target)
    target_list = list([data[i, target_index] for i in range(data_len) if rows[i] == 1])
    target_set = set(target_list)
    len_of_each_target_val = []
    for target_val in target_set:
        len_of_each_target_val.append(target_list.count(target_val))

    entropy_data = 0.0

    for target_count in len_of_each_target_val:
        entropy_data += -target_count * 1.0 / sum(len_of_each_target_val) * np.log(
            target_count * 1.0 / sum(len_of_each_target_val))

    return entropy_data * sum(rows) * 1.0 / len(rows)


def get_expected_entropy_data(data, attributes, attri, target):
    attri_index = attributes.index(attri)
    attri_value_set = set(data[:, attri_index])
    data_len = data.shape[0]
    sum_expected_entropy = 0.0

    for attri_value in attri_value_set:
        attri_selected_rows = np.zeros(data_len)
        for i in range(data_len):
            if data[i, attri_index] == attri_value:
                attri_selected_rows[i] = 1
        sum_expected_entropy += get_entropy_data(data, attributes, target, attri_selected_rows)

    return sum_expected_entropy


def infoGain(data, attributes, attri, target):
    entropy_data = get_entropy_data(data, attributes, target, np.ones(data.shape[0]))
    expected_entropy_data = get_expected_entropy_data(data, attributes, attri, target)
    return entropy_data - expected_entropy_data


# id3
def best_split(data, attributes, target):
    max_info = 0.000001  # Also can be seen as a threshold
    best_attri = ""
    print
    "best_split attributes: ", attributes
    print
    "data_len: ", data.shape[0]
    for attri in attributes:
        if attri != target:
            attri_infoGain = infoGain(data, attributes, attri, target)
            if attri_infoGain > max_info:
                max_info = attri_infoGain
                best_attri = attri

    print
    "max_info_gain: ", attri_infoGain
    print
    "best_attri: ", best_attri

    # if attri_infoGain <= 0.0:
    # a = 1
    return best_attri


# get the possible value of best_attri in the data
def getValue(data, attributes, best_attri):
    best_attri_index = attributes.index(best_attri)
    return set(data[:, best_attri_index])


# get the data that best_attri==val from parent-data
def getExample(data, attributes, best_attri, val):
    best_attri_index = attributes.index(best_attri)
    data_len = data.shape[0]
    subset_data = []
    for i in range(data_len):
        if data[i, best_attri_index] == val:
            subset_data.append(np.concatenate([data[i, 0:best_attri_index], data[i, (best_attri_index + 1):]]))

    return np.array(subset_data)


# data: np.ndarray, training data, each row is a piece of data, each column is a feature
# attributes: list , feature name list
# target: target name
def makeTree(data, attributes, target, depth):
    print
    "depth: ", depth
    depth += 1
    val = [record[attributes.index(target)] for record in data]  # val is the value of target
    label_prediction = majority(data, attributes, target)

    # if data is empty or attributes is empty
    # len(attributes) <= 1, 1 is from the target
    if len(attributes) <= 1:
        return label_prediction
    elif val.count(val[0]) == len(val):
        return val[0]
    else:
        best_attri = best_split(data, attributes, target)
        print
        "best_attri: ", best_attri
        if best_attri == "":
            return label_prediction

        # create a new decision tree
        tree = {best_attri: {}}

        for val in getValue(data, attributes, best_attri):
            examples = getExample(data, attributes, best_attri, val)
            if examples.shape[0] == 0:  # if the data_len ==0, then this is leaf node whose value is the majority
                tree[best_attri][val] = label_prediction
            else:
                newAttr = copy.copy(attributes)
                newAttr.remove(best_attri)
                subTree = makeTree(examples, newAttr, target, depth)
                tree[best_attri][val] = subTree

    return tree
