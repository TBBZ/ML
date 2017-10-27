'''
Created on Oct 21, 2015

@author: joeyqzhou
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import csv

from DecisionTrees import *


def prepareData(training_data_file_path, testing_data_file_path):
    print ("prepare adult data")
    # To split continues data and have the same split standard, we have to discretize the
    # training data and testing data in one batch. And use the panda.cut to discretize .

    continue_feature_list = [0, 2, 4, 10, 11, 12]  # The index of continues feature
    bins = [10, 12, 8, 12, 12, 12]  # The bins of each continues feature
    data = []
    training_size = 0
    with open(training_data_file_path) as f:
        dataList = f.read().splitlines()
    for datai in dataList:
        training_size += 1
        datai_feature_list = datai.split(", ")
        data.append(np.array(datai_feature_list))

    with open(testing_data_file_path) as f:
        dataList = f.read().splitlines()
    for datai in dataList:
        datai_feature_list = datai.split(", ")
        data.append(np.array(datai_feature_list))
    data = np.array(data)
    discretizedData = discretizeData(data, continue_feature_list, bins)

    # return training data and testing data
    print ("training_size: ",training_size)
    return discretizedData[0:training_size, :], discretizedData[training_size:, :]


# data_of_feature:np.array, the data of a feature
# bin_num: to discretize to how many bins
def discretizeFeature(data_of_feature, bin_num):
    return pd.cut(data_of_feature, bin_num)

    # data: np.ndarray, the training data
    # continue_attr: list
    # bins: the length of each discretized feature
    # To discretize the continues attribute/feature of data


def discretizeData(data, continue_feature_list, bins):
    for feature_i_index in range(len(continue_feature_list)):
        feature = continue_feature_list[feature_i_index]
        data_of_feature_i = np.array([float(rowi) for rowi in data[:, feature]])  # str to float
        discretized_feature_i = discretizeFeature(data_of_feature_i, bins[feature_i_index])
        print (discretized_feature_i)
        data[:, feature] = np.array(discretized_feature_i)  # Use the discretized feature replace the continues feature
    return data


def testing_piece_of_data(apiece_of_data, tree, attributes, target):
    target_index = attributes.index(target)
    true_label = apiece_of_data[target_index]
    while (isinstance(tree, dict)):
        if(not isinstance(tree.keys(), list)):
            return False
        tree_key = tree.keys()[0]
        tree_key_index = attributes.index(tree_key)
        data_value_of_key = apiece_of_data[tree_key_index]
        try:
            if tree[tree_key].has_key(data_value_of_key):
                if isinstance(tree[tree_key][data_value_of_key], dict):
                    tree = tree[tree_key][data_value_of_key]
                else:
                    if tree[tree_key][data_value_of_key] == true_label:
                        return True
                    else:
                        return False
            else:  # To those that decision tree can't predict, we predict them as the majority of all training data
                return '<=50K' == true_label
        except:
            print ("error here")

    return


def testing(testing_data, tree, attributes, target):
    all_count = 0.0
    right_count = 0.0
    data_len = testing_data.shape[0]
    for i in range(data_len):
        all_count += 1
        if testing_piece_of_data(testing_data[i, :], tree, attributes, target):
            right_count += 1

    return right_count / all_count  # precision


def main():
    print ("Begin")
    training_data_file_path = "adult.data"
    testing_data_file_path = "adult.test"
    training_data, testing_data = prepareData(training_data_file_path, testing_data_file_path)

    M = 15  # feature length + 1 (1 is target)
    attributes = []
    depth = 0  # The tree's depth

    # create the feature's name
    for i in range(M):
        attributes.append('#' + str(1 + i))
    target = '#15'
    print("attributes: ",attributes)
    tree = makeTree(training_data, attributes, target, depth)

    print("Finish making trees")
    print (tree)

    print ("Begin To test")

    precision = testing(testing_data, tree, attributes, target)

    print ("Precision: ",precision)


if __name__ == '__main__':
    main()
