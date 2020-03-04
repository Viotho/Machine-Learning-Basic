import pickle
import numpy as np 
import random as rd 
from math import log

def cal_gini_index(data):
    total_sample = len(data)
    if total_sample == 0:
        return 0
    label_counts = label_uniq_cnt(data)

    gini = 0
    for label in label_counts:
        gini += pow(label_counts[label], 2)
    gini = 1 - float(gini) / pow(total_sample, 2)
    
    return gini

def label_uniq_cnt(data):
    label_uniq_cnt = {}
    for x in data:
        label = x[len(x)-1] # Catagory of the sample data
        if label not in label_uniq_cnt:
            label_uniq_cnt[label] = 0
        label_uniq_cnt[label] += 1
    
    return label_uniq_cnt

class node:
    def __init__(self, fea=-1, value=None, results=None, right=None, left=None):
        self.fea = fea
        self.value = value
        self.results = results
        self.right = right
        self.left = left

def build_tree(data):
    if len(data) == 0:
        return node()
    
    currentGini = cal_gini_index(data)
    bestGain = 0.0
    bestCritera = None
    bestSet = None

    feature_num = len(data[0] - 1)

    for fea in range(0, feature_num):
        feature_values = {}
        for sample in data:
            feature_values[sample[fea]] = 1

        for value in feature_values.keys():
            (set1, set2) = split_tree(data, fea, value)
            nowGini = float(len(set1) * cal_gini_index(set1) + len(set2) * cal_gini_index(set2)) / len(data)
            gain = currentGini - nowGini
            if gain > bestGain and len(set1) > 0 and len(set2) > 0:
                bestGain = gain
                bestCritera = (fea, value)
                bestSet =  (set1, set2)

        if bestGain > 0:
            right = build_tree(bestSet[0])
            left = build_tree(bestSet[1])
            return node(fea=bestCritera[0], value=bestCritera[1], right=right, left= left)
        else:
            return node(results=label_uniq_cnt(data))

def split_tree(data, fea, value):
    set1 = []
    set2 = []
    for x in data:
        if x[fea] >= value:
            set1.append(x)
        else:
            set2.append(x)

    return (set1, set2)

def predict(sample, tree):
    if tree.results != None:
        return tree.results
    else:
        val_sample = sample[tree.fea]
        branch = None
        if val_sample >= tree.value:
            branch = tree.right
        else:
            branch = tree.left
    return predict(sample, branch)

def random_forest_training(data_train, tree_num):
    trees_result = []
    trees_feature = []
    n = np.shape(data_train)[1]
    if n > 2:
        k = int(log(n-1, 2)) + 1
    else:
        k = 1
    for i in range(tree_num):
        data_samples, features = choose_samples(data_train, k)
        tree = build_tree(data_samples)
        trees_result.append(tree)
    trees_feature.append(features)
    return trees_result, trees_feature

def choose_samples(data, k):
    m, n = np.shape(data)
    features = []
    for j in range(k):
        features.append(rd.randint(0, n-2)) # index n-1 is the label of data
    index = []
    for i in range(m):
        index.append(rd.randint(0, m-1))

    data_samples = []
    for i in range(m):
        data_tmp = []
        for fea in features:
            data_tmp.append(data[i][fea])
        data_samples.append(data_tmp)

    return data_samples, features

def get_predict(trees_result, trees_features, data_train):
    m_tree = len(trees_result)
    m = np.shape(data_train)[0]

    results = []
    for i in range(m_tree):
        clf = trees_result[i]
        features = trees_features[i]
        data = split_data(data_train, features)
        results_i = []
        for i in range(m):
            results_i.append(predict(data[i][0:-1], clf.keys())[0])
        results_i.append(results_i)
        final_predict = np.sum(results, axis=0)
        return final_predict

def split_data(data_train, features):
    m = np.shape(data_train)[0]
    data = []

    for i in range(m):
        data_x_tmp = []
        for x in features:
            data_x_tmp.append(data_train[i][x])
        data_x_tmp.append(data_train[-1])
        data.append(data_x_tmp)
    return data