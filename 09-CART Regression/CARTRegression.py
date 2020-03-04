import pickle
import numpy as np 
import random as rd 

def err_cnt(dataSet):
    data = np.mat(dataSet)
    return np.var(data[:,-1]) * np.shape(data)[0]

def split_tree(data, fea, value):
    set_1 = []
    set_2 = []
    for x in data:
        if x[fea] >= value:
            set_1.append(x)
        else:
            set_2.append(x)
    return (set_1, set_2)

class Node:
    def  __init__(self, fea=-1, value=None, result=None, right=None, left=None):
        self.fea = fea
        self.value = value
        self.result = result
        self.right = right
        self.left = left

def build_tree(data, min_sample, min_err):
    if len(data) <= min_sample:
        return Node(result=leaf(data))

    best_err = err_cnt(data)
    bestCriteria = None
    bestSets = None

    feature_num = len(data[0]) - 1
    for fea in range(0, feature_num):
        feature_values = {}
        for sample in data:
            feature_values[sample[fea]] = 1

        for value in feature_values.keys():
            (set_1, set_2) = split_tree(data, fea, value)
            if len(set_1) < 2 or len(set_2)  < 2:
                continue
            now_err = err_cnt(set_1) + err_cnt(set_2)
            if now_err < best_err and len(set_1) > 0 and len(set_2) > 0:
                best_err = now_err
                bestCriteria = (fea, value)
                bestSets = (set_1, set_2)

        if best_err > min_err:
            right = build_tree(bestSets[0], min_sample, min_err)
            left = build_tree(bestSets[1], min_sample, min_err)
            return Node(fea=bestCriteria[0], value=bestCriteria[1], right=right, left=left)
        else:
            return Node(result=leaf(data))

def leaf(dataSet):
    data = np.mat(dataSet)
    return np.mean(data[:, -1])

def predict(sample, tree):
    if tree.result != None:
        return tree.result
    else:
        val_sample = sample[tree.fea]
        branch = None
        if val_sample >= tree.value:
            branch = tree.right
        else:
            branch = tree.left
        return predict(sample, branch)

def cal_error(data, tree):
    m = len(data)  
    n = len(data[0]) - 1
    err = 0.0
    for i in range(m):
        tmp = []
        for j in range(n):
            tmp.append(data[i][j])
        pre = predict(tmp, tree)
        err += (data[i][-1] - pre) * (data[i][-1] - pre)
    return err / m

def load_data(data_file):
    data = []
    with open(data_file) as f:
        for line in f.readlines():
            sample = []
            lines = line.strip().split("\t")
            for x in lines:
                sample.append(float(x))
            data.append(sample)
    return data

def save_model(regression_tree, result_file):
    with open(result_file, 'w') as f:
        pickle.dump(regression_tree, f)

def load_test_data():
    data_test = []
    for i in range(400):
        tmp = []
        tmp.append(rd.random())
        data_test.append(tmp)
    return data_test

def load_model(tree_file):
    with open(tree_file, 'r') as f:
        regression_tree = pickle.load(f)
    return regression_tree    

def get_prediction(data_test, regression_tree):
    result = []
    for x in data_test:
        result.append(predict(x, regression_tree))
    return result

def save_result(data_test, result, prediction_file):
    with open(prediction_file, "w") as f:
        for i in range(len(result)):
            a = str(data_test[i][0]) + "\t" + str(result[i]) + "\n"
            f.write(a)

if __name__ == "__main__":
    print("----------- 1、load data -------------")
    data = load_data("sine.txt")
    print("----------- 2、build CART ------------")
    regression_tree = build_tree(data, 30, 0.3)
    print("----------- 3、cal err -------------")
    err = cal_error(data, regression_tree)
    print("\t--------- err : ", err)
    print("----------- 4、save result -----------") 
    save_model(regression_tree, "regression_tree")

    print("--------- 1、load data ----------")
    data_test = load_test_data()
    print("--------- 2、load regression tree ---------")
    regression_tree = load_model("regression_tree")
    print("--------- 3、get prediction -----------")
    prediction = get_prediction(data_test, regression_tree)
    print("--------- 4、save result ----------")
    save_result(data_test, prediction, "prediction")