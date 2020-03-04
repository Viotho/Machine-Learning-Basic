import numpy as np
import random as rd

def load_weights(weights_path):
    with open(weights_path) as f:
        w = []
        for line in f.readlines():
            w_temp = []
            lines = line.strip().split('\t')
            for item in lines:
                w_temp.append(float(item))
            w.append(w_temp)
        weights = np.mat(w)
        m, n = np.shape(weights)
        return weights, m, n

def load_data(num, m):
    # num:测试样本个数
    # m:测试样本维度
    testDataSet = np.mat(np.ones((num, m)))
    for i in range(num):
        testDataSet[i, 1] = rd.random() * 6 - 3
        testDataSet[i, 2] = rd.random() * 15
    return testDataSet

def predict(test_data, weights):
    h = test_data * weights
    return h.argmax(axis=1)

def save_result(file_name, result):
    with open(file_name, 'w') as f_result:
        m = np.shape(result)[0]
        for i in range(m):
            f_result.write(str(result[i, 0]) + '\n')

if __name__ == "__main__":
    weights, m, n  = load_weights('weights')
    test_data = load_data(4000, m)
    result = predict(test_data, weights)
    save_result('result', result)