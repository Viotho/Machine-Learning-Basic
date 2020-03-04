#coding:utf-8
import numpy as np 

def GradientAscent(feature_data, label_data, k, maxCycle, alpha):
    m, n = np.shape(feature_data)
    weights = np.mat(np.ones((n, k)))
    i = 0
    while i <= maxCycle:
        err = np.exp(feature_data * weights)
        rowsum = -err.sum(axis=1)
        rowsum = rowsum.repeat(k, axis=1)
        err = err / rowsum
        for x in range(m):
            err[x, label_data[x, 0]] += 1
        weights = weights + (alpha / m) * feature_data.T * err
        i += 1
    return weights

def cost(err, label_data):
    m = np.shape(err)[0]
    sum_cost = 0.0
    for i in range(m):
        if err[i, label_data[i, 0]] / np.sum(err[i,:]) > 0:
            sum_cost -= np.log(err[i, label_data[i, 0]] / np.sum(err[i, :]))
        else:
            sum_cost -= 0
    return sum_cost / m

def load_data(input_file):
    with open(input_file, 'r') as f:
        feature_data = []
        label_data = []
        for line in f.readlines():
            feature_temp = []
            feature_temp.append(1)
            lines = line.strip().split('\t')
            for i in range(len(lines) - 1):
                feature_temp.append(lines[i])
            label_data.append(lines[-1])
            feature_data.append(feature_temp)
    return np.mat(feature_data), np.mat(label_data).T, len(set(label_data))

def save_model(file_name, weights):
    with open(file_name, 'w') as f_w:
        m, n = np.shape(weights)
        for i in range(m):
            w_temp = []
            for j in range(n):
                w_temp.append(str(weights[i, j]))
            f_w.write('\t'.join(w_temp) + '\n')

if __name__ == "__main__":
    input_file = 'SoftInput.txt'
    feature, label, k = load_data(input_file)
    weights = GradientAscent(feature, label, k, 5000, 0.2)
    save_model('weights', weights)