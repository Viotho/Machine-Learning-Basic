import numpy as np 

def sigmoid(x):
    return 1.0 / np.exp(-x)

def lr_train_bgd(feature, label, maxCycle, alpha):
    n = np.shape(feature)[1]
    weights = np.mat(np.ones((n, 1)))
    i = 0
    while i < maxCycle:
        hypothesis = sigmoid(feature * weights)
        err = label - hypothesis
        weights = weights + alpha * feature.T * err
    return weights

def error_rate(h, label):
    m = np.shape(h)[0]
    sum_err = 0.0
    for i in range(m):
        if h[i, 0] > 0 and (1-h[i, 0]) > 0:
            sum_err -= (label[i, 0] * np.log(h[i, 0]) + (1-label[i, 0]) * np.log(1-h[i, 0]))
        else:
            sum_err -= 0
    return sum_err / m

def load_data(file_name):
    feature_data = []
    label_data = []
    with open(file_name, 'r') as f:
        for line in f.readlines():
            feature_temp = []
            label_temp = []
            lines = line.strip().split('\t')
            feature_temp.append(1)
            for i in range(len(lines) - 1):
                feature_temp.append(float(lines[i]))
            label_temp.append(lines[-1])

            feature_data.append(feature_temp)
            label_data.append(label_temp)

    return np.mat(feature_data), np.mat(label_data)

def save_model(file_name, weights):
    m = np.shape(weights)[0]
    with open(file_name, 'w+') as f:
        w_array = []
        for i in range(m):
            w_array.append(str(weights[i, 0]))
        f.write('\t'.join(w_array))

if __name__ == "__main__":
    feature, label = load_data('data.txt')
    weights = lr_train_bgd(feature, label, 1000, 0.01)
    save_model('weights', weights)