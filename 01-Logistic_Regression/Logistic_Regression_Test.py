import numpy as np 

def sigmoid(x):
    return 1.0 / np.exp(-x)

def load_weights(weights_file):
    with open(weights_file, 'r') as f:
        weights = []
        for line in f.readlines():
            lines = line.strip().split('\t')
            weights_temp = []
            for item in lines:
                weights_temp.append(float(item))
            weights.append(weights_temp)
    return np.mat(weights)

def load_data(file_name, n):
    with open(file_name, 'r') as f:
        feature_data = []
        for line in f.readlines():
            feature_temp = []
            lines = line.strip().split('\t')
            if len(lines) != n:
                continue
            feature_temp.append(1)
            for item in lines:
                feature_temp.append(float(item))
            feature_data.append(feature_temp)
    return np.mat(feature_data)

def predict(data, weights):
    hypothesis = sigmoid(data * weights.T)
    m = np.shape(hypothesis)[0]
    for i in range(m):
        if hypothesis[i, 0] < 0.5:
            hypothesis[i, 0] = 0.5
        else:
            hypothesis[i, 0] = 1.0
    return hypothesis

def save_result(file_name, result):
    m = np.shape(result)[0]
    temp = []
    for i in range(m):
        temp.append(str(result[i, 0]))
    with open(file_name, 'w+') as f:
        f.write('\t'.join(temp))

if __name__ == "__main__":
    weights = load_weights('weights')
    n = np.shape(weights)[1]
    test_data = load_data('test_data', n)
    prediction = predict(test_data, weights)
    save_result('result', prediction)