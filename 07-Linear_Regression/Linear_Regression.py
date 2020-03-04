import numpy as np 
import numpy.linalg as linalg

def least_square(feature, label):
    w = linalg.pinv(feature.T * feature) * feature.T * label
    return w

def get_error(feature, label, w):
    return (label - feature * w).T * (label - feature * w) / 2

def first_derivative(feature, label, w):
    m, n = np.shape(feature)
    g = np.mat(np.zeros((n, 1)))
    for i in range(m):
        err = label[i, 0] - feature[i,:] * w
        for j in range(n):
            g[j, 0] -= err * feature[i, j]
    return g

def second_derivative(feature):
    m, n = np.shape(feature)
    for i in range(m):
        G += feature[i, :].T * feature[i, :]
    return G

def get_min_m(feature, label, sigma, delta, d, w, g):
    m = 0
    while True:
        w_new = w + pow(sigma, m) * d
        left_part = get_error(feature, label, w_new)
        right_part =  get_error(feature, label, w) + pow(delta, m) * g.T * d
        if left_part <= right_part:
            break
        else:
            m += 1
    return m

def newton(feature, label, iterMax, sigma, delta):
    m, n = np.shape(feature)
    w = np.mat(np.zeros((m, 1)))
    it = 0
    while it < iterMax:
        g = first_derivative(feature, label, w)
        G = second_derivative(feature)
        d = -linalg.pinv(G) * g
        m = get_min_m(feature, label, sigma, delta, d, w, g)
        w += pow(delta, m) * d
        it = it + 1
    return w

def lwlr(feature, label, k):
    m = np.shape(feature)[0]
    predict = np.zeros(m)
    weights = np.mat(np.eye(m))
    for i in range(m):
        for j in range(m):
            diff = feature[i, :] - feature[j, :]
            weights[j, j] = np.exp(np.sqrt(diff * diff.T) / -2.0 * pow(k, 2))
            xTx = feature.T * (weights * feature)
            ws = linalg.pinv(xTx) * (feature.T * (weights * label))
            predict[i] = feature[i, :] * ws
    return predict

def load_data(file_path):
    f = open(file_path)
    feature = []
    label = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        feature_tmp.append(1)  # x0
        for i in range(len(lines) - 1):
            feature_tmp.append(float(lines[i]))
        feature.append(feature_tmp)
        label.append(float(lines[-1]))
    f.close()
    return np.mat(feature), np.mat(label).T

def save_model(file_name, w):
    f_result = open(file_name, "w")
    m, n = np.shape(w)
    for i in range(m):
        w_tmp = []
        for j in range(n):
            w_tmp.append(str(w[i, j]))
        f_result.write("\t".join(w_tmp) + "\n")
    f_result.close()

def load_test_data(file_path):
    f = open(file_path)
    feature = []
    for line in f.readlines():
        feature_tmp = []
        lines = line.strip().split("\t")
        feature_tmp.append(1)  # x0
        for i in range(len(lines)):
            feature_tmp.append(float(lines[i]))
        feature.append(feature_tmp)
    f.close()
    return np.mat(feature)

def load_model(model_file):
    w = []
    f = open(model_file)
    for line in f.readlines():
        w.append(float(line.strip()))
    f.close()
    return np.mat(w).T

def get_prediction(data, w):
    return data * w

def save_predict(file_name, predict):
    m = np.shape(predict)[0]
    result = []
    for i in range(m):
        result.append(str(predict[i,0]))
    f = open(file_name, "w")
    f.write("\n".join(result))
    f.close()   

if __name__ == "__main__":
    feature, label = load_data("data.txt")
    w_newton = newton(feature, label, 50, 0.1, 0.5)
    save_model("weights", w_newton)

    testData = load_test_data("data_test.txt")
    w = load_model("weights")
    predict = get_prediction(testData, w)
    save_predict("predict_result", predict)