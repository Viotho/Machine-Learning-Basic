#coding:utf-8
import numpy as np 
from math import sqrt

def load_data(file_name):
    with open(file_name) as f:  
        feature_data = []
        label_tmp = []
        for line in f.readlines():
            feature_tmp = []
            lines = line.strip().split("\t")
            for i in range(len(lines) - 1):
                feature_tmp.append(float(lines[i]))
            label_tmp.append(int(lines[-1]))      
            feature_data.append(feature_tmp)
    
    m = len(label_tmp)
    n_class = len(set(label_tmp)) 
    
    label_data = np.mat(np.zeros((m, n_class)))
    for i in range(m):
        label_data[i, label_tmp[i]] = 1
    
    return np.mat(feature_data), label_data, n_class

def sig(x):
    return 1.0 / (1 + np.exp(-x))

def partial_sig(x):
    m, n = np.shape(x)
    out = np.mat(np.zeros((m, n)))
    for i in range(m):
        for j in range(n):
            out[i, j] = sig(x[i, j]) * (1 - sig(x[i, j]))
    return out

def hidden_in(feature, w0, b0):
    m = np.shape(feature)[0]
    hidden_in = feature * w0
    for i in range(m):
        hidden_in[i, ] += b0
    return hidden_in

def hidden_out(hidden_in):
    hidden_output = sig(hidden_in)
    return hidden_output;

def predict_in(hidden_out, w1, b1):
    m = np.shape(hidden_out)[0]
    predict_in = hidden_out * w1
    for i in range(m):
        predict_in[i, ] += b1
    return predict_in
    
def predict_out(predict_in):
    result = sig(predict_in)
    return result

def bp_train(feature, label, n_hidden, maxCycle, alpha, n_output):
    m, n = np.shape(feature)

    w0 = np.mat(np.random.rand(n, n_hidden))
    w0 = w0 * (8.0 * sqrt(6) / sqrt(n + n_hidden)) - \
     np.mat(np.ones((n, n_hidden))) * \
      (4.0 * sqrt(6) / sqrt(n + n_hidden))
    b0 = np.mat(np.random.rand(1, n_hidden))
    b0 = b0 * (8.0 * sqrt(6) / sqrt(n + n_hidden)) - \
     np.mat(np.ones((1, n_hidden))) * \
      (4.0 * sqrt(6) / sqrt(n + n_hidden))
    w1 = np.mat(np.random.rand(n_hidden, n_output))
    w1 = w1 * (8.0 * sqrt(6) / sqrt(n_hidden + n_output)) - \
     np.mat(np.ones((n_hidden, n_output))) * \
      (4.0 * sqrt(6) / sqrt(n_hidden + n_output))
    b1 = np.mat(np.random.rand(1, n_output))
    b1 = b1 * (8.0 * sqrt(6) / sqrt(n_hidden + n_output)) - \
     np.mat(np.ones((1, n_output))) * \
      (4.0 * sqrt(6) / sqrt(n_hidden + n_output))
    

    i = 0
    while i <= maxCycle:

        hidden_input = hidden_in(feature, w0, b0)
        hidden_output = hidden_out(hidden_input)
        output_in = predict_in(hidden_output, w1, b1)
        output_out = predict_out(output_in)
        
        delta_output = -np.multiply((label - output_out), partial_sig(output_in))
        delta_hidden = np.multiply((delta_output * w1.T), partial_sig(hidden_input))
        
        w1 = w1 - alpha * (hidden_output.T * delta_output)
        b1 = b1 - alpha * np.sum(delta_output, axis=0) * (1.0 / m)
        w0 = w0 - alpha * (feature.T * delta_hidden)
        b0 = b0 - alpha * np.sum(delta_hidden, axis=0) * (1.0 / m)

        i += 1           
    return w0, w1, b0, b1

def get_cost(cost):
    m,n = np.shape(cost)
    cost_sum = 0.0
    for i in range(m):
        for j in range(n):
            cost_sum += cost[i,j] * cost[i,j]
    return cost_sum / m

def get_predict(feature, w0, w1, b0, b1):
    return predict_out(predict_in(hidden_out(hidden_in(feature, w0, b0)), w1, b1))    

def save_model(w0, w1, b0, b1):
    def write_file(file_name, source):   
        with open(file_name, "w") as f:
            m, n = np.shape(source)
            for i in range(m):
                tmp = []
                for j in range(n):
                    tmp.append(str(source[i, j]))
                f.write("\t".join(tmp) + "\n")
    
    write_file("weight_w0", w0)
    write_file("weight_w1", w1)
    write_file("weight_b0", b0)
    write_file("weight_b1", b1)
    
def err_rate(label, pre):
    m = np.shape(label)[0]
    err = 0.0
    for i in range(m):
        if label[i, 0] != pre[i, 0]:
            err += 1
    rate = err / m
    return rate

def load_test_data(file_name):
    with open(file_name) as f:
        feature_data = []
        for line in f.readlines():
            feature_tmp = []
            lines = line.strip().split("\t")
            for i in range(len(lines)):
                feature_tmp.append(float(lines[i]))        
            feature_data.append(feature_tmp)

    return np.mat(feature_data)

def generate_data():
    data = np.mat(np.zeros((20000, 2)))
    m = np.shape(data)[0]
    x = np.mat(np.random.rand(20000, 2))
    for i in range(m):
        data[i, 0] = x[i, 0] * 9 - 4.5
        data[i, 1] = x[i, 1] * 9 - 4.5

    with open("test_data", "w") as f:
        m,n = np.shape(data)
        for i in range(m):
            tmp =[]
            for j in range(n):
                tmp.append(str(data[i,j]))
            f.write("\t".join(tmp) + "\n")     

def load_model(file_w0, file_w1, file_b0, file_b1):
    
    def get_model(file_name):
        f = open(file_name)
        model = []
        for line in f.readlines():
            lines = line.strip().split("\t")
            model_tmp = []
            for x in lines:
                model_tmp.append(float(x.strip()))
            model.append(model_tmp)
        f.close()
        return np.mat(model)
    
    w0 = get_model(file_w0)
    w1 = get_model(file_w1)
    b0 = get_model(file_b0)
    b1 = get_model(file_b1)

    return w0, w1, b0, b1

def save_predict(file_name, pre):
    with open(file_name, "w") as f:
        m = np.shape(pre)[0]
        result = []
        for i in range(m):
            result.append(str(pre[i, 0]))
        f.write("\n".join(result))

if __name__ == "__main__":
    print("--------- 1.load data ------------")
    feature, label, n_class = load_data("data.txt")
    print("--------- 2.training ------------")
    w0, w1, b0, b1 = bp_train(feature, label, 20, 1000, 0.1, n_class)
    print("--------- 3.save model ------------")
    save_model(w0, w1, b0, b1)
    print("--------- 4.get prediction ------------")
    result = get_predict(feature, w0, w1, b0, b1)
    print("训练准确性为：", (1 - err_rate(np.argmax(label, axis=1), np.argmax(result, axis=1))))

    generate_data()
    print("--------- 1.load data ------------")
    dataTest = load_test_data("test_data")
    print("--------- 2.load model ------------")
    w0, w1, b0, b1 = load_model("weight_w0", "weight_w1", "weight_b0", "weight_b1")
    print("--------- 3.get prediction ------------")
    result = get_predict(dataTest, w0, w1, b0, b1)
    print("--------- 4.save result ------------")
    pre = np.argmax(result, axis=1)
    save_predict("result", pre)