import numpy as np 
from random import normalvariate

def initialize_v(n, k):
    v = np.mat(np.zeros((n, k)))
    for i in range(n):
        for j in range(k):
            v[i, j] = normalvariate(0, 0.2)
    return v

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def getCost(predict, classLabels):
    m = len(predict)
    error = 0.0
    for i in range(m):
        error -= np.log(sigmoid(predict[i] * classLabels[i]))
    return error

def stocGradDescent(dataMatrix, classLabels, k, max_iter, alpha):
    m, n = np.shape(dataMatrix)
    w = np.zeros((n, 1))
    w0 = 0
    v = initialize_v(n, k)

    for it in range(max_iter):
        for x in range(m):
            inter_1 = dataMatrix[x] * v
            inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * np.multiply(v, v)
            interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2
            p = w0 + w * dataMatrix[x] + interaction
            loss = sigmoid(classLabels[x] * p[0, 0]) - 1

            w0 = w0 - alpha * loss * classLabels[x]
            for i in range(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] - alpha * loss * classLabels[x] * dataMatrix[x, i]

                    for j in range(k):
                        v[i, j]  = v[i, j] - alpha * loss * classLabels * dataMatrix[x, i] * (dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])
    return w0, w, v

def getPrediction(dataMatrix, w0, w, v):
    m = np.shape(dataMatrix)[0]   
    result = []
    for x in range(m):
        
        inter_1 = dataMatrix[x] * v
        inter_2 = np.multiply(dataMatrix[x], dataMatrix[x]) * \
         np.multiply(v, v)  # multiply对应元素相乘
        # 完成交叉项
        interaction = np.sum(np.multiply(inter_1, inter_1) - inter_2) / 2.
        p = w0 + dataMatrix[x] * w + interaction  # 计算预测的输出        
        pre = sigmoid(p[0, 0])        
        result.append(pre)        
    return result

def getAccuracy(predict, classLabels):
    m = len(predict)
    allItem = 0
    error = 0
    for i in range(m):
        allItem += 1
        if float(predict[i]) < 0.5 and classLabels[i] == 1.0:
            error += 1
        elif float(predict[i]) >= 0.5 and classLabels[i] == -1.0:
            error += 1
        else:
            continue
    return float(error) / allItem    

def loadDataSet(data):
    dataMat = []
    labelMat = []   
    with open(data, 'r+') as fr: 
        for line in fr.readlines():
            lines = line.strip().split("\t")
            lineArr = []
            for i in range(len(lines) - 1):
                lineArr.append(float(lines[i]))
            dataMat.append(lineArr)
            labelMat.append(float(lines[-1]) * 2 - 1)  # 转换成{-1,1}
    return dataMat, labelMat

def save_model(file_name, w0, w, v):
    with open(file_name, 'w') as f:
        # 1、保存w0
        f.write(str(w0) + "\n")
        # 2、保存一次项的权重
        w_array = []
        m = np.shape(w)[0]
        for i in range(m):
            w_array.append(str(w[i, 0]))
        f.write("\t".join(w_array) + "\n")
        # 3、保存交叉项的权重
        m1 , n1 = np.shape(v)
        for i in range(m1):
            v_tmp = []
            for j in range(n1):
                v_tmp.append(str(v[i, j]))
            f.write("\t".join(v_tmp) + "\n")

def loadTestDataSet(data):
    dataMat = []
    with open(data) as fr:
        for line in fr.readlines():
            lines = line.strip().split("\t")
            lineArr = [] 
            for i in range(len(lines)):
                lineArr.append(float(lines[i]))
            dataMat.append(lineArr)  
    return dataMat

def loadModel(model_file):
    with open(model_file, 'r+') as f:
        line_index = 0
        w0 = 0.0
        w = []
        v = []
        for line in f.readlines():
            lines = line.strip().split("\t")
            if line_index == 0:  # w0
                w0 = float(lines[0].strip())
            elif line_index == 1:  # w
                for x in lines:
                    w.append(float(x.strip()))
            else:
                v_tmp = []
                for x in lines:
                    v_tmp.append(float(x.strip()))
                v.append(v_tmp)
            line_index += 1     
    return w0, np.mat(w).T, np.mat(v)

def save_result(file_name, result):
    with open(file_name, 'w') as f:
        f.write("\n".join(str(x) for x in result))

if __name__ == "__main__":
    # Training
    print("---------- 1.load data ---------")
    dataTrain, labelTrain = loadDataSet("data_1.txt")
    print("---------- 2.learning ---------")
    w0, w, v = stocGradDescent(np.mat(dataTrain), labelTrain, 3, 10000, 0.01)
    predict_result = getPrediction(np.mat(dataTrain), w0, w, v)
    print("----------training accuracy: %f" % (1 - getAccuracy(predict_result, labelTrain)))
    print("---------- 3.save result ---------")
    save_model("weights", w0, w, v)

    # Testing
    dataTest = loadTestDataSet("test_data.txt")
    w0, w , v = loadModel("weights")
    result = getPrediction(dataTest, w0, w, v)
    save_result("predict_result", result)