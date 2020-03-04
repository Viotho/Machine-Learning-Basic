import numpy as np 

def gradAscent(dataMat, k, alpha, beta, maxCycles):
    m, n = np.shape(dataMat)
    p = np.mat(np.random.rand((m, k)))
    q = np.mat(np.random.rand((k, n)))
    for step in range(maxCycles):
        for i in range(m):
            for j in range(n):
                if dataMat[i, j] > 0:
                    error = dataMat[i, j]
                    for r in range(k):
                        error = error - p[i, r] * q[r, j]
                    for r in range(k):
                        p[i, r] = p[i, r] + alpha * (2 * error * q[r, j] - beta * p[i, r])
                        q[r, j] = q[r, j] + alpha * (2 * error * p[i, r] - beta * q[r, j])
        loss = 0.0
        for i in range(m):
            for j in range(n):
                if dataMat[i, j]  > 0:
                    error = 0.0
                    for r in range(k):
                        error += p[i, r] * q[r, j]
                    loss = (dataMat[i, j] - error) * (dataMat[i, j])
                    for r in range(k):
                        loss += beta * (p[i, r] * p[i, r] + q[r, j] + q[r, j]) / 2
        if loss < 0.001:
            break

    return p, q

def prediction(dataMatrix, p, q, user):
    n = np.shape(dataMatrix)[1]
    predict = {}
    for j in range(n):
        if dataMatrix[user, j] == 0:
            predict[j] = (p[user, ] * q[:, j])[0, 0]
    return sorted(predict.items(), key=lambda d: d[1], reverse=True)

def top_k(predict, k):
    top_recom = []
    len_result = len(predict)
    if k >= len_result:
        top_recom = predict
    else:
        for i in range(k):
            top_recom.append(predict[i])
    return top_recom

def train(V, r, maxCycles, e):
    m, n = np.shape(V)
    W = np.mat(np.random.rand((m, r)))
    H = np.mat(np.random.rand((r, n)))

    for step in range(maxCycles):
        V_pre = W * H
        E = V - V_pre
        err = 0.0
        for i in range(m):
            for j in range(n):
                err += E[i, j] * E[i, j]

        if err < e:
            break
        a = W.T * V
        b = W.T * W * H
        for i_1 in range(r):
            for j_1 in range(n):
                if b[i_1, j_1] != 0:
                    H[i_1, j_1] = H[i_1, j_1] * a[i_1, j_1] / b[i_1, j_1]

        c = V * H.T
        d = W * H * H.T
        for i_2 in range(m):
            for j_2 in range(n):
                if d[i_2, j_2] != 0:
                    W[i_2, j_2] = W[i_2, j_2] * c[i_2, j_2] / d[i_2, j_2]
    return W, H

def load_data(path):
    data = []
    with open(path, 'r+') as f:
        for line in f.readlines():
            arr = []
            lines = line.strip().split("\t")
            for x in lines:
                if x != "-":
                    arr.append(float(x))
                else:
                    arr.append(float(0))
            data.append(arr)
    return np.mat(data)

def save_file(file_name, source):
    with open(file_name, 'w+') as f:
        m, n = np.shape(source)
        for i in range(m):
            tmp = []
            for j in range(n):
                tmp.append(str(source[i, j]))
            f.write("\t".join(tmp) + "\n")

if __name__ == "__main__":
    print("----------- 1、load data -----------")
    dataMatrix = load_data("data.txt")
    print("----------- 2、training -----------")
    p, q = gradAscent(dataMatrix, 5, 0.0002, 0.02, 5000)
    print("----------- 3、save decompose -----------")
    save_file("p", p)
    save_file("q", q)
    print("----------- 4、prediction -----------")
    predict = prediction(dataMatrix, p, q, 0)
    print("----------- 5、top_k recommendation ------------")
    top_recom = top_k(predict, 2)
    print(top_recom)
    print(p*q)

    print("----------- 1、load data -----------")
    V = load_data("data.txt")
    print("----------- 2、training -----------")    
    W, H = train(V, 5, 10000, 1e-5)
    print("----------- 3、save decompose -----------")
    save_file("W", W)
    save_file("H", H)
    print("----------- 4、prediction -----------")
    predict = prediction(V, W, H, 0)
    print("----------- 5、top_k recommendation ------------")
    top_recom = top_k(predict, 2)
    print(top_recom)
    print(W * H)