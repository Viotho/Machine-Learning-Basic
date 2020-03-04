import math
import numpy as np 

MinPts = 5

def epsilon(data, MinPts):
    m, n = np.shape(data)
    xMax = np.max(data, 0)
    xMin = np.min(data, 0)
    eps = ((np.prod(xMax - xMin) * MinPts * math.gamma(0.5 * n + 1)) / (m * math.sqrt(math.pi ** n))) ** (1.0 / n)
    return eps

def distance(data):
    m, n = np.shape(data)
    dis = np.mat(np.zeros((m, m)))
    for i in range(m):
        for j in range(i, m):
            tmp = 0
            for k in range(n):
                tmp += (data[i, k] - data[j, k]) * (data[i, k] - data[j, k])
            dis[i, j] = np.sqrt(tmp)
            dis[j, i] = dis[i, j]
    return dis

def dbscan(data, eps, MinPts):
    m = np.shape(data)[0]
    types = np.mat(np.zeros((1, m)))
    sub_class = np.mat(np.zeros((1, m)))
    dealed = np.mat(np.zeros((m, 1)))
    dis = distance(data)
    number = 1 # Label of Groups

    for i in range(m):
        if dealed[0, i] == 0:
            D = dis[i]
            ind = find_eps(D, eps)
            if len(ind) > 1 and len(ind) < MinPts + 1:
                types[0, i] = 0
                sub_class[0, i] = 0

            if len(ind) == 1:
                types[0, i] = -1
                sub_class[0, i] = -1
                dealed[i, 0] = 1
            
            if len(ind) >= MinPts + 1:
                types[i, 0] = 1
                for x in ind:
                    sub_class[0, x] = number

            while len(ind) > 0:
                dealed[ind[0], 0] = 1
                D = dis[ind[0]]
                tmp = ind[0]
                del ind[0]
                ind_1 = find_eps(D, eps)
                if len(ind_1) > 1:
                    for x1 in ind_1:
                        sub_class[0, x1] = number
                    if len(ind_1) >= MinPts + 1:
                        types[0, tmp] = 1
                    else:
                        types[0, tmp] = 0
                    for j in range(len(ind_1)):
                        if dealed[ind_1[j], 0] == 0:
                            dealed[ind_1[j], 0] = 1
                            ind.append(ind_1[j])
                            sub_class[0, ind_1[j]] = number
            number += 1

            ind_2 = ((sub_class == 0).nonzero())[1]
            for x in ind_2:
                sub_class[0, x] = -1
                types[0, x] = -1
    return types, sub_class

def find_eps(distance_D, eps):
    ind = []
    n = np.shape(distance_D)[1]
    for  j in range(n):
        if distance_D[0, j] <= eps:
            ind.append(j)
    return ind

def load_data(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f.readlines():
            data_tmp = []
            lines = line.strip().split("\t")
            for x in lines:
                data_tmp.append(float(x.strip()))
            data.append(data_tmp)
    return np.mat(data)

def save_result(file_name, source):
    with open(file_name, "w") as f:
        n = np.shape(source)[1]
        tmp = []
        for i in range(n):
            tmp.append(str(source[0, i]))
        f.write("\n".join(tmp))

if __name__ == "__main__":
    print("----------- 1、load data ----------")
    data = load_data("data.txt")
    print("----------- 2、calculate eps ----------")
    eps = epsilon(data, MinPts)
    print("----------- 3、DBSCAN -----------")
    types, sub_class = dbscan(data, eps, MinPts)
    print("----------- 4、save result -----------")
    save_result("types", types)
    save_result("sub_class", sub_class)