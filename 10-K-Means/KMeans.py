import numpy as np 
from random import random

FLOAT_MAX = 1e100

def distance(vecA, vecB):
    dist = (vecA - vecB) * (vecA - vecB).T
    return dist[0, 0]

def randCent(data, k):
    n  = np.shape(data)[1]
    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = np.min(data[:, j])
        rangeJ = np.max(data[:, j]) - minJ
        centroids[:, j] = minJ * np.ones((k, 1)) + np.random.rand(k, 1) * rangeJ
    return centroids

def kmeans(data, k, centroids):
    m, n = np.shape(data)
    subCenter = np.mat(np.zeros((m, 2))) # [Index, Distance]
    change = True
    while change:
        change= False
        for i in range(m):
            minDist = np.inf
            minIndex = 0
            for j in range(k):
                dist = distance(data[i,], centroids[j,])
                if dist < minDist:
                    minDist = dist
                    minIndex = j
            
            if subCenter[i, 0] != minIndex:
                change = True
                subCenter[i, ] = np.mat([minIndex, minDist])

        for j in range(k):
            sum_all = np.mat(np.zeros((1, n)))
            r = 0
            for i in range(m):
                if subCenter[i, 0] == j:
                    sum_all += data[i, ]
                    r += 1
            for z in range(n):
                try:
                    centroids[j, z] = sum_all[0, z] / r
                except:
                    print('r is zero.')
        return centroids, subCenter

def get_centroids(points, k):
    m, n = np.shape(points)
    cluster_center = np.mat(np.zeros((k, n)))
    index = np.random.randint(0, m)
    cluster_center[0, ] = np.copy(points[index, ])
    d = [0.0 for _ in range(m)]

    for i in range(1, k):
        sum_all = 0
        for j in range(m):
            d[j] = nearest(points[j, ], cluster_center[0:i, ])
            sum_all += d[j]
        sum_all *= random()
        for j, di in enumerate(d):
            sum_all -= di
            if sum_all > 0:
                continue
            cluster_center[i] = np.copy(points[j, ])
            break
    return cluster_center

def nearest(point, cluster_center):
    min_dist = FLOAT_MAX
    m = np.shape(cluster_center)[0]
    for i in range(m):
        d = distance(point, cluster_center[i])
    if min_dist > d:
        min_dist = d
    return min_dist

def load_data(file_path):
    data = []
    with open(file_path) as f:
        for line in f.readlines():
            row = []
            lines = line.strip().split("\t")
            for x in lines:
                row.append(float(x))
            data.append(row)
    return np.mat(data)

def save_result(file_name, source):
    m, n = np.shape(source)
    with open(file_name, "w") as f:
        for i in range(m):
            tmp = []
            for j in range(n):
                tmp.append(str(source[i, j]))
            f.write("\t".join(tmp) + "\n")

if __name__ == "__main__":
    # KMeans Algorithm
    k = 4
    file_path = "data.txt"
    print("---------- 1.load data ------------")
    data = load_data(file_path)
    print("---------- 2.random center ------------")
    centroids = randCent(data, k)
    print("---------- 3.kmeans ------------")
    subCenter = kmeans(data, k, centroids)  
    print("---------- 4.save subCenter ------------")
    save_result("sub", subCenter)
    print("---------- 5.save centroids ------------")
    save_result("center", centroids) 

    # KMeans++ Algorithm
    k = 4
    file_path = "data.txt"
    print("---------- 1.load data ------------")
    data = load_data(file_path)
    print("---------- 2.K-Means++ generate centers ------------")
    centroids = get_centroids(data, k)
    print("---------- 3.kmeans ------------")
    subCenter = kmeans(data, k, centroids)
    print("---------- 4.save subCenter ------------")
    save_result("sub_pp", subCenter)
    print("---------- 5.save centroids ------------")
    save_result("center_pp", centroids)