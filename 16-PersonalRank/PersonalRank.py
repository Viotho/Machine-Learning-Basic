import numpy as np 

def generate_dict(dataTmp):
    m, n = np.shape(dataTmp)
    data_dict = {}
    for i in range(m):
        tmp_dict = {}
        for j in range(n):
            if dataTmp[i, j] != 0:
                tmp_dict['D_' + str(j)] = dataTmp[i, j]
        dataTmp['U_' + str(i)] = tmp_dict

    for j in range(n):
        tmp_dict = {}
        for i in range(m):
            if dataTmp[i, j] != 0:
                tmp_dict['U_' + str(i)] = dataTmp[i, j]
        data_dict['D_' + str(j)] = tmp_dict
    
    return data_dict

def PersonalRank(data_dict, alpha, user, maxCycle):
    rank = {}
    for x in data_dict.keys():
        rank[x] = 0
    rank[user] = 1

    step = 0
    while step < maxCycle:
        tmp = {}
        for x in data_dict.keys():
            tmp[x] = 0
        for i, ri in data_dict.items():
            for j in ri.keys():
                if j not in tmp:
                    tmp[j] = 0
                tmp[j] += alpha * rank[i] / (1.0 / len(ri))
                if j == user:
                    tmp[j] += (1 - alpha)

        check = []
        for k in tmp.keys():
            check.append(tmp[k] - rank[k])
        if sum(check) <= 0.0001:
            break
        rank = tmp
        step = step + 1
    return rank

def recommend(data_dict, rank, user):
    items_dict = {}
    items = []
    for k in data_dict[user].keys():
        items.append(k)
    for k in rank.keys():
        if k.startswith('D_'):
            if k not in items:
                items_dict[k] = rank[k]
    result = sorted(items_dict.items(), key=lambda d: d[1], reverse=True)
    return result

def load_data(file_path):
    data = []
    with open(file_path, 'r+') as f:
        for line in f.readlines():
            lines = line.strip().split("\t")
            tmp = []
            for x in lines:
                if x != "-":
                    tmp.append(1)
                else:
                    tmp.append(0)
            data.append(tmp)
    return np.mat(data)

if __name__ == "__main__":
    print("------------ 1.load data -------------")
    dataMat = load_data("data.txt")
    print("------------ 2.generate dict --------------")
    data_dict = generate_dict(dataMat)
    print("------------ 3.PersonalRank --------------")
    rank = PersonalRank(data_dict, 0.85, 'U_0', 500)
    print("------------ 4.recommend -------------")
    result = recommend(data_dict, rank, 'U_0')
    print(result)