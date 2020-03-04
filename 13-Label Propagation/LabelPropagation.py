import numpy as np 

def loadData(filePath):
    vector_dict = {}
    edge_dict = {}
    with open(filePath, 'r') as f:
        for line in f.readlines():
            lines = line.strip().split('\t')
            for i in range(2):
                if lines[i] not in vector_dict:
                    vector_dict[lines[i]] = int(lines[i])
                    edge_list = []
                    if len(lines) == 3:
                        edge_list.append(lines[1 - i] + ':' + lines[2])
                    else:
                        edge_list.append(lines[1 - i] + ':' + '1')
                    edge_dict[lines[i]] = edge_list
                else:
                    edge_list = edge_dict[lines[i]]
                    if len(lines) == 3:
                        edge_list.append(lines[1 - i] + ':' + lines[2])
                    else:
                        edge_list.append(lines[1 - i] + ':' + '1')
                    edge_dict[lines[i]] = edge_list
    return vector_dict, edge_dict

def label_propagation(vector_dict, edge_dict):
    while True:
        if (check(vector_dict, edge_dict) == 0):
            for node in vector_dict.keys():
                adjacency_node_list = edge_dict[node]
                vector_dict[node] = get_max_community_label(vector_dict, adjacency_node_list)
        else:
            break
    return vector_dict

def check(vector_dict, edge_dict):
    for node in vector_dict.keys():
        adjacency_node_list = edge_dict[node]
        node_label = vector_dict[node]
        label = get_max_community_label(vector_dict, adjacency_node_list)
        if node_label == label:
            continue
        else:
            return 0
        return 1

def get_max_community_label(vector_dict, adjacency_node_list):
    label_dict = {}
    for node in adjacency_node_list:
        node_id_weight = node.strip().split(':')
        node_id = node_id_weight[0]
        node_weight = int(node_id_weight[1])
        if vector_dict[node_id] not in label_dict:
            label_dict[vector_dict[node_id]] = node_weight
        else:
            label_dict[vector_dict[node_id]] += node_weight
        sort_list = sorted(label_dict.items(), key=lambda d: d[1], reverse=True)
        return sort_list[0][0]

def save_result(file_name, vec_new):
    with open(file_name, "w") as f_result:
        for key in vec_new.keys():
            f_result.write(str(key) + "\t" + str(vec_new[key]) + "\n")

if __name__ == "__main__":
    print("----------1.load data ------------")
    vector_dict, edge_dict = loadData("cd_data.txt")
    print("original community: \n", vector_dict)
    print("----------2.label propagation ------------")
    vec_new = label_propagation(vector_dict, edge_dict)
    print("----------3.save result ------------")
    save_result("result1", vec_new)
    print("final_result:", vec_new)