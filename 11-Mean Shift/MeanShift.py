import math
import numpy as np 

MIN_DISTANCE = 0.000001

def gaussian_kernal(distance, bandwidth):
    m = np.shape(distance)[0]
    right = np.mat(np.zeros((m, 1)))
    for i in range(m):
        right[i, 0] = (-0.5 * distance[i] * distance[i].T) / (bandwidth * bandwidth)
        right[i, 0] = np.exp(right[i, 0])
    left = 1 / (bandwidth * math.sqrt(2 * math.pi))
    gaussian_val = left * right
    return gaussian_val

def euclidean_dist(pointA, pointB):
    total = (pointA - pointB) * (pointA - pointB).T
    return math.sqrt(total)

def train_mean_shift(points, kernel_bandwidth=2):
    mean_shift_points = np.mat(points)
    max_min_dist = 1
    iteration = 0
    m = np.shape(mean_shift_points)[0]
    need_shift = [True] * m

    while max_min_dist > MIN_DISTANCE:
        max_min_dist = 0 # The longest distance in the current for loop
        iteration += 1
        for i in range(0, m):
            if not need_shift[i]:
                continue
            p_new = mean_shift_points[i]
            p_new_start = p_new
            p_new = shift_point(p_new, points, kernel_bandwidth)
            dist = euclidean_dist(p_new, p_new_start) # Shifted Distance

            if dist > max_min_dist:
                max_min_dist = dist
            if dist < MIN_DISTANCE:
                need_shift[i] = False

            mean_shift_points[i] = p_new
        group = group_points(mean_shift_points)

    return np.mat(points), mean_shift_points, group

def shift_point(point, points, kernel_bandwidth):
    points = np.mat(points)
    m = np.shape(points)[0]
    point_distances = np.mat(np.zeros((m, 1)))
    for i in range(m):
        point_distances[i, 0] = euclidean_dist(point, points[i])
    points_weights = gaussian_kernal(point_distances, kernel_bandwidth) # M×1 Matrix
    all_sum = 0.0
    for i in range(m):
        all_sum += points_weights[i, 0]
    point_shifted = points_weights.T * points / all_sum

    return point_shifted

def group_points(mean_shift_points):
    group_assignment = []
    m, n = np.shape(mean_shift_points)
    index = 0
    index_dict = {}
    for i in range(m):
        item = []
        for j in range(n):
            item.append(str('%5.2f' % mean_shift_points[i, j]))
        item_1 = '_'.join(item)
        if item_1 not in index_dict:
            index_dict[item_1] = index
            index += 1
    for i in range(m):
        item = []
        for j in range(n):
            item.append(str('%5.2f' % mean_shift_points[i, j]))
        item_1 = '_'.join(item)
        group_assignment.append(index_dict[item_1])

    return group_assignment

def load_data(path, feature_num=2):
    data = []
    with open(path) as f:
        for line in f.readlines():
            lines = line.strip().split("\t")
            data_tmp = []
            if len(lines) != feature_num:
                continue
            for i in range(feature_num):
                data_tmp.append(float(lines[i]))
            data.append(data_tmp)
    return data

def save_result(file_name, data):
    with open(file_name, "w") as f:
        m, n = np.shape(data)
        for i in range(m):
            tmp = []
            for j in range(n):
                tmp.append(str(data[i, j]))
            f.write("\t".join(tmp) + "\n")

if __name__ == "__main__":
    print("----------1.load data ------------")
    data = load_data("data", 2)
    print("----------2.training ------------")
    points, shift_points, cluster = train_mean_shift(data, 2)
    print("----------3.1.save sub ------------")
    save_result("sub_1", np.mat(cluster))
    print("----------3.2.save center ------------")
    save_result("center_1", shift_points)    