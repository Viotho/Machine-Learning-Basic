import numpy as np 
import pickle 

class SVM:
    def __init__(self, dataSet, labels, C, toler, kernal_option):
        self.train_x = dataSet
        self.train_y = labels
        self.C = C
        self.toler = toler
        self.n_samples = np.shape(dataSet)[0]
        self.alphas = np.mat(np.zeros((self.n_samples, 1)))
        self.b = 0.0
        self.error_tmp = np.mat(np.zeros((self.n_samples, 2)))
        self.kernal_opt = kernal_option
        self.kernl_mat = calc_kernel(self.train_x, self.kernal_opt)

def calc_kernel(train_x, kernal_option):
    m = np.shape(train_x)[0]
    kernal_matrix = np.mat(np.zeros((m, m)))
    for i in range(m):
        kernal_matrix[:, i] = cal_kernel_value(train_x, train_x[i, :], kernal_option)
    return kernal_matrix

def cal_kernel_value(train_x, train_x_i, kernel_option):
    kernel_type = kernel_option[0]
    m = np.shape(train_x)[0]
    kernel_value = np.mat(np.zeros((m, 1)))

    if kernel_type == 'rbf':
        sigma = kernel_option[1]
        if sigma == 0:
            sigma = 1.0
        for i in range(m):
            diff = train_x[i, :] - train_x_i
            kernel_value[i] = np.exp(diff * diff.T / (-2.0 * pow(sigma, 2)))
    else:
        kernel_value = train_x * train_x_i.T

    return kernel_value

def SVM_training(train_x, train_y, C, toler, max_iter, kernel_option = ('rbf', 0.431029)):
    svm = SVM(train_x, train_y, C, toler, kernel_option)
    entireSet = True
    alpha_pairs_changed = 0
    iteration = 0

    while(iteration < max_iter) and (alpha_pairs_changed > 0) or (entireSet):
        alpha_pairs_changed = 0
        if entireSet:
            for x in range(svm.n_samples):
                alpha_pairs_changed += choose_and_update(svm, x)
            iteration += 1
        else:
            bound_samples = []
            for i in range(svm.n_samples):
                if 0 < svm.alphas[i] < svm.C:
                    bound_samples.append(i)
            for x in bound_samples:
                alpha_pairs_changed += choose_and_update(svm, x)
            iteration += 1

        if entireSet:
            entireSet = False
        elif alpha_pairs_changed == 0:
            entireSet = True

    return svm

def choose_and_update(svm, alpha_i):
    error_i = cal_error(svm, alpha_i)
    # KKT Conditions
    if(svm.train_y[alpha_i] * error_i < -svm.toler) and (svm.train_y[alpha_i] < svm.C) or (svm.train_y[alpha_i] * error_i > svm.toler) and (svm.train_y[alpha_i] > 0):
        alpha_j, error_j = select_second_sample_j(svm, alpha_i, error_i)
        alpha_i_old = svm.alphas[alpha_i].copy()
        alpha_j_old = svm.alphas[alpha_j].copy()

        if svm.train_y[alpha_i] != svm.train_y(alpha_j):
            L = max(0, svm.alphas[alpha_j] - svm.alphas[alpha_i])
            H = max(svm.C, svm.alphas[alpha_j] + svm.alphas[alpha_i])
        else:
            L = max(0, svm.alphas[alpha_i] + svm.alphas[alpha_j] - svm.C)
            H = min(svm.C, svm.alphas[alpha_i] + svm.alphas[alpha_j])
        if L == H:
            return 0

        eta = svm.kernl_mat[alpha_i, alpha_i] + svm.kernl_mat[alpha_j, alpha_j] - 2.0 * svm.kernl_mat[alpha_i, alpha_j]
        if eta <= 0:
            return 0

        svm.alphas[alpha_j] += svm.train_y[alpha_j] * (error_i - error_j) / eta

        if svm.alphas[alpha_j] > H:
            svm.alphas[alpha_j] = H
        if svm.alphas[alpha_j] < L:
            svm.alphas[alpha_j] = L

        if abs(alpha_j_old - svm.alphas[alpha_j]) < 0.00001:
            update_error_tmp(svm, alpha_j)
            return 0

        svm.alphas[alpha_i] += svm.train_y[alpha_i] * svm.train_y[alpha_j] * (alpha_j_old - svm.alphas[alpha_j])

        b1 = svm.b - error_i - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) * svm.kernl_mat[alpha_i, alpha_i] - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) * svm.kernl_mat[alpha_i, alpha_j]
        b2 = svm.b - error_j - svm.train_y[alpha_i] * (svm.alphas[alpha_i] - alpha_i_old) * svm.kernl_mat[alpha_i, alpha_j] - svm.train_y[alpha_j] * (svm.alphas[alpha_j] - alpha_j_old) * svm.kernl_mat[alpha_j, alpha_j]

        if (0 < svm.alphas[alpha_i]) and (svm.alphas[alpha_i] < svm.C):
            svm.b = b1
        elif (0 < svm.alphas[alpha_j]) and (svm.alphas[alpha_j] < svm.C):
            svm.b = b2
        else:
            svm.b = (b1 + b2) / 2.0

        update_error_tmp(svm, alpha_i)
        update_error_tmp(svm, alpha_j)

        return 1
    
    else:
        return 0
        
def cal_error(svm, alpha_k):
    output_k= float(np.multiply(svm.alphas, svm.train_y).T * svm.kernl_mat[:alpha_k] + svm.b)
    error_k = output_k - float(svm.train_y[alpha_k])
    return error_k

def select_second_sample_j(svm, alpha_i, error_i):
    svm.error_tmp[alpha_i] = [1, error_i]
    candidateAlphaList = np.nonzero(svm.error_tmp[:, 0].A)[0]
    maxStep = 0
    alpha_j = 0
    error_j = 0
    if len(candidateAlphaList) > 1:
        for alpha_k in candidateAlphaList:
            if alpha_k == alpha_i:
                continue
            error_k = cal_error(svm, alpha_k)
            if abs(error_k - error_i) > maxStep:
                maxStep = abs(error_k - error_i)
                alpha_j = alpha_k
                error_j = error_k

            else:
                alpha_j = alpha_i
                while(alpha_j == alpha_i):
                    alpha_j = int(np.random.uniform(0, svm.n_samples))
                error_j = cal_error(svm, alpha_j)
    return alpha_j, error_j

def update_error_tmp(svm, alpha_k):
    error = cal_error(svm, alpha_k)
    svm.error_tmp[alpha_k] = [1, error]

def svm_predict(svm, test_sample_x):
    kernel_value = cal_kernel_value(svm.train_x, test_sample_x, svm.kernel_opt)
    predict = kernel_value.T * np.multiply(svm.train_y, svm.alphas) + svm.b
    return predict

def cal_accuracy(svm, test_x, test_y):
    n_samples = np.shape(test_x)[0]
    correct = 0.0
    for i in range(n_samples):
        predict=svm_predict(svm, test_x[i, :])
        if np.sign(predict) == np.sign(test_y[i]):
            correct += 1
    accuracy = correct / n_samples
    return accuracy

def save_svm_model(svm_model, model_file):
    with open(model_file, 'w') as f:
        pickle.dump(svm_model, f)

def load_data_libsvm(data_file):
    data = []
    label = []
    with open(data_file, 'r+') as f:
        for line in f.readlines():
            lines = line.strip().split(' ')

            label.append(float(lines[0]))
            index = 0
            tmp = []
            for i in range(1, len(lines)):
                li = lines[i].strip().split(":")
                if int(li[0]) - 1 == index:
                    tmp.append(float(li[1]))
                else:
                    while(int(li[0]) - 1 > index):
                        tmp.append(0)
                        index += 1
                    tmp.append(float(li[1]))
                index += 1
            while len(tmp) < 13:
                tmp.append(0)
            data.append(tmp)
    return np.mat(data), np.mat(label).T

def load_test_data(test_file):
    data = []
    with open(test_file, 'r+') as f:
        for line in f.readlines():
            lines = line.strip().split(' ')
            index = 0
            tmp = []
            for i in range(0, len(lines)):
                li = lines[i].strip().split(":")
                if int(li[0]) - 1 == index:
                    tmp.append(float(li[1]))
                else:
                    while(int(li[0]) - 1 > index):
                        tmp.append(0)
                        index += 1
                    tmp.append(float(li[1]))
                index += 1
            while len(tmp) < 13:
                tmp.append(0)
            data.append(tmp)
        f.close()
    return np.mat(data)

def load_svm_model(svm_model_file):
    with open(svm_model_file, 'r') as f:
        svm_model = pickle.load(f)
    return svm_model

def get_prediction(test_data, svm):
    m = np.shape(test_data)[0]
    prediction = []
    for i in range(m):
        predict = svm_predict(svm, test_data[i, :])
        prediction.append(str(np.sign(predict)[0, 0]))
    return prediction

def save_prediction(result_file, prediction):
    with open(result_file, 'w') as f:
        f.write(" ".join(prediction))


if __name__ == "__main__":
    print("------------ 1、load data --------------")
    dataSet, labels = load_data_libsvm("heart_scale")
    print("------------ 2、training ---------------")
    C = 0.6
    toler = 0.001
    maxIter = 500
    kernel_option = ('rbf', 0.431029)
    svm = SVM(dataSet, labels, C, toler, kernel_option)
    svm_model = svm.SVM_training(dataSet, labels, C, toler, maxIter)
    print("------------ 3、cal accuracy --------------")
    accuracy = svm.cal_accuracy(svm_model, dataSet, labels)  
    print("The training accuracy is: %.3f%%" % (accuracy * 100))
    print("------------ 4、save model ----------------")
    svm.save_svm_model(svm_model, "model_file")


    print("--------- 1.load data ---------")
    test_data = load_test_data("svm_test_data")
    print("--------- 2.load model ----------")
    svm_model = load_svm_model("model_file")
    print("--------- 3.get prediction ---------")
    prediction = get_prediction(test_data, svm_model)
    print("--------- 4.save result ----------")
    save_prediction("result", prediction)