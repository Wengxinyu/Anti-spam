# -*- coding: utf-8 -*-
#!/usr/bin/python

import sys
from numpy import *
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import math
import decimal

data_temp = []
num_attribute = 57
num_fold = 10
test_data = [[]]
train_data = [[]]
D = decimal.Decimal


'''precondition data
   return data with z-score'''


def preCondition(num_attribute, data):
    value_attribute = [[] for n in range(num_attribute)]
    for i in range(len(data)):
        '''the last number is the label for each item'''
        for j in range(0, num_attribute):
            value_attribute[j].append(data[i][j+1])
    mu = [mean(a) for a in value_attribute]
    sd = [std(a) for a in value_attribute]

    '''print the mean value and sd'''
    # print(mu)
    # print(sd)

    for i in range(len(data)):
        for j in range(0, num_attribute):#0-56
            '''replace attribute used z-score'''
            data[i][j+1] = (data[i][j+1] - mu[j]) / sd[j]

    # print(data)

    '''returen z-score instead of original data'''
    return data


'''implement k-cross validation'''


def cross_validation(pre_data, index_fold):
    test_line = 0
    train_line = 0
    train_data = [[0] * 59 for i in range(int(len(pre_data)/num_fold * (num_fold-1))+1)]
    test_data = [[0] * 59 for i in range(int(len(pre_data)/num_fold)+1)]

    for num_line in range(len(pre_data)):
        if num_line % 10 == index_fold:
            test_data[test_line] = pre_data[num_line]
            test_line += 1
        else:
            train_data[train_line] = pre_data[num_line]
            train_line += 1

    '''return training set, test set, the number of train, the number of test'''
    return train_data, test_data, train_line, test_line


def logistic_batch_gd(data, theta, learning_rate):
    temp = [0]*58
    for each_data in data:
        h_temp = 0.0
        for j in range(num_attribute + 1):
            h_temp += each_data[j] * theta[j]
        h = sigmoid(h_temp)
        for j in range(num_attribute + 1):
            temp[j] += (each_data[num_attribute+1]-h) * each_data[j]
    for j in range(num_attribute + 1):
        theta[j] += learning_rate * temp[j]


def logistics_regression(each_data, theta):
        hypothesis_temp = 0.0
        for j in range(num_attribute + 1):
            hypothesis_temp += each_data[j] * theta[j]
        hypothesis = sigmoid(hypothesis_temp)
        return (-1.0) * each_data[num_attribute + 1] * math.log(hypothesis) - (1.0-each_data[num_attribute + 1]) * math.log((1.0-hypothesis))


def prediction(each_test, theta):
    prediction_label = 0.0
    for j in range(num_attribute + 1):
        prediction_label += each_test[j] * theta[j]
    return prediction_label


def sigmoid(x):
    if x > 20:
        ep = math.exp(-20)
    else:
        ep = math.exp(-x)
    return 1.0 / (1.0 + ep)


def runAndPrint(num_fold, filename):
    print ("Loading data from file %s" % filename)
    print ("Partition the data into %d fold \n" % num_fold)
    iterations = 10000
    BGD_learning_rate = 0.001
    y_score = []
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    y_true = []
    with open(filename) as infile:
        for line in infile:
            line = line.strip()
            data_temp.append(line)

        # print(lines[0][1])

    data = [[0]*59 for i in range(len(data_temp))]

    # print(data)

    '''change to float'''
    for i in range(len(data_temp)):
        data[i][0] = 1
        for j in range(0, num_attribute + 1):# 0-57
            data[i][j+1] = float(data_temp[i].split(',')[j])
    # print(data)

    pre_data = preCondition(num_attribute, data)

    # print(pre_data)

    # f = open('Logistics_BGD_0.01.csv', 'w')
    fig = plt.figure()
    for k in range(10):

        train_set, test_set, num_train, num_test = cross_validation(pre_data, k)
        random.shuffle(train_set)
        random.shuffle(test_set)
        theta = [0.0] * (num_attribute + 1 )

        # print(train_set[1][0])

        ''' stochastic gradient descent'''
        J = 0
        for epoch in range(iterations):

                error = 0
                old_J = J
                logistic_batch_gd(train_set, theta, BGD_learning_rate)
                for index_data in range(num_train):
                    error += logistics_regression(train_set[index_data], theta)
                J = error / num_train
                print('epoch : %d' % (epoch+1) + '   ' + 'mean_square_error : %f' % J)

                # f.write(str(str(epoch+1)+',' + str(J)))
                # f.write('\n')

                '''threshold of convergence'''

                if abs(J - old_J) < 0.00001:
                    print('theta is : ' + '%s' % theta)
                    print('learning rate is : ' + '%f' % BGD_learning_rate)
                    print('mean square error is : ' + '%f' % J)
                    print('the number of epoch is : ' + '%d' % epoch)
                    break

        for index_test in range(num_test):
            y_score.append(prediction(test_set[index_test], theta))
            y_true.append(test_set[index_test][num_attribute+1])
        fpr[k], tpr[k], thresholds = roc_curve(y_true, y_score)
        roc_auc[k] = auc(fpr[k], tpr[k])
        print(roc_auc[k])
        plt.plot(fpr[k], tpr[k], label= 'ROC curve of fold{0} (area = {1:0.2f})'.format(k, roc_auc[k]))

    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(10)]))
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(10):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])
    mean_tpr /= 10
    fpr["mean"] = all_fpr
    tpr["mean"] = mean_tpr
    roc_auc["mean"] = auc(fpr["mean"], tpr["mean"])
    plt.plot(fpr["mean"], tpr["mean"],label='average ROC curve (area = {0:0.2f})'.format(roc_auc["mean"]),linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    fig.patch.set_facecolor('white')
    plt.show()
    # f.close()


if __name__ == '__main__':
        runAndPrint(num_fold, "spambase.data")