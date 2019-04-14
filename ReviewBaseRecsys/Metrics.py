# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2019/4/7 3:11
@Function:
"""
from surprise import SVD, evaluate, accuracy, CoClustering
from surprise.model_selection import cross_validate


def test():
    target = [1.5, 2.1, 3.3, -4.7, -2.3, 0.75]
    prediction = [0.5, 1.5, 2.1, -2.2, 0.1, -0.5]

    error = []
    for i in range(len(target)):
        error.append(target[i] - prediction[i])

    print("Errors: ", error)
    print(error)

    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值

    print("Square Error: ", squaredError)
    print("Absolute Value of Error: ", absError)

    print("MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE

    from math import sqrt

    print("RMSE = ", sqrt(sum(squaredError) / len(squaredError)))  # 均方根误差RMSE
    print("MAE = ", sum(absError) / len(absError))  # 平均绝对误差MAE

    targetDeviation = []
    targetMean = sum(target) / len(target)  # target平均值
    for val in target:
        targetDeviation.append((val - targetMean) * (val - targetMean))
    print("Target Variance = ", sum(targetDeviation) / len(targetDeviation))  # 方差


def get_res(label, predict):
    error = []
    for i in range(len(label)):
        error.append(label[i] - predict[i])
    # print("Errors: ", error)

    squaredError = []
    absError = []
    for val in error:
        squaredError.append(val * val)  # target-prediction之差平方
        absError.append(abs(val))  # 误差绝对值

    # print("Square Error: ", squaredError)
    # print("Absolute Value of Error: ", absError)
    # print("MSE = ", sum(squaredError) / len(squaredError))  # 均方误差MSE
    from math import sqrt
    print("RMSE = ", sqrt(sum(squaredError) / len(squaredError)))  # 均方根误差RMSE
    print("MAE = ", sum(absError) / len(absError))  # 平均绝对误差MAE

import ReviewBaseRecsys.Utils as utils
import pandas as pd


def get_data(file_name):
    return pd.read_csv(utils.save_path + file_name)



def get_base_res():
    """
    计算损失 所有 base方法
    :return:
    """

    base_method = utils.BASE_METHOD

    for method in base_method:
        data = get_data("pred_score_" + method + ".csv")
        get_res(data['rating'], data['pred_score'])


def get_ours_res():
    """
    计算基于评论方案的损失
    :return:
    """
    data = get_data("pred_score.csv")
    get_res(data['rating'], data['pred_score'])


def get_ours_items_res():

    max_items = utils.max_items
    for item in max_items:

        print('item:', item)
        data = get_data("pred_score_ours" + str(item) + ".csv")
        get_res(data['rating'], data['pred_score'])


def get_our_and_base_res():
    """
    计算词向量方案和基本方案的加权结果
    :return:
    """
    data_our = get_data("pred_score_ours10.csv")
    data_base = get_data("pred_score_ SVDpp.csv")
    weights = utils.WEIGHTS

    our_score = data_our['pred_score']
    base_score = data_base['pred_score']

    for w in weights:
        print("The weight is:", w)
        pred_score = w * our_score + (1 - w) * base_score

        get_res(data_our['rating'], base_score)
        get_res(data_our['rating'], our_score)
        get_res(data_our['rating'], pred_score)


if __name__ == '__main__':

    # get_base_res()
    # get_ours_res()
    get_ours_items_res()
    # get_our_and_base_res()

