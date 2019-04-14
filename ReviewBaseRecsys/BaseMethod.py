# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2019/4/6 14:53
@Function:
"""
from surprise import SVD, evaluate, accuracy, CoClustering, NMF, SVDpp, KNNBasic, KNNWithMeans, BaselineOnly, KNNWithZScore, NormalPredictor
from surprise.model_selection import cross_validate, train_test_split

import ReviewBaseRecsys.Utils as utils
import pandas as pd




def get_baseMethon_res():
    """
    计算全部base方法的评分结果并保存文件
    :param std_format:
    :return:
    """
    base_method = utils.BASE_METHOD
    std_format = utils.get_std_format()
    # base_method = ['SVDpp']
    print('get_baseMethon_res start...')
    for method in base_method:
        print('----------------------this is method:', method, '----------------------------------------')
        data = utils.get_base_data_format()
        # data = data.build_full_trainset()

        trainset, testset = train_test_split(data, test_size=.5)
        object = __import__("surprise")
        algo_method = getattr(object, method)

        algo = algo_method()
        algo.fit(trainset)

        pred_scores = []
        for uid, iid, rating in zip(std_format['user_id'], std_format['item_id'], std_format['rating']):
            pred = algo.predict(uid, iid, r_ui=rating, verbose=True)
            # print(pred[3])
            pred_scores.append(pred[3])

        scores = pd.DataFrame(pd.Series(pred_scores), columns=['pred_score'])
        # scores = scores.reset_index().rename(columns={'index': 'user_id'})
        std_format['pred_score'] = scores
        std_format.to_csv(utils.save_path + "pred_score_" + method + ".csv", index=False)

        predictions = algo.test(testset)
        accuracy.rmse(predictions)
        accuracy.mae(predictions)
    print('get_baseMethon_res over...')


def get_baseMethon_CV_res(std_format):
    """
    计算全部base方法的CV评分结果或者直接 train test结果
    :param std_format:
    :return:
    """
    base_method = utils.BASE_METHOD
    for method in base_method:
        print('----------------------this is method:', method, '----------------------------------------')
        data = utils.get_base_data_format()
        # data = data.build_full_trainset()
        trainset, testset = train_test_split(data, test_size=.5)
        object = __import__("surprise")
        algo_method = getattr(object, method)

        cross_validate(algo_method, data, measures=['RMSE', 'MAE'], cv=2, verbose=True)
        # algo = algo_method()
        # algo.fit(trainset)
        # predictions = algo.test(testset)
        # accuracy.rmse(predictions)
        # accuracy.mae(predictions)


def get_knn_items_res():
    """
    计算近邻数对结果的影响
    :return:
    """
    max_items = utils.max_items
    # base_method = ['SVDpp']
    print('get_baseMethon_res start...')
    for item in max_items:
        print('----------------------this is method:', item, '----------------------------------------')
        data = utils.get_base_data_format()

        trainset, testset = train_test_split(data, test_size=.3)

        algo = KNNBasic(k=item)
        algo.fit(trainset)

        predictions = algo.test(testset)
        accuracy.rmse(predictions)
        accuracy.mae(predictions)


if __name__ == '__main__':
    # get_knn_items_res()
    # get_CoClustering_res(std_format)
    # get_NMF_res(std_format)
    # get_SVD_res(std_format)
    # get_SVDpp_res(std_format)

    get_knn_items_res()
    # get_baseMethon_CV_res(std_format)