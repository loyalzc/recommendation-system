# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2019/4/7 1:38
@Function:
"""

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

import ReviewBaseRecsys.Utils as utils
from numpy import *


def get_user_items(file_name):
    return pd.read_csv(utils.save_path + file_name)


def get_item_vec(file_name):
    return pd.read_csv(utils.save_path + file_name)


def get_std_format():
    """
    获取surprise标准的数据格式 user_id  item_id  rating  (time)
    :return:
    """
    return pd.read_csv(utils.save_path + "std_format.csv")


def get_user_items_dict(user_items):
    """
    获取user和其所打分item的字典
    :param user_items:
    :return:
    """
    user_dict = {}
    for user_id, item_list in zip(user_items['user_id'], user_items['item_list']):
        user_dict[user_id] = item_list
    return user_dict


def get_item_vec_dict(item_vec):
    """
    获取商品的词向量
    :param item_vec:
    :return:
    """
    item_vec_dict = {}
    for item_id, vec in zip(item_vec['item_id'], item_vec['vector']):
        item_vec_dict[item_id] = vec
    return item_vec_dict


def predict_score(std_format_data, user_items_dict, item_vec_dict, neighbours):
    """
    对商品进行评分
    :param std_format_data: 标准的数据，为了计算损失，需要用户原始的评分信息
    :param user_items_dict:
    :param item_vec_dict:
    :return:
    """
    pred_scores = []
    for user_id, item_id, rating in zip(std_format_data['user_id'], std_format_data['item_id'], std_format_data['rating']):
        item_list = user_items_dict[user_id]
        item_vec = item_vec_dict[item_id]

        ratings = []
        sims = []
        # print(item_list)
        for items in item_list.split(" "):

            items = items.split(":")
            iid = items[0]
            irating = items[1]
            if iid == item_id:
                continue
            ivec = item_vec_dict[iid]
            sim = cosine_similarity(mat(item_vec), mat(ivec))[0][0]

            ratings.append(irating)
            sims.append(sim)
        score = 0.0
        sum_sim = sum(sims)
        for irating, isim in zip(ratings, sims):
            score += float(irating) * isim / sum_sim

        pred_scores.append(score)
        # print(user_id, item_id, rating, score)
    scores = pd.DataFrame(pd.Series(pred_scores), columns=['pred_score'])
    # scores = scores.reset_index().rename(columns={'index': 'user_id'})
    std_format_data['pred_score'] = scores
    std_format_data.to_csv(utils.save_path + "pred_score_ours" + str(neighbours) + ".csv", index=False)


def do_exp():

    max_neighbours = utils.max_neighbours
    max_items = utils.max_items
    print('do_exp start...')
    for item in max_items:
        for neighbours in max_neighbours:
            print('item:', item)
            user_items = get_user_items("user_items" + str(item) + ".csv")
            user_items_dict = get_user_items_dict(user_items)

            item_vec = get_item_vec("item_vec" + str(neighbours) + ".csv")
            item_vec_dict = get_item_vec_dict(item_vec)

            std_format = utils.get_std_format()

            predict_score(std_format, user_items_dict, item_vec_dict, item)
    print('do_exp over...')

if __name__ == '__main__':
    do_exp()