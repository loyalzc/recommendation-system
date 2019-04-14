# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2019/4/6 14:53
@Function:
"""
import pandas as pd
from surprise import Reader, Dataset
save_path = "D:\python\Data\\"
file_path = "D:\python\Data\\reviews_Cell_Phones_and_Accessories_5.csv"

BASE_METHOD = ['BaselineOnly', 'KNNWithMeans', 'KNNBasic', 'SVDpp', 'SVD', 'NMF', 'CoClustering', 'KNNWithZScore', 'NormalPredictor']

WEIGHTS = [0, 0.2, 0.4, 0.6, 0.8, 1]
# 用户商品列表的数目
max_neighbours = [10]

# 选择的评论个数
max_items = [1, 3, 5, 10, 20, 30, 1000]


def get_all_data():
    """
    获取所有的数据
    :return:
    """

    return pd.read_csv(file_path)


def get_base_data_format():
    """
    获取Surprise标准格式的数据类型， 不含time 字段
    :return:
    """
    data = get_all_data()
    std_format = pd.DataFrame()
    # print(data.head(5))
    # print(data.columns.values)
    std_format['user_id'] = data['reviewerID']
    std_format['item_id'] = data['asin']
    std_format['rating'] = data['overall']
    #
    # std_format.to_csv(save_path + "std_format.csv", index=False)
    reader = Reader(rating_scale=(1, 5))
    #
    # # The columns must correspond to user id, item id and ratings (in that order).
    std_format = Dataset.load_from_df(std_format[['user_id', 'item_id', 'rating']], reader)

    return std_format


def get_std_format():
    """
    获取Surprise标准格式的数据类型， 含time 字段
    :return:
    """
    data = get_all_data()
    std_format = pd.DataFrame()
    std_format['user_id'] = data['reviewerID']
    std_format['item_id'] = data['asin']
    std_format['rating'] = data['overall']
    return std_format


def get_review_clean_data():
    return pd.read_csv(save_path+"item_review_clean.csv")

if __name__ == '__main__':

    # get_all_data()
    # get_time_format()
    # get_review_data()
    get_all_data()
    # get_std_format()
    pass