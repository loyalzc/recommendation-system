# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/4/2 17:13
@Function:
"""
import pickle

from surprise import SVD, Reader, NormalPredictor, KNNBaseline
from surprise import Dataset
from surprise import evaluate, print_perf
import pandas as pd


def getData(file_path, sep=',', n_filds=5):
    reader = Reader(line_format='user item rating timestamp', sep=sep)
    data = Dataset.load_from_file(file_path, reader=reader)
    # data.split(n_folds=n_filds)
    data = data.build_full_trainset()
    return data


def getModel(train_data):

    sim_option = {'name': 'pearson_baseline', 'user_based': False}
    algo = KNNBaseline(sim_options=sim_option)
    algo.train(train_data)
    return algo


def test(model):
    # song_dic = pickle.load(open('data/song_dic', 'rb'))
    # id_item = song_dic.keys()[0]
    # print(id_item, song_dic[id_item])
    # iid_inner = model.trainset.to_inner_iid()
    # k_neighbors = model.get_neighbors(iid_inner, k=5)
    # print(k_neighbors)
    # for inner_id in k_neighbors:
    #     raw_id = model.trainset.to_raw_id(inner_id)
    #     print(song_dic[raw_id])
    movies_to_id = {}
    id_to_movie = {}
    with open("data/movielens/ml-1m/movies.dat") as in_f:
        for line in in_f.readlines():
            list = line.split('::')
            movies_to_id[list[1]] = list[0]
            id_to_movie[list[0]] = list[1]

    key = "Toy Story (1995)"
    raw_id = movies_to_id[key]
    model_id = model.trainset.to_inner_iid(raw_id)
    k_neighbors = model.get_neighbors(model_id, k=5)
    for neighbor in k_neighbors:
        ne_raw_id = model.trainset.to_raw_id(neighbor)
        print(id_to_movie[ne_raw_id])


def processData():
    file_path = "data/std_song_list.csv"
    import pandas as pd

    data = pd.read_csv(file_path)
    data = data.dropna()
    data.to_csv("data/std_song_list2.csv", index=False, sep=',')


if __name__ == '__main__':

    # processData()

    file_path = "data/movielens/ml-1m/ratings.dat"
    # i = 0
    # with open(file_path) as in_f:
    #     for line in in_f.readlines():
    #         i += 1
    #         datas = line.split(',')
    #         if len(datas) != 4:
    #             print(i)
    # print('end...')
    #
    data = getData(file_path, sep='::')
    model = getModel(data)
    test(model)