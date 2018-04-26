# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/4/2 20:39
@Function:
"""

import multiprocessing
import pickle
import gensim
from random import shuffle


def parse_song_list_get_sequence(in_line, song_list_sequence):
    """
    获得所有歌单中歌曲的序列，并且歌曲的序列在歌单中的顺序是无关的
    :param in_line:
    :param song_list_sequence:
    :return:
    """
    song_sequence = []
    contents = in_line.split("\t")
    for song_info in contents[1:]:
        try:
            song_id, song_name, song_atrists, song_popularity = song_info.split(":::")
            song_sequence.append(song_id)
        except:
            print("song format error!")
            # print(song_info)
    for i in range(len(song_sequence)):
        # 应为加入歌单的顺序是无关的，所以这里打乱歌单顺序，表示所有的排列都是正常的
        shuffle(song_sequence)
        song_list_sequence.append(song_sequence)


def train_song2vec(in_file, out_file):
    song_list_sequence = []

    with open(in_file, encoding='utf8') as in_f:
        for line in in_f:
            parse_song_list_get_sequence(line, song_list_sequence)
    # 使用word2vec训练模型
    cores = multiprocessing.cpu_count()  # 多进程训练，cpu核数
    print("using all " + str(cores) + "cores")
    print("train song2vec model")
    model = gensim.models.Word2Vec(sentences=song_list_sequence, size=150, min_count=3, window=7, workers=cores-2)
    print("saving model")
    model.save(out_file)


def test():
    model = gensim.models.Word2Vec.load('song2vec.model')
    # playlist_dic = pickle.load(open("playlist.pkl", "rb"))
    song_dic = pickle.load(open('song_dic', 'rb'))
    song_ids = list(song_dic.keys())[1500:2500:50]
    for song in song_ids:
        result = model.most_similar(song)
        print(song, song_dic[song])
        print('similar songs:')

        for sim_song in result:
            print(sim_song, song_dic[sim_song[0]])


if __name__ == '__main__':
    # train_song2vec("all_song_list.csv", "song2vec.model")
    test()
    pass