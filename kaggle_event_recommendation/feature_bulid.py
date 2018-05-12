# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/5/12 15:26
@Function:
"""

import pickle
import numpy as np
import pandas as pd
import scipy.io as sio

class Feature_bulid:

    def __init__(self):
        self.user_index = pickle.load(open(''))
        self.event_index = pickle.load(open(''))
        self.user_event_scores = sio.mmread(open(''))
        self.user_sim_matrix = sio.mmread(open(''))
        self.event_sim_matrix = sio.mmread(open(''))
        self.event_cont_sim = sio.mmread(open(''))
        self.user_friend_num = sio.mmread(open(''))
        self.user_friend_matrix = sio.mmread(open(''))
        self.event_popularity = sio.mmread(open(''))

    def _user_recom(self, user_id, event_id):
        """user based CF 得到event的推荐度"""

        user_index = self.user_index[user_id]
        event_index = self.event_index[event_id]

        event_scores = self.user_event_scores[:, event_index]
        users_sim = self.user_sim_matrix[user_index, :]

        prod = users_sim * event_scores

        try:
            # 所有用户对物品的相似度加权评分 - 用户自己对物品的评分
            return prod[0, 0] - self.user_event_scores[user_index, event_index]
        except IndexError:
            return 0

    def _event_recom(self, user_id, event_id):
        """event based CF 得到event的推荐度"""
        user_index = self.user_index[user_id]
        event_index = self.event_index[event_id]

        user_scores = self.user_event_scores[user_index, :]
        event_psim = self.event_sim_matrix[:, event_index]
        event_csim = self.event_cont_sim[:, event_index]

        pprod = user_scores * event_psim
        cprod = user_scores * event_csim
        pscore = 0
        cscore = 0
        try:
            pscore = pprod[0, 0] - self.user_event_scores[user_index, event_index]
        except IndexError:
            pass
        try:
            cscore = cprod[0, 0] - self.user_event_scores[user_index, event_index]
        except IndexError:
            pass
        return pscore, cscore

    def _user_friend_num(self, user_id):
        """用户朋友数量"""
        if self.user_index.__contains__(user_id):
            user_index = self.user_index[user_id]

            try:
                return self.user_friend_num[0, user_index]
            except IndexError:
                return 0
        else:
            return 0

    def _user_friend_activ(self, user_id):
        """观察user周围的friend的活跃程度 作为特征"""
        user_index = self.user_index[user_id]
        num_users = np.shape(self.user_friend_matrix)[1]

        return (self.user_friend_num[user_index].sum(axis=0) / num_users)[0, 0]

    def _event_poplar(self, event_id):

        event_index = self.event_index[event_id]
        return self.event_popularity[event_index]

    def bulid_feature(self, start=1):
        train_data = pd.read_csv('data/train.csv')
        test_data = pd.read_csv('data/test.csv')

        train_data = pd.concat([train_data, test_data])

        user_recom = []
        event_recom = []
        user_friend_num = []
        user_friend_activ = []
        event_poplar = []
        for user_id, event_id in zip(train_data['user'], train_data['event']):
            user_recom.append(self._user_recom(user_id, event_id))
            event_recom.append(self._event_recom(user_id, event_id))
            user_friend_num.append(self._user_friend_num(user_id))
            user_friend_activ.append(self._user_friend_activ(user_id))
            event_poplar.append(self._event_poplar(event_id))

        train_data['user_recom'] = user_recom
        train_data['event_precom'] = event_recom[:, 0]
        train_data['event_crecom'] = event_recom[:, 1]
        train_data['user_friend_num'] = user_friend_num
        train_data['user_friend_activ'] = user_friend_activ
        train_data['event_poplar'] = event_poplar

        cols = ['invited', 'user_recom', 'event_precom', 'event_crecom', 'user_friend_num', 'user_friend_activ', 'event_poplar']

        train_x = train_data[cols].values
        train_y = train_data['interested'].values

        return train_x, train_y





