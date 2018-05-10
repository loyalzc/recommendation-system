# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/5/10 17:47
@Function:
"""
import numpy as np
import scipy.sparse as ss
import scipy.io as sio
from sklearn.preprocessing import normalize


class Friends:
    """
    找出用户的朋友
    1）朋友多，更容易参加活动
    2）朋友参加活动，很可能跟随参加
    """
    def __init__(self, user_event_entity):
        num_user = len(user_event_entity.user_index.keys())

        self.user_friends_num = np.zeros(num_user)
        self.user_friends_matrix = ss.dok_matrix((num_user, num_user))

        with open('user_friends.csv', 'r') as friends_f:
            friends_f.readline()
            for line in friends_f.readlines():
                cols = line.strip().split(',')
                user = cols[0]

                if user_event_entity.has_key(user):
                    friends = cols[1].split(' ')
                    user_index = user_event_entity.user_index[user]
                    # user 的 friend 数量
                    self.user_friends_num = len(friends)
                    for friend in friends:
                        if user_event_entity.user_index.has_key(friend):
                            friend_index = user_event_entity.user_index[friend]
                            # friend 在所有event的score的平均， 代表了friend对活动的兴趣程度
                            events_for_user = user_event_entity.user_event_scores.getrow(friend_index).todense()
                            avg_score = events_for_user.sum() / np.shape(events_for_user)[1]
                            self.user_friends_matrix[user_index, friend_index] += avg_score
                            self.user_friends_matrix[friend_index, user_index] += avg_score
        # 归一化
        sum_num_friends = self.user_friends_num.sum(axis=0)
        self.user_friends_num = self.user_friends_num / sum_num_friends

        sio.mmwrite('user_friends_num', np.matrix(self.user_friends_num))
        self.user_friends_matrix = normalize(self.user_friends_matrix, norm='l1', axis=0, copy=False)

        sio.mmwrite('user_friends_matrix', self.user_friends_matrix)
