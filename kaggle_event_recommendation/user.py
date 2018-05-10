# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/5/10 15:38
@Function:
"""
import scipy.spatial.distance as ssd
import scipy.sparse as ss
from kaggle_event_recommendation import data_clean
from sklearn.preprocessing import normalize
import scipy.io as sio


class Users:
    """
    user/user 相似度矩阵
    """
    def __init__(self, user_event_entity, sim=ssd.correlation):
        cleaner = data_clean.DataCleaner()
        num_users = len(user_event_entity.user_index.keys())

        with open('data/users.csv', 'r') as user_f:
            col_names = user_f.readline().strip().split(',')
            self.user_feature = ss.dok_matrix((num_users, len(col_names) - 1))
            for line in user_f.readlines():
                cols = line.strip().split(',')
                if cols[0] in user_event_entity.user_index.keys():
                    u_index = user_event_entity.user_index[cols[0]]
                    self.user_feature[u_index, 0] = cleaner.getLocaleId(cols[1])
                    self.user_feature[u_index, 1] = cleaner.getBirthYearInt(cols[2])
                    self.user_feature[u_index, 2] = cleaner.getGenderId(cols[3])
                    self.user_feature[u_index, 3] = cleaner.getJoinedYearMonth(cols[4])
                    self.user_feature[u_index, 4] = cleaner.getCountryId(cols[5])
                    self.user_feature[u_index, 5] = cleaner.getTimezoneInt(cols[6])
        self.user_feature = normalize(self.user_feature, norm='l1', axis=0, copy=False)
        sio.mmwrite("user_feature", self.user_feature)

        # 计算用户相似度矩阵
        self.user_sim_matrix = ss.dok_matrix((num_users, num_users))
        for i in range(num_users):
            self.user_sim_matrix[i, i] = 1
        for u1, u2 in user_event_entity.unique_user_pairs:
            u1_index = user_event_entity.user_index[u1]
            u2_index = user_event_entity.user_index[u2]

            if (u1_index, u2_index) not in self.user_sim_matrix.keys():
                u_sim = sim(self.user_feature.getrow(u1_index).todense(), self.user_feature.getrow(u2_index).todense())
                self.user_sim_matrix[u1_index, u2_index] = u_sim
                self.user_sim_matrix[u2_index, u1_index] = u_sim
        sio.mmwrite('user_sim_matrix', self.user_sim_matrix)

