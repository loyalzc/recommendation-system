# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/5/10 18:37
@Function:
"""

import scipy.spatial.distance as ssd
import scipy.sparse as ss
from sklearn.preprocessing import normalize
import scipy.io as sio
from kaggle_event_recommendation import data_clean


class Events:
    """
    user-event 相似度
    event-event 相似度
    event_feature_matrix: event基本特征相似度
    event_cont_feture： event col 特征
    event_sim_matrix:  event基本特征相似度矩阵
    event_cont_sim:    event col 特征相似度矩阵
    """
    def __init__(self, user_event_entity, psim=ssd.correlation, csim=ssd.cosine):
        cleaner = data_clean.DataCleaner()
        with open('data/events.csv', 'r') as events_f:
            col_names = events_f.readline().strip().split(',')
            num_event = len(user_event_entity.event_index.keys())
            # event 基本feature
            self.event_feature_matrix = ss.dok_matrix((num_event, 7))
            # event c_ feature
            self.event_cont_feture = ss.dok_matrix((num_event, 100))
            for line in events_f.readlines():
                cols = line.strip().split(',')
                event_id = cols[0]
                if event_id in user_event_entity.event_index.keys():
                    event_index = user_event_entity.event_index[event_id]
                    self.event_feature_matrix[event_index, 0] = cleaner.getJoinedYearMonth(cols[2])
                    self.event_feature_matrix[event_index, 1] = cleaner.getFeatureHash(cols[3])
                    self.event_feature_matrix[event_index, 2] = cleaner.getFeatureHash(cols[4])
                    self.event_feature_matrix[event_index, 3] = cleaner.getFeatureHash(cols[5])
                    self.event_feature_matrix[event_index, 4] = cleaner.getFeatureHash(cols[6])
                    self.event_feature_matrix[event_index, 5] = cleaner.getFloatValue(cols[7])
                    self.event_feature_matrix[event_index, 6] = cleaner.getFloatValue(cols[8])

                    for i in range(9, 100):
                        self.event_cont_feture[event_index, i - 9] = cols[i]

            self.event_feature_matrix = normalize(self.event_feature_matrix, norm='l1', axis=0, copy=False)
            sio.mmwrite('event_feature_matrix', self.event_feature_matrix)

            self.event_cont_feture = normalize(self.event_cont_feture, norm='l1', axis=0, copy=False)
            sio.mmwrite('event_feature_matrix', self.event_cont_feture)

            self.event_sim_matrix = ss.dok_matrix((num_event, num_event))
            self.event_cont_sim = ss.dok_matrix((num_event, num_event))

            for e1, e2 in user_event_entity.unique_event_pairs:
                e1_index = user_event_entity.event_index[e1]
                e2_index = user_event_entity.event_index[e2]
                if (e1_index, e2_index) not in self.event_sim_matrix.keys():
                    event_sim = psim(self.event_feature_matrix.getrow(e1_index).todense(),
                                     self.event_feature_matrix.getrow(e2_index).todense())

                    self.event_sim_matrix[e1_index, e2_index] = event_sim
                    self.event_sim_matrix[e2_index, e1_index] = event_sim
                if (e1_index, e2_index) not in self.event_cont_sim.keys():

                    event_cont_sim = csim(self.event_cont_feture.getrow(e1_index).todense(),
                                          self.event_cont_feture.getrow(e2_index).todense())

                    self.event_cont_sim[e1_index, e2_index] = event_cont_sim
                    self.event_cont_sim[e2_index, e1_index] = event_cont_sim

        sio.mmwrite("event_sim_matrix", self.event_sim_matrix)
        sio.mmwrite("event_cont_sim", self.event_cont_sim)
