# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/5/10 14:54
@Function:
"""
from collections import defaultdict
import pickle
import itertools
import scipy.sparse as ss
import scipy.io as sio


class User_Event_Entity:
    """
    user-event 信息：
    user_index ： 用户id
    event_index： 事件id
    user_event_scores： user --> event 感兴趣程度
    unique_user_pairs：  有关系的用户对
    unique_event_pairs： 有关系的事件对
    """
    def __init__(self):
        unique_users = set()
        unique_events = set()
        events_for_user = defaultdict(set)
        users_for_event = defaultdict(set)

        for filename in ['data/train.csv', 'data/test.csv']:
            with open(filename, 'r') as in_f:
                in_f.readline().strip().split(',')
                for line in in_f.readlines():
                    cols = line.strip().split(',')
                    unique_users.add(cols[0])
                    unique_events.add(cols[1])
                    events_for_user[cols[0]].add(cols[1])
                    users_for_event[cols[1]].add(cols[0])
        self.user_event_scores = ss.dok_matrix((len(unique_users), len(unique_events)))
        self.user_index = dict()
        self.event_index = dict()

        for i, user in enumerate(unique_users):
            self.user_index[user] = i
        for i, event in enumerate(unique_events):
            self.event_index[event] = i
        # 构造user to event 评分矩阵
        with open('data/train.csv', 'r') as train_f:
            train_f.readline()
            for line in train_f.readlines():
                cols = line.strip().split(',')
                user_i = self.user_index[cols[0]]
                event_j = self.event_index[cols[1]]
                # 用户评分为 三级： 感兴趣、无所谓、不感兴趣
                self.user_event_scores[user_i, event_j] = int(cols[4]) - int(cols[5])
        sio.mmwrite('prep_data/matrix_user_event_scores', self.user_event_scores)
        # 找到所有关联的user event; 指具有多user 对 evenet 或者多event 对 user
        self.unique_user_pairs = set()
        self.unique_event_pairs = set()
        for event in unique_events:
            users = users_for_event[event]
            if len(users) > 2:
                self.unique_user_pairs.update(itertools.combinations(users, 2))
        for user in unique_users:
            events = events_for_user[user]
            if len(events) > 2:
                self.unique_event_pairs.update(itertools.combinations(events, 2))
        pickle.dump(self.user_index, open("prep_data/user_index.pkl", 'wb'))
        pickle.dump(self.event_index, open("prep_data/event_index.pkl", 'wb'))
