# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/5/10 21:07
@Function: event 热度 活跃度
"""
import scipy.sparse as ss
import scipy.io as sio
from sklearn.preprocessing import normalize


class EventAttendees:
    """
    统计活动 参加和不参加的人数
    """

    def __init__(self, user_event_entity):
        num_event = len(user_event_entity.event_index.keys())

        self.event_poplarity = ss.dok_matrix((num_event, 1))

        with open('event_attendees.csv', 'r') as event_att_f:
            event_att_f.readline()
            for line in event_att_f.readlines():
                cols = line.strip().split(',')
                event_id = cols[0]
                if event_id in user_event_entity.event_inde.keys():
                    event_index = user_event_entity.event_index[event_id]
                    # event 流行度  num_yes - num_no
                    self.event_poplarity[event_index, 0] = len(cols[1].split(' ')) - len(cols[4].split(' '))
        self.event_poplarity = normalize(self.event_poplarity, norm='l1', axis=0, copy=False)
        sio.mmwrite('event_popularlity', self.event_poplarity)
