# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/5/10 21:51
@Function:
"""

from kaggle_event_recommendation.data_clean import DataCleaner
from kaggle_event_recommendation.events import Events
from kaggle_event_recommendation.user import Users
from kaggle_event_recommendation.friends import Friends
from kaggle_event_recommendation.event_attendees import EventAttendees
from kaggle_event_recommendation.user_evevt_entity import User_Event_Entity


def data_prepare():
    """
    计算生成所有的数据，用矩阵或者其他形式存储方便后续提取特征和建模
    """
    print("---1：get user and event index...")
    user_event_entity = User_Event_Entity()
    print("   ---end...")
    print("2：get user similarity matrix...")
    Users(user_event_entity)
    print("   ---end...")
    print("3：get friend matrix...")
    Friends(user_event_entity)
    print("   ---end...")
    print("4：get event similarity matrix...")
    Events(user_event_entity)
    print("   ---end...")
    print("5：get event popularity...")
    EventAttendees(user_event_entity)
    print("   ---end...")


# 运行进行数据准备
data_prepare()