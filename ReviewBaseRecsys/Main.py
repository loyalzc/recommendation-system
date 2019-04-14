# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2019/4/11 16:12
@Function:
"""


import ReviewBaseRecsys.Utils as Utils
import ReviewBaseRecsys.Items as Items
import ReviewBaseRecsys.User as User
import ReviewBaseRecsys.BaseMethod as BaseMethod
import ReviewBaseRecsys.PredictScore as PredictScore
import ReviewBaseRecsys.Metrics as Metrics



if __name__ == '__main__':

    data = Utils.get_all_data()

    BaseMethod.get_baseMethon_res()

    user_items = User.get_user_items()

    Items.get_data_cleaning()
    Items.get_reviewers()
    Items.get_w2v()

    PredictScore.do_exp()

    Metrics.get_our_and_base_res()
