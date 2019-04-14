# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2019/4/7 1:29
@Function:
"""


import ReviewBaseRecsys.Utils as utils
import pandas as pd

def get_user_items():
    """
    获取用户所购买的商品列表
    :return:
    """
    data = utils.get_all_data()
    max_items = utils.max_items
    print('user item start...')
    for items in max_items:
        print('items', items)
        dict = {}
        dict_count = {}
        user_dict = {}
        for uid, iid, rating in zip(data['reviewerID'], data['asin'], data['overall']):
            if uid not in user_dict:
                # user_dict[uid] = [[iid, rating]]
                user_dict[uid] = [str(iid) + ":" + str(rating)]
                dict_count[uid] = 1
            else:
                if dict_count[uid] >= items:
                    continue
                else:
                    user_dict[uid].append(str(iid) + ":" + str(rating))
                    dict_count[uid] += 1
        for key in user_dict.keys():
            user_dict[key] = ' '.join(user_dict[key])
        users = pd.DataFrame(pd.Series(user_dict), columns=['item_list'])
        users = users.reset_index().rename(columns={'index': 'user_id'})
        # print(users.head(5))

        users.to_csv(utils.save_path + "user_items" + str(items) + ".csv", index=False)

    print('user item over...')

if __name__ == '__main__':
    get_user_items()