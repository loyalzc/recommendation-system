# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2018/4/3 9:48
@Function:
"""

import numpy as np
import pandas as pd
import tensorflow as tf
import time
from collections import deque
from six import next


np.random.seed(2018)

BATCH_SIZE = 2000  # 一批数据的大小
USER_NUM = 6040  # user
ITEM_NUM = 3952  # item
DIM = 15  # factor维度
EPOCH_MAX = 200  # 最大迭代轮数
DEVICE = "/cpu:0"  # 使用cpu做训练


def read_and_process_data(filename, sep="::"):
    col_names = ["user", "item", "rate", "st"]
    df = pd.read_csv(filename, sep=sep, header=None, names=col_names, engine='python')
    df["user"] -= 1
    df["item"] -= 1
    for col in ("user", "item"):
        df[col] = df[col].astype(np.int32)
    df["rate"] = df["rate"].astype(np.float32)
    return df


class ShuffleDataIterator(object):
    """
    随机生成一个batch的数据
    """

    # 初始化
    def __init__(self, inputs, batch_size=100):
        self.inputs = inputs
        self.batch_size = batch_size
        self.num_cols = len(self.inputs)
        self.num_rows = len(self.inputs[0])
        self.inputs = np.transpose(np.vstack([np.array(self.inputs[i]) for i in range(self.num_cols)]))

    # 总样本量
    def __len__(self):
        return self.num_rows

    def __iter__(self):
        return self

    # 取出下一个batch
    def __next__(self):
        return self.next()

    # 随机生成batch_size个下标，取出对应的样本
    def next(self):
        ids = np.random.randint(0, self.num_rows, (self.batch_size,))
        out = self.inputs[ids, :]
        return [out[:, i] for i in range(self.num_cols)]


class OneEpochDataIterator(ShuffleDataIterator):
    """
    顺序产出一个epoch的数据，在测试中可能会用到
    """

    def __init__(self, inputs, batch_size=10):
        super(OneEpochDataIterator, self).__init__(inputs, batch_size=batch_size)
        if batch_size > 0:
            self.idx_group = np.array_split(np.arange(self.num_rows), np.ceil(self.num_rows / batch_size))
        else:
            self.idx_group = [np.arange(self.num_rows)]
        self.group_id = 0

    def next(self):
        if self.group_id >= len(self.idx_group):
            self.group_id = 0
            raise StopIteration
        out = self.inputs[self.idx_group[self.group_id], :]
        self.group_id += 1
        return [out[:, i] for i in range(self.num_cols)]


def model_SVD(user_batch, item_batch, user_num, item_num, dim=10, device='/cpu:0'):

    with tf.device(device):
        global_bias = tf.get_variable("globa_bias", shape=[])
        w_bias_user = tf.get_variable("w_bias_user", shape=[user_num])
        w_bias_item = tf.get_variable("w_bias_item", shape=[item_num])

        bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name="bias_user")
        bias_item = tf.nn.embedding_lookup(w_bias_item, item_batch, name="bias_item")
        w_user = tf.get_variable("embd_user", shape=[user_num, dim], initializer=tf.truncated_normal_initializer(stddev=0.02))
        w_item = tf.get_variable("embd_item", shape=[item_num, dim], initializer=tf.truncated_normal_initializer(stddev=0.02))
        # user向量与item向量
        embd_user = tf.nn.embedding_lookup(w_user, user_batch, name="embedding_user")
        embd_item = tf.nn.embedding_lookup(w_item, item_batch, name="embedding_item")
    with tf.device(device):
        # 按照实际公式进行计算
        # 先对user向量和item向量求内积
        model = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)
        # 加上几个偏置项
        model = tf.add(model, global_bias)
        model = tf.add(model, bias_user)
        model = tf.add(model, bias_item, name="svd_model")
        # 加上正则化项
        regularizer = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item), name="svd_regularizer")
    return model, regularizer


def optimization(model, regularizer, rate_batch, learning_rate=0.001, reg=0.1, device='/cpu:0'):
    global_step = tf.train.get_global_step()
    with tf.device(device):
        cost_l2 = tf.nn.l2_loss(tf.subtract(model, rate_batch))
        penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="l2")
        cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))
        train_op = tf.train.AdamOptimizer(learning_rate).minimize(cost, global_step=global_step)

    return cost, train_op


def get_data():
    df_data = read_and_process_data('data/movielens/ml-1m/ratings.dat')
    rows = len(df_data)
    df_data = df_data.iloc[np.random.permutation(rows)].reset_index(drop=True)

    split_index = int(rows * 0.9)
    df_train = df_data[0:split_index]
    df_test = df_data[split_index:].reset_index(drop=True)
    print(df_train.shape, df_test.shape)
    return df_train, df_test


def SVD(train, test):
    samples_ebatch = len(train) // BATCH_SIZE
    # 一个batch的训练数据
    iter_train = ShuffleDataIterator([train["user"],
                                      train["item"],
                                      train["rate"]],
                                     batch_size=BATCH_SIZE)
    # 测试数据
    iter_test = OneEpochDataIterator([test["user"],
                                      test["item"],
                                      test["rate"]],
                                     batch_size=-1)

    # user和item batch
    user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
    item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
    rate_batch = tf.placeholder(tf.float32, shape=[None])


    # 构建graph和训练
    model, regularizer = model_SVD(user_batch, item_batch, user_num=USER_NUM, item_num=ITEM_NUM, dim=DIM, device=DEVICE)
    global_step = tf.contrib.framework.get_or_create_global_step()
    _, train_op = optimization(model, regularizer, rate_batch, learning_rate=0.001, reg=0.05, device=DEVICE)

    # 初始化所有变量
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)
        summary_writer = tf.summary.FileWriter(logdir="/tmp/svd/log", graph=sess.graph)
        print("{} {} {} {}".format("epoch", "train_error", "val_error", "elapsed_time"))
        start = time.time()
        errors = deque(maxlen=samples_ebatch)
        for i in range(EPOCH_MAX * samples_ebatch):
            users, items, rates = next(iter_train)

            _, pred_batch = sess.run([train_op, model], feed_dict={user_batch: users,
                                                                   item_batch: items,
                                                                   rate_batch: rates})

            pred_batch = [np.clip(pred_batch, 1.0, 5.0)]
            errors.append(np.power(pred_batch - rates, 2))
            if i % samples_ebatch == 0:
                train_err = np.sqrt(np.mean(errors))
                test_err2 = np.array([])
                for users, items, rates in iter_test:
                    pred_batch = sess.run(model, feed_dict={user_batch: users, item_batch: items})
                    pred_batch = [np.clip(pred_batch, 1.0, 5.0)]
                    test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
                end = time.time()
                test_err = np.sqrt(np.mean(test_err2))
                print("{:3d} {:f} {:f} {:f}(s)".format(i // samples_ebatch, train_err, test_err, end - start))
                start = end


if __name__ == '__main__':
    df_train, df_test = get_data()
    # 完成实际的训练
    SVD(df_train, df_test)
