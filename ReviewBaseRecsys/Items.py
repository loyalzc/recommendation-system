# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2019/4/6 16:18
@Function:
"""
import nltk
import pandas as pd
from gensim import corpora, models
from gensim.corpora import dictionary
from gensim.models import Word2Vec, Doc2Vec
from nltk import RegexpTokenizer, WordNetLemmatizer
from nltk.corpus import stopwords
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from numpy import *
import ReviewBaseRecsys.Utils as utils

import scipy.io as sio
import scipy.sparse as ss

def get_reviewers():
    """
    获取商品的所有评论信息  加入近邻信息
    :return:
    """
    data = utils.get_review_clean_data()
    max_neighbours = utils.max_neighbours
    print('revewers start...')
    for neighbours in max_neighbours:
        print('neighbours', neighbours)
        dict = {}
        dict_count = {}
        for iid, review in zip(data['asin'], data['review_clean']):
            # print(iid, review)
            if not iid in dict:
                dict[iid] = review
                dict_count[iid] = 1
            else:
                if dict_count[iid] > neighbours:
                    continue
                else:
                    dict[iid] += review
                    dict_count[iid] += 1

        iid_review = pd.DataFrame(pd.Series(dict), columns=['review'])
        iid_review = iid_review.reset_index().rename(columns={'index': 'item_id'})
        # print(iid_review.head(5))

        iid_review.to_csv(utils.save_path + "item_review.csv", index=False)
    print('revewers over...')


def get_data_cleaning():
    """
    根据全部的数据 对 评论数据进行清洗
    :return:
    """
    def data_cleaning(data):
        data["review_clean"] = data["reviewText"].str.lower()
        # 分词
        tokenizer = RegexpTokenizer(r'[a-zA-Z]+')
        # print(data.head())
        data["review_clean"] = data["review_clean"].apply(tokenizer.tokenize)

        # print('cleanning...')
        # 去停用词
        stop_words = stopwords.words('english')

        def word_clean_stopword(word_list):
            words = [word for word in word_list if word not in stop_words]
            # print(words)
            return words

        data["review_clean"] = data["review_clean"].apply(word_clean_stopword)

        # 词形还原
        lemmatizer = WordNetLemmatizer()
        def word_reduction(word_list):
            words = [lemmatizer.lemmatize(word) for word in word_list]
            return words

        data["review_clean"] = data["review_clean"].apply(word_reduction)

        # 词干化
        stemmer = nltk.stem.SnowballStemmer('english')

        def word_stemming(word_list):
            words = [stemmer.stem(word) for word in word_list]
            return words

        data["review_clean"] = data["review_clean"].apply(word_stemming)

        return data

    print("data clean start...")
    review_data = utils.get_all_data()
    # print(review_data.head(5))
    cleaned_data = data_cleaning(review_data)
    cleaned_data["review_clean"] = cleaned_data["review_clean"].apply(lambda x: ' '.join(x))
    # print(cleaned_data.head(5))
    cleaned_data.to_csv(utils.save_path + "item_review_clean.csv", index=False)
    print("data clean over...")


def get_w2v():
    """
    获取商品评论的词向量
    :return: 保存文件
    """
    def word2vec_vector(data):
        """word2vec"""
        vec_size = 100
        model = Word2Vec(data['review'], size=vec_size, min_count=1, iter=20, window=10)
        model.save(utils.save_path + "word2vec.model")

        model = Word2Vec.load(utils.save_path + "word2vec.model")
        def get_matr(sents):
            sents_vec = []
            for sent in sents:
                s_len = len(sent)
                vec = np.zeros(vec_size)
                for word in sent:
                    vec += model.wv[word]
                if s_len > 1:
                    vec = vec / s_len
                # print(list(vec))
                sents_vec.append(list(vec))
            return sents_vec

        sent = get_matr(data['review'])
        return sent
    print('word2vec start...')
    max_neighbours = utils.max_neighbours
    for neighbours in max_neighbours:
        print('neighbours', neighbours)
        cleaned_data = pd.read_csv(utils.save_path + "item_review.csv")
        cleaned_data['review'] = cleaned_data['review'].apply(lambda x: x.split(" "))
        w2v_data = word2vec_vector(cleaned_data)

        cleaned_data["vector"] = w2v_data

        cleaned_data.to_csv(utils.save_path + "item_all" + str(neighbours) + ".csv", index=False)
        # del cleaned_data['review_clean']
        del cleaned_data['review']
        # print(review_data.head(5))
        cleaned_data.to_csv(utils.save_path + "item_vec" + str(neighbours) + ".csv", index=False)
    print('word2vec over...')


def get_doc2vec():

    def doc2vec_vector(data):

        words = [[word for word in review] for review in data['review_clean']]
        # print(words)
        dictionary = corpora.Dictionary(words)
        # print(dictionary)
        corpus = [dictionary.doc2bow(text) for text in data['review_clean']]
        tfidf = models.TfidfModel(corpus)
        for i in tfidf[corpus]:
            print(i)

    cleaned_data = pd.read_csv(utils.save_path + "item_review_clean.csv")
    cleaned_data['review_clean'] = cleaned_data['review_clean'].apply(lambda x: x.split(" "))
    doc_data = doc2vec_vector(cleaned_data)
    print("data to vetor over...")

    cleaned_data["vector"] = doc_data
    del cleaned_data['review_clean']
    del cleaned_data['review']
    # print(review_data.head(5))
    cleaned_data.to_csv(utils.save_path + "item_doc2vec.csv", index=False)


def get_sim_matrix():
    """
    获取相似度矩阵，计算量巨大
    :return:
    """
    data = pd.read_csv(utils.save_path + 'item_vec.csv')
    data['vector'] = data['vector'].apply(lambda x: x[1:-1])
    row, col = data.shape
    sim_matrix = ss.dok_matrix((row, row))

    for i in range(row):
        for j in range(i, row):
            sim_matrix[data.iat[i, 0], data.iat[j, 0]] = cosine_similarity(mat(data.iat[i, 1]), mat(data.iat[j, 1]))[0][0]
            print(sim_matrix[data.iat[i, 0], data.iat[j, 0]])
    sio.mmwrite(utils.save_path + "sim_matrix", sim_matrix)

