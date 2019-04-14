# -*- coding: utf-8 -*-
"""
@author: Infaraway
@time: 2019/4/3 22:34
@Function: transform the json data format to csv
"""

import tensorflow
import pandas as pd
import gzip

import ReviewBaseRecsys.Utils as utils
path = "D:\python\Data\\"

def parse(path):
  g = gzip.open(path, 'rb')
  for l in g:
    yield eval(l)

def getDF(path):
  i = 0
  df = {}
  for d in parse(path):
    df[i] = d
    i += 1
  return pd.DataFrame.from_dict(df, orient='index')


df = getDF(path + 'reviews_Cell_Phones_and_Accessories_5.json.gz')
# df['reviewText'].dropna(inplace=True)
df['reviewText'] = df['reviewText'] + "sss"
df.to_csv(path + "reviews_Cell_Phones_and_Accessories_5.csv", index=None)