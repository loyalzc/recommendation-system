

## 基于用户评论的推荐
Review based Reconmmendation System
     通过word2vec模型计算评论相似度，并且基于评论的相似度来进行用户打分预测


### json格式数据项含义
* reviewerID - ID of the reviewer, e.g. A2SUAM1J3GNN3B
* asin - ID of the product, e.g. 0000013714
* reviewerName - name of the reviewer
* helpful - helpfulness rating of the review, e.g. 2/3
* reviewText - text of the review
* overall - rating of the product
* summary - summary of the review
* unixReviewTime - time of the review (unix time)
* reviewTime - time of the review (raw)
[Amazon Review Data](http://jmcauley.ucsd.edu/data/amazon/)

