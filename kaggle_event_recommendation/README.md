## kaggle  Event Recommendation Engine Challenge

#### page & data : https://www.kaggle.com/c/event-recommendation-engine-challenge

#### 主要计算信息：

- user_similartity_matrix  用户相似度矩阵
- event_similartity_matrix  事件相似度矩阵（基于协同过滤的相似度，基于事件本身的相似度）
- user_event_scores matrix  用户事件评分（参与度）
- user friend matrix  用户朋友关系矩阵（并记录朋友对event 的热度、 是否热衷于参与）
- user friend num  用户朋友数量
- event popularity  event的流行度，考虑event参与人数的多少

#### 主要特征信息：

- user based CF: 基于用户的协同过滤，得到event 的推荐度
- event based CF: 基于event的协同过滤， 得到event 的推荐度
- event based similarity: 基于event 的相似度， 得到event 的推荐度
- user friends number : 用户朋友的数量
- user friends active：用户朋友的活跃度
- event popularity: event的流行度


