# 此程序从bike_fence和bike_order中提取单车结点的属性
# 属性包括deposit、instrength、outstrength
# indegree、outdegree、PageRank、betweenness
import os, codecs
import pandas as pd
import numpy as np
import geohash
# 数据存储路径
PATH = '../../data/'

# 读取单车围栏数据
def bike_fence_format(s):
    s = s.replace('[', '').replace(']', '').split(',')
    print(s)
    s = np.array(s).astype(float).reshape(5, -1)
    return s

# 共享单车停车点位（电子围栏）数据
bike_fence = pd.read_csv(PATH + 'gxdc_tcd.csv')
bike_fence['FENCE_LOC'] = bike_fence['FENCE_LOC'].apply(bike_fence_format)
# 根据停车点 计算中心经纬度
bike_fence['FENCE_CENTER'] = bike_fence['FENCE_LOC'].apply(
    lambda x: np.mean(x[:-1, ::-1], 0)
)

# 读取单车订单的数据
bike_order = pd.read_csv(PATH + 'gxdc_dd.csv')

# 处理时间数据
bike_order['UPDATE_TIME'] = pd.to_datetime(bike_order['UPDATE_TIME'])
bike_order['DAY'] = bike_order['UPDATE_TIME'].dt.day.astype(object)
bike_order['DAY'] = bike_order['DAY'].apply(str)

bike_order['HOUR'] = bike_order['UPDATE_TIME'].dt.hour.astype(object)
bike_order['HOUR'] = bike_order['HOUR'].apply(str)
bike_order['HOUR'] = bike_order['HOUR'].str.pad(width=2,side='left',fillchar='0')
# 日期和时间进行拼接
bike_order['DAY_HOUR'] = bike_order['DAY'] + bike_order['HOUR']


# 使用geohash将经纬度转化成字符串，方便匹配单车位置和停车点位置,精度设置为6
bike_order['geohash'] = bike_order.apply(
    lambda x: geohash.encode(x['LATITUDE'], x['LONGITUDE'], precision=6), 
axis=1)
bike_fence['geohash'] = bike_fence['FENCE_CENTER'].apply(
    lambda x: geohash.encode(x[0], x[1], precision=6)
)

# 这段代码计算速度慢，运行一次即可，之后将处理的订单数据进行保存
# 使用保存的order数据
# 匹配订单经纬度和停车点经纬度
# 使用sklearn中的NearestNeighbors，通过设置haversine距离完成最近停车点的计算

# from sklearn.neighbors import NearestNeighbors
# knn = NearestNeighbors(metric = "haversine", n_jobs=-1, algorithm='brute')
# knn.fit(np.stack(bike_fence['FENCE_CENTER'].values))
# # 计算订单中对应的停车点位置
# dist, index = knn.kneighbors(bike_order[['LATITUDE','LONGITUDE']].values[:], n_neighbors=1)
# # 订单中的fence属性是该订单所属的停车点
# bike_order['fence'] = bike_fence.iloc[index.flatten()]['FENCE_ID'].values
# # 保存order数据
# bike_order.to_csv('bike_order(all).csv')

# 读取订单数据进行处理
bike_order = pd.read_csv('bike_order(all).csv')
# 筛选指定日期的订单数据，这里以25号为例
bike_order = bike_order.loc[bike_order.DAY == 24]


# 计算deposit属性
bike_inflow = pd.pivot_table(bike_order.loc[bike_order['LOCK_STATUS'] == 1], 
                   values='LOCK_STATUS', index=['fence'],
                    aggfunc='count', fill_value=0
)
bike_outflow = pd.pivot_table(bike_order.loc[bike_order['LOCK_STATUS'] == 0], 
                   values='LOCK_STATUS', index=['fence'],
                    aggfunc='count', fill_value=0
)
bike_remain = (bike_inflow - bike_outflow).fillna(0)
bike_remain = bike_remain.sum(1)
# 将deposit属性匹配到相应的fence停车点数据中
bike_fence['deposit'] = bike_fence['FENCE_ID'].map(bike_remain).fillna(0)
# 从fence数据中提取需要的信息
bike_feature = bike_fence.loc[:,['FENCE_ID','FENCE_CENTER','geohash','deposit']]


# 计算instrength和outstrength属性
# 将订单数据按照ID和时间排序
bike_order = bike_order.sort_values(['BICYCLE_ID','UPDATE_TIME'])

#创建fence的geohash字典：
geo_dict = {}
for geo in bike_fence['geohash']:
    if geo not in geo_dict.keys():
        geo_dict[geo] = 1
    else:
        geo_dict[geo] += 1

#创建order的geohash字典
geo_dict2 = {}
for geo in bike_order['geohash']:
    if geo not in geo_dict2.keys():
        geo_dict2[geo] = 1
    else:
        geo_dict2[geo] += 1

#创建有向图邻接矩阵
row = len(geo_dict)
col = len(geo_dict)
geo_mat = np.zeros((row,col))

col_name = list(geo_dict.keys())
row_name = col_name
#创建邻接矩阵dataframe格式
geo_m = pd.DataFrame(geo_mat,columns=col_name,index=row_name)

# 遍历order表，计算instrength和outstrength，并建立邻接矩阵表
# 首先要去确定一对点，代表start和end
# 同一个Id的0和1认为是一个有效数据
bike_feature['instrength'] = 0
bike_feature['outstrength'] = 0

item_couple = []
for index,item in bike_order.iterrows():
    if len(item_couple) != 2:
        item_couple.append(item)
    else:
        #取到连续的两个记录时，判断是否符合条件
        start_point = item_couple[0]
        end_point = item_couple[1]
        if start_point.BICYCLE_ID != end_point.BICYCLE_ID:
            # 说明不是同一台自行车的记录，所以删除第一个记录，将第二个记录保存为起始点
            item_couple.clear()
            item_couple.append(end_point)
            continue
        elif start_point.LOCK_STATUS!=0 or end_point.LOCK_STATUS !=1:
            # 说明不是起始点和终止点
            item_couple.clear()
            item_couple.append(end_point)
        else:
            bike_feature.loc[bike_feature.FENCE_ID==start_point.fence,'outstrength'] += 1
            bike_feature.loc[bike_feature.FENCE_ID==end_point.fence,'instrength'] += 1
            # 修改邻接矩阵
            start = bike_feature.loc[bike_feature.FENCE_ID==start_point['fence'], 'geohash'].iloc[0]
            end = bike_feature.loc[bike_feature.FENCE_ID==end_point['fence'], 'geohash'].iloc[0]
            geo_m.loc[start][end] = 1
            # 做完操作之后清除couple
            item_couple.clear()

# 按照geohash统计数据
temp = pd.pivot_table(bike_feature, index = ['geohash'],aggfunc=[np.sum])
res = pd.DataFrame(columns = ['deposit','instrength','outstrength'])
res['deposit'] = temp[('sum',   'deposit')]
res['instrength'] = temp[('sum',  'instrength')]
res['outstrength'] = temp[('sum', 'outstrength')]

# 存储邻接矩阵和当前属性
path = '24'
res.to_csv('res'+path+'.csv')
#geo_m.to_csv('adj'+path+'.csv')


# 计算indegree和outdegree属性
# 删除第一行、列之后导入adj矩阵
geo_m = pd.read_csv('adj'+path+'.csv',header=None)
#每一行的和作为该geohash的出度
outdegree = np.sum(geo_m,axis=1)
#每一列的和作为该geohash的入度数
indegree = np.sum(geo_m,axis=0)
print(outdegree)
#根据geohash匹配indegree和outdegree
res['indegree'] = 0
res['outdegree'] = 0

for ind,value in outdegree.items():
    res.loc[res.index == col_name[ind], 'outdegree'] = value

for ind,value in indegree.items():
    res.loc[res.index == row_name[ind], 'indegree'] = value


# 计算PageRank属性
# 邻接矩阵
adj = pd.read_csv('adj'+path+'.csv',header=None)
adj = np.array(adj,dtype = float)
print(adj)
# 设置确定随机跳转概率的alpha、网页结点数
alpha = 0.9
N = len(adj)
# 初始化随机跳转概率的矩阵
jump = np.full([2,1], [[alpha], [1-alpha]], dtype=float)
# 对邻接矩阵进行归一化
row_sums = adj.sum(axis=1)
# 对每一行求和
row_sums[row_sums == 0] = 0.1
# 防止由于分母出现0而导致的Nan
adj = adj / row_sums[:, np.newaxis] # 除以每行之和的归一化
# 初始的PageRank值，通常是设置所有值为1.0
pr = np.full([1,N], 1, dtype=float)

# PageRank算法本身是采样迭代方式进行的，当最终的取值趋于稳定后结束。
for i in range(0, 40):    
    # 进行点乘，计算Σ(PR(pj)/L(pj))
    pr = np.dot(pr, adj)
    # 转置保存Σ(PR(pj)/L(pj))结果的矩阵，并增加长度为N的列向量，其中每个元素的值为1/N，便于下一步的点乘.。
    pr_jump = np.full([N, 2], [[0, 1/N]])
    pr_jump[:,:-1] = pr.transpose()
    # 进行点乘，计算α(Σ(PR(pj)/L(pj))) + (1-α)/N)
    pr = np.dot(pr_jump, jump)
    # 归一化PageRank得分
    pr = pr.transpose()
    pr = pr / pr.sum()
    print("第",i + 1,"轮" ": ", pr)

df = pd.DataFrame(pr.transpose())
res['pagerank'] = 0
i = 0
for p in pr.transpose():
    #print(p[0])
    res.loc[res.index==col_name[i],'pagerank'] = p[0]
    i += 1
res.to_csv('res'+path+'.csv')

# 根据geohash匹配相应的中心经纬度
#创建geohash字典：
geo_dict3 = {}
for geo in bike_fence['geohash']:
    if geo not in geo_dict3.keys():
        geo_dict3[geo] = 1
    else:
        geo_dict3[geo] += 1
res['fence_center'] = ''
s = []
for geo in geo_dict3.keys():
    for index,item in bike_fence.iterrows():
        #print(item.geohash,item.FENCE_CENTER)
        if geo == item.geohash:
            s.append(item.FENCE_CENTER)
            break
res['fence_center'] = s
res.to_csv('res'+path+'.csv')
