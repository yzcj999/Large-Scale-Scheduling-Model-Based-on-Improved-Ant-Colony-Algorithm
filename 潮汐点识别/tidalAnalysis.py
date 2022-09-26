import os, codecs
import pandas as pd
import numpy as np
import geohash
import matplotlib.pyplot as plt

PATH = '../data/'

# 读取单车围栏数据
def bike_fence_format(s):
    s = s.replace('[', '').replace(']', '').split(',')
    print(s)
    s = np.array(s).astype(float).reshape(5, -1)
    return s

# 共享单车停车点位（电子围栏）数据
bike_fence = pd.read_csv(PATH + 'gxdc_tcd.csv')
bike_fence['FENCE_LOC'] = bike_fence['FENCE_LOC'].apply(bike_fence_format)

for i in bike_fence['FENCE_LOC'].values[:10]:
    temp = list(i)
    lon = temp[0][0]
    lat = temp[0][1]
    # print(lat,lon)

for data in bike_fence['FENCE_LOC'].values[:10]:
    temp = list(data)
    lon = temp[0][0]
    lat = temp[0][1]
    folium.Marker([lat, lon]).add_to(m)
    
m.show()

# 对停车点进行处理，计算停车点的面积和中心经纬度
# 得出停车点 LATITUDE 范围
bike_fence['MIN_LATITUDE'] = bike_fence['FENCE_LOC'].apply(lambda x: np.min(x[:, 1]))
bike_fence['MAX_LATITUDE'] = bike_fence['FENCE_LOC'].apply(lambda x: np.max(x[:, 1]))

# 得到停车点 LONGITUDE 范围
bike_fence['MIN_LONGITUDE'] = bike_fence['FENCE_LOC'].apply(lambda x: np.min(x[:, 0]))
bike_fence['MAX_LONGITUDE'] = bike_fence['FENCE_LOC'].apply(lambda x: np.max(x[:, 0]))

from geopy.distance import geodesic
# 根据停车点 范围 计算具体的面积
bike_fence['FENCE_AREA'] = bike_fence.apply(lambda x: geodesic(
    (x['MIN_LATITUDE'], x['MIN_LONGITUDE']), (x['MAX_LATITUDE'], x['MAX_LONGITUDE'])
).meters, axis=1)

index.append('FENCE_AREA')
bike_fence[index].head(3)

# 根据停车点 计算中心经纬度
bike_fence['FENCE_CENTER'] = bike_fence['FENCE_LOC'].apply(
    lambda x: np.mean(x[:-1, ::-1], 0)
)
index.append('FENCE_CENTER')
bike_fence[index].head(3)

# 读取单车订单的数据
bike_order = pd.read_csv(PATH + 'gxdc_dd.csv')
bike_order = bike_order.sort_values(['BICYCLE_ID', 'UPDATE_TIME'])

# 使用geohash将经纬度转化成字符串，方便匹配单车位置和停车点位置
import geohash
bike_order['geohash'] = bike_order.apply(
    lambda x: geohash.encode(x['LATITUDE'], x['LONGITUDE'], precision=6), 
axis=1)

bike_fence['geohash'] = bike_fence['FENCE_CENTER'].apply(
    lambda x: geohash.encode(x[0], x[1], precision=6)
)

# 对订单的时间数据进行提取
bike_order['UPDATE_TIME'] = pd.to_datetime(bike_order['UPDATE_TIME'])
bike_order['DAY'] = bike_order['UPDATE_TIME'].dt.day.astype(object)
bike_order['DAY'] = bike_order['DAY'].apply(str)

bike_order['HOUR'] = bike_order['UPDATE_TIME'].dt.hour.astype(object)
bike_order['HOUR'] = bike_order['HOUR'].apply(str)
bike_order['HOUR'] = bike_order['HOUR'].str.pad(width=2,side='left',fillchar='0')

# 日期和时间进行拼接
bike_order['DAY_HOUR'] = bike_order['DAY'] + bike_order['HOUR']

# 使用透视表统计流量
bike_inflow = pd.pivot_table(bike_order[bike_order['LOCK_STATUS'] == 1], 
                   values='LOCK_STATUS', index=['geohash'],
                    columns=['DAY_HOUR'], aggfunc='count', fill_value=0
)

bike_outflow = pd.pivot_table(bike_order[bike_order['LOCK_STATUS'] == 0], 
                   values='LOCK_STATUS', index=['geohash'],
                    columns=['DAY_HOUR'], aggfunc='count', fill_value=0
)

# 展示其中一个经纬度位置的流量
plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
plt.rcParams['axes.unicode_minus']=False   #这两行需要手动设置

bike_inflow.loc['ws7fzz'].plot()
bike_outflow.loc['ws7fzz'].plot()
plt.xticks(list(range(bike_inflow.shape[1])), bike_inflow.columns, rotation=40)
plt.legend(['入流量', '出流量'])

# 展示另外一个点
num = 11
bike_inflow.loc['ws7fzr'].plot()
bike_outflow.loc['ws7fzr'].plot()
plt.xticks(list(range(bike_inflow.shape[1])), bike_inflow.columns, rotation=40)
plt.legend(['入流量', '出流量'])

# 使用经纬度距离匹配的方法来进行尝试，具体的思路为计算订单最近的停车点，进而计算具体的潮汐情况

# 使用sklearn中的NearestNeighbors，通过设置haversine距离完成最近停车点的计算
from sklearn.neighbors import NearestNeighbors

knn = NearestNeighbors(metric = "haversine", n_jobs=-1, algorithm='brute')
knn.fit(np.stack(bike_fence['FENCE_CENTER'].values))

# 计算订单中对应的停车点位置
dist, index = knn.kneighbors(bike_order[['LATITUDE','LONGITUDE']].values[:], n_neighbors=1)

# 订单中的fence属性是该订单所属的停车点
bike_order['fence'] = bike_fence.iloc[index.flatten()]['FENCE_ID'].values

bike_order.to_csv('bikeorder(all).csv')

bike_remain = (bike_inflow - bike_outflow).fillna(0)
#bike_remain[bike_remain < 0] = 0  
bike_remain = bike_remain.sum(1)

# 计算停车点密度
bike_density = bike_remain / bike_fence.set_index('FENCE_ID')['FENCE_AREA']
bike_density = bike_density.sort_values(ascending=False).reset_index()
bike_density = bike_density.fillna(0)

bike_density.to_csv('bike_density.csv')
