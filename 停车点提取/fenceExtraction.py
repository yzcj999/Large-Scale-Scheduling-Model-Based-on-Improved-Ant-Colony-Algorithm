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
