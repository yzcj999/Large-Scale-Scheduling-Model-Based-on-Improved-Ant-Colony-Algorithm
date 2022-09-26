import os, codecs
import pandas as pd
import numpy as np
import geohash
import matplotlib.pyplot as plt

PATH = './'

# 共享单车轨迹数据
bike_track = pd.concat([
    pd.read_csv(PATH + 'gxdc_gj20201221.csv'),
    pd.read_csv(PATH + 'gxdc_gj20201222.csv'),
    pd.read_csv(PATH + 'gxdc_gj20201223.csv'),
    pd.read_csv(PATH + 'gxdc_gj20201224.csv'),
    pd.read_csv(PATH + 'gxdc_gj20201225.csv')

])

# 按照单车ID和时间进行排序
bike_track = bike_track.sort_values(['BICYCLE_ID', 'LOCATING_TIME'])

bike_dict = {}
# 生成所有的自行车ID字典
for item in bike_track['BICYCLE_ID'][:]:
    if item not in bike_dict.keys():
        bike_dict[item] = 1
    else:
        bike_dict[item] += 1
bike_key_li = list(bike_dict.keys())

import folium
# 绘制出行轨迹，这里仅画出前10个单车轨迹
m = folium.Map(location=[24.482426, 118.157606], zoom_start=12)
for item in bike_key_li[:10]:
    my_PolyLine=folium.PolyLine(locations=bike_track[bike_track['BICYCLE_ID'] == item][['LATITUDE', 'LONGITUDE']].values,weight=5)
    m.add_child(my_PolyLine)

m.show()
