##### 代码功能介绍：

getFeature.py：主要根据共享单车停车点数据和订单数据实现对XGBoost所需要的图结构特征进行提取并生成特征变量文件；

calculateBetweenness.m：MATLAB代码程序，功能是使用邻接矩阵数据计算点的介数中心性值；

XGBoost.py：主要根据提取的图结构化特征变量训练XGBoost回归模型，用于对未来订单量的预测。

##### 注意事项：

1.运行代码文件时需要修改文件导入路径；

2.运行所需要安装的python库有：

numpy==1.19.2 pandas==1.2.4 geohash==1.0 folium==0.12.1 Django==3.2 xgboost==1.3.3 scipy==1.6.1

