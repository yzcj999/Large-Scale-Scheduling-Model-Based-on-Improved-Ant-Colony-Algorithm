##### 代码功能介绍：

GA.py代码是用来将所设定的优化函数通过遗传迭代的方法来进行不断优化，以求得最优解。
ACO.py代码通过信息素衰减的机制确定最短路径。
结合两者的特点，我们将蚁群算法的参数组作为优化函数输入给遗传算法，遗传算法通过遗传更迭来输出给蚁群算法参数组，蚁群算法再利用这个参数组进行寻找最优路径并将其发送给遗传算法计算适应度值，直至到达遗传次数。

##### 注意事项：

1.ACO.py读取test.csv文件时，需要确定正确的文件路径；

2.ACO.py代码需要安装 的python库有：
 random
 copy
 sys
 math
 tkinter
 threading
 numpy ==1.19.2
 pandas ==1.2.4

3.GA.py代码需要安装的python库有：
matplotlib
numpy

