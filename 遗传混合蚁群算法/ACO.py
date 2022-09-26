import random
import copy
import sys
import math
import tkinter
import threading
import numpy as np
from functools import reduce
import pandas as pd
h=pd.read_csv('test.csv')
# 城市坐标
distance_x =[]
for i in h['x']:
    distance_x.append(i)
    
distance_y =[]
for i in h['y']:
    distance_y.append(i)
# 预定义参数
# 信息素重要程度因子α
# 启发函数重要程度因子β
# 信息素挥发因子ρ
#初始携带量T
#最大站点数state
(ALPHA,BETA,RHO,Q,city_num,ant_num,T,state)=(2,1,0.6,100,len(distance_x),len(distance_x),480,15)
# 城市之间的距离和初始信息素
distance_graph=[[0 for col in range(city_num)] for raw in range(city_num)]
pheromone_graph=[[1 for col in range(city_num)] for raw in range(city_num)]
# 每个城市的deposit
deposit_city =[]
for i in h['d']:
    deposit_city.append(i)
# 所有蚂蚁未经过的城市表
open_table_city=[True for i in range(city_num)] #标记哪个城市去过，哪个没去过；也称为禁忌表
# 设置最后一个地点为中心，所有蚂蚁均从这里出发并最终回到这里
open_table_city[8] = False
class Ant():
    def __init__(self,ID):
        self.ID=ID #蚂蚁ID
        self.__clean_data() #蚂蚁初始化
        
    def __clean_data(self):
        self.path=[]#小蚂蚁的路径
        self.total_distance=0 #小蚂蚁当前路径的总距离
        self.move_count=0 #小蚂蚁的移动次数
        self.current_city=-1 #小蚂蚁的当前所在城市
        
        # city_index=random.randint(0,city_num-1)#随机初始化出生地
        city_index=8 #所有蚂蚁都从一个点出发，8号位置
        self.current_city=city_index
        self.path.append(city_index)
        # global open_table_city
        self.my_open_table = copy.deepcopy(open_table_city) #每只蚂蚁维护自己的城市禁忌表
        #self.open_table_city[city_index]=False
        self.move_count=1 #初始步数为1
        self.carry = T #初始化每只蚂蚁的carry
    
    def update_my_open_table(self):
        # global open_table_city
        self.my_open_table = copy.deepcopy(open_table_city) #每只蚂蚁维护自己的城市禁忌表
    
    #小蚂蚁选择下一个城市
    def __choice_next_city(self):
        next_city=-1
        select_citys_prob=[0 for i in range(city_num)]#去下一个城市的概率
        total_prob=0

        #去下一个城市的概率
        for i in range(city_num):
            if self.my_open_table[i]: #如果下一个城市没去过，还可以去
                try:
                    #选中概率：与信息素浓度呈正比，与距离呈反比
                    select_citys_prob[i]=pow(pheromone_graph[self.current_city][i],ALPHA)*\
                                         pow(1/distance_graph[self.current_city][i],BETA)
                    total_prob+=select_citys_prob[i]
                except ZeroDivisionError as e:
                    print('Ant ID: {ID}, current city: {current}, target city: {target}'.
                          format(ID=self.ID,current=self.current_city,target=i))
                    sys.exit(1)

        #轮盘对赌
        if total_prob>0:
            #产生随机概率
            temp_prob=random.uniform(0,total_prob)
            for i in range(city_num):
                if self.my_open_table[i]:
                    temp_prob-=select_citys_prob[i]
                    if temp_prob<0:
                        next_city=i
                        break
        #如果没有从轮盘对赌中选出，则顺序选择一个未访问城市
        if next_city==-1:
            for i in range(city_num):
                if self.my_open_table[i]:
                    next_city=i
                    break
                    
        #每只蚂蚁走50步，即完成对50个城市的遍历，每次至少都会从禁忌表中找到下一个城市
#         if next_city==-1:
#             next_city=random.randint(0,city_num-1)
#             while False==self.my_open_table[next_city]:
#                 next_city=random.randint(0,city_num-1)
        
        
        #返回下一个城市序号
        return next_city
    
    #小蚂蚁通过惩罚机制综合考虑选择下一个城市
    def __choice_next_city2(self):
        next_city=-1
        select_citys_prob=[0 for i in range(city_num)]#去下一个城市的概率
        total_prob=0
        punishment=0
        #去下一个城市的概率
        for i in range(city_num):
            if deposit_city[i]>0:
                punishment=0.0001
            else:
                if self.carry+deposit_city[i]<0:
                    punishment=0-(self.carry+deposit_city[i])
                else:
                    punishment=0.0001
            if self.my_open_table[i]: #如果下一个城市没去过，还可以去
                try:
                    #选中概率：与信息素浓度呈正比，与距离呈反比
                    select_citys_prob[i]=pow(pheromone_graph[self.current_city][i],ALPHA)*\
                                         pow(1/(distance_graph[self.current_city][i]+10*punishment),BETA)
                    total_prob+=select_citys_prob[i]
                except ZeroDivisionError as e:
                    print('Ant ID: {ID}, current city: {current}, target city: {target}'.
                          format(ID=self.ID,current=self.current_city,target=i))
                    sys.exit(1)

        #轮盘对赌
        if total_prob>0:
            #产生随机概率
            temp_prob=random.uniform(0,total_prob)
            for i in range(city_num):
                if self.my_open_table[i]:
                    temp_prob-=select_citys_prob[i]
                    if temp_prob<0:
                        next_city=i
                        break
        #如果没有从轮盘对赌中选出，则顺序选择一个未访问城市
        if next_city==-1:
            for i in range(city_num):
                if self.my_open_table[i]:
                    next_city=i
                    break
                    
        #每只蚂蚁走50步，即完成对50个城市的遍历，每次至少都会从禁忌表中找到下一个城市
#         if next_city==-1:
#             next_city=random.randint(0,city_num-1)
#             while False==self.my_open_table[next_city]:
#                 next_city=random.randint(0,city_num-1)
        
        
        #返回下一个城市序号
        return next_city

    #计算路径总距离
    def __cal_total_distance(self):
        temp_distance=0
        # 根据每只蚂蚁步数计算距离
        for i in range(1,self.move_count):
            start,end =self.path[i],self.path[i-1]
            temp_distance+=distance_graph[start][end]
        #回路
        end=self.path[0]
        temp_distance+=distance_graph[start][end]
        # 蚂蚁的路径中不显示返回到起点，默认最终返回起点
        # self.path.append(self.path[0])
        self.total_distance=temp_distance
    #小蚂蚁移动操作
    def __move(self,next_city):
        self.path.append(next_city)
        # global open_table_city
        open_table_city[next_city]=False
        self.my_open_table[next_city]=False
        self.total_distance+=distance_graph[self.current_city][next_city]
        self.carry += deposit_city[next_city] #更新蚂蚁携带量
        self.current_city=next_city
        self.move_count+=1
    #小蚂蚁踏上寻路之旅
    def search_path(self):
        #初始化数据
        self.__clean_data()
        # 修改逻辑如下：如果选择出的下一个城市的deposit大于蚂蚁目前的携带量carry，则重新选择下一个城市
        # 如果剩下的城市都无法满足，则蚂蚁回到出发点
        # 并且每只蚂蚁走完15站之后也不再走
        while self.move_count<state:
            next_city=self.__choice_next_city()
            # 如果蚂蚁当前携带量不满足下一个城市的deposit，重新选择
            while self.carry+deposit_city[next_city] < 0:
                # 无法满足目前next_city的需求，更新蚂蚁自己的禁忌表
                self.my_open_table[next_city]=False
                next_city = -1
                # 判断是否有城市可以选择
                # 如果剩下的所有城市都无法满足，则不再寻找
                if True in self.my_open_table:
                    next_city=self.__choice_next_city2()
                else:
                    break
            # 当下一步不为-1时，蚂蚁移动，否则停止移动
            if next_city != -1:
                self.__move(next_city)
            else:
                break
        
        #计算路径总长度
        self.__cal_total_distance()
        print("蚂蚁{ID}完成自己的工作，路径为{path}".format(ID=self.ID,path=self.path))
class TSP():

    def __init__(self,root,width=800,height=600,n=city_num):
        self.root=root
        self.width=width #画布的宽度和高度
        self.height=height
        self.n=n #城市数目

        #tkinter 来画布
        self.canvas=tkinter.Canvas(root,width=self.width,height=self.height,bg= "#EBEBEB", xscrollincrement=1,yscrollincrement=1)
        self.canvas.pack(expand=tkinter.YES,fill=tkinter.BOTH)
        self.title("TSP蚁群算法(n:初始化 e:开始搜索 s:停止搜索 q:退出程序)")
        self.__r = 5 #城市节点的半径
        self.__lock = threading.RLock()  # 线程锁

        self.__bindEvents()
        self.new()

    

    # 按键响应程序
    def __bindEvents(self):

        self.root.bind("q", self.quite)  # 退出程序
        self.root.bind("n", self.new)  # 初始化
        self.root.bind("e", self.search_path)  # 开始搜索
        self.root.bind("s", self.stop)  # 停止搜索
        
    # 更改标题
    def title(self, s):
        self.root.title(s)

    # 初始化
    def new(self, evt=None):
        # 停止线程
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()

        self.clear()  # 清除信息
        self.nodes = []  # 节点坐标
        self.nodes2 = []  # 节点对象

        # 初始化城市节点
        for i in range(len(distance_x)):
            # 在画布上随机初始坐标
            x = distance_x[i]
            y = distance_y[i]
            self.nodes.append((x, y))
            # 生成节点椭圆，半径为self.__r
            node = self.canvas.create_oval(x - self.__r,
                                           y - self.__r, x + self.__r, y + self.__r,
                                           fill="#ff0000",  # 填充红色
                                           outline="#000000",  # 轮廓白色
                                           tags="node",
                                           )
            self.nodes2.append(node)
            # 显示坐标
            self.canvas.create_text(x, y - 10,  # 使用create_text方法在坐标（302，77）处绘制文字
                                    text='(' + str(x) + ',' + str(y) + ')',  # 所绘制文字的内容
                                    fill='black'  # 所绘制文字的颜色为灰色
                                    )

        # 初始城市之间的距离和信息素
        for i in range(city_num):
            for j in range(city_num):
                temp_distance = pow((distance_x[i] - distance_x[j]), 2) + pow((distance_y[i] - distance_y[j]), 2)
                temp_distance = pow(temp_distance, 0.5)
                distance_graph[i][j] = float(int(temp_distance + 0.5))
        # 信息素最开始都是100
        for i in range(city_num):
            for j in range(city_num):
                pheromone_graph[i][j] = 100.0

        self.ants = [Ant(ID) for ID in range(ant_num)]  # 初始蚁群
        #self.temp_ants = [] # 临时蚂蚁群
        self.best_ants = [] # 最佳蚂蚁群
        self.best_total_distance = 1 << 31  # 初始最大距离
        self.iter = 1  # 初始化迭代次数
        self.k = 0 #记录参与的蚂蚁个数

    # 将节点按order顺序连线
    def line(self):
        self.canvas.delete("line")
        
        def line2(i1, i2):
            p1, p2 = self.nodes[i1], self.nodes[i2]
            self.canvas.create_line(p1, p2, fill="#000000", tags="line")
            return i2
        for ant in self.best_ants:
            order = copy.deepcopy(ant.path)
            reduce(line2, order, order[-1])
        
    # 清除画布
    def clear(self):
        for item in self.canvas.find_all():
            self.canvas.delete(item)

    # 退出程序
    def quite(self, evt):
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()
        self.root.destroy()
        print(u"\n程序已退出...")
        sys.exit()

    # 停止搜索
    def stop(self, evt):
        self.__lock.acquire()
        self.__running = False
        self.__lock.release()

    #开始搜索
    def search_path(self, evt=None):
        # 开启线程
        self.__lock.acquire()
        self.__running = True
        self.__lock.release()
        best=[]
        iters=[]
        distance=[]
        while self.iter <= Q:
            #遍历每一只蚂蚁
            # 当所有城市都遍历时退出
            for ant in self.ants:
                # 蚂蚁首先根据整体的禁忌表更新自己的禁忌表
                ant.update_my_open_table()
                ant.search_path() #小蚂蚁ant完成自己的任务
                # 一个蚂蚁完成后的整体禁忌表
                #print(open_table_city)
                # 蚂蚁自身的open_table
                #print(ant.my_open_table)
                
                #self.best_ants[]copy.deepcopy(ant)
                #self.line(self.temp_ant.path)
                self.k += 1
                global open_table_city
                if True not in open_table_city:
                    # 完成一次遍历，更新信息素时已经计算本次总距离
                    self.__update_pheromone_gragh()
                    best.append(self.best_total_distance)
                    iters.append(self.iter)
                    distance.append(self.total_distance)
                    if self.total_distance <self.best_total_distance:
                        self.best_total_distance=self.total_distance 
                        # 更新最佳蚂蚁群
                        self.best_ants=[]
                        for i in range(self.k):
                            self.best_ants.append(copy.deepcopy(self.ants[i]))
                    elif self.total_distance==self.best_total_distance:
                        if self.k<len(self.best_ants):
                            self.best_total_distance=self.total_distance 
                        # 更新最佳蚂蚁群
                        self.best_ants=[]
                        for i in range(self.k):
                            self.best_ants.append(copy.deepcopy(self.ants[i]))

                    # 输出结果
                    print('{}只蚂蚁完成第{}次遍历,本次距离为：{} 历史最优距离为：{}'.
                         format(self.k,self.iter,self.total_distance,self.best_total_distance))
                    self.iter += 1
                    self.line()
                    # 为下一次做初始化操作
                    self.ants = [Ant(ID) for ID in range(ant_num)]  # 初始蚁群
                    open_table_city=[True for i in range(city_num)] # 初始化禁忌表
                    open_table_city[8] = False
                    
                    self.k = 0 # 更新参与的蚂蚁个数
                    break
                    
            # 设置标题
            self.title("TSP蚁群算法(n:随机初始 e:开始搜索 s:停止搜索 q:退出程序) 迭代次数: %d" % self.iter)
            # 更新画布
            self.canvas.update()
        #展示最佳路径与距离
        print("最佳路径为:")
        a=1
        dis=[]
        for ant in self.best_ants:
            order = copy.deepcopy(ant.path)
            print(a,end=":")
            a=a+1
            for i in order:
                print(i,'->',end='')
            print("8;")
            k=0
            for i in range(len(order)-1):
                k+=distance_graph [order[i]][order[i+1]]
            k+=distance_graph[order[len(order)-1]][8]
            dis.append(k)
        print(dis)
        print(sum(dis))
        
        #验证车辆数
        deposit_citycopy=[]
        for i in deposit_city:
            deposit_citycopy.append(i)
        pre=sum(deposit_citycopy)
        t=0
        n=0
        for ant in self.best_ants:
            order = copy.deepcopy(ant.path)
            antcarry=T
            n=n+1
            for i in order:
                if antcarry+deposit_citycopy[i]<0:
                    deposit_citycopy[i]=antcarry+deposit_citycopy[i]
                    antcarry=0
                else:
                    antcarry+=deposit_citycopy[i]
                    deposit_citycopy[i]=0
            t=t+antcarry
        now=sum(deposit_citycopy)
        h=[pre,n*T,now,t]
        print("行驶之前各地的deposit，每辆车初始携带量总和，行驶后各地的deposit，每辆车携带量总和分别为",end=':')
        print(h)
        if(pre+n*T==now+t):#行驶之前各地的deposit+每辆车初始携带量=行驶后各地的deposit+每辆车携带量
            print("调度成功，车辆数正确")
        else:
            print("调度有问题，车辆数不对")
       
        import matplotlib.pyplot as plt

        import numpy as np
        del best[0]
        del iters[0]
        plt.plot(iters,best,linestyle='--',color='green')
        plt.xlabel('iters')
        plt.ylabel('distance')
        plt.show()  
        del distance[0]
        plt.plot(iters,distance,linestyle='--',color='green')
        plt.xlabel('iters')#迭代图
        plt.ylabel('value')#距离图
        plt.show()  
        
    # 更新信息素
    def __update_pheromone_gragh(self):
        temp_distance = 0.0
        # 获取每只蚂蚁在其路径上留下的信息素
        temp_pheromone = [[0.0 for col in range(city_num)] for raw in range(city_num)]
        # k只蚂蚁参与遍历
        for count in range(self.k):
            ant = self.ants[count]
            temp_distance += ant.total_distance
            for i in range(len(ant.path)):
                start, end = ant.path[i - 1], ant.path[i]
                # 在路径上的每两个相邻城市间留下信息素，与路径总距离反比
                temp_pheromone[start][end] += Q / (ant.total_distance+0.01)
                temp_pheromone[end][start] = temp_pheromone[start][end]

        # 更新所有城市之间的信息素，旧信息素衰减加上新迭代信息素
        for i in range(city_num):
            for j in range(city_num):
                pheromone_graph[i][j] = pheromone_graph[i][j] * RHO + temp_pheromone[i][j]
        self.total_distance = temp_distance
    def mainloop(self):
        self.root.mainloop()
        
f=TSP(tkinter.Tk())
f.mainloop()
        
