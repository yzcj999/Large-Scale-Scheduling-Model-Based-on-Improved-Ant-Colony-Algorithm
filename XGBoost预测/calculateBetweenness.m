% 导入数据
% geom
% 从有向邻接矩阵转化成无向邻接矩阵
geom = max(geom,geom');

% 调用函数计算介数（需要额外安装图的计算库）
[node,arc]=betweenness_centrality(sparse(geom));
% node是结点的介数，arc是边的介数

% 将矩阵变量转为table
data_table = array2table(node);
% 2、将data_table写入csv文件
writetable(data_table, "bet.csv");