% ��������
% geom
% �������ڽӾ���ת���������ڽӾ���
geom = max(geom,geom');

% ���ú��������������Ҫ���ⰲװͼ�ļ���⣩
[node,arc]=betweenness_centrality(sparse(geom));
% node�ǽ��Ľ�����arc�ǱߵĽ���

% ���������תΪtable
data_table = array2table(node);
% 2����data_tableд��csv�ļ�
writetable(data_table, "bet.csv");