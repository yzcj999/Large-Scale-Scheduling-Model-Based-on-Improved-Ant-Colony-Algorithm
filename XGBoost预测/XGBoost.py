import xgboost
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn import linear_model, svm, gaussian_process
from sklearn.ensemble import RandomForestRegressor
# 原始数据处理
df21 = pd.read_csv("res21.csv")
df22 = pd.read_csv('res22.csv')
df23 = pd.read_csv('res23.csv')
df24 = pd.read_csv('res24.csv')

df21['y'] = df22['deposit']
df22['y'] = df23['deposit']
df23['y'] = df24['deposit']

data = pd.concat([df21,df22,df23])
# 保存数据
data.to_csv('data_raw.csv')

# 载入数据
df = pd.read_csv('data_raw.csv')

# 切分训练集
X = np.array(df.loc[:,['deposit','instrength','outstrength','indegree','outdegree','pagerank']])
Y = np.array(df.y)


# 把数据集拆分成训练集和测试集
seed = 7
test_size = 0.15
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)

# 拟合XGBoost模型
model = linear_model.BayesianRidge()
model.fit(X_train, y_train)

# 对测试集做预测
y_pred = model.predict(X_test)
mse_test=np.sum((y_pred-y_test)**2)/len(y_test)
print("MSE:",mse_test)
mae_test=np.sum(abs(y_pred-y_test))/len(y_test)
print("MAE:",mae_test)
rmse_tset=pow(mse_test,0.5)
print("RMSE:",rmse_tset)
from sklearn.metrics import r2_score#R square
print('R2:',r2_score(y_test,y_pred))

# 使用模型进行预测
x = pd.read_csv('res24.csv')
x = np.array(x.loc[:,['deposit','instrength','outstrength','indegree','outdegree','pagerank']])
y = model.predict(x)
# 保存预测数据
yd = pd.DataFrame(y)
yd.to_csv('y_pre.csv')
