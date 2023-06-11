import pandas as pd
import joblib
from sklearn import metrics
import numpy as np

# 准备训练集、测试集
df1 = pd.DataFrame(pd.read_excel("0.9058383528980465trainset.xlsx")) #训练集
df2 = pd.DataFrame(pd.read_excel("0.9058383528980465testset.xlsx")) #测试集
xtrain = df1.iloc[:, :9]
ytrain = df1.iloc[:, 9:10]
xtest = df2.iloc[:, :9]
ytest = df2.iloc[:, 9:10]



# 导入模型，计算预测值
rf_model = joblib.load("0.90583+0.96589model.pkl")
ypre = rf_model.predict(xtrain)
ypre2 = rf_model.predict(xtest)
# df_out = pd.DataFrame(ypre)
# df_out.to_excel("γ'尺寸变化预测值-3.xlsx")

# 在训练集上
# 拟合优度R2
print('train_R^2:', rf_model.score(xtrain, ytrain))
# 计算平均绝对误差MAE的保存与输出
print("train_MAE: ", metrics.mean_absolute_error(ytrain, ypre))
# 计算均方根误差RMSE的保存与输出
print("train_RMSE: ", np.sqrt(metrics.mean_squared_error(ytrain, ypre)))

# 在测试集上
# 拟合优度R2
print('R^2:', rf_model.score(xtest, ytest))
# 计算平均绝对误差MAE的保存与输出
print("MAE: ", metrics.mean_absolute_error(ytest, ypre2))
# 计算均方根误差RMSE的保存与输出
print("RMSE: ", np.sqrt(metrics.mean_squared_error(ytest, ypre2)))
print()