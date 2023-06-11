import pandas as pd
import joblib
from sklearn import metrics
import numpy as np

# prepare training set and test set
df1 = pd.DataFrame(pd.read_excel("0.8457525399896143trainset.xlsx")) #训练集
df2 = pd.DataFrame(pd.read_excel("0.8457525399896143testset.xlsx")) #测试集
xtrain = df1.iloc[:, :16]
ytrain = df1.iloc[:, 16:17]
xtest = df2.iloc[:, :16]
ytest = df2.iloc[:, 16:17]



# Import the model and calculate the predicted life
rf_model = joblib.load("0.84575+0.98380model.pkl")
ypre = rf_model.predict(xtrain)
ypre2 = rf_model.predict(xtest)
# df_out = pd.DataFrame(ypre)


# R2 on training set
print('train_R^2:', rf_model.score(xtrain, ytrain))
# calculate and print MAE
print("train_MAE: ", metrics.mean_absolute_error(ytrain, ypre))
# calculate and print RMSE
print("train_RMSE: ", np.sqrt(metrics.mean_squared_error(ytrain, ypre)))

# R2 on test set
print('R^2:', rf_model.score(xtest, ytest))
# calculate and print MAE
print("MAE: ", metrics.mean_absolute_error(ytest, ypre2))
# calculate and print RMSE
print("RMSE: ", np.sqrt(metrics.mean_squared_error(ytest, ypre2)))
print()