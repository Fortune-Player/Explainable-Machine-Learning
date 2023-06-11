import pandas as pd
import joblib
from sklearn import metrics
import numpy as np

# prepare training set and test set
df1 = pd.DataFrame(pd.read_excel("0.9452249987547019trainset.xlsx"))  # train set
df2 = pd.DataFrame(pd.read_excel("0.9452249987547019testset.xlsx"))  # test set
xtrain = df1.iloc[:, :10]
ytrain = df1.iloc[:, 10:11]
xtest = df2.iloc[:, :10]
ytest = df2.iloc[:, 10:11]

# Import the model and calculate the predicted life
rf_model = joblib.load("0.9452249987547019+0.9754641709321241model.pkl")
ypre = rf_model.predict(xtrain)
ypre2 = rf_model.predict(xtest)


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
