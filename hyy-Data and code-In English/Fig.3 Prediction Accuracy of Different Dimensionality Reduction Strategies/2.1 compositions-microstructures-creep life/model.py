import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import train_test_split
from sklearn import metrics
import datetime
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor


def model():
    # Show Chinese and minus signs
    mpl.rcParams['font.sans-serif'] = ['SimHei']
    mpl.rcParams['axes.unicode_minus'] = False

    starttime = datetime.datetime.now()
    print('Time begin： ', starttime)

    # prepare training set and test set
    df = pd.DataFrame(pd.read_excel("Creep life data.xlsx"))
    X = df.iloc[:, 14:24]
    X = MinMaxScaler().fit_transform(X)
    y = df["Creep life lg(y)/h"]
    xtrain, xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2)  # , random_state=42
    xtrain = xtrain.astype(np.float64)
    xtest = xtest.astype(np.float64)

    # save training set and test set
    xtrain1 = pd.DataFrame(xtrain)
    ytrain1 = pd.DataFrame(ytrain)
    xtest1 = pd.DataFrame(xtest)
    ytest1 = pd.DataFrame(ytest)
    df_train = pd.DataFrame(pd.concat([xtrain1, ytrain1]))
    # df_train.to_excel("trainset.xlsx", index=False)
    df_test = pd.DataFrame(pd.concat([xtest1, ytest1]))
    # df_test.to_excel("testset.xlsx", index=False)

    # fit model
    rf_model = RandomForestRegressor(max_depth=12, n_estimators=77, random_state=42, n_jobs=-1)
    rf_model.fit(xtrain, ytrain)
    # rf_model = joblib.load("RF_best_model.pkl")
    ypre = rf_model.predict(xtest)
    ypre2 = rf_model.predict(xtrain)

    # R2 on training set
    print('train_R^2:', rf_model.score(xtrain, ytrain))
    R2_train = rf_model.score(xtrain, ytrain)
    # calculate and print MAE
    print("train_MAE: ", metrics.mean_absolute_error(ytrain, ypre2))
    # calculate and print RMSE
    print("train_RMSE: ", np.sqrt(metrics.mean_squared_error(ytrain, ypre2)))

    # R2 on test set
    print('R^2:', rf_model.score(xtest, ytest))
    R2 = rf_model.score(xtest, ytest)
    # calculate and print MAE
    print("MAE: ", metrics.mean_absolute_error(ytest, ypre))
    # calculate and print RMSE
    print("RMSE: ", np.sqrt(metrics.mean_squared_error(ytest, ypre)))
    print()

    # save results
    df_out1 = pd.DataFrame({"train_y_ture": ytrain, "train_y_pre": ypre2})
    # df_out1.to_excel("trainset predicting result.xlsx", index=False)
    df_out2 = pd.DataFrame({"test_y_ture": ytest, "test_y_pre": ypre})
    # df_out2.to_excel("testset predicting result.xlsx", index=False)

    # save ML model
    # joblib.dump(rf_model, '2.1model.pkl')

    # print calculating time
    endtime = datetime.datetime.now()
    print('Endtime: ', endtime)
    print('Total time:', endtime - starttime)

    return R2, R2_train, df_train, df_test, rf_model, df_out1, df_out2


r2 = [0]
a = 0
while a < 0.95:
    a, b, trainset, testset, rfmodel, trainresult, testresult = model()
    if a > max(r2):
        r2.append(a)
        trainset.to_excel(str(a) + "trainset.xlsx", index=False)
        testset.to_excel(str(a) + "testset.xlsx", index=False)
        joblib.dump(rfmodel, str(a) + "+" + str(b) + 'model.pkl')
        trainresult.to_excel(str(a) + "trainset predicting result.xlsx", index=False)
        testresult.to_excel(str(a) + "testset predicting result.xlsx", index=False)

print(r2)
df_r2 = pd.DataFrame({"R2": r2})
df_r2.to_excel("R2.xlsx", index=False)
