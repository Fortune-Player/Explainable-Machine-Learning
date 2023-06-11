import pandas as pd
import shap
import joblib
import matplotlib as mpl
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np

# Show Chinese and minus signs
mpl.rcParams['font.sans-serif'] = ['Times New Roman']
mpl.rcParams['axes.unicode_minus'] = False

model = joblib.load("2.1model.pkl")

df = pd.DataFrame(pd.read_excel("designed alloys.xlsx"))
X = df.iloc[:, 14:24]
features = df.columns[14:24]

shap.initjs()
explainer = shap.TreeExplainer(model)
shap_values = explainer(X)

shap.plots.waterfall(shap_values[0])
shap.plots.force(shap_values[0])

# # 绘制shap全局分析条形图
# plt.figure(figsize=(3,3),dpi=600)
# shap.summary_plot(shap_values, X, feature_names=features,plot_type="bar",show=False)
# plt.savefig("shap_bar.png")
#
# # 绘制shap全局分析图
# plt.figure(figsize=(3,3),dpi=600)
# shap.summary_plot(shap_values, X,feature_names=features,show=False)
# plt.savefig("shap_scatter.png")
df_out = pd.DataFrame(list(shap_values))
df_out.to_excel("shap.xlsx",index=False)


###
# plt.figure(figsize=(3,3),dpi=600)
# shap.force_plot(explainer.expected_value,shap_values[0,:],X.iloc[0,:])
# plt.savefig("shap_one.png")
# shap.force_plot(explainer.expected_value, shap_values, X)
# shap.dependence_plot("Dγ", shap_values, X)
# shap.summary_plot(shap_values, X)
# shap.plots.waterfall(shap_values[0])
# shap.plots.force(shap_values[0])

# print(shap_values[0].base_values)