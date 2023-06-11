import pandas as pd
import numpy as np

data = pd.DataFrame(pd.read_excel("E:\IMR日常\数据管理\机器学习数据\小论文-数据及代码\数据汇总.xlsx"))
data = data.iloc[:, 16:24]
print(data)

corr1 = data.corr(method='pearson')
corr_out1 = pd.DataFrame(corr1)
corr_out1.to_excel('相关性分析（Pearson）.xlsx', index=False)
#
# corr2 = data.corr(method='spearman')
# corr_out2 = pd.DataFrame(corr2)
# corr_out2.to_excel('相关性分析（Spearman）.xlsx', index=False)

# corr3 = data.corr(method='kendall')
# corr_out3 = pd.DataFrame(corr3)
# corr_out3.to_excel('相关性分析（Kendall）.xlsx')

# print(corr['蠕变寿命lg(y)/h'].sort_values(ascending=False))
