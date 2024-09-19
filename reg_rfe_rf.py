import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
import warnings
warnings.filterwarnings("ignore")
import os
os.chdir(r"D:\Users\阿哲\Desktop\文章\修稿上传")


# 1) 获取数据
df_path = "RegressionData_AChE.csv"
df = pd.read_csv(df_path, index_col = 'ID')
X = df.drop(columns = ['Y','Smiles'])
y = df['Y']
X_original = X.copy()

#移除常量特征
constant_features = [feat for feat in X.columns if X[feat].std() == 0]
X.drop(labels=constant_features, axis=1, inplace=True)

# 移除准常量特征
sel = VarianceThreshold(threshold=0.01)
sel.fit(X)
features_to_keep = X.columns[sel.get_support()]
X = sel.transform(X)
X = pd.DataFrame(X)

X.columns = features_to_keep


# Instantiate RFECV visualizer with a random forest regressor
rfe = RFE(RandomForestRegressor(),n_features_to_select=150, step=1)#默认使用R²交叉验证3
rfe.fit(X, y) # Fit the data to the visualizer

#获取最佳特征子集的掩码
feature_mask = rfe.support_
# 根据掩码从原特征表格中创建新数据帧
X_selected = X.loc[:, feature_mask]
X_selected.index = X_original.index
X_selected.insert(0,"Y",y)
print(X_selected)
# 将新数据帧保存为一个新的 Excel 文件
X_selected.to_csv('RegressionData_AChE_RFE50.csv', index=True)