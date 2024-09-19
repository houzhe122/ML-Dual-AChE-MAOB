import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#from sklearn.svm import SVR  # 使用 SVM 回归模型
from lightgbm import LGBMRegressor  # 使用 LGBM 回归模型
import matplotlib.pyplot as plt
import shap
import os
#更改工作目录
os.chdir(r"D:\Users\阿哲\Desktop\文章\修稿上传")

# 1) 获取数据
file_path = "RegressionData_MAO-B_RFE.csv"
df = pd.read_csv(file_path, index_col='ID')

X = df.drop(columns=['Y'])  # 特征数据
y = df['Y']  # 目标变量

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test1 = X_test.copy()

# 标准化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# 使用 LGBMRegressor，填入最佳超参数
best_params = {'num_leaves': 30, 'n_estimators': 1000, 'min_child_samples': 10, 'learning_rate': 0.05, 'lambda_l2': 0.5, 'lambda_l1': 0.5}

# 初始化 LGBM 回归模型
lgbm_model = LGBMRegressor(**best_params)

# 训练模型
lgbm_model.fit(X_train, y_train)

# 假设 X_test1 是一个 DataFrame，X_test 是有效的输入
features = pd.DataFrame(X_test, columns=X_test1.columns)

# 创建 SHAP 解释器
explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer.shap_values(features)

# 绘制 SHAP summary plot
shap.summary_plot(shap_values, features, max_display=20, show=False)

# 设置字体
plt.rc('font', family='Times New Roman')
plt.yticks(fontsize=15, fontname='Times New Roman')
plt.xticks(fontsize=15, fontname='Times New Roman')
plt.xlabel("SHAP value (impact on model output)", fontsize=15, fontname='Times New Roman')

# 保存图片
plt.savefig("lgbm-shap图.jpg", dpi=500, bbox_inches='tight')
