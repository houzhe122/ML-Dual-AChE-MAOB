import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR  # 使用 SVM 回归模型
import matplotlib.pyplot as plt
import shap
import os

# 更改工作目录
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

# 使用 SVR，填入最佳超参数
best_params = {'kernel': 'rbf', 'gamma': 'scale', 'epsilon': 0.3, 'degree': 5, 'C': 10}

# 初始化 SVM 回归模型
svm_model = SVR(**best_params)

# 训练模型
svm_model.fit(X_train, y_train)

# 使用 shap.sample 从背景数据中随机抽样 K 个实例
background_data_summary = shap.sample(X_train, 100)  # 调整 100 为所需的样本数

# 使用 shap.KernelExplainer 解释 SVM 回归模型
explainer = shap.KernelExplainer(model=svm_model.predict, data=background_data_summary, link='identity')
# 计算测试集的 SHAP 值
shap_values = explainer.shap_values(X_test)
# 汇总图
shap.summary_plot(shap_values, X_test, feature_names=X.columns, show=False)
# 设置字体
plt.rc('font', family='Times New Roman')
plt.yticks(fontsize=15, fontname='Times New Roman')
plt.xticks(fontsize=15, fontname='Times New Roman')
plt.xlabel("SHAP value (impact on model output)", fontsize=15, fontname='Times New Roman')

# 保存图片
plt.savefig("svm-shap.jpg", dpi=500, bbox_inches='tight')
