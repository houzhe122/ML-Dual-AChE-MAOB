import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from lightgbm import LGBMRegressor  # 使用 LGBM 回归模型
import matplotlib.pyplot as plt
import shap

# 读取数据
df_path =  r"D:\Users\阿哲\Desktop\工作簿1.xlsx"
df = pd.read_excel(df_path, sheet_name='Sheet2')
X = df.iloc[:, 3:]  # 从第三列开始读取特征
y = df['Y']  # 目标变量

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test1 = X_test.copy()  # 复制一份测试集原始数据

# 标准化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 使用 LGBMRegressor，填入最佳超参数
best_params = {'subsample': 0.8, 'n_estimators': 300, 'min_child_samples': 20, 'num_leaves': 50, 'max_depth': 5, 'learning_rate': 0.1}

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
plt.savefig(r"D:\Users\阿哲\Desktop\lgbm-shap图.jpg", dpi=500, bbox_inches='tight')
