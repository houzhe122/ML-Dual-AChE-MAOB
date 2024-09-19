import numpy as np #数据处理
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier

import matplotlib.pyplot as plt
import shap

df_path =  r"D:\Users\阿哲\Desktop\工作簿1.xlsx"
df = pd.read_excel(df_path, sheet_name='Sheet2')
X = df.iloc[:, 3:]  #从第三列开始读取
y = df['Y']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_test1 = X_test.copy()
# 标准化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# 创建gbdt模型
best_params ={'subsample': 0.8, 'n_estimators': 300, 'min_samples_split': 5, 'min_samples_leaf': 2, 'max_depth': 5, 'learning_rate': 0.1}

gbdt_model = GradientBoostingClassifier(**best_params)#根据优化结果调整

# 训练模型
gbdt_model.fit(X_train, y_train)

# 假设 X_test1 是一个DataFrame，X_test 是有效的输入。
features = pd.DataFrame(X_test, columns=X_test1.columns)

explainer = shap.TreeExplainer(gbdt_model)
shap_values = explainer.shap_values(features)
shap.summary_plot(shap_values,features ,max_display=20,show=False)
plt.rc('font',family='Times New Roman')
plt.yticks(fontsize=15, fontname='Times New Roman')
plt.xticks(fontsize=15, fontname='Times New Roman')
plt.xlabel("SHAP value(impact on model output)", fontsize=15, fontname='Times New Roman')
plt.savefig(r"D:\Users\阿哲\Desktop\gbdt-shap图.jpg",dpi=500,bbox_inches = 'tight')



