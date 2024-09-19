import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
import warnings
warnings.filterwarnings("ignore")
import os
os.chdir(r"D:\Users\阿哲\Desktop\文章\修稿上传")
# 创建一个空的DataFrame并存储每次循环的评估结果
evaluation_results = pd.DataFrame(columns=[ 'MSE', 'MAE', 'R2', 'Best Params'])

# 1) 获取数据
file_path = "RegressionData_MAO-B_RFE.csv"
df = pd.read_csv(file_path, index_col = 'ID')

X = df.drop(columns = ['Y'])
y = df['Y']

    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 # 标准化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
    
# KNN回归模型和超参数优化
model = KNeighborsRegressor()
param_grid = {
        'n_neighbors': [3, 5, 7, 9],
        'weights': ['uniform', 'distance'],
        'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
        'leaf_size': [10, 20, 30, 40],
        'p': [1, 2]  # 1 for Manhattan distance, 2 for Euclidean distance
    }
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')
grid_search.fit(X_train, y_train)
best_knn_model = grid_search.best_estimator_
best_params = grid_search.best_params_

# 预测
y_pred = best_knn_model.predict(X_test)
    
# 计算评价指标
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
    
# 存储评估结果到DataFrame中
evaluation_results = evaluation_results._append({
    'MSE': mse,
    'MAE': mae,
    'R2': r2,
    'Best Params': best_params
    }, ignore_index=True)

# 打印评估结果
print(evaluation_results)
evaluation_results.to_csv('KNNresults_rfe.csv', index=False)
