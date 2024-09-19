import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from tensorflow.keras.wrappers.scikit_learn import KerasRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import warnings
import os
os.chdir(r"D:\Users\阿哲\Desktop\文章\修稿上传")
warnings.filterwarnings("ignore")

# 创建一个空的DataFrame并存储每次循环的评估结果
evaluation_results = pd.DataFrame(columns=[ 'MSE', 'MAE', 'R2', 'Best Params'])


# 定义构建模型的函数
def create_model(optimizer='adam', activation='relu', neurons=32):
    model = Sequential()
    model.add(Dense(neurons, input_dim=X_train.shape[1], activation=activation))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(1))  # 输出层不使用激活函数
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    return model


# 1) 获取数据
file_path = "RegressionData_AChE_RFE.csv"
df = pd.read_csv(file_path, index_col = 'ID')

X = df.drop(columns = ['Y'])
y = df['Y']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# DNN回归模型和超参数优化
model = KerasRegressor(build_fn=create_model, verbose=0)
param_dist = {
    'batch_size': [16, 32, 64],
    'epochs': [50, 100, 200],
    'optimizer': ['adam', 'rmsprop'],
    'activation': ['relu', 'tanh'],
    'neurons': [16, 32, 64]
    }
random_search = RandomizedSearchCV(estimator=model, param_distributions=param_dist, n_iter=50, cv=3,
                                       scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)
random_search.fit(X_train, y_train)
best_dnn_model = random_search.best_estimator_
best_params = random_search.best_params_

# 预测
y_pred = best_dnn_model.predict(X_test)

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
evaluation_results.to_csv('DNNresults_rfe.csv', index=False)
