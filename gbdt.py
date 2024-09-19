import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
import warnings
from imblearn.combine import SMOTETomek

import os
os.chdir(r"D:\Users\阿哲\Desktop\文章\修稿上传")
warnings.filterwarnings("ignore")

# 创建一个空的DataFrame并存储每次循环的评估结果
evaluation_results = pd.DataFrame(
    columns=['Fingerprint', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC', 'MCC', 'Best Params'])

# 1) 获取数据
file_path = "cla_data_rfe.xlsx"
df = pd.read_excel(file_path, sheet_name=None)

for fp in df:
    X = df[fp].iloc[:, 2:]  #从第三列开始读取
    y = df[fp]['Y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # 使用 SMOTETomek 平衡数据
    smt = SMOTETomek(random_state=42)
    X_train, y_train = smt.fit_resample(X_train, y_train)
    # 标准化处理
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    # 使用 SMOTETomek 平衡数据
    smt = SMOTETomek(random_state=42)
    X_train_resampled, y_train_resampled = smt.fit_resample(X_train, y_train)
    # GBDT模型和超参数优化
    gbdt = GradientBoostingClassifier()
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 4, 5],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(gbdt, param_grid, cv=5, scoring='f1')
    grid_search.fit(X_train_resampled, y_train_resampled)
    best_gbdt_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # 预测
    y_pred = best_gbdt_model.predict(X_test)
    y_pred_prob = best_gbdt_model.predict_proba(X_test)[:, 1]

    # 计算评价指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_pred_prob)

    # 存储评估结果到DataFrame中
    evaluation_results = pd.concat([evaluation_results, pd.DataFrame({
        'Fingerprint': [fp],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1 Score': [f1],
        'AUC-ROC': [auc_roc],
        'MCC': [mcc],
        'Best Params': [best_params]
    })], ignore_index=True)

# 打印评估结果
print(evaluation_results)
evaluation_results.to_csv('GBDT_results.csv', index=False)
