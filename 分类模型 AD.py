import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
from lightgbm import LGBMRegressor
from sklearn.ensemble import GradientBoostingClassifier
import os
os.chdir(r"D:\Users\阿哲\Desktop\文章\修稿上传")
from sklearn.preprocessing import StandardScaler
from math import sqrt

from sklearn.model_selection import train_test_split

def show_metrics(y_true, y_pred):
    R2 = r2_score(y_true, y_pred)
    MAE = mean_absolute_error(y_true, y_pred)
    RMSE = sqrt(mean_squared_error(y_true, y_pred))
    return [R2, MAE, RMSE]

# 优化后的 cal_h 函数，提前计算伪逆
def cal_h(tr_x, X):
    tr_arr = tr_x.values
    XTX_inv = np.linalg.pinv(np.dot(tr_arr.T, tr_arr))  # 预先计算伪逆
    X_arr = X.values
    h = np.einsum('ij,jk,ik->i', X_arr, XTX_inv, X_arr)  # 使用爱因斯坦求和加速矩阵运算
    return h

# 使用矢量化操作替代循环
def cal_performance_AD(df, h_):
    in_AD = (df['h'] <= h_) & (-3 <= (df['y_true'] - df['y_pred'])) & ((df['y_true'] - df['y_pred']) <= 3)
    coverage = in_AD.mean()  # 使用平均值计算覆盖率
    metrics = show_metrics(df['y_true'][in_AD], df['y_pred'][in_AD])
    metrics.append(coverage)
    return metrics

def cal_williams(tr_x, tr_y, tr_pred, te_x, te_y, te_pred, williams_path):
    # 计算训练和测试集的 h 值
    tr_h = cal_h(tr_x, tr_x)
    h_ = 3 * (len(tr_x.columns) + 1) / len(tr_x)
    te_h = cal_h(tr_x, te_x)

    # 创建训练和测试集的 DataFrame
    df_tr = pd.DataFrame({'h': tr_h, 'y_true': tr_y, 'y_pred': tr_pred})
    df_te = pd.DataFrame({'h': te_h, 'y_true': te_y, 'y_pred': te_pred})

    # 计算训练和测试集的适用性域性能
    results_tr = cal_performance_AD(df_tr, h_)
    results_te = cal_performance_AD(df_te, h_)

    # 绘制威廉姆斯图
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 16
    plt.xlim([-0.2, 2])
    plt.ylim([-4, 4])
    plt.plot(tr_h, (df_tr['y_true'] - df_tr['y_pred']), 'bo', markerfacecolor='none', label='training set', alpha=0.5, markersize=5)
    plt.plot(te_h, (df_te['y_true'] - df_te['y_pred']), 'rx', label='test set', alpha=0.5, markersize=5)
    plt.axvline(h_, color='black', linestyle="--", lw=1)
    plt.axhline(-3, color='black', linestyle="--", lw=1)
    plt.axhline(3, color='black', linestyle="--", lw=1)
    plt.legend(loc='best')
    plt.xlabel("hi")
    plt.ylabel("Standardized residual")
    plt.savefig(williams_path, dpi=600, bbox_inches="tight")

    # 返回结果
    result_dict = {
        'h_': h_,
        'tr_coverage': results_tr[-1], 'tr_R2': results_tr[0], 'tr_MAE': results_tr[1], 'tr_RMSE': results_tr[2],
        'te_coverage': results_te[-1], 'te_R2': results_te[0], 'te_MAE': results_te[1], 'te_RMSE': results_te[2]
    }
    return result_dict

# 1) 获取数据
file_path = "cla_data_rfe.xlsx"
df = pd.read_excel(file_path,sheet_name="ECFP4", index_col = 'ID')

X = df.drop(columns = ['Y'])
y = df['Y']

# 分割训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state= 42)
# 标准化特征数据
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 将标准化后的 numpy 数组转换回 DataFrame
X_train_df = pd.DataFrame(X_train, columns=df.drop(columns=['Y']).columns)
X_test_df = pd.DataFrame(X_test, columns=df.drop(columns=['Y']).columns)

# 创建和训练模型
Best_para = {'learning_rate': 0.1, 'max_depth': 5, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 300}


#gbdt.fit(X_train, y_train)LGBMRegressor(**Best_para)
best_model = GradientBoostingClassifier(**Best_para)
best_model.fit(X_train, y_train)
# 对训练集和测试集进行预测
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# 指定保存威廉姆斯图的路径
williams_path = 'williams_plot.png分类'

# 调用 cal_williams 函数
result_dict = cal_williams(X_train_df, y_train, y_train_pred, X_test_df, y_test, y_test_pred, williams_path)

# 打印结果
print(result_dict)