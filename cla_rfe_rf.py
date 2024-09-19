import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import warnings
import os
from openpyxl import Workbook
from openpyxl.utils.dataframe import dataframe_to_rows

warnings.filterwarnings("ignore")
os.chdir(r"D:\Users\阿哲\Desktop\文章")

# 创建一个新的 Excel 工作簿
output_path = "cla_data_rfe平衡.xlsx"
wb = Workbook()

# 定义需要处理的 Excel 文件路径
input_path = "工作簿1.xlsx"
# 加载Excel文件，获取所有表单
xls = pd.ExcelFile(input_path)
sheet_names = xls.sheet_names

# 遍历每个表单
for sheet in sheet_names:
    # 读取当前表单数据
    df = pd.read_excel(input_path, sheet_name=sheet, index_col='Smiles')

    # 分离特征和目标
    X = df.drop(columns=['Y'])
    y = df['Y']

    # 使用分类模型进行特征筛选
    classifier = RandomForestClassifier(class_weight='balanced')
    rfe = RFE(estimator=classifier, n_features_to_select=50, step=1)
    rfe.fit(X, y)

    # 获取经过筛选的特征
    feature_mask = rfe.support_
    X_selected = X.loc[:, feature_mask]
    X_selected.index = df.index
    X_selected.insert(0, "Y", y)

    # 将特征筛选后的数据保存到新的工作簿的不同表单
    ws = wb.create_sheet(title=sheet)
    for r in dataframe_to_rows(X_selected, index=True, header=True):
        ws.append(r)

# 删除默认创建的空表单
del wb['Sheet']

# 保存新的 Excel 文件
wb.save(output_path)
print(f"特征筛选完成，已将结果保存到 {output_path}。")

# 辅助函数：将DataFrame转换为Excel表格行
from openpyxl.utils.dataframe import dataframe_to_rows
