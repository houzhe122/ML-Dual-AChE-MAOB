import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib

# 设置 matplotlib 后端为 'Agg'
matplotlib.use('Agg')

# 假设你有多个 CSV 文件，每个文件包含一个模型的结果
file_paths = [
    "KNN_results.csv",
    "SVM_results.csv",
    "RF_results.csv",
    "GBDT_results.csv",
    "LGBM_results.csv",
]

# 读取所有文件并合并到一个 DataFrame
dfs = []
for file in file_paths:
    df = pd.read_csv(file)
    model_name = file.split('_')[0]  # 从文件名中提取模型名称
    df['Model'] = model_name
    dfs.append(df)

all_results = pd.concat(dfs)

# 定义需要绘制的指标
metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC', 'MCC']

# 创建一个包含模型和指标的交叉表
pivot_tables = {metric: all_results.pivot(index='Fingerprint', columns='Model', values=metric) for metric in metrics}

# 设置画布和子图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

# 绘制每个指标的热图
for i, metric in enumerate(metrics):
    sns.heatmap(pivot_tables[metric], annot=True, cmap='viridis', fmt='.2f', ax=axes[i])
    axes[i].set_title(f'Heatmap of {metric}')
    axes[i].set_ylabel('')
    axes[i].set_xlabel('')

# 调整布局
plt.tight_layout()

# 保存热图为文件
plt.savefig("combined_heatmaps.png")
plt.close()

print("Combined heatmap saved as 'combined_heatmaps.png'.")
