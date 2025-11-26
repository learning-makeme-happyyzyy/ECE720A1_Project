import csv
import numpy as np
import glob

# 获取所有特征重要性文件
file_names = glob.glob('*no_perm')

feature_importance_dict = {}
for file_name in file_names:
    with open(file_name, newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar="\"")
        for row in reader:
            for feature, importance in row.items():
                if feature not in feature_importance_dict:
                    feature_importance_dict[feature] = []
                feature_importance_dict[feature].append(float(importance))

# 计算平均特征重要性
avg_feature_importance = {feature: np.mean(importances) for feature, importances in feature_importance_dict.items()}
sorted_avg_feature_importance = sorted(avg_feature_importance.items(), key=lambda x: x[1], reverse=True)

# 保存平均特征重要性
with open('avg_feature_importance.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['feature', 'average_importance'])
    for feature, avg_imp in sorted_avg_feature_importance:
        writer.writerow([feature, avg_imp])

# 打印前10个最重要的特征
print("Top 10 features:")
for feature, avg_imp in sorted_avg_feature_importance[:10]:
    print(f"{feature}: {avg_imp}")