import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 数据准备
xgb_data = {
    "Feature": [
        "TYPE_DECLARATION",
        "Dev Refactoring Experience",
        "# Previous Refactorings",
        "# of Files",
        "LOC Left Side",
        "Average number of files",
        "Dev Refactoring Commit Experience",
        "SINGLE_VARIABLE_DECLARATION",
        "refType_Add Parameter",
        "Age"
    ],
    "SHAP": [0.2288, 0.2175, 0.2022, 0.1997, 0.1761, 0.1488, 0.1362, 0.1360, 0.1354, 0.1346]
}

rf_data = {
    "Feature": [
        "TYPE_DECLARATION",
        "# of Files",
        "Average number of files",
        "SINGLE_VARIABLE_DECLARATION",
        "TYPE",
        "METHOD_DECLARATION",
        "LOC Right Side",
        "FIELD_DECLARATION",
        "LOC Left Side",
        "refactoryCount"
    ],
    "SHAP": [0.0201, 0.0190, 0.0187, 0.0176, 0.0166, 0.0153, 0.0151, 0.0147, 0.0142, 0.0141]
}

# 转换为 DataFrame
df_xgb = pd.DataFrame(xgb_data)
df_rf = pd.DataFrame(rf_data)

# 设置样式
sns.set(style="whitegrid", font_scale=0.9)
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=False)

# XGBoost
sns.barplot(
    ax=axes[0],
    y="Feature", x="SHAP",
    data=df_xgb,
    color="sandybrown"
)
axes[0].set_title("Top 10 Average SHAP Importance - XGBoost")
axes[0].set_xlabel("Mean |SHAP value|")
axes[0].set_ylabel("")

# Random Forest
sns.barplot(
    ax=axes[1],
    y="Feature", x="SHAP",
    data=df_rf,
    color="steelblue"
)
axes[1].set_title("Top 10 Average SHAP Importance - Random Forest")
axes[1].set_xlabel("Mean |SHAP value|")
axes[1].set_ylabel("")

plt.tight_layout()
plt.show()
fig.savefig("shap_feature_importance_comparison.png", dpi=300)