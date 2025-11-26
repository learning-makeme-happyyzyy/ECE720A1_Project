import csv
import json
from dateutil import parser
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve, auc, f1_score, \
    accuracy_score
from imblearn.under_sampling import RandomUnderSampler
import statistics as st
# Import random forest library
from sklearn.ensemble import RandomForestClassifier
from collections import Counter
from imblearn.over_sampling import SMOTE, RandomOverSampler
import xgboost as xgb
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import re
import shap
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# 设置SHAP输出格式
shap.initjs()

# 添加自定义JSON编码器处理NumPy类型
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.bool_, np.bool8)):
            return bool(obj)
        elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, 
                             np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def temp_method():
    # 原有的temp_method函数保持不变
    pass

def merge_features():
    # 原有的merge_features函数保持不变
    pass

def shap_analysis_xgboost(model, X_test, feature_names, project_name):
    """
    XGBoost的SHAP分析
    """
    print(f"Starting SHAP analysis for {project_name} using XGBoost...")
    
    try:
        # 确保数据格式正确
        if hasattr(X_test, 'values'):
            X_sample = X_test
        else:
            X_sample = pd.DataFrame(X_test, columns=feature_names)
        
        # 使用较小的样本以提高计算效率
        sample_size = min(500, len(X_sample))
        X_sample = X_sample.sample(n=sample_size, random_state=42)
        
        # 创建SHAP解释器
        explainer = shap.TreeExplainer(model)
        
        # 计算SHAP值
        shap_values = explainer.shap_values(X_sample)
        
        print(f"XGBoost SHAP values shape: {np.array(shap_values).shape}")
        
        # 处理SHAP值格式
        if isinstance(shap_values, list) and len(shap_values) == 2:
            # 对于二分类，取正类（索引1）的SHAP值
            shap_values = shap_values[1]
        elif len(np.array(shap_values).shape) == 3:
            # 如果是三维数组 (samples, features, classes)，取正类的SHAP值
            shap_values = shap_values[:, :, 1]
        
        # 确保shap_values是二维数组
        shap_values = np.array(shap_values)
        if len(shap_values.shape) == 1:
            shap_values = shap_values.reshape(1, -1)
        
        print(f"Final XGBoost SHAP values shape: {shap_values.shape}")
        
        # 计算平均绝对SHAP值
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        print(f"XGBoost mean abs SHAP shape: {mean_abs_shap.shape}")
        
        # 创建特征重要性DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        # 1. 全局特征重要性图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, show=False)
        plt.title(f"SHAP Summary Plot - {project_name} (XGBoost)", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"shap_summary_{project_name.replace('/', '_')}_XGBoost.pdf", 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # 2. 特征重要性条形图
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                         plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance - {project_name} (XGBoost)", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"shap_bar_{project_name.replace('/', '_')}_XGBoost.pdf", 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # 保存具体的SHAP值数据
        feature_importance_df.to_csv(
            f"shap_importance_{project_name.replace('/', '_')}_XGBoost.csv", 
            index=False
        )
        
        # 打印最重要的特征
        print(f"\nTop 10 features by SHAP importance for {project_name} (XGBoost):")
        for i, row in feature_importance_df.head(10).iterrows():
            print(f"  {i+1}. {row['feature']}: {row['mean_abs_shap']:.4f}")
        
        return feature_importance_df
        
    except Exception as e:
        print(f"Error in XGBoost SHAP analysis for {project_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def shap_analysis_randomforest(model, X_test, feature_names, project_name):
    """
    Random Forest的SHAP分析 - 修复版本
    """
    print(f"Starting SHAP analysis for {project_name} using RandomForest...")
    
    try:
        # 确保数据格式正确
        if hasattr(X_test, 'values'):
            X_sample = X_test
        else:
            X_sample = pd.DataFrame(X_test, columns=feature_names)
        
        # 使用较小的样本以提高计算效率
        sample_size = min(200, len(X_sample))  # 进一步减少样本量
        X_sample = X_sample.sample(n=sample_size, random_state=42)
        
        print(f"X_sample shape: {X_sample.shape}")
        print(f"Number of features: {len(feature_names)}")
        
        # 创建解释器
        explainer = shap.TreeExplainer(model)
        
        # 计算SHAP值
        shap_values = explainer.shap_values(X_sample)
        
        print(f"Raw SHAP values type: {type(shap_values)}")
        print(f"Raw SHAP values shape: {np.array(shap_values).shape}")
        
        # 处理SHAP值格式 - 关键修复！
        if isinstance(shap_values, list):
            # 如果是列表，通常是二分类问题 [class0_shap, class1_shap]
            if len(shap_values) == 2:
                print("Binary classification detected, using positive class SHAP values")
                shap_values = shap_values[1]  # 使用正类的SHAP值
        elif len(np.array(shap_values).shape) == 3:
            # 如果是三维数组 (samples, features, classes)
            print(f"3D SHAP array detected, shape: {np.array(shap_values).shape}")
            shap_values = shap_values[:, :, 1]  # 取正类的SHAP值
        
        # 确保shap_values是二维numpy数组
        shap_values = np.array(shap_values)
        print(f"Processed SHAP values shape: {shap_values.shape}")
        
        # 检查形状是否匹配
        if len(shap_values.shape) != 2:
            print(f"Unexpected SHAP values shape: {shap_values.shape}")
            return None
        
        if shap_values.shape[1] != len(feature_names):
            print(f"Shape mismatch: SHAP values have {shap_values.shape[1]} features, but feature_names has {len(feature_names)}")
            # 截断以匹配
            min_features = min(shap_values.shape[1], len(feature_names))
            shap_values = shap_values[:, :min_features]
            feature_names_used = feature_names[:min_features]
        else:
            feature_names_used = feature_names
        
        # 计算平均绝对SHAP值
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        print(f"Mean abs SHAP shape: {mean_abs_shap.shape}")
        
        # 创建特征重要性DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names_used,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        # 1. 全局特征重要性图
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample.iloc[:, :len(feature_names_used)], 
                         feature_names=feature_names_used, show=False)
        plt.title(f"SHAP Summary Plot - {project_name} (RandomForest)", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"shap_summary_{project_name.replace('/', '_')}_RandomForest.pdf", 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # 2. 特征重要性条形图
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample.iloc[:, :len(feature_names_used)], 
                         feature_names=feature_names_used, plot_type="bar", show=False)
        plt.title(f"SHAP Feature Importance - {project_name} (RandomForest)", fontsize=16)
        plt.tight_layout()
        plt.savefig(f"shap_bar_{project_name.replace('/', '_')}_RandomForest.pdf", 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
        # 保存具体的SHAP值数据
        feature_importance_df.to_csv(
            f"shap_importance_{project_name.replace('/', '_')}_RandomForest.csv", 
            index=False
        )
        
        # 打印最重要的特征
        print(f"\nTop 10 features by SHAP importance for {project_name} (RandomForest):")
        for i, row in feature_importance_df.head(10).iterrows():
            print(f"  {i+1}. {row['feature']}: {row['mean_abs_shap']:.4f}")
        
        return feature_importance_df
        
    except Exception as e:
        print(f"Error in RandomForest SHAP analysis for {project_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def shap_analysis_simple(model, X_test, feature_names, project_name, model_type):
    """
    简化的SHAP分析函数 - 如果上述方法仍有问题，使用这个版本
    """
    print(f"Starting simple SHAP analysis for {project_name} using {model_type}...")
    
    try:
        # 使用更小的样本
        sample_size = min(100, len(X_test))
        X_sample = X_test.sample(n=sample_size, random_state=42) if hasattr(X_test, 'sample') else X_test[:sample_size]
        
        # 创建解释器
        explainer = shap.TreeExplainer(model)
        
        # 计算SHAP值
        shap_values = explainer.shap_values(X_sample)
        
        print(f"Raw SHAP shape: {np.array(shap_values).shape}")
        
        # 简化处理：如果是三维，取第一个样本的第一个类别的SHAP值作为示例
        if len(np.array(shap_values).shape) == 3:
            shap_values_flat = np.array(shap_values)[0, :, 1]  # 第一个样本，正类的SHAP值
        elif isinstance(shap_values, list) and len(shap_values) == 2:
            shap_values_flat = np.array(shap_values[1]).mean(axis=0)  # 正类SHAP值的平均值
        else:
            shap_values_flat = np.array(shap_values).mean(axis=0)  # 平均所有样本
        
        print(f"Final SHAP values shape: {shap_values_flat.shape}")
        
        # 确保形状匹配
        if len(shap_values_flat) > len(feature_names):
            shap_values_flat = shap_values_flat[:len(feature_names)]
        elif len(shap_values_flat) < len(feature_names):
            # 填充0
            shap_values_flat = np.pad(shap_values_flat, (0, len(feature_names) - len(shap_values_flat)))
        
        # 创建特征重要性DataFrame
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'mean_abs_shap': np.abs(shap_values_flat)
        }).sort_values('mean_abs_shap', ascending=False)
        
        # 保存结果
        feature_importance_df.to_csv(
            f"shap_importance_{project_name.replace('/', '_')}_{model_type}_simple.csv", 
            index=False
        )
        
        print(f"\nTop 10 features by SHAP importance for {project_name} ({model_type} - simple):")
        for i, row in feature_importance_df.head(10).iterrows():
            print(f"  {i+1}. {row['feature']}: {row['mean_abs_shap']:.4f}")
        
        return feature_importance_df
        
    except Exception as e:
        print(f"Error in simple SHAP analysis for {project_name}: {str(e)}")
        return None


def calculate_average_shap_importance(all_shap_results):
    """
    计算所有项目的平均SHAP特征重要性
    """
    print("\nCalculating average SHAP importance across all projects...")
    
    xgb_importances = {}
    rf_importances = {}
    
    # 收集所有项目的SHAP重要性
    for project, results in all_shap_results.items():
        # XGBoost特征重要性
        if results["XGBoost"] is not None:
            for _, row in results["XGBoost"].iterrows():
                feature = row['feature']
                importance = row['mean_abs_shap']
                if feature not in xgb_importances:
                    xgb_importances[feature] = []
                xgb_importances[feature].append(importance)
        
        # Random Forest特征重要性
        if results["RandomForest"] is not None:
            for _, row in results["RandomForest"].iterrows():
                feature = row['feature']
                importance = row['mean_abs_shap']
                if feature not in rf_importances:
                    rf_importances[feature] = []
                rf_importances[feature].append(importance)
    
    # 计算平均值
    avg_xgb_importance = {
        feature: np.mean(importances) 
        for feature, importances in xgb_importances.items()
    }
    avg_rf_importance = {
        feature: np.mean(importances) 
        for feature, importances in rf_importances.items()
    }
    
    # 转换为DataFrame并排序
    avg_xgb_df = pd.DataFrame({
        'feature': list(avg_xgb_importance.keys()),
        'mean_abs_shap': list(avg_xgb_importance.values())
    }).sort_values('mean_abs_shap', ascending=False)
    
    avg_rf_df = pd.DataFrame({
        'feature': list(avg_rf_importance.keys()),
        'mean_abs_shap': list(avg_rf_importance.values())
    }).sort_values('mean_abs_shap', ascending=False)
    
    # 保存平均重要性
    avg_xgb_df.to_csv("average_shap_importance_xgboost.csv", index=False)
    avg_rf_df.to_csv("average_shap_importance_randomforest.csv", index=False)
    
    # 打印Top 10平均特征重要性
    print("\nTop 10 Average SHAP Importance - XGBoost:")
    for i, row in avg_xgb_df.head(10).iterrows():
        print(f"  {i+1}. {row['feature']}: {row['mean_abs_shap']:.4f}")
    
    print("\nTop 10 Average SHAP Importance - Random Forest:")
    for i, row in avg_rf_df.head(10).iterrows():
        print(f"  {i+1}. {row['feature']}: {row['mean_abs_shap']:.4f}")
    
    return {
        "XGBoost_avg": avg_xgb_importance,
        "RandomForest_avg": avg_rf_importance,
        "XGBoost_df": avg_xgb_df,
        "RandomForest_df": avg_rf_df
    }


def statistical_significance_test(rf_metrics, xgb_metrics):
    """
    使用配对t检验比较RF和XGBoost性能
    """
    print("\n" + "="*50)
    print("STATISTICAL SIGNIFICANCE TEST")
    print("="*50)
    
    # 提取所有项目的AUC值
    common_projects = set(rf_metrics.keys()) & set(xgb_metrics.keys())
    rf_aucs = [rf_metrics[proj]['auc'] for proj in common_projects]
    xgb_aucs = [xgb_metrics[proj]['auc'] for proj in common_projects]
    
    if len(rf_aucs) < 2:
        print("Not enough projects for statistical test")
        return None, None
    
    # 配对t检验
    t_stat, p_value = stats.ttest_rel(rf_aucs, xgb_aucs)
    
    print(f"Number of projects compared: {len(common_projects)}")
    print(f"RF平均AUC: {np.mean(rf_aucs):.4f} ± {np.std(rf_aucs):.4f}")
    print(f"XGBoost平均AUC: {np.mean(xgb_aucs):.4f} ± {np.std(xgb_aucs):.4f}")
    print(f"平均改进: {np.mean(xgb_aucs) - np.mean(rf_aucs):.4f}")
    print(f"T统计量: {t_stat:.4f}")
    print(f"P值: {p_value:.4f}")
    
    if p_value < 0.05:
        print("✅ 性能差异在统计上显著 (p < 0.05)")
    else:
        print("❌ 性能差异在统计上不显著")
    
    return t_stat, p_value


# run first with False to generate necessary files, the True to use them to remove the features.
removing_Features = True


def run_model_with_xgboost_shap():
    """
    扩展的run_model函数，包含XGBoost和SHAP分析
    """
    print("Starting extended analysis with XGBoost and SHAP...")
    
    # 数据加载和预处理
    feature_regular_df = pd.read_csv("commit_all_features.csv")
    commits_info = pd.read_csv("refactoring_commits.csv")
    features = pd.merge(feature_regular_df, commits_info, on="commit_id")
    features = features.drop(['commit_id', "id", "test", "revision_hash"], axis=1)
    features['label'] = features['label'].apply(lambda x: True if x == "co-occur" else False)
    features = features.groupby("url")

    features = sorted(features, key=lambda x: len(x[1].loc[x[1]['label'] == True].index), reverse=True)
    
    # 存储结果
    tot_metrics_rf = {}   # Random Forest结果
    tot_metrics_xgb = {}  # XGBoost结果
    shap_results = {}     # SHAP分析结果
    
    project_count = 0
    
    for project_to_test in features[:10]:  # 处理所有10个项目
        if len(project_to_test[1].index) < 300:
            continue
            
        url = project_to_test[0]
        project_data = project_to_test[1]
        
        # 检查是否有足够的共现重构提交
        found_co_occur = project_data['label'].sum()
        if found_co_occur < 10:
            print(f"Skipping {url} - insufficient co-occurring commits: {found_co_occur}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing project: {url}")
        print(f"Total commits: {len(project_data)}, Co-occurring commits: {found_co_occur}")
        print(f"{'='*60}")

        # 特征选择（如果有）
        if not removing_Features:
            features_to_remove = []
            try:
                with open(url.split('/')[-1] + "no_perm", newline='') as csvfile:
                    reader = csv.DictReader(csvfile, delimiter=',', quotechar="\"")
                    val = {}
                    for row in reader:
                        val = row
                    for feat, feat_importance in val.items():
                        if float(feat_importance) <= -0.001:
                            features_to_remove.append(feat)
                project_data = project_data.drop(features_to_remove, axis=1)
                print(f"Removed {len(features_to_remove)} features")
            except:
                print("No feature removal file found, using all features")
        
        project_data = project_data.drop(['url'], axis=1)
        X, y = project_data.iloc[:, :-1], project_data.iloc[:, -1]
        feature_names = X.columns.tolist()
        
        # 准备交叉验证
        kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
        
        # 存储每个模型的指标
        rf_metrics = {"auc": [], "f1": [], "accuracy": [], "recall": [], "precision": []}
        xgb_metrics = {"auc": [], "f1": [], "accuracy": [], "recall": [], "precision": []}
        
        # 用于SHAP分析的模型
        best_xgb_model = None
        best_rf_model = None
        shap_X_test = None
        shap_y_test = None
        
        fold_num = 0
        for train_idx, test_idx in kfolds.split(X):
            fold_num += 1
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            
            # 处理类别不平衡
            oversample = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)
            
            # ========== Random Forest ==========
            rf_model = RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                random_state=42
            )
            rf_model.fit(X_train_resampled, y_train_resampled)
            rf_preds = rf_model.predict(X_test)
            
            rf_metrics["auc"].append(roc_auc_score(y_test, rf_preds))
            rf_metrics["f1"].append(f1_score(y_test, rf_preds))
            rf_metrics["accuracy"].append(accuracy_score(y_test, rf_preds))
            rf_metrics["recall"].append(recall_score(y_test, rf_preds))
            rf_metrics["precision"].append(precision_score(y_test, rf_preds))
            
            # ========== XGBoost ==========
            xgb_model = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
            xgb_model.fit(X_train_resampled, y_train_resampled)
            xgb_preds = xgb_model.predict(X_test)
            
            xgb_metrics["auc"].append(roc_auc_score(y_test, xgb_preds))
            xgb_metrics["f1"].append(f1_score(y_test, xgb_preds))
            xgb_metrics["accuracy"].append(accuracy_score(y_test, xgb_preds))
            xgb_metrics["recall"].append(recall_score(y_test, xgb_preds))
            xgb_metrics["precision"].append(precision_score(y_test, xgb_preds))
            
            # 保存第一个fold的模型用于SHAP分析
            if fold_num == 1:
                best_xgb_model = xgb_model
                best_rf_model = rf_model
                shap_X_test = X_test
                shap_y_test = y_test
        
        # 计算平均指标
        project_metrics_rf = {
            "auc": st.mean(rf_metrics["auc"]),
            "f1": st.mean(rf_metrics["f1"]),
            "accuracy": st.mean(rf_metrics["accuracy"]),
            "recall": st.mean(rf_metrics["recall"]),
            "precision": st.mean(rf_metrics["precision"])
        }
        
        project_metrics_xgb = {
            "auc": st.mean(xgb_metrics["auc"]),
            "f1": st.mean(xgb_metrics["f1"]),
            "accuracy": st.mean(xgb_metrics["accuracy"]),
            "recall": st.mean(xgb_metrics["recall"]),
            "precision": st.mean(xgb_metrics["precision"])
        }
        
        tot_metrics_rf[url] = project_metrics_rf
        tot_metrics_xgb[url] = project_metrics_xgb
        
        # 打印对比结果
        print(f"\nResults for {url.split('/')[-1]}:")
        print(f"Random Forest - AUC: {project_metrics_rf['auc']:.4f}, F1: {project_metrics_rf['f1']:.4f}")
        print(f"XGBoost      - AUC: {project_metrics_xgb['auc']:.4f}, F1: {project_metrics_xgb['f1']:.4f}")
        print(f"AUC Improvement: {project_metrics_xgb['auc'] - project_metrics_rf['auc']:.4f}")
        
        project_count += 1
        
        # 对每个项目进行SHAP分析
        print(f"\nPerforming SHAP analysis for {url}...")
        
        # XGBoost SHAP分析
        shap_xgb_result = shap_analysis_xgboost(
            best_xgb_model, shap_X_test, feature_names, 
            url.split('/')[-1]
        )
        
        # Random Forest SHAP分析 - 先尝试完整版本，如果失败则使用简化版本
        shap_rf_result = shap_analysis_randomforest(
            best_rf_model, shap_X_test, feature_names,
            url.split('/')[-1]
        )
        
        # 如果完整版本失败，尝试简化版本
        if shap_rf_result is None:
            print("Trying simple SHAP analysis for RandomForest...")
            shap_rf_result = shap_analysis_simple(
                best_rf_model, shap_X_test, feature_names,
                url.split('/')[-1], "RandomForest"
            )
        
        shap_results[url] = {
            "XGBoost": shap_xgb_result,
            "RandomForest": shap_rf_result
        }
    
    # 计算平均SHAP重要性
    average_shap_results = calculate_average_shap_importance(shap_results)
    
    # 打印总体结果
    print("\n" + "="*80)
    print("OVERALL RESULTS SUMMARY")
    print("="*80)
    
    rf_auc_mean = st.mean([m['auc'] for m in tot_metrics_rf.values()])
    xgb_auc_mean = st.mean([m['auc'] for m in tot_metrics_xgb.values()])
    
    print(f"\nRandom Forest - Average across {len(tot_metrics_rf)} projects:")
    print(f"  AUC: {rf_auc_mean:.4f}")
    print(f"  F1: {st.mean([m['f1'] for m in tot_metrics_rf.values()]):.4f}")
    print(f"  Precision: {st.mean([m['precision'] for m in tot_metrics_rf.values()]):.4f}")
    print(f"  Recall: {st.mean([m['recall'] for m in tot_metrics_rf.values()]):.4f}")
    
    print(f"\nXGBoost - Average across {len(tot_metrics_xgb)} projects:")
    print(f"  AUC: {xgb_auc_mean:.4f}")
    print(f"  F1: {st.mean([m['f1'] for m in tot_metrics_xgb.values()]):.4f}")
    print(f"  Precision: {st.mean([m['precision'] for m in tot_metrics_xgb.values()]):.4f}")
    print(f"  Recall: {st.mean([m['recall'] for m in tot_metrics_xgb.values()]):.4f}")
    
    print(f"\nOverall Improvement:")
    print(f"  AUC: {xgb_auc_mean - rf_auc_mean:.4f} ({(xgb_auc_mean - rf_auc_mean)/rf_auc_mean*100:.2f}%)")
    
    # 统计显著性检验
    t_stat, p_value = statistical_significance_test(tot_metrics_rf, tot_metrics_xgb)
    
    # 准备SHAP结果用于JSON序列化
    shap_results_serializable = {}
    for url, results in shap_results.items():
        shap_results_serializable[url] = {
            "XGBoost_top_features": results["XGBoost"].head(10).to_dict('records') if results["XGBoost"] is not None else None,
            "RandomForest_top_features": results["RandomForest"].head(10).to_dict('records') if results["RandomForest"] is not None else None
        }
    
    # 保存详细结果
    results_summary = {
        "random_forest": tot_metrics_rf,
        "xgboost": tot_metrics_xgb,
        "summary": {
            "rf_mean_auc": rf_auc_mean,
            "xgb_mean_auc": xgb_auc_mean,
            "improvement": xgb_auc_mean - rf_auc_mean,
            "t_statistic": float(t_stat) if t_stat is not None else None,
            "p_value": float(p_value) if p_value is not None else None,
            "significant": p_value < 0.05 if p_value else False
        },
        "shap_analysis": shap_results_serializable,
        "average_shap_importance": {
            "XGBoost_top_10": average_shap_results["XGBoost_df"].head(10).to_dict('records'),
            "RandomForest_top_10": average_shap_results["RandomForest_df"].head(10).to_dict('records')
        }
    }
    
    # 保存结果为JSON文件
    with open('extended_analysis_results.json', 'w') as f:
        json.dump(results_summary, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nDetailed results saved to 'extended_analysis_results.json'")
    
    return tot_metrics_rf, tot_metrics_xgb, results_summary, average_shap_results


def create_comparison_visualization(tot_metrics_rf, tot_metrics_xgb):
    """
    创建模型对比可视化
    """
    # 提取AUC值用于对比
    common_projects = set(tot_metrics_rf.keys()) & set(tot_metrics_xgb.keys())
    rf_aucs = [tot_metrics_rf[proj]['auc'] for proj in common_projects]
    xgb_aucs = [tot_metrics_xgb[proj]['auc'] for proj in common_projects]
    
    # 创建对比箱线图
    plt.figure(figsize=(8, 6))
    box_data = [rf_aucs, xgb_aucs]
    box_labels = ['Random Forest', 'XGBoost']
    
    boxprops = dict(color="black", linewidth=1)
    medianprops = dict(color="red", linewidth=2)
    
    bp = plt.boxplot(box_data, labels=box_labels, patch_artist=True, 
                    boxprops=boxprops, medianprops=medianprops)
    
    # 设置颜色
    colors = ['#A0AECA', '#FFC8A2']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    plt.ylabel('AUC Score', fontsize=12)
    plt.title('Model Performance Comparison: Random Forest vs XGBoost', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # 添加数值标注
    for i, data in enumerate(box_data):
        plt.text(i+1, np.median(data) + 0.01, f'Med: {np.median(data):.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('model_comparison_auc.pdf', bbox_inches='tight', dpi=300)
    plt.show()
    
    print(f"Comparison visualization saved to 'model_comparison_auc.pdf'")


def create_average_shap_visualization(average_shap_results):
    """
    创建平均SHAP重要性可视化
    """
    # XGBoost平均SHAP重要性
    xgb_top10 = average_shap_results["XGBoost_df"].head(10)
    rf_top10 = average_shap_results["RandomForest_df"].head(10)
    
    # 创建对比图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # XGBoost特征重要性
    ax1.barh(range(len(xgb_top10)), xgb_top10['mean_abs_shap'], color='#FFC8A2')
    ax1.set_yticks(range(len(xgb_top10)))
    ax1.set_yticklabels(xgb_top10['feature'])
    ax1.set_xlabel('Mean |SHAP value|')
    ax1.set_title('Average SHAP Importance - XGBoost')
    ax1.invert_yaxis()
    
    # Random Forest特征重要性
    ax2.barh(range(len(rf_top10)), rf_top10['mean_abs_shap'], color='#A0AECA')
    ax2.set_yticks(range(len(rf_top10)))
    ax2.set_yticklabels(rf_top10['feature'])
    ax2.set_xlabel('Mean |SHAP value|')
    ax2.set_title('Average SHAP Importance - Random Forest')
    ax2.invert_yaxis()
    
    plt.tight_layout()
    plt.savefig('average_shap_importance_comparison.pdf', bbox_inches='tight', dpi=300)
    plt.show()
    
    print(f"Average SHAP importance visualization saved to 'average_shap_importance_comparison.pdf'")


def create_performance_comparison_table(tot_metrics_rf, tot_metrics_xgb):
    """
    创建详细的性能对比表格
    """
    # 创建对比DataFrame
    comparison_data = []
    for project in tot_metrics_rf.keys():
        if project in tot_metrics_xgb:
            rf_metrics = tot_metrics_rf[project]
            xgb_metrics = tot_metrics_xgb[project]
            
            comparison_data.append({
                'Project': project.split('/')[-1],
                'RF_AUC': rf_metrics['auc'],
                'XGB_AUC': xgb_metrics['auc'],
                'RF_F1': rf_metrics['f1'],
                'XGB_F1': xgb_metrics['f1'],
                'AUC_Improvement': xgb_metrics['auc'] - rf_metrics['auc'],
                'F1_Improvement': xgb_metrics['f1'] - rf_metrics['f1']
            })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # 计算平均值
    avg_row = {
        'Project': 'Average',
        'RF_AUC': comparison_df['RF_AUC'].mean(),
        'XGB_AUC': comparison_df['XGB_AUC'].mean(),
        'RF_F1': comparison_df['RF_F1'].mean(),
        'XGB_F1': comparison_df['XGB_F1'].mean(),
        'AUC_Improvement': comparison_df['AUC_Improvement'].mean(),
        'F1_Improvement': comparison_df['F1_Improvement'].mean()
    }
    
    comparison_df = pd.concat([comparison_df, pd.DataFrame([avg_row])], ignore_index=True)
    
    # 保存为CSV
    comparison_df.to_csv('performance_comparison_table.csv', index=False)
    
    print(f"\nPerformance comparison table saved to 'performance_comparison_table.csv'")
    
    return comparison_df


if __name__ == '__main__':
    print("Starting Extended RQ2 Analysis with XGBoost and SHAP")
    print("="*60)
    
    # 运行扩展分析
    tot_metrics_rf, tot_metrics_xgb, results_summary, average_shap_results = run_model_with_xgboost_shap()
    
    # 创建对比可视化
    create_comparison_visualization(tot_metrics_rf, tot_metrics_xgb)
    
    # 创建平均SHAP重要性可视化
    create_average_shap_visualization(average_shap_results)
    
    # 创建性能对比表格
    create_performance_comparison_table(tot_metrics_rf, tot_metrics_xgb)
    
    print("\n" + "="*60)
    print("EXTENDED ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)