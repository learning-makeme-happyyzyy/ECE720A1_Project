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

# run first with False to generate necessary files, the True to use them to remove the features.
removing_Features = True

def feature_name_mapping(feature_names):
    """
    将特征名称映射到原文使用的名称
    """
    mapping = {
        'num_touched_before': '# Previous Refactorings',
        'age': 'Age',
        'dev_ref_exp': 'Dev Refactoring Experience',
        'dev_ref_com_exp': 'Dev Refactoring Commit Experience',
        'leftLocationCount': '# of left side locations',
        'rightLocationCount': '# of right side locations',
        'leftLocationDiff': 'LOC Left Side',
        'rightLocationDiff': 'LOC Right Side',
        'fileTouchCount': '# of Files',
        'fileTouchAverage': 'Average number of files',
        'refactoryTypeCount': '# of Refactorings',
        'codeElementCount': '# of unique code elements'
    }
    
    mapped_names = []
    for feat in feature_names:
        if feat in mapping:
            mapped_names.append(mapping[feat])
        else:
            # 对于代码元素类型特征，保持原样
            mapped_names.append(feat)
    
    return mapped_names

def accurate_shap_analysis_cross_fold(models, X_tests, feature_names, project_name, model_type):
    """
    跨fold平均的SHAP分析
    """
    print(f"进行跨fold SHAP分析 for {project_name} ({model_type})...")
    
    try:
        all_shap_importances = []
        
        for i, (model, X_test) in enumerate(zip(models, X_tests)):
            print(f"  Fold {i+1} SHAP分析...")
            
            # 使用较小的样本以提高计算效率
            sample_size = min(100, len(X_test))
            X_sample = X_test.sample(n=sample_size, random_state=42) if hasattr(X_test, 'sample') else X_test[:sample_size]
            
            # 创建解释器
            explainer = shap.TreeExplainer(model)
            
            # 计算SHAP值
            shap_values = explainer.shap_values(X_sample)
            
            # 处理SHAP输出格式
            if isinstance(shap_values, list) and len(shap_values) == 2:
                shap_values = shap_values[1]  # 二分类取正类
            elif len(np.array(shap_values).shape) == 3:
                shap_values = shap_values[:, :, 1]  # 三维数组取正类
            
            shap_values = np.array(shap_values)
            
            # 计算该fold的SHAP重要性
            if len(shap_values.shape) == 2:
                fold_shap_importance = np.abs(shap_values).mean(axis=0)
            else:
                fold_shap_importance = np.abs(shap_values)
            
            all_shap_importances.append(fold_shap_importance)
        
        # 跨fold平均SHAP重要性
        if len(all_shap_importances) > 0:
            # 确保所有fold的特征数量一致
            min_features = min(imp.shape[0] for imp in all_shap_importances)
            all_shap_importances = [imp[:min_features] for imp in all_shap_importances]
            feature_names_used = feature_names[:min_features]
            
            avg_shap_importance = np.mean(all_shap_importances, axis=0)
            
            # 创建特征重要性DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': feature_names_used,
                'mean_abs_shap': avg_shap_importance
            }).sort_values('mean_abs_shap', ascending=False)
            
            # 映射特征名称
            feature_importance_df['feature_mapped'] = feature_name_mapping(feature_importance_df['feature'].tolist())
            
            print(f"\n{project_name} ({model_type}) - 跨fold平均SHAP重要性 Top 10:")
            for i, row in feature_importance_df.head(10).iterrows():
                print(f"  {i+1}. {row['feature_mapped']}: {row['mean_abs_shap']:.4f}")
            
            return feature_importance_df
        
        return None
        
    except Exception as e:
        print(f"跨fold SHAP分析失败 for {project_name}: {str(e)}")
        return None

def run_model_with_xgboost_shap_improved():
    """
    改进的扩展分析，保持与原文一致的数据处理流程
    """
    print("Starting improved extended analysis with XGBoost and SHAP...")
    
    # 数据加载和预处理（与原文完全一致）
    feature_regular_df = pd.read_csv("commit_all_features.csv")
    commits_info = pd.read_csv("refactoring_commits.csv")
    features = pd.merge(feature_regular_df, commits_info, on="commit_id")
    features = features.drop(['commit_id', "id", "test", "revision_hash"], axis=1)
    features['label'] = features['label'].apply(lambda x: True if x == "co-occur" else False)
    features = features.groupby("url")

    features = sorted(features, key=lambda x: len(x[1].loc[x[1]['label'] == True].index), reverse=True)
    
    # 存储结果
    tot_metrics_rf = {}
    tot_metrics_xgb = {}
    shap_results = {}
    
    project_count = 0
    
    for project_to_test in features[:10]:  # 处理所有10个项目
        if len(project_to_test[1].index) < 300:
            continue
            
        url = project_to_test[0]
        project_data = project_to_test[1]
        
        # 检查是否有足够的共现重构提交（与原文一致）
        found_co_occur = project_data['label'].sum()
        if found_co_occur < 10:
            print(f"Skipping {url} - insufficient co-occurring commits: {found_co_occur}")
            continue

        print(f"\n{'='*60}")
        print(f"Processing project: {url}")
        print(f"Total commits: {len(project_data)}, Co-occurring commits: {found_co_occur}")
        print(f"{'='*60}")

        # 特征选择（与原文一致）
        if not removing_Features:
            features_to_remove = []
            try:
                with open(url.split('/')[-1] + "no_perm", newline='') as csvfile:
                    reader = csv.DictReader(csvfile, delimiter=',', quotechar="\"")
                    val = {}
                    for row in reader:
                        val = row
                    for feat, feat_importance in val.items():
                        if float(feat_importance) <= -0.001:  # 与原文相同的阈值
                            features_to_remove.append(feat)
                project_data = project_data.drop(features_to_remove, axis=1)
                print(f"Removed {len(features_to_remove)} features")
            except:
                print("No feature removal file found, using all features")
        
        project_data = project_data.drop(['url'], axis=1)
        X, y = project_data.iloc[:, :-1], project_data.iloc[:, -1]
        feature_names = X.columns.tolist()
        
        # 映射特征名称
        mapped_feature_names = feature_name_mapping(feature_names)
        
        # 准备交叉验证（与原文相同的10-fold CV）
        kfolds = KFold(n_splits=10, shuffle=True, random_state=42)
        
        # 存储每个模型的指标
        rf_metrics = {"auc": [], "f1": [], "accuracy": [], "recall": [], "precision": []}
        xgb_metrics = {"auc": [], "f1": [], "accuracy": [], "recall": [], "precision": []}
        
        # 用于SHAP分析的模型和测试集
        rf_models = []
        xgb_models = []
        rf_X_tests = []
        xgb_X_tests = []
        
        fold_num = 0
        for train_idx, test_idx in kfolds.split(X):
            fold_num += 1
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]
            
            # 处理类别不平衡（与原文相同的SMOTE）
            oversample = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = oversample.fit_resample(X_train, y_train)
            
            # ========== Random Forest with GridSearch (与原文一致) ==========
            rf_param_grid = {
                'bootstrap': [True],
                'max_depth': [50, 100, 150],
                'max_features': ['auto', 'sqrt', 'log2'],
                'n_estimators': [50, 100, 200, 300]
            }
            
            rf_grid_search = GridSearchCV(
                estimator=RandomForestClassifier(),
                param_grid=rf_param_grid,
                cv=3, n_jobs=2, verbose=0
            )
            rf_grid_search.fit(X_train_resampled, y_train_resampled)
            
            rf_model = RandomForestClassifier(
                bootstrap=rf_grid_search.best_params_['bootstrap'],
                n_estimators=rf_grid_search.best_params_['n_estimators'],
                max_depth=rf_grid_search.best_params_['max_depth'],
                max_features=rf_grid_search.best_params_['max_features'],
                random_state=42
            )
            rf_model.fit(X_train_resampled, y_train_resampled)
            rf_preds = rf_model.predict(X_test)
            
            rf_metrics["auc"].append(roc_auc_score(y_test, rf_preds))
            rf_metrics["f1"].append(f1_score(y_test, rf_preds))
            rf_metrics["accuracy"].append(accuracy_score(y_test, rf_preds))
            rf_metrics["recall"].append(recall_score(y_test, rf_preds))
            rf_metrics["precision"].append(precision_score(y_test, rf_preds))
            
            rf_models.append(rf_model)
            rf_X_tests.append(X_test)
            
            # ========== XGBoost with GridSearch ==========
            xgb_param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
            
            xgb_grid_search = GridSearchCV(
                estimator=XGBClassifier(random_state=42, eval_metric='logloss'),
                param_grid=xgb_param_grid,
                cv=3, n_jobs=2, verbose=0
            )
            xgb_grid_search.fit(X_train_resampled, y_train_resampled)
            
            xgb_model = XGBClassifier(
                n_estimators=xgb_grid_search.best_params_['n_estimators'],
                max_depth=xgb_grid_search.best_params_['max_depth'],
                learning_rate=xgb_grid_search.best_params_['learning_rate'],
                subsample=xgb_grid_search.best_params_['subsample'],
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
            
            xgb_models.append(xgb_model)
            xgb_X_tests.append(X_test)
        
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
        
        # 跨fold SHAP分析
        print(f"\nPerforming cross-fold SHAP analysis for {url}...")
        
        # Random Forest SHAP分析
        shap_rf_result = accurate_shap_analysis_cross_fold(
            rf_models, rf_X_tests, mapped_feature_names,
            url.split('/')[-1], "RandomForest"
        )
        
        # XGBoost SHAP分析
        shap_xgb_result = accurate_shap_analysis_cross_fold(
            xgb_models, xgb_X_tests, mapped_feature_names,
            url.split('/')[-1], "XGBoost"
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
    
    print(f"\nXGBoost - Average across {len(tot_metrics_xgb)} projects:")
    print(f"  AUC: {xgb_auc_mean:.4f}")
    print(f"  F1: {st.mean([m['f1'] for m in tot_metrics_xgb.values()]):.4f}")
    
    print(f"\nOverall Improvement:")
    print(f"  AUC: {xgb_auc_mean - rf_auc_mean:.4f} ({(xgb_auc_mean - rf_auc_mean)/rf_auc_mean*100:.2f}%)")
    
    # 统计显著性检验
    t_stat, p_value = statistical_significance_test(tot_metrics_rf, tot_metrics_xgb)
    
    # 保存详细结果
    save_detailed_results(tot_metrics_rf, tot_metrics_xgb, shap_results, average_shap_results, t_stat, p_value)
    
    return tot_metrics_rf, tot_metrics_xgb, shap_results, average_shap_results

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
                feature = row['feature_mapped']
                importance = row['mean_abs_shap']
                if feature not in xgb_importances:
                    xgb_importances[feature] = []
                xgb_importances[feature].append(importance)
        
        # Random Forest特征重要性
        if results["RandomForest"] is not None:
            for _, row in results["RandomForest"].iterrows():
                feature = row['feature_mapped']
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

def save_detailed_results(tot_metrics_rf, tot_metrics_xgb, shap_results, average_shap_results, t_stat, p_value):
    """
    保存详细结果
    """
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
            "rf_mean_auc": st.mean([m['auc'] for m in tot_metrics_rf.values()]),
            "xgb_mean_auc": st.mean([m['auc'] for m in tot_metrics_xgb.values()]),
            "improvement": st.mean([m['auc'] for m in tot_metrics_xgb.values()]) - st.mean([m['auc'] for m in tot_metrics_rf.values()]),
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
    with open('extended_analysis_results_improved.json', 'w') as f:
        json.dump(results_summary, f, indent=2, cls=NumpyEncoder)
    
    print(f"\nDetailed results saved to 'extended_analysis_results_improved.json'")

if __name__ == '__main__':
    print("Starting Improved Extended RQ2 Analysis with XGBoost and SHAP")
    print("="*60)
    
    # 运行改进的分析
    tot_metrics_rf, tot_metrics_xgb, shap_results, average_shap_results = run_model_with_xgboost_shap_improved()
    
    print("\n" + "="*60)
    print("IMPROVED EXTENDED ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*60)