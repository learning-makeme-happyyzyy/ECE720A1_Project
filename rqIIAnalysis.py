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

from rqIAnalysis import get_repos_to_commits_to_refactors, extract_rminer_data


def temp_method():
    rcr = get_repos_to_commits_to_refactors(lambda refactor_list: refactor_list)

    s = {}
    with open('commit.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar="\"")
        for row in reader:
            commit_id = row['_id']
            for repo, commit_list in rcr.items():
                if commit_id in commit_list:
                    if repo not in s:
                        s[repo] = {}
                    if commit_id not in s[repo]:
                        s[repo][commit_id] = {}
                    s[repo][commit_id]["ref_list"] = rcr[repo][commit_id]
                    s[repo][commit_id]["data"] = row

    projects_to_commits = {}
    for repo, commit_dict in s.items():
        projects_to_commits[repo] = sorted(commit_dict.values(), key=lambda d: d["data"]["author_date"])

    the_commit_list = []
    for repo, commit_list in projects_to_commits.items():
        file_list = {}
        developer_list = {}
        for commit in commit_list:
            files_touched = {}

            non_test_count = 0  # to check if not test-only commit, so we remove
            for refactor in commit["ref_list"]:
                files_touched_in_refactor = set()
                if refactor['test'] == 'True':
                    continue
                else:
                    non_test_count += 1
                for hunk in json.loads(refactor['leftSideLocations']):
                    filePath = hunk['filePath']
                    # for num times file has been touched by refactors
                    files_touched_in_refactor.add(filePath)

                    if filePath in files_touched:
                        continue
                    if filePath in file_list:
                        # file_list[filePath]['num_touched'] += 1

                        commit_date = parser.parse(commit['data']['author_date'])
                        previous_file_touched_date = parser.parse(file_list[filePath]['last_touched'])
                        file_age = (commit_date - previous_file_touched_date).days

                        files_touched[filePath] = {"age": file_age}
                    else:
                        files_touched[filePath] = {"age": 0}

                for hunk in json.loads(refactor['rightSideLocations']):
                    filePath = hunk['filePath']
                    # for num times file has been touched by refactors
                    files_touched_in_refactor.add(filePath)

                    if filePath in files_touched:
                        continue
                    if filePath in file_list:
                        # file_list[filePath]['num_touched'] += 1

                        commit_date = parser.parse(commit['data']['author_date'])
                        previous_file_touched_date = parser.parse(file_list[filePath]['last_touched'])
                        file_age = (commit_date - previous_file_touched_date).days

                        files_touched[filePath] = {"age": file_age}
                    else:
                        files_touched[filePath] = {"age": 0}

                for file in files_touched_in_refactor:
                    if 'num_touched' not in files_touched[file]:
                        files_touched[file]['num_touched'] = 1
                    else:
                        files_touched[file]['num_touched'] += 1

            if non_test_count == 0:  # skip test-only commit
                continue

            average_age = np.average([file['age'] for path, file in files_touched.items()])
            avg_touched_before = np.average([file_list[path]["num_touched"]
                                             for path, file in files_touched.items() if path in file_list])

            average_age = 0 if np.isnan(average_age) else average_age
            avg_touched_before = 0 if np.isnan(avg_touched_before) else avg_touched_before

            dev_ref_exp = 0
            dev_ref_com_exp = 0
            author_id = commit['data']['author_id']
            if author_id in developer_list:
                dev_ref_exp = developer_list[author_id]['dev_ref_exp']
                dev_ref_com_exp = developer_list[author_id]['dev_ref_com_exp']

                developer_list[author_id]['dev_ref_exp'] += len([rf for rf in commit["ref_list"]
                                                                 if rf["test"] != "True"])
                developer_list[author_id]['dev_ref_com_exp'] += 1
            else:
                developer_list[author_id] = {}
                developer_list[author_id]['dev_ref_exp'] = len([rf for rf in commit["ref_list"]
                                                                if rf["test"] != "True"])
                developer_list[author_id]['dev_ref_com_exp'] = 1

            cmt = {"id": commit["data"]["_id"], "num_touched_before": avg_touched_before, "age": average_age,
                   "dev_ref_exp": dev_ref_exp, "dev_ref_com_exp": dev_ref_com_exp}
            the_commit_list.append(cmt)
            for file, stats in files_touched.items():
                # adding number of times files are touched by refactoring
                if file in file_list:
                    file_list[file]["num_touched"] += stats['num_touched']
                else:
                    file_list[file] = {}
                    file_list[file]["num_touched"] = stats['num_touched']

                # updating last time touched by this file
                file_list[file]['last_touched'] = commit['data']['author_date']

    with open('commit_prev_experience.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['id', 'num_touched_before', 'age', 'dev_ref_exp',
                                                     'dev_ref_com_exp'])
        writer.writeheader()
        for value in the_commit_list:
            writer.writerow(value)


def merge_features():
    features = []
    with open('commit_level_features.csv', newline="") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar="\"")
        commit_features = [row for row in reader]
    with open('commit_prev_experience.csv', newline="") as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',', quotechar="\"")
        prev_features = [row for row in reader]
        for commit in commit_features:
            for prev_feature in prev_features:
                if prev_feature["id"] == commit["commit_id"]:
                    features.append({**prev_feature, **commit})

    with open('commit_all_features.csv', 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=features[0].keys())
        writer.writeheader()
        for value in features:
            writer.writerow(value)


# run first with False to generate necessary files, the True to use them to remove the features.
removing_Features = True


def run_model():
    feature_regular_df = pd.read_csv("commit_all_features.csv")
    commits_info = pd.read_csv("refactoring_commits.csv")
    features = pd.merge(feature_regular_df, commits_info, on="commit_id")
    features = features.drop(['commit_id', "id", "test", "revision_hash"], axis=1)
    features['label'] = features['label'].apply(lambda x: True if x == "co-occur" else False)
    features = features.groupby("url")

    features = sorted(features, key=lambda x: len(x[1].loc[x[1]['label'] == True].index),
                      # /len(x[1].index), reverse=True)
                      reverse=True)
    # for proj in features:
    #
    #     print(proj[0])
    #     print(proj[1])
    # projects = [(group[0], group[1].drop(['url'], axis=1)) for group in features]

    tot_metrics = {}  # {"auc": [], "f1": [], "accuracy": [], "recall": [], "precision": []}
    for project_to_test in features[:10]:
        if len(project_to_test[1].index) < 300:
            continue
        url = project_to_test[0]
        project_to_test = project_to_test[1]
        found_co_occur = 0
        for i, commit in project_to_test.iterrows():
            if commit[-2]:
                found_co_occur += 1
        if found_co_occur < 10:
            print("Skipping no co-occuring commits: ")
            print(project_to_test)
            continue

        if not removing_Features:
            features_to_remove = []
            # with open(url.split('/')[-1], newline='') as csvfile:
            with open(url.split('/')[-1] + "no_perm", newline='') as csvfile:
                reader = csv.DictReader(csvfile, delimiter=',', quotechar="\"")
                val = {}
                for row in reader:
                    val = row
                for feat, feat_importance in val.items():
                    # print(feat)
                    # print(float(feat_importance))
                    if float(feat_importance) <= -0.001: #feature importance threshold was found to be best at -0.001
                        features_to_remove.append(feat)
            project_to_test = project_to_test.drop(features_to_remove, axis=1)
            print(features_to_remove)
        project_to_test = project_to_test.drop(['url'], axis=1)
        X, y = project_to_test.iloc[:, :-1], project_to_test.iloc[:, -1]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)
        # Create the grid of parameters
        param_grid = {
            'bootstrap': [True],
            'max_depth': [50, 100, 150],
            'max_features': ['auto', 'sqrt', 'log2'],
            'n_estimators': [50, 100, 200, 300]
        }

        # # Create a based model
        rf = RandomForestClassifier()

        # Instantiate the grid search model
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid,
                                   cv=3, n_jobs=2, verbose=0)

        # Perform search and print the best parameters
        grid_search.fit(X_train, y_train)
        print(grid_search.best_params_)

        my_features = [c for c in project_to_test.columns if c != 'label']
        print(my_features)
        target = 'label'

        kfolds = KFold(n_splits=10, shuffle=True)

        metrics = {"auc": [], "f1": [], "accuracy": [], "recall": [], "precision": [], "importances": []}

        for train_idx, test_idx in kfolds.split(X):
            model = RandomForestClassifier(bootstrap=grid_search.best_params_['bootstrap'],
                                           n_estimators=grid_search.best_params_['n_estimators'],
                                           max_depth=grid_search.best_params_['max_depth'],
                                           max_features=grid_search.best_params_['max_features'])

            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_test, y_test = X.iloc[test_idx], y.iloc[test_idx]

            # counter = Counter(y_train)
            # undersample = RandomOverSampler(sampling_strategy='minority')
            oversample = SMOTE()
            X_train, y_train = oversample.fit_resample(X_train, y_train)
            # counter = Counter(y_train)

            # instantiate the model (using the default parameters)
            # logreg = LogisticRegression(solver='liblinear')
            # logreg.fit(X_train, y_train)
            # preds = logreg.predict_proba(X_test)[::, 1]

            # clf = svm.SVC(max_iter=1500)
            # clf.fit(X_train, y_train)
            # preds = clf.predict(X_test)

            # clf = DecisionTreeClassifier()
            # clf.fit(X_train, y_train)
            # preds = clf.predict(X_test)

            # xgb_clf = XGBClassifier()

            # fit the classifier to the training data
            # xgb_clf.fit(X_train, y_train)
            # preds = xgb_clf.predict(X_test)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            if removing_Features:
                # result = permutation_importance(
                #     model, X_test, y_test, n_repeats=10, random_state=42, n_jobs=2
                # )
                result = model.feature_importances_

                # metrics['importances'].append(result.importances_mean)
                metrics['importances'].append(result)

            metrics["auc"].append(roc_auc_score(y_test, preds))
            metrics["f1"].append(f1_score(y_test, preds))
            metrics["accuracy"].append(accuracy_score(y_test, preds))
            metrics["recall"].append(recall_score(y_test, preds))
            metrics["precision"].append(precision_score(y_test, preds))

        project_metrics = {"auc": st.mean(metrics["auc"]), "f1": st.mean(metrics["f1"]),
                           "accuracy": st.mean(metrics["accuracy"]), "recall": st.mean(metrics["recall"]),
                           "precision": st.mean(metrics["precision"])}
        print(project_metrics)
        print(url)
        tot_metrics[url] = project_metrics
        if removing_Features:
            k = []
            for val in metrics["importances"][0]:
                k.append([])
            for importance in metrics["importances"]:
                for index, val in enumerate(importance):
                    k[index].append(val)
            feature_importance_means = []
            for valu in k:
                feature_importance_means.append(st.mean(valu))
            # with open(url.split("/")[-1], 'w', newline='') as csvfile:
            with open(url.split("/")[-1] + "no_perm", 'w', newline='') as csvfile:
                writer = csv.writer(csvfile, delimiter=',', quotechar='"', lineterminator='\n')
                writer.writerow(my_features)
                writer.writerow(feature_importance_means)

    print(tot_metrics)
    print('mean AUC: {:.04f}'.format(st.mean([proj["auc"] for proj in tot_metrics.values()])))
    print('mean f1: {:.04f}'.format(st.mean([proj["f1"] for proj in tot_metrics.values()])))
    print('mean accuracy: {:.04f}'.format(st.mean([proj["accuracy"] for proj in tot_metrics.values()])))
    print('mean recall: {:.04f}'.format(st.mean([proj["recall"] for proj in tot_metrics.values()])))
    print('mean precision: {:.04f}'.format(st.mean([proj["precision"] for proj in tot_metrics.values()])))

def plot_project_model_results():
    # obtained from run_model print statement
    stats = {'https://github.com/apache/kafka.git': {'auc': 0.7816685229752448, 'f1': 0.755403639886282,
                                                     'accuracy': 0.7851420586865394, 'recall': 0.7273821841181239,
                                                     'precision': 0.7895994698760656},
             'https://github.com/apache/archiva.git': {'auc': 0.7502614645590333, 'f1': 0.6703518335612324,
                                                       'accuracy': 0.7836836283185841, 'recall': 0.6431902316955522,
                                                       'precision': 0.7029098883195387},
             'https://github.com/apache/activemq.git': {'auc': 0.6722934626689445, 'f1': 0.4846959445092124,
                                                        'accuracy': 0.8367828304292393, 'recall': 0.3950969660075606,
                                                        'precision': 0.6569778948909384},
             'https://github.com/apache/kylin.git': {'auc': 0.6930082256535932, 'f1': 0.5226453729045557,
                                                     'accuracy': 0.8332884207088029, 'recall': 0.45885493553608453,
                                                     'precision': 0.624700777676621},
             'https://github.com/apache/phoenix.git': {'auc': 0.9240199617153595, 'f1': 0.9060106732698068,
                                                       'accuracy': 0.9831014890282131, 'recall': 0.8508208786208155,
                                                       'precision': 0.9714718040580109},
             'https://github.com/apache/tez.git': {'auc': 0.7728664958084609, 'f1': 0.6883951840509218,
                                                   'accuracy': 0.8004591265397536, 'recall': 0.6901107065371771,
                                                   'precision': 0.6963240728190918},
             'https://github.com/apache/helix.git': {'auc': 0.6946995174062658, 'f1': 0.581404941846765,
                                                     'accuracy': 0.7540969899665552, 'recall': 0.5252218420766808,
                                                     'precision': 0.6575912820864194},
             'https://github.com/apache/commons-math': {'auc': 0.7710931697288462, 'f1': 0.6871910930975202,
                                                        'accuracy': 0.8135416666666667, 'recall': 0.6600035333909573,
                                                        'precision': 0.728217219458425},
             'https://github.com/apache/mahout': {'auc': 0.7281971884678619, 'f1': 0.6513511803795621,
                                                  'accuracy': 0.7604395604395604, 'recall': 0.6092486645601252,
                                                  'precision': 0.7033060612550868},
             'https://github.com/apache/nifi.git': {'auc': 0.6859778989218263, 'f1': 0.5449502091793551,
                                                    'accuracy': 0.7780769230769231, 'recall': 0.48340888502178825,
                                                    'precision': 0.6442966912165323}}
    the_stats = {'auc': [], 'f1': [], 'recall': [], 'precision': []}
    for statistic in stats.values():
        for metric in the_stats.keys():
            the_stats[metric].append(statistic[metric])
    data = [the_stats[metric] for metric in the_stats.keys()]
    fig = plt.figure(figsize=(6, 2))
    # #a3a1cb # logan #A1AFCB #rock blue
    boxprops = dict(color="black", linewidth=1, facecolor="#A0AECA")
    medianprops = dict(color="black", linewidth=1)
    # scaling data points to be on a percentage scale
    bplot = plt.boxplot([[dp*100 for dp in box] for box in data], showfliers=True, vert=False, boxprops=boxprops,
                medianprops=medianprops, patch_artist=True)
    plt.yticks(range(1, len(data) + 1), ['AUC', 'F1 Score', 'Recall', 'Precision'])
    plt.xlabel("Percentage")
    plt.savefig("model-eval.pdf", bbox_inches="tight")

def plot_info_gain():
    stats = {'https://github.com/apache/kafka.git': {'auc': 0.7816685229752448, 'f1': 0.755403639886282,
                                                     'accuracy': 0.7851420586865394, 'recall': 0.7273821841181239,
                                                     'precision': 0.7895994698760656},
             'https://github.com/apache/archiva.git': {'auc': 0.7502614645590333, 'f1': 0.6703518335612324,
                                                       'accuracy': 0.7836836283185841, 'recall': 0.6431902316955522,
                                                       'precision': 0.7029098883195387},
             'https://github.com/apache/activemq.git': {'auc': 0.6722934626689445, 'f1': 0.4846959445092124,
                                                        'accuracy': 0.8367828304292393, 'recall': 0.3950969660075606,
                                                        'precision': 0.6569778948909384},
             'https://github.com/apache/kylin.git': {'auc': 0.6930082256535932, 'f1': 0.5226453729045557,
                                                     'accuracy': 0.8332884207088029, 'recall': 0.45885493553608453,
                                                     'precision': 0.624700777676621},
             'https://github.com/apache/phoenix.git': {'auc': 0.9240199617153595, 'f1': 0.9060106732698068,
                                                       'accuracy': 0.9831014890282131, 'recall': 0.8508208786208155,
                                                       'precision': 0.9714718040580109},
             'https://github.com/apache/tez.git': {'auc': 0.7728664958084609, 'f1': 0.6883951840509218,
                                                   'accuracy': 0.8004591265397536, 'recall': 0.6901107065371771,
                                                   'precision': 0.6963240728190918},
             'https://github.com/apache/helix.git': {'auc': 0.6946995174062658, 'f1': 0.581404941846765,
                                                     'accuracy': 0.7540969899665552, 'recall': 0.5252218420766808,
                                                     'precision': 0.6575912820864194},
             'https://github.com/apache/commons-math': {'auc': 0.7710931697288462, 'f1': 0.6871910930975202,
                                                        'accuracy': 0.8135416666666667, 'recall': 0.6600035333909573,
                                                        'precision': 0.728217219458425},
             'https://github.com/apache/mahout': {'auc': 0.7281971884678619, 'f1': 0.6513511803795621,
                                                  'accuracy': 0.7604395604395604, 'recall': 0.6092486645601252,
                                                  'precision': 0.7033060612550868},
             'https://github.com/apache/nifi.git': {'auc': 0.6859778989218263, 'f1': 0.5449502091793551,
                                                    'accuracy': 0.7780769230769231, 'recall': 0.48340888502178825,
                                                    'precision': 0.6442966912165323}}
    file_names = [url.split('/')[-1] + "no_perm" for url in stats.keys()]
    feat_imp_per_project = []
    for f in file_names:
        with open(f, newline='') as csvfile:
            reader = csv.DictReader(csvfile, delimiter=',', quotechar="\"")
            val = {}
            for row in reader:
                feat_imp_per_project.append(row)

    tot_importances = {}
    feat_imp_list = {}
    for feat_import in feat_imp_per_project:
        for feat, importance in feat_import.items():
            importance = float(importance)
            if feat not in tot_importances:
                tot_importances[feat] = importance
                feat_imp_list[feat] = [importance]
            else:
                tot_importances[feat] += importance
                feat_imp_list[feat].append(importance)
    tot_importances = [{"name": feat, "importance": importance} for feat, importance in tot_importances.items()]
    top_importances = sorted(tot_importances, key=lambda x: x["importance"], reverse=True)[:10]

    data = [feat_imp_list[feat["name"]] for feat in top_importances]
    print(data)
    found_titles = True  # set manually to True once the top features are found, otherwise set to False
    feature_names = [feat["name"] for feat in top_importances] if not found_titles else \
        ["LOC Left Side", "LOC Right Side", "Dev Refactoring Experience", "# Previous Refactorings",
         "Dev Refactoring Commit Experience", "Method Declaration", "Single Variable Declaration", "# of Refactorings",
         "Age", "# of Files"]
    fig = plt.figure(figsize=(6, 4))
    boxprops = dict(color="black", linewidth=1, facecolor="#A0AECA")
    medianprops = dict(color="black", linewidth=1)
    plt.boxplot(data, showfliers=True, vert=False, boxprops=boxprops, medianprops=medianprops, patch_artist=True)
    plt.yticks(range(1, len(data) + 1), [re.sub(r'((\w+|#) \w+ )', '\\1\n', feature) for feature in feature_names])
    plt.gca().invert_yaxis()
    plt.xlabel("Mean decrease in impurity")
    plt.savefig("feat-importance.pdf", bbox_inches="tight")

if __name__ == '__main__':
    # step 1
    # (before this step, export the commit collection to commit.csv in this folder from the smart shark dataset v2.1)
    #temp_method()
    # step 2
    #merge_features()
    # step 3
    run_model()
    # copy the output of the model to the stats variables in the methods of step 4 and 5.
    # note: all other model configurations are commented out, simply comment and uncomment the appropriate
    # models for the model comparison phase.
    # step 4
    plot_project_model_results()
    # step 5
    plot_info_gain()
